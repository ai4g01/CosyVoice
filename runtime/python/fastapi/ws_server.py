# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import os
import re
import sys
import argparse
import logging
import threading
import time
import uuid
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
# from cosyvoice.cli.rt_cosyvoice import RTCosyVoice2
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

app = FastAPI()

@app.get("/")
async def index():
    return FileResponse('runtime/python/fastapi/index.html')


@app.get("/prompt_voice/{type}")
async def get_prompt_voice(type):
    if type == "sft":
        return cosyvoice.list_available_spks()
    else:
        return await promptVoice.get_list()


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    synthesizer = SpeechSynthesizer(websocket)
    try:
        await websocket.accept()
        while True:
            data = await websocket.receive_json()
            if data['action'] == 'start':
                await synthesizer.start(data)
            elif data['action'] == 'put_text':
                await synthesizer.put_text(data)
            elif data['action'] == 'finish':
                await synthesizer.finish()

    except WebSocketDisconnect:
        logger.info("Websocket disconnected")
    except Exception as e:
        websocket.send_json({
            "event" : "error",
            "message": str(e),
        })
        logger.exception(e)
    finally:
        await synthesizer.finish()
        logger.info("Websocket connection closed")

class PromptVoice:
    def __init__(self):
        self.prompt_voice = json.load(open('asset/prompt_voice.json'))
        
    async def get_list(self):
        return list(self.prompt_voice.keys())

    def get(self, spk_id):
        if spk_id in self.prompt_voice:
            return self.prompt_voice[spk_id]["text"], load_wav(self.prompt_voice[spk_id]["path"], 16000)
        raise Exception(f"{spk_id} not found")

promptVoice = PromptVoice()

class SpeechSynthesizer:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._uuid = str(uuid.uuid1())
        self._input_text = ""
        self._intput_text_lock = threading.Lock()
        self._finished = False

    async def start(self, data: dict):
        self._finished = False
        self._thread = threading.Thread(target=self._run, args=(data,))
        self._thread.start()

    async def close(self):
        pass

    async def finish(self):        
        self._finished = True

    async def put_text(self, data: dict):
        with self._intput_text_lock:
            self._input_text += data["text"]

    def _run(self, data: dict):
        logger.info(f"{self._uuid} started, {data}")
        asyncio.run(self.websocket.send_json({"event": "started", "uuid": self._uuid}))  #
        
        remaining_text = ""
        def get_input_text():
            with self._intput_text_lock:
                text = remaining_text + self._input_text
                self._input_text = ""
                return text

        while not self._finished:
            text = get_input_text()
            end_index = self.find_last_punctuations(text)  ##找到最后一个标点符号
            if end_index > 5:
                tts_text = text[:end_index]
                remaining_text = text[end_index:]
                data["text"] = tts_text
                self._synthesis(data)
            else:
                remaining_text = text

            time.sleep(0.01)

        data["text"] = get_input_text()  ## 剩余的文本
        if len(data["text"]) > 0:
            self._synthesis(data)

        asyncio.run(self.websocket.send_json({"event": "finished", "uuid": self._uuid}))
        self._thread = None
        logger.info(f"{self._uuid} finished")

    def find_last_punctuations(self, s):
        punctuation = r'.,?!"【】。，；：？！～…'                      
        match = re.search(f"[{punctuation}]", s[::-1])
        if match:
            return len(s) - match.start() - 1
        else:
            return - 1

    def _synthesis(self, data: dict):
        # logger.info(f"{self._uuid} synthesis, {data}")
        set_all_random_seed(42)
        if data["method"] == "sft":
            model_output = cosyvoice.inference_sft(tts_text=data['text'], spk_id=data["spk_id"], stream=True)
        elif data["method"] == "zero_shot":
            prompt_text, prompt_speech_16k = promptVoice.get(data["spk_id"])
            model_output = cosyvoice.inference_zero_shot(data['text'], prompt_text, prompt_speech_16k, stream=True)               
        elif data["method"] == "cross_lingual":
            _, prompt_speech_16k = promptVoice.get(data["spk_id"])
            model_output = cosyvoice.inference_cross_lingual(data['text'], prompt_speech_16k, stream=True)                
        elif data["method"] == "instruct2":
            _, prompt_speech_16k = promptVoice.get(data["spk_id"])
            model_output = cosyvoice.inference_instruct2(data['text'], data["instruct"], prompt_speech_16k, stream=True)

        for i in model_output:
            tts_audio = (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
            asyncio.run(self.websocket.send_bytes(tts_audio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8100)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
