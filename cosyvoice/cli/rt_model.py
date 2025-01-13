import re
from .model import CosyVoice2Model
import torch
import threading
import time
import uuid
from cosyvoice.utils.file_utils import logging


class RTCosyVoice2Model(CosyVoice2Model):
    '''
    双向流式支持
    
    '''
    def __init__(
        self, llm: torch.nn.Module, flow: torch.nn.Module, hift: torch.nn.Module, fp16: bool, frontend
    ):
        super().__init__(llm, flow, hift, fp16)
        self.tts_speech_input_text = {}
        self.tts_speech_input_end = {}
        self.tts_speech_closed = {}
        self.frontend = frontend

    def find_last_punctuations(self, s):
        punctuation = r'.,?!"【】。，；：？！～…'
        match = re.search(f"[{punctuation}]", s[::-1])
        if match:
            return len(s) - match.start() - 1
        else:
            return len(s) - 1

    def llm_job(self, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        text_offset = 0
        last_text = ""
        sequence_min_len = 5
        while not self.tts_speech_input_end[uuid]:
            text_chunk = self.tts_speech_input_text[uuid][text_offset:]
            text = last_text + "".join(text_chunk)  ## 拼接上次剩余文本
            if len(text) > sequence_min_len:
                end_index = self.find_last_punctuations(text)  ##找到最后一个标点符号
                if end_index > sequence_min_len:
                    text = text[:end_index]
                    last_text = text[end_index:]
                else:
                    last_text = ""

                self.llm_inference(text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid)
                text_offset += len(text_chunk)
            time.sleep(0.01)

        if len(self.tts_speech_input_text[uuid]) > text_offset and not self.tts_speech_closed[uuid]:
            text_chunk = self.tts_speech_input_text[uuid][text_offset:]
            text = last_text +"".join(text_chunk)
            self.llm_inference(text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid)

        self.llm_end_dict[uuid] = True

    def llm_inference(self, input_text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):        
        with self.llm_context:
            logging.info(f"Prepare to normalize: {input_text}")
            for text in self.frontend.text_normalize(input_text):
                logging.info(f"Been normalized: {text}")
                if self.tts_speech_closed[uuid]:
                    break
                text_token, _ = self.frontend._extract_text_token(text)
                for i in self.llm.inference(text=text_token.to(self.device),
                                            text_len=torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_text=prompt_text.to(self.device),
                                            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                            embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), speed=1.0, uuid=str(uuid.uuid1()), **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = uuid

        p = threading.Thread(target=self.llm_job, args=(prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()

        token_offset = 0
        while True:
            time.sleep(0.001)
            if self.tts_speech_closed[uuid]:
                break
            if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                    prompt_token=flow_prompt_speech_token,
                                                    prompt_feat=prompt_speech_feat,
                                                    embedding=flow_embedding,
                                                    uuid=this_uuid,
                                                    token_offset=token_offset,
                                                    finalize=False)
                token_offset += self.token_hop_len
                yield {'tts_speech': this_tts_speech.cpu()}
            if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < self.token_hop_len + self.flow.pre_lookahead_len:
                break
        p.join()
        # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=flow_embedding,
                                            uuid=this_uuid,
                                            token_offset=token_offset,
                                            finalize=True)
        yield {'tts_speech': this_tts_speech.cpu()}

        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.tts_speech_input_text.pop(this_uuid)
            self.tts_speech_input_end.pop(this_uuid)
            self.tts_speech_closed.pop(this_uuid)
        torch.cuda.empty_cache()
        
    def start(self, uuid):
        with self.lock:
            self.tts_speech_input_text[uuid], self.tts_speech_token_dict[uuid] = [], [] # 输入文本、 token队列
            self.tts_speech_input_end[uuid], self.llm_end_dict[uuid], self.hift_cache_dict[uuid] = False, False, None
            self.tts_speech_closed[uuid]=False

    def close(self, uuid):
        if uuid in self.tts_speech_closed:
            self.tts_speech_input_end[uuid] = True
            self.tts_speech_closed[uuid] = True

    def put_text(self, text, uuid):
        if uuid in self.tts_speech_input_text:
            self.tts_speech_input_text[uuid].append(text)

    def finish(self, uuid):
        self.tts_speech_input_end[uuid] = True
