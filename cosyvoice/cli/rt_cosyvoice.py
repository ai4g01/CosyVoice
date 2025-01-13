import os
import time
import uuid
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from .cosyvoice import CosyVoice2
from .rt_model import RTCosyVoice2Model
from cosyvoice.utils.file_utils import logging

class RTCosyVoice2(CosyVoice2):

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        #assert get_model_type(configs) == RTCosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = RTCosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16, self.frontend)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.v100.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        del configs
        
        
    def inference_sft(self, spk_id, speed=1.0, uuid=str(uuid.uuid1())):
        #for i in tqdm(self.frontend.text_normalize("sft", split=True, text_frontend=text_frontend)):
        start_time = time.time()
        speech_len = 0
        model_input = self.frontend.frontend_sft('sft', spk_id)
        for model_output in self.model.tts(**model_input, speed=speed, uuid=uuid):
            speech_len += model_output['tts_speech'].shape[1] / self.sample_rate
            yield model_output
        logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))

    def inference_zero_shot(self, prompt_text, prompt_speech_16k, speed=1.0, text_frontend=True, uuid=str(uuid.uuid1())):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        # if len(i) < 0.5 * len(prompt_text):
        #     logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
        speech_len = 0
        start_time = time.time()
        model_input = self.frontend.frontend_zero_shot('zero_shot', prompt_text, prompt_speech_16k, self.sample_rate)
        for model_output in self.model.tts(**model_input, speed=speed, uuid=uuid):
            speech_len += model_output['tts_speech'].shape[1] / self.sample_rate
            yield model_output
        logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))

    def inference_cross_lingual(self, prompt_speech_16k, speed=1.0, uuid=str(uuid.uuid1())):
        speech_len = 0
        start_time = time.time()
        model_input = self.frontend.frontend_cross_lingual('lingual', prompt_speech_16k, self.sample_rate)
        for model_output in self.model.tts(**model_input, speed=speed, uuid=uuid):
            speech_len += model_output['tts_speech'].shape[1] / self.sample_rate
            yield model_output
        logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))


    def inference_instruct2(self, instruct_text, prompt_speech_16k, speed=1.0, uuid=str(uuid.uuid1())):
        #for i in tqdm(self.frontend.text_normalize("instruct2", split=True, text_frontend=text_frontend)):
        speech_len = 0
        start_time = time.time()
        model_input = self.frontend.frontend_instruct2('instruct2', instruct_text, prompt_speech_16k, self.sample_rate)
        for model_output in self.model.tts(**model_input, speed=speed, uuid=uuid):
            speech_len += model_output['tts_speech'].shape[1] / self.sample_rate
            yield model_output
        logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')


    def streaming_start(self, uuid):
        self.model.start(uuid)
        return uuid
    
    

    def put_text(self, text, uuid):
        #for s in self.frontend.text_normalize(text):
        self.model.put_text(text=text, uuid=uuid)

    def streaming_finish(self, uuid):
        self.model.finish(uuid)
        
    def streaming_close(self, uuid):
        self.model.close(uuid)
