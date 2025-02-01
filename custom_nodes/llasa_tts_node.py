import torch
import torchaudio
import numpy as np
import os
import sys
import folder_paths

# 添加依赖检查和导入处理
def import_dependencies():
    try:
        global AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, XCodec2Model
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        from xcodec2.modeling_xcodec2 import XCodec2Model
    except ImportError:
        print("[LLaSA TTS] 正在安装必要依赖...")
        import subprocess
        requirements_path = os.path.join(os.path.dirname(__file__), "llasa_requirements.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        
        # 重新导入
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        from xcodec2.modeling_xcodec2 import XCodec2Model

class LLaSATTS:
    def __init__(self):
        self.loaded_models = False
        self.tokenizer = None
        self.model = None
        self.codec_model = None
        self.whisper_pipe = None
        # 初始化时导入依赖
        import_dependencies()
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_tts"
    CATEGORY = "audio"

    def load_models(self):
        if not self.loaded_models:
            # 设置模型路径
            models_dir = os.path.join(folder_paths.base_path, "models")
            local_model_path = os.path.join(models_dir, "llasa-3b")
            local_codec_path = os.path.join(models_dir, "xcodec2")
            local_whisper_path = os.path.join(models_dir, "whisper-large-v3-turbo")

            # 加载模型
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                device_map='cuda',
                quantization_config=quantization_config,
                low_cpu_mem_usage=True
            )
            
            self.codec_model = XCodec2Model.from_pretrained(local_codec_path)
            self.codec_model.eval().cuda()
            
            self.whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=local_whisper_path,
                torch_dtype=torch.float16,
                device='cuda',
            )
            
            self.loaded_models = True

    def generate_tts(self, audio_path: str, text: str):
        self.load_models()
        
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)
            if len(waveform[0])/sample_rate > 15:
                waveform = waveform[:, :sample_rate*15]

            if waveform.size(0) > 1:
                waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
            else:
                waveform_mono = waveform

            prompt_wav = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=16000
            )(waveform_mono)
            
            # 使用与 tts_api.py 相同的处理逻辑
            prompt_text = self.whisper_pipe(prompt_wav[0].numpy())['text'].strip()

            if len(text) == 0:
                raise ValueError("Target text cannot be empty")
            elif len(text) > 300:
                text = text[:300]
                
            input_text = prompt_text + ' ' + text

            with torch.no_grad():
                vq_code_prompt = self.codec_model.encode_code(input_waveform=prompt_wav)
                vq_code_prompt = vq_code_prompt[0,0,:]
                speech_ids_prefix = self.ids_to_speech_tokens(vq_code_prompt)

                formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

                chat = [
                    {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                    {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
                ]

                input_ids = self.tokenizer.apply_chat_template(
                    chat, 
                    tokenize=True, 
                    return_tensors='pt', 
                    continue_final_message=True
                ).cuda()

                speech_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
                outputs = self.model.generate(
                    input_ids,
                    max_length=2048,
                    eos_token_id=speech_end_id,
                    do_sample=True,
                    top_p=1,           
                    temperature=0.8
                )

                generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
                speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   
                speech_tokens = self.extract_speech_ids(speech_tokens)
                speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
                gen_wav = self.codec_model.decode_code(speech_tokens) 
                gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

            # 保存生成的音频
            output_dir = os.path.join(folder_paths.get_output_directory(), "llasa_tts")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, "generated_audio.wav")
            import soundfile as sf
            sf.write(output_path, gen_wav[0, 0, :].cpu().numpy(), 16000)
            
            return (output_path,)
            
        except Exception as e:
            print(f"Error in TTS generation: {str(e)}")
            return (None,)

    def ids_to_speech_tokens(self, speech_ids):
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return speech_tokens_str

    def extract_speech_ids(self, speech_tokens_str):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith('<|s_') and token_str.endswith('|>'):
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            else:
                print(f"Unexpected token: {token_str}")
        return speech_ids