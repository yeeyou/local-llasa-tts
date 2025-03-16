import os
from pathlib import Path
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import torch
import torchaudio
import tempfile
import os
import logging
import sys
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from xcodec2.modeling_xcodec2 import XCodec2Model

# 修改日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tts_service.log', encoding='utf-8')
    ],
    force=True  # 强制重新配置日志
)

# 移除uvicorn logger配置，避免重复日志
logger = logging.getLogger(__name__)

# 创建输出目录
OUTPUT_DIR = Path.home() / "tts_out"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI()

# 模型加载
logger.info("=================== 开始加载模型 ===================")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# 修改为本地模型路径
LOCAL_MODEL_PATH = "./models/Llasa-8b"
LOCAL_CODEC_PATH = "./models/xcodec2"

logger.info("加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
logger.info("加载 LLASA 模型...")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    device_map='cuda',
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)

logger.info("加载 XCodec2 模型...")
Codec_model = XCodec2Model.from_pretrained(LOCAL_CODEC_PATH)
Codec_model.eval().cuda()

logger.info("加载 Whisper 模型...")
whisper_turbo_pipe = pipeline(
    "automatic-speech-recognition",
    model="./models/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device='cuda',
)
logger.info("所有模型加载完成！")

def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

async def process_tts(audio_path: str, target_text: str):
    try:
        logger.info(f"开始处理音频，目标文本长度：{len(target_text)}")
        waveform, sample_rate = torchaudio.load(audio_path)

        if len(waveform[0])/sample_rate > 15:
            logger.info("音频长度超过15秒，将被截断")
            waveform = waveform[:, :sample_rate*15]

        if waveform.size(0) > 1:
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform_mono = waveform

        prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)
        logger.info("正在进行语音识别...")
        prompt_text = whisper_turbo_pipe(prompt_wav[0].numpy())['text'].strip()
        logger.info(f"语音识别结果: {prompt_text}")

        if len(target_text) == 0:
            raise HTTPException(status_code=400, detail="Target text cannot be empty")
        elif len(target_text) > 300:
            target_text = target_text[:300]

        input_text = prompt_text + ' ' + target_text

        with torch.no_grad():
            vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
            vq_code_prompt = vq_code_prompt[0,0,:]
            speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
            ]

            input_ids = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                return_tensors='pt',
                continue_final_message=True
            ).cuda()

            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            outputs = model.generate(
                input_ids,
                max_length=2048,
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=1,
                temperature=0.8
            )

            generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
            gen_wav = Codec_model.decode_code(speech_tokens)
            gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

            logger.info("语音生成完成")
            return gen_wav[0, 0, :].cpu().numpy()

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 添加允许的音频格式
ALLOWED_FORMATS = {'.wav', '.mp3'}

@app.post("/tts/")
async def create_tts(
    audio: UploadFile = File(...),
    text: str = Form(default="")
):
    temp_audio_path = None
    try:
        logger.info("="*50)
        logger.info(f"收到新的TTS请求:")
        logger.info(f"音频文件: {audio.filename}")
        logger.info(f"原始文本内容: '{text}'")

        # 验证音频文件
        file_extension = Path(audio.filename).suffix.lower()
        if file_extension not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的音频格式。支持的格式: {', '.join(ALLOWED_FORMATS)}"
            )

        # 验证文本
        text = text.strip()

        # 保存上传的音频文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        try:
            logger.info("开始处理 TTS...")
            audio_array = await process_tts(temp_audio_path, text)

            # 使用时间戳创建唯一的输出文件名
            output_filename = f"output_{int(time.time())}.wav"
            output_path = OUTPUT_DIR / output_filename

            logger.info(f"生成音频文件: {output_path}")
            import soundfile as sf
            sf.write(str(output_path), audio_array, 16000)

            logger.info("TTS 处理完成，返回音频文件")
            return FileResponse(
                str(output_path),
                media_type="audio/wav",
                filename=output_filename
            )

        except Exception as e:
            logger.error(f"TTS 处理失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"TTS 处理失败: {str(e)}")

    finally:
        # 只清理临时音频文件
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"清理临时文��失败: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*50)
    logger.info("TTS 服务正在启动...")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info("API文档地址: http://localhost:8008/docs")
    print("="*50 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8008,
        log_level="info",
        access_log=False  # 禁用访问日志以避免重复
    )
