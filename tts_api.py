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
import numpy as np
import sys
import uvicorn
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from xcodec2.modeling_xcodec2 import XCodec2Model

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="启动 LLASA TTS API 服务")
parser.add_argument("--model", choices=["3b", "8b"], default="3b", help="选择 LLASA 模型版本 (3b 或 8b)")
parser.add_argument("--port", type=int, default=8008, help="API 服务端口")
parser.add_argument("--host", default="0.0.0.0", help="API 服务地址")
parser.add_argument("--output-dir", default=str(Path.home() / "tts_out"), help="输出音频保存目录")
args = parser.parse_args()

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
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI()

# 根据命令行参数选择模型路径
MODEL_VERSION_MAP = {
    "3b": "./models/llasa-3b",
    "8b": "./models/Llasa-8B"
}

# 获取所选模型路径
LOCAL_MODEL_PATH = MODEL_VERSION_MAP[args.model]
LOCAL_CODEC_PATH = "./models/xcodec2"

# 模型加载
logger.info("=================== 开始加载模型 ===================")
logger.info(f"选择的模型版本: {args.model} (路径: {LOCAL_MODEL_PATH})")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

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

        # 优化短文本处理 - 确保目标文本不超过300字符
        if len(target_text) == 0:
            raise HTTPException(status_code=400, detail="Target text cannot be empty")
        elif len(target_text) > 300:
            logger.info(f"文本超过300字符，截断至300字符")
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

            # 计算合适的生成长度
            input_len = input_ids.shape[1]
            margin = 100  # 参考音频模式使用100的margin
            dynamic_max_length = (input_len + margin) * 1.5
            
            logger.info(f"动态生成长度: {dynamic_max_length} tokens (输入长度: {input_len})")

            speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            outputs = model.generate(
                input_ids,
                max_length=dynamic_max_length,
                min_length=int(dynamic_max_length * 0.7),  # 设置最小长度为最大长度的一半
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=1.0,     # 使用默认温度
                top_p=1.0,          # 使用默认 top_p
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

            generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
            gen_wav = Codec_model.decode_code(speech_tokens)
            gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

            # 获取解码后的音频
            audio_array = gen_wav[0, 0, :].cpu().numpy()
            
            # 短文本音频后处理 - 使用更严格的截断参数
            # audio_array = trim_silence_end(audio_array, threshold=0.008, buffer_duration=0.15)
            
            logger.info("语音生成完成")
            return audio_array

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 修改 trim_silence_end 函数，支持更精确的缓冲区控制
def trim_silence_end(audio_array, threshold=0.01, buffer_duration=0.2, sample_rate=16000):
    """
    修剪音频末尾的静音或杂音 - 优化版
    
    参数:
        audio_array: 音频数据数组
        threshold: 振幅阈值，低于此值视为静音
        buffer_duration: 留下的缓冲区长度（秒）
        sample_rate: 采样率
    """
    # 计算音频能量
    frame_length = int(0.02 * sample_rate)  # 20ms帧
    hop_length = int(0.01 * sample_rate)    # 10ms跳跃
    
    # 如果音频太短，直接返回
    if len(audio_array) < frame_length:
        return audio_array
    
    # 计算帧能量
    energy = []
    for i in range(0, len(audio_array) - frame_length, hop_length):
        frame = audio_array[i:i+frame_length]
        energy.append(np.sqrt(np.mean(frame**2)))
    
    energy = np.array(energy)
    
    # 找到最后一个高于阈值的帧
    active_frames = np.where(energy > threshold)[0]
    
    if len(active_frames) == 0:
        # 如果没有活跃帧，返回前1秒的音频（如果有）
        return audio_array[:min(len(audio_array), sample_rate)]
    
    last_active = active_frames[-1]
    
    # 确定截断位置（最后一个活跃帧后加上指定的缓冲区）
    buffer_frames = int(buffer_duration * sample_rate / hop_length)
    end_frame = min(last_active + buffer_frames, len(energy) - 1)
    
    # 转换回样本索引
    end_sample = (end_frame + 1) * hop_length
    
    # 确保不超出音频长度
    end_sample = min(end_sample + frame_length, len(audio_array))
    
    # 在确定的位置截断音频
    return audio_array[:end_sample]

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
                logger.warning(f"清理临时文件失败: {str(e)}")

if __name__ == "__main__":
    print("\n" + "="*50)
    logger.info("TTS 服务正在启动...")
    logger.info(f"LLASA 模型版本: {args.model}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"API文档地址: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print("="*50 + "\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False  # 禁用访问日志以避免重复
    )
