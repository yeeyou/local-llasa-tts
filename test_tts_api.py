import os
import httpx
import asyncio
from pathlib import Path

async def test_tts_api(audio_path, text):
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # API端点
    url = "http://localhost:8008/tts/"
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            print(f"发送请求到 {url}")
            print(f"音频文件: {audio_path}")
            print(f"文本内容: {text}")
            
            files = {
                'audio': open(audio_path, 'rb'),
                'text': (None, text)
            }
            
            response = await client.post(url, files=files)
            
            if response.status_code == 200:
                content_disposition = response.headers.get('content-disposition', '')
                filename = content_disposition.split('filename=')[-1].strip('"\'')
                if not filename:
                    filename = f"output_{Path(audio_path).stem}.wav"
                
                output_path = output_dir / filename.strip('"')
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"生成的音频已保存到: {output_path}")
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                    
        except Exception as e:
            print(f"发生错误: {str(e)}")
        finally:
            files['audio'].close()

async def main():
    # 测试参数
    audio_path = "input/elonmusk.wav"  # 修改为你的音频文件路径
    text = "Audio longer than 10s will be truncated due to computing resources"
    
    await test_tts_api(audio_path, text)

if __name__ == "__main__":
    asyncio.run(main())