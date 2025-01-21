# local-llasa-tts
Examples of using the llasa-tts models locally

# System Reqs
- If you cant run it locally use the colab notebook below
- It takes 8.5GB of vram if you are also loading whisper large turbo with the llm in 4bit
- 6.5 GB of vram if you dont load whisper and the llm in 4bit
- you could also use the 1B model but I havent tested it.

# Colab
<a href="https://colab.research.google.com/github/nivibilla/local-llasa-tts/blob/main/colab_notebook_4bit.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Local

## Native HF Transformers Inference (4 Bit with nvidia gpu)
Runs a gradio app similar to https://huggingface.co/spaces/srinivasbilla/llasa-3b-tts
```
%sh
cd /local_disk0
git clone https://github.com/nivibilla/local-llasa-tts.git
pip install -r ./requirements_native_hf.txt
pip install -r ./requirements_base.txt
python ./hf_app.py
```
