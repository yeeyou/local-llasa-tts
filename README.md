# local-llasa-tts
Examples of using the llasa-tts models locally

# Colab
<a href="https://colab.research.google.com/github/nivibilla/local-llasa-tts/blob/main/llasa_tts.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Local - Native HF Transformers Inference (4 Bit with nvidia gpu)
Runs a gradio app similar to https://huggingface.co/spaces/srinivasbilla/llasa-3b-tts
```
%sh
cd /local_disk0
git clone https://github.com/nivibilla/local-llasa-tts.git
pip install -r ./requirements_native_hf.txt
pip install -r ./requirements_base.txt
python ./hf_app.py
```
