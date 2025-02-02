# Llasa TTS Dashboard

A powerful, local text-to-speech system powered by Llasa-tts models with a modern, interactive dashboard interface. This project supports both 3B and 8B models and features advanced caching, GPU optimizations, and customizable generation parameters.

---

## Overview

**Llasa TTS Dashboard** transforms the traditional TTS pipeline into a user-friendly, robust application. Key improvements include:

- **Revamped Dashboard UI:** A sleek, two-panel interface for synthesis and history, built with Gradio.
- **Multi-Model Support:** Easily switch between 3B and 8B models (the 1B model is available but not fully tested).
- **Enhanced Inference Options:** Generate speech from plain text or using a reference audio prompt.
- **Optimized Performance:** Model caching, efficient CUDA initialization, and precise GPU memory management ensure smooth performance.
- **Advanced Generation Controls:** Adjust parameters such as max length, temperature, top-p, and set a random seed for reproducibility.
- **Comprehensive Environment Setup:** Pre-configured for optimal HuggingFace cache management and progress reporting.

---

## System Requirements

- **VRAM Requirements:**
  - **8.5 GB VRAM:** When loading Whisper Large Turbo alongside the LLM in 4-bit.
  - **6.5 GB VRAM:** When running without Whisper and using the LLM in 4-bit.
- **GPU:** An NVIDIA GPU with CUDA support is required.

---

## Installation and Setup

### Native HF Transformers Inference (4-Bit with NVIDIA GPU)

Clone the repository, install the dependencies, and run the application using the following commands:

```bash
git clone https://github.com/nivibilla/local-llasa-tts.git
cd local-llasa-tts
pip install -r requirements_native_hf.txt
pip install -r requirements_base.txt
python hf_app.py
```

### Using Google Colab

If you are unable to run the application locally, try our [Colab Notebook](https://colab.research.google.com/github/nivibilla/local-llasa-tts/blob/main/colab_notebook_4bit.ipynb):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nivibilla/local-llasa-tts/blob/main/colab_notebook_4bit.ipynb)

---

## Additional Resources

- **Long Text Inference with VLLM and Chunking:**  
  See the [llasa_vllm_longtext_inference.ipynb](llasa_vllm_longtext_inference.ipynb) notebook for details on handling long text inputs.

## Acknowledgements

- **Inspiration & Contributions:**  
  - Based on the original [LLaSA training repository](https://github.com/zhenye234/LLaSA_training).
  - Grateful to [mrfakename](https://huggingface.co/spaces/mrfakename/E2-F5-TTS) for the Gradio demo code.
  
Enjoy the new dashboard and the enhanced text-to-speech generation experience!