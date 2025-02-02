print("Initializing system...", flush=True)

# Core Python imports
import os
import io
import sys
import base64
import tempfile
import argparse
import numpy as np
import random

print("Checking CUDA availability...", flush=True)
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This application requires a CUDA-capable GPU.")
    sys.exit(1)
print(f"CUDA is available. Using device: {torch.cuda.get_device_name()}", flush=True)

print("Initializing CUDA backend...", flush=True)
# Force CUDA initialization
torch.cuda.init()
_ = torch.zeros(1).cuda()
print("CUDA initialized successfully", flush=True)

print("Loading audio libraries...", flush=True)
import torchaudio
import soundfile as sf
import gradio as gr

print("Configuring HuggingFace environment...", flush=True)
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.makedirs(cache_dir, exist_ok=True)

# Configure HuggingFace environment for better progress reporting
os.environ['HF_HOME'] = cache_dir  # New recommended way to set cache directory
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "true"
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = "0"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "1"  # Suppress deprecation warnings

print("Loading ML libraries...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from transformers import logging as transformers_logging
from transformers.utils import move_cache

# Pre-initialize transformers
print("Pre-initializing transformers library...", flush=True)
try:
    move_cache()  # Ensure cache is properly initialized
    print("Cache migration completed if needed", flush=True)
except Exception as e:
    print(f"Cache migration skipped: {e}", flush=True)

print("Loading codec model...", flush=True)
from xcodec2.modeling_xcodec2 import XCodec2Model

###############################################################################
#                               CONFIG / SETUP                                #
###############################################################################

MAX_HISTORY = 5  # how many previous generations to keep in the dashboard

# In-memory list of dictionaries storing generation results for the session:
#   e.g. [ { "mode": str, "text": str, "audio_url": str, "temperature": float, ... }, ... ]
history_data = []

# Optional environment variable for the HF API key
HF_KEY_ENV_VAR = "LLASA_API_KEY"

# Use quantization to reduce memory usage and ensure compute dtype matches input
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# In-memory caches for loaded models/tokenizers
loaded_models = {}
loaded_tokenizers = {}

def get_llasa_model(model_choice: str, hf_api_key: str = None):
    """
    Load and cache the specified model (3B or 8B). 
    If an API key is provided, it will be used to authenticate with Hugging Face.
    """
    if model_choice not in loaded_models:
        if model_choice == "3B":
            repo = "srinivasbilla/llasa-3b"
        elif model_choice == "8B":
            repo = "HKUSTAudio/Llasa-8B"
        else:
            repo = "srinivasbilla/llasa-3b"  # fallback

        print(f"Preparing to load {repo}...", flush=True)
        print(f"Current GPU memory usage: {get_gpu_memory():.2f}GB", flush=True)
        
        if check_model_cache(repo):
            print(f"Loading {repo} from cache...", flush=True)
        else:
            print(f"Model {repo} not found in cache.", flush=True)
            print("Starting first-time download process...", flush=True)
            print("Note: Initial downloads can take several minutes depending on your internet speed.", flush=True)
            print("The system is actively downloading. You will see progress updates shortly...", flush=True)
            print("(If this is your first run, multiple models need to be downloaded)", flush=True)
            print(f"Downloading {repo} model and tokenizer...", flush=True)

        print("Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, use_auth_token=hf_api_key)
        print("Tokenizer loaded successfully!", flush=True)
        
        print(f"Loading {model_choice} model into memory (this may take a moment)...", flush=True)
        print("Note: Loading large models can take several minutes on first run.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            device_map='cuda',
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_auth_token=hf_api_key,
            torch_dtype=torch.float16  # Ensure consistent dtype throughout
        )
        torch.cuda.empty_cache()  # Clear any temporary GPU memory
        print(f"{model_choice} model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)", flush=True)
        
        loaded_tokenizers[model_choice] = tokenizer
        loaded_models[model_choice] = model
    return loaded_tokenizers[model_choice], loaded_models[model_choice]

###############################################################################
#                           MODEL LOADING FUNCTIONS                           #
###############################################################################

# Global variables to store loaded models
Codec_model = None
whisper_turbo_pipe = None

def check_model_cache(model_id):
    """Check if a model exists in the HuggingFace cache."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    # Convert model_id to cache path format
    cache_path = os.path.join(cache_dir, model_id.replace("/", "--"))
    return os.path.exists(cache_path)

def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3

def initialize_models():
    """Initialize all required models with progress feedback."""
    global Codec_model, whisper_turbo_pipe
    
    print("Step 1/3: Preparing XCodec2 model...", flush=True)
    model_path = "srinivasbilla/xcodec2"
    if check_model_cache(model_path):
        print(f"Loading XCodec2 model from cache...", flush=True)
    else:
        print(f"Model {model_path} not found in cache.", flush=True)
        print("Starting first-time download process...", flush=True)
        print("Note: Initial downloads can take several minutes depending on your internet speed.", flush=True)
        print("The system is actively downloading. You will see progress updates shortly...", flush=True)
        print("(If this is your first run, multiple models need to be downloaded)", flush=True)
        print(f"Downloading {model_path} (this may take a few minutes)...", flush=True)
    print("Loading XCodec2 model into memory...", flush=True)
    Codec_model = XCodec2Model.from_pretrained(model_path)
    print("Moving XCodec2 model to GPU and optimizing...", flush=True)
    Codec_model.eval().cuda()
    torch.cuda.empty_cache()  # Clear any temporary GPU memory
    print(f"XCodec2 model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)")
    
    print("\nStep 2/3: Preparing Whisper model...", flush=True)
    whisper_model = "openai/whisper-large-v3-turbo"
    if check_model_cache(whisper_model):
        print(f"Loading Whisper model from cache...", flush=True)
    else:
        print(f"Model {whisper_model} not found in cache.", flush=True)
        print("Starting first-time download process...", flush=True)
        print("Note: Initial downloads can take several minutes depending on your internet speed.", flush=True)
        print("The system is actively downloading. You will see progress updates shortly...", flush=True)
        print("(If this is your first run, multiple models need to be downloaded)", flush=True)
        print(f"Downloading {whisper_model} (this may take a few minutes)...", flush=True)
    print("Loading Whisper model and preparing pipeline...", flush=True)
    whisper_turbo_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        torch_dtype=torch.float16,
        device='cuda',
        model_kwargs={"use_cache": True}  # Enable caching for better performance
    )
    torch.cuda.empty_cache()  # Clear any temporary GPU memory
    print(f"Whisper model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)")

###############################################################################
#                             UTILITY FUNCTIONS                               #
###############################################################################

def ids_to_speech_tokens(speech_ids):
    """Convert list of integers to <|s_#|> tokens."""
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def extract_speech_ids(speech_tokens_str):
    """Extract integer speech IDs from <|s_#|> tokens."""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            try:
                num = int(num_str)
                speech_ids.append(num)
            except ValueError:
                print(f"Failed to convert {num_str} to int")
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def generate_audio_data_url(audio_np, sample_rate=16000, format='WAV'):
    """
    Encode the NumPy audio array into a base64 data URL so it can be embedded 
    directly in HTML <audio> tags. Ensures proper audio format conversion.
    """
    # Convert to float32 in range [-1, 1] if not already
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    if np.abs(audio_np).max() > 1.0:
        audio_np = audio_np / np.abs(audio_np).max()
    
    # Convert to int16 for WAV format
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    with io.BytesIO() as buf:
        sf.write(buf, audio_int16, sample_rate, format=format, subtype='PCM_16')
        audio_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:audio/wav;base64,{audio_data}"

def render_previous_generations(history_list, is_generating=False):
    """
    Generate an HTML string with a modern, clean layout for previous generations.
    Displays newest first. Shows a skeleton loader when generating.
    """
    if not history_list and not is_generating:
        return "<div style='color: #999; font-style: italic;'>No previous generations yet.</div>"

    html = """
    <div style="display: flex; flex-direction: column; gap: 1rem;">
    """
    
    # Add skeleton loader if generation is in progress
    if is_generating:
        skeleton_html = """
        <div class="skeleton-loader">
            <div class="skeleton-title"></div>
            <div class="skeleton-text"></div>
            <div class="skeleton-text" style="width: 70%;"></div>
            <div class="skeleton-audio"></div>
        </div>
        """
        html += skeleton_html

    html += """
    <style>
    .audio-controls {
        width: 100%;
        margin-top: 8px;
        background: #2E2F46;
        border-radius: 4px;
        padding: 8px;
    }
    .audio-controls audio {
        width: 100%;
    }
    .audio-controls audio::-webkit-media-controls-panel {
        background-color: #38395A;
    }
    .audio-controls audio::-webkit-media-controls-play-button,
    .audio-controls audio::-webkit-media-controls-mute-button {
        background-color: #3F61EF;
        border-radius: 50%;
        width: 32px;
        height: 32px;
    }
    .audio-controls audio::-webkit-media-controls-current-time-display,
    .audio-controls audio::-webkit-media-controls-time-remaining-display {
        color: #EAEAEA;
    }
    .audio-controls audio::-webkit-media-controls-timeline {
        background-color: #4A4B6F;
    }
    </style>
    <div style="display: flex; flex-direction: column; gap: 1rem;">
    """

    for entry in reversed(history_list):
        text = entry["text"]
        audio_url = entry["audio_url"]
        mode = entry["mode"]
        temperature = entry["temperature"]
        top_p = entry["top_p"]
        max_length = entry["max_length"]

        card_html = f"""
        <div style="background: #33344D; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            <h3 style="margin: 0; font-size: 1.1rem;">Mode: {mode}</h3>
            <p style="margin: 0.5rem 0;"><strong>Text:</strong> {text}</p>
            <p style="margin: 0.5rem 0;"><strong>Params:</strong> max_len={max_length}, temp={temperature}, top_p={top_p}{', seed=' + str(entry.get('seed')) if entry.get('seed') is not None else ''}</p>
            <div class="audio-controls">
                <audio controls src="{audio_url}"></audio>
            </div>
        </div>
        """
        html += card_html

    html += "</div>"
    return html

###############################################################################
#                          MAIN INFERENCE FUNCTION                            #
###############################################################################

def set_seed(seed):
    """Set seeds for reproducible generation across PyTorch, NumPy, and Python's random."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

def infer(
    generation_mode,         # "Text only" or "Reference audio"
    ref_audio_path,          # Reference audio path (if any)
    target_text,             # The text to be synthesized
    model_version,           # Model size: "3B" or "8B"
    hf_api_key,              # Hugging Face API key
    trim_audio,              # Whether to trim reference audio to 15s
    max_length,              # Generation param
    temperature,             # Generation param
    top_p,                   # Generation param
    whisper_language,        # Language for Whisper transcription
    seed,                    # Random seed for reproducible generation
    prev_history,            # The stored history in State
    progress=gr.Progress()
):
    # Set seed if provided
    set_seed(seed)
    # If user doesn't supply an API key in the UI, try environment variable
    if not hf_api_key or not hf_api_key.strip():
        env_key = os.environ.get(HF_KEY_ENV_VAR, "").strip()
        if env_key:
            hf_api_key = env_key

    tokenizer, model = get_llasa_model(model_version, hf_api_key=hf_api_key)

    # Basic text checks
    if len(target_text) == 0:
        return None, render_previous_generations(prev_history), prev_history
    elif len(target_text) > 1000:
        gr.warning("Text is too long. Truncating to 1000 characters.")
        target_text = target_text[:1000]

    # If we have a reference mode, gather a prefix
    speech_ids_prefix = []
    prompt_text = ""
    if generation_mode == "Reference audio" and ref_audio_path:
        progress(0, "Loading & trimming reference audio...")
        waveform, sample_rate = torchaudio.load(ref_audio_path)
        # Trim to 15 seconds if requested
        if trim_audio and len(waveform[0]) / sample_rate > 15:
            waveform = waveform[:, :sample_rate * 15]
        # Convert to mono if needed
        if waveform.size(0) > 1:
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform_mono = waveform

        # Resample to 16kHz
        prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)
        # Transcribe with selected language
        prompt_text = whisper_turbo_pipe(
            prompt_wav[0].numpy(),
            generate_kwargs={"language": whisper_language} if whisper_language != "auto" else {}
        )['text'].strip()

        # Encode the reference audio into speech tokens
        with torch.no_grad():
            vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
            vq_code_prompt = vq_code_prompt[0, 0, :]
            speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
    elif generation_mode == "Reference audio" and not ref_audio_path:
        gr.warning("No reference audio provided. Proceeding in text-only mode.")

    progress(0.5, "Generating speech...")
    combined_input_text = prompt_text + " " + target_text
    prefix_str = "".join(speech_ids_prefix) if speech_ids_prefix else ""
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{combined_input_text}<|TEXT_UNDERSTANDING_END|>"

    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + prefix_str},
    ]

    with torch.no_grad():
        # Create input tensors with attention mask
        model_inputs = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True
        )
        
        input_ids = model_inputs.to("cuda")
        attention_mask = torch.ones_like(input_ids).to("cuda")
        
        speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            max_length=int(max_length),
            min_length=int(max_length * 0.5),  # Ensure at least half of max_length is generated
            eos_token_id=speech_end_id,
            do_sample=True,
            num_beams=2,  # Use beam search but with fewer beams for efficiency
            length_penalty=1.5,  # Encourage longer sequences
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=1.2,  # Penalize repetition
            early_stopping=True,  # Allow early stopping for efficiency
            no_repeat_ngram_size=3,  # Prevent exact 3-gram repetitions
        )

        # The portion we want is from the end of the prompt (minus the prefix) to the second-last token
        prefix_len = len(speech_ids_prefix)
        generated_ids = outputs[0][(input_ids.shape[1] - prefix_len) : -1]
        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_tokens = extract_speech_ids(speech_tokens)
        speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

        # Decode the tokens to a waveform
        gen_wav = Codec_model.decode_code(speech_tokens)

        # Remove the reference prompt from the final audio if needed
        if speech_ids_prefix:
            gen_wav = gen_wav[:, :, prompt_wav.shape[1]:]

    sr = 16000
    out_audio_np = gen_wav[0, 0, :].cpu().numpy()

    progress(0.9, "Finalizing audio...")
    audio_data_url = generate_audio_data_url(out_audio_np, sample_rate=sr)

    # Build new entry for the history
    new_entry = {
        "mode": generation_mode,
        "text": target_text,
        "audio_url": audio_data_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "seed": seed,  # Track seed in history
    }
    if len(prev_history) >= MAX_HISTORY:
        prev_history.pop(0)
    prev_history.append(new_entry)

    updated_dashboard_html = render_previous_generations(prev_history)

    # Return new audio, updated history HTML, and state
    updated_dashboard_html = render_previous_generations(prev_history, is_generating=False)
    return (sr, out_audio_np), updated_dashboard_html, prev_history

###############################################################################
#                             NEW DASHBOARD UI                                #
###############################################################################

# New CSS for a clean, modern dashboard with a top header and two panels.
NEW_CSS = """
/* Remove Gradio branding/footer */
#footer, .gradio-container a[target="_blank"] { display: none; }

/* Skeleton Loader Animation */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.skeleton-loader {
    background: #33344D;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    margin-bottom: 1rem;
}

.skeleton-loader .skeleton-title {
    height: 24px;
    width: 120px;
    background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite linear;
    border-radius: 4px;
    margin-bottom: 12px;
}

.skeleton-loader .skeleton-text {
    height: 16px;
    width: 100%;
    background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite linear;
    border-radius: 4px;
    margin: 8px 0;
}

.skeleton-loader .skeleton-audio {
    height: 48px;
    width: 100%;
    background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite linear;
    border-radius: 4px;
    margin-top: 12px;
}

/* Global styles */
body, .gradio-container {
    margin: 0;
    padding: 0;
    background-color: #1E1E2A;
    color: #EAEAEA;
    font-family: 'Segoe UI', sans-serif;
}

/* Header styling */
#header {
    background-color: #2E2F46;
    padding: 1rem 2rem;
    text-align: center;
}
#header h1 {
    margin: 0;
    font-size: 2rem;
}

/* Main content row styling */
#content-row {
    display: flex;
    flex-direction: row;
    gap: 1rem;
    padding: 1rem 2rem;
    height: calc(100vh - 80px);  /* Adjust for header height */
}

/* Synthesis panel styling */
#synthesis-panel {
    flex: 2;
    background-color: #222233;
    border-radius: 8px;
    padding: 1.5rem;
    overflow-y: auto;
}

/* History panel styling */
#history-panel {
    flex: 1;
    background-color: #222233;
    border-radius: 8px;
    padding: 1.5rem;
    overflow-y: auto;
}

/* Form elements styling */
.gr-textbox input, .gr-textbox textarea, .gr-dropdown select {
    background-color: #38395A;
    border: 1px solid #4A4B6F;
    color: #F1F1F1;
    border-radius: 4px;
    padding: 0.5rem;
}

/* Button styling */
button, .gr-button {
    background-color: #3F61EF;
    color: #fff;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    cursor: pointer;
}
button:hover, .gr-button:hover {
    background-color: #2F51DF;
}

/* History card styling */
.card-grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
.card {
    background-color: #33344D;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.card h3 { margin: 0; font-size: 1.1rem; }
.card p { margin: 0.5rem 0; font-size: 0.9rem; }
.empty-msg { font-size: 0.9rem; color: #999; font-style: italic; }

/* Audio component styling */
.audio-input, .audio-output {
    background-color: #2E2F46 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
}

/* Audio controls container */
.audio-input > div, .audio-output > div {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 8px !important;
    background-color: #38395A !important;
    border-radius: 4px !important;
}

/* Audio buttons */
.audio-input button, .audio-output button {
    background-color: #3F61EF !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 8px !important;
    min-width: 32px !important;
    height: 32px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    margin: 0 4px !important;
}

/* Button hover state */
.audio-input button:hover, .audio-output button:hover {
    background-color: #2F51DF !important;
}

/* Audio progress bar */
.audio-input input[type="range"], .audio-output input[type="range"] {
    flex: 1 !important;
    height: 4px !important;
    background-color: #4A4B6F !important;
    border-radius: 2px !important;
    margin: 0 8px !important;
}

/* Time display */
.audio-input .time, .audio-output .time {
    color: #EAEAEA !important;
    font-size: 14px !important;
    min-width: 60px !important;
    text-align: center !important;
}

/* Ensure icons are visible */
.audio-input button svg, .audio-output button svg {
    width: 16px !important;
    height: 16px !important;
    fill: currentColor !important;
    opacity: 1 !important;
    visibility: visible !important;
}
"""

def build_dashboard():
    """Build the entire Gradio interface with a redesigned, dashboard-like appearance."""
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        background_fill_primary="#1E1E2A",
        background_fill_secondary="#222233",
        border_color_primary="#4A4B6F",
        button_primary_background_fill="#3F61EF",
        button_primary_background_fill_hover="#2F51DF",
        button_primary_text_color="white",
        body_text_color="#EAEAEA",
        block_title_text_color="#EAEAEA",
        block_label_text_color="#EAEAEA",
        input_background_fill="#38395A",
    )

    with gr.Blocks(theme=theme) as demo:
        # Top header
        gr.Markdown("<div id='header'><h1>Llasa TTS Dashboard</h1></div>", elem_id="header")
        
        # Main content row: synthesis panel (left) and history panel (right)
        with gr.Row(elem_id="content-row"):
            with gr.Column(elem_id="synthesis-panel"):
                gr.Markdown("## Synthesize Speech")
                model_choice = gr.Dropdown(
                    label="Select llasa Model",
                    choices=["3B", "8B"],
                    value="3B"
                )
                generation_mode = gr.Radio(
                    label="Generation Mode",
                    choices=["Text only", "Reference audio"],
                    value="Text only",
                    type="value"
                )
                with gr.Group():
                    ref_audio_input = gr.Audio(
                        label="Reference Audio (Optional)",
                        sources=["upload", "microphone"],
                        type="filepath",
                        interactive=True,
                        elem_classes="audio-component"
                    )
                trim_audio_checkbox = gr.Checkbox(
                    label="Trim Reference Audio to 15s?",
                    value=False
                )
                gen_text_input = gr.Textbox(
                    label="Text to Generate",
                    lines=4,
                    placeholder="Enter text here..."
                )
                with gr.Accordion("Advanced Generation Settings", open=False):
                    max_length_slider = gr.Slider(
                        minimum=64, maximum=4096, value=2048, step=64,
                        label="Max Length (tokens) - Higher values allow longer audio generation"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                        label="Temperature"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                        label="Top-p"
                    )
                    whisper_language = gr.Dropdown(
                        label="Whisper Language (for reference audio)",
                        choices=["en", "auto", "ja", "zh", "de", "es", "ru", "ko", "fr", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "jw", "bn", "et", "sr", "sk", "sl", "sw", "af", "fa", "am", "mr", "cy", "gu", "is", "mk", "be", "ne", "si", "kn", "az", "tl", "lv", "mt", "ga", "eu", "gl", "hy", "ka", "lb", "sq", "my", "yi", "bs", "km", "kk", "mn", "sd", "su", "ps", "ky", "ku", "uz", "bo", "sa", "tk", "sh", "yo", "mg", "ha", "as", "ny", "so", "pa", "ka", "te", "tg", "ug", "zu", "sn", "ig", "xh", "st", "tn", "ak", "ht", "ln", "om", "rw", "ti", "gd", "oc", "kw", "br", "fo", "fy", "mi", "qu", "rm", "sc", "gn", "ay", "tt", "cv", "iu", "dv", "or", "lo", "ks", "wo", "ba", "ce", "na", "co", "sm", "bi", "to", "ty", "ss", "sg", "tw", "ff", "dz", "aa", "nn", "nv", "kl", "ki", "ve", "ng", "cr", "ee", "ab", "av", "os", "sc", "li", "ia", "ie", "ik", "io", "vo", "za"],
                        value="en",
                        type="value"
                    )
                    seed_number = gr.Number(
                        label="Random Seed (optional, for reproducible generation)",
                        value=None,
                        precision=0,
                        minimum=0,
                        maximum=2**32-1,
                        step=1
                    )
                api_key_input = gr.Textbox(
                    label="Hugging Face API Key (Optional, but needed for 8B model)",
                    type="password",
                    placeholder="Enter your HF token or leave blank"
                )
                generate_btn = gr.Button("Synthesize", variant="primary")
                with gr.Group():
                    audio_output = gr.Audio(
                        label="Synthesized Audio",
                        type="numpy",
                        interactive=True,
                        show_label=True,
                        autoplay=False,
                        elem_classes="audio-component"
                    )
            
            with gr.Column(elem_id="history-panel"):
                gr.Markdown("## Previous Generations")
                dashboard_html = gr.HTML(value="", show_label=False)
        
        # Gradio State to keep track of previous generations
        prev_history_state = gr.State([])

        def show_loading_state(history):
            """Show skeleton loader immediately when generation starts."""
            return render_previous_generations(history, is_generating=True)

        generate_btn.click(
            fn=show_loading_state,
            inputs=[prev_history_state],
            outputs=[dashboard_html],
        ).then(
            fn=infer,
            inputs=[
                generation_mode,
                ref_audio_input,
                gen_text_input,
                model_choice,
                api_key_input,
                trim_audio_checkbox,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                whisper_language,
                seed_number,
                prev_history_state
            ],
            outputs=[audio_output, dashboard_html, prev_history_state],
        )
    return demo

###############################################################################
#                             MAIN ENTRY POINT                                #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Run the redesigned Llasa TTS dashboard.")
    parser.add_argument("--share", help="Enable gradio share", action="store_true")
    args = parser.parse_args()

    print("Step 1/3: Loading XCodec2 and Whisper models...", flush=True)
    initialize_models()
    
    print("\nStep 2/3: Pre-loading Llasa 3B model...", flush=True)
    get_llasa_model("3B")  # Load default 3B model without API key
    print("Llasa 3B model loaded successfully!")
    
    print("\nStep 3/3: Starting Gradio interface...", flush=True)
    app = build_dashboard()
    app.launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=7860,
    )

if __name__ == "__main__":
    print("\n=== Llasa TTS Dashboard ===", flush=True)
    main()
