#!/usr/bin/env python3
import os
import io
import sys
import base64
import tempfile
import argparse
import json
import numpy as np
import random

print("Initializing system...", flush=True)

# Check Python version
if sys.version_info < (3, 10):
    print("ERROR: Python 3.10 or higher is required to run this application.")
    sys.exit(1)

# Core Python imports
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This application requires a CUDA-capable GPU.")
    sys.exit(1)
print(f"CUDA is available. Using device: {torch.cuda.get_device_name()}", flush=True)

print("Initializing CUDA backend...", flush=True)
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
os.environ['HF_HOME'] = cache_dir
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "true"
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = "0"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "1"

print("Loading ML libraries...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from transformers import logging as transformers_logging
from transformers.utils import move_cache

print("Pre-initializing transformers library...", flush=True)
try:
    move_cache()
    print("Cache migration completed if needed", flush=True)
except Exception as e:
    print(f"Cache migration skipped: {e}", flush=True)

print("Loading codec model...", flush=True)
from xcodec2.modeling_xcodec2 import XCodec2Model

###############################################################################
#                               CONFIG / SETUP                                #
###############################################################################

MAX_HISTORY = 5  # How many previous generations to keep
history_data = []  # In-memory history list
HF_KEY_ENV_VAR = "LLASA_API_KEY"

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# In-memory caches for loaded models/tokenizers
loaded_models = {}
loaded_tokenizers = {}

def unload_model(model_choice: str):
    """Unload a model from GPU and clear from cache."""
    if model_choice in loaded_models:
        print(f"Unloading {model_choice} model from GPU...", flush=True)
        if hasattr(loaded_models[model_choice], 'cpu'):
            loaded_models[model_choice].cpu()
        del loaded_models[model_choice]
        if model_choice in loaded_tokenizers:
            del loaded_tokenizers[model_choice]
        torch.cuda.empty_cache()
        print(f"{model_choice} model unloaded successfully!", flush=True)

def get_llasa_model(model_choice: str, hf_api_key: str = None):
    """
    Load and cache the specified model (3B or 8B).
    If an API key is provided, it is used to authenticate with Hugging Face.
    """
    for existing_model in list(loaded_models.keys()):
        if existing_model != model_choice:
            unload_model(existing_model)
    if model_choice not in loaded_models:
        repo = "srinivasbilla/llasa-3b" if model_choice == "3B" else "HKUSTAudio/Llasa-8B"
        print(f"Preparing to load {repo}...", flush=True)
        print(f"Current GPU memory usage: {get_gpu_memory():.2f}GB", flush=True)
        hub_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--" + repo.replace("/", "--"))
        if os.path.exists(hub_path):
            print(f"Loading {repo} from local cache...", flush=True)
        else:
            print(f"Model {repo} not found in cache. Starting download...", flush=True)
        print("Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, use_auth_token=hf_api_key)
        print("Tokenizer loaded successfully!", flush=True)
        print(f"Loading {model_choice} model into memory...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            device_map='cuda',
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_auth_token=hf_api_key,
            torch_dtype=torch.float16
        )
        torch.cuda.empty_cache()
        print(f"{model_choice} model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)", flush=True)
        loaded_tokenizers[model_choice] = tokenizer
        loaded_models[model_choice] = model
    return loaded_tokenizers[model_choice], loaded_models[model_choice]

def get_gpu_memory():
    """Return current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def initialize_models():
    """Initialize XCodec2 and Whisper models."""
    global Codec_model, whisper_turbo_pipe
    print("Step 1/3: Preparing XCodec2 model...", flush=True)
    model_path = "srinivasbilla/xcodec2"
    hub_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--" + model_path.replace("/", "--"))
    if os.path.exists(hub_path):
        print(f"Loading XCodec2 model from local cache...", flush=True)
    else:
        print(f"Model {model_path} not found in cache. Starting download...", flush=True)
    print("Loading XCodec2 model into memory...", flush=True)
    Codec_model = XCodec2Model.from_pretrained(model_path)
    Codec_model.eval().cuda()
    torch.cuda.empty_cache()
    print(f"XCodec2 model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)")
    
    print("\nStep 2/3: Preparing Whisper model...", flush=True)
    whisper_model = "openai/whisper-large-v3-turbo"
    hub_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--" + whisper_model.replace("/", "--"))
    if os.path.exists(hub_path):
        print(f"Loading Whisper model from local cache...", flush=True)
    else:
        print(f"Model {whisper_model} not found in cache. Starting download...", flush=True)
    print("Loading Whisper model and preparing pipeline...", flush=True)
    whisper_turbo_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        torch_dtype=torch.float16,
        device='cuda'
    )
    torch.cuda.empty_cache()
    print(f"Whisper model loaded successfully! (GPU memory: {get_gpu_memory():.2f}GB)")

###############################################################################
#                             UTILITY FUNCTIONS                               #
###############################################################################

def toggle_auto_optimize_checkbox(mode):
    return gr.update(interactive=(mode=="Text only"))

def ids_to_speech_tokens(speech_ids):
    """Convert list of integers to token strings."""
    return [f"<|s_{speech_id}|>" for speech_id in speech_ids]

def extract_speech_ids(speech_tokens_str):
    """Extract integer IDs from tokens."""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            try:
                num = int(token_str[4:-2])
                speech_ids.append(num)
            except ValueError:
                print(f"Failed to convert token: {token_str}")
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def generate_audio_data_url(audio_np, sample_rate=16000, format='WAV'):
    """Encode NumPy audio array into a base64 data URL for HTML audio tags."""
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)
    if np.abs(audio_np).max() > 1.0:
        audio_np = audio_np / np.abs(audio_np).max()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    with io.BytesIO() as buf:
        sf.write(buf, audio_int16, sample_rate, format=format, subtype='PCM_16')
        audio_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:audio/wav;base64,{audio_data}"

def render_previous_generations(history_list, is_generating=False):
    """Render history entries as HTML."""
    if not history_list and not is_generating:
        return "<div style='color: #999; font-style: italic;'>No previous generations yet.</div>"
    html = """
    <style>
    #footer, .gradio-container a[target="_blank"] { display: none !important; }
    .audio-controls { width: 100%; margin-top: 8px; background: #2E2F46; border-radius: 4px; padding: 8px; }
    .audio-controls audio { width: 100%; }
    .audio-controls audio::-webkit-media-controls-panel { background-color: #38395A; }
    .audio-controls audio::-webkit-media-controls-play-button,
    .audio-controls audio::-webkit-media-controls-mute-button { background-color: #3F61EF; border-radius: 50%; width: 32px; height: 32px; }
    .audio-controls audio::-webkit-media-controls-current-time-display,
    .audio-controls audio::-webkit-media-controls-time-remaining-display { color: #EAEAEA; }
    .audio-controls audio::-webkit-media-controls-timeline { background-color: #4A4B6F; }
    @keyframes shimmer { 0% { background-position: -1000px 0; } 100% { background-position: 1000px 0; } }
    .skeleton-loader { background: #33344D; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-bottom: 1rem; }
    .skeleton-loader .skeleton-title { height: 24px; width: 120px; background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%); background-size: 1000px 100%; animation: shimmer 2s infinite linear; border-radius: 4px; margin-bottom: 12px; }
    .skeleton-loader .skeleton-text { height: 16px; width: 100%; background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%); background-size: 1000px 100%; animation: shimmer 2s infinite linear; border-radius: 4px; margin: 8px 0; }
    .skeleton-loader .skeleton-audio { height: 48px; width: 100%; background: linear-gradient(90deg, #38395A 25%, #4A4B6F 50%, #38395A 75%); background-size: 1000px 100%; animation: shimmer 2s infinite linear; border-radius: 4px; margin-top: 12px; }
    </style>
    """
    if is_generating:
        html += """
        <div class="skeleton-loader">
            <div class="skeleton-title"></div>
            <div class="skeleton-text"></div>
            <div class="skeleton-text" style="width: 70%;"></div>
            <div class="skeleton-audio"></div>
        </div>
        """
    if history_list:
        html += "<div style='display: flex; flex-direction: column; gap: 1rem;'>"
        for entry in reversed(history_list):
            card_html = f"""
            <div style="background: #33344D; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                <h3 style="margin: 0; font-size: 1.1rem;">Mode: {entry['mode']}</h3>
                <p style="margin: 0.5rem 0;"><strong>Text:</strong> {entry['text']}</p>
                <p style="margin: 0.5rem 0;"><strong>Params:</strong> max_len={entry['max_length']}, temp={entry['temperature']}, top_p={entry['top_p']}{', seed=' + str(entry.get('seed')) if entry.get('seed') is not None else ''}</p>
                <div class="audio-controls">
                    <audio controls src="{entry['audio_url']}"></audio>
                </div>
            </div>
            """
            html += card_html
        html += "</div>"
    return html

###############################################################################
#                  HELPER FUNCTIONS FOR PODCAST MODE                          #
###############################################################################

def parse_conversation(transcript: str):
    """
    Parse the transcript into a list of (speaker, message) tuples and a list of unique speaker names.
    Expected format per line: "Speaker Name: message"
    """
    lines = transcript.splitlines()
    conversation = []
    speakers = set()
    for line in lines:
        if ':' not in line:
            continue
        speaker, text = line.split(":", 1)
        speaker = speaker.strip()
        text = text.strip()
        conversation.append((speaker, text))
        speakers.add(speaker)
    return conversation, list(speakers)

def join_audio_segments(segments, sample_rate=16000, crossfade_duration=0.05):
    """
    Concatenate a list of 1D NumPy audio arrays with a brief crossfade.
    """
    if not segments:
        return np.array([], dtype=np.float32)
    crossfade_samples = int(sample_rate * crossfade_duration)
    joined_audio = segments[0]
    for seg in segments[1:]:
        if crossfade_samples > 0 and len(joined_audio) >= crossfade_samples and len(seg) >= crossfade_samples:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            joined_audio[-crossfade_samples:] = joined_audio[-crossfade_samples:] * fade_out + seg[:crossfade_samples] * fade_in
            joined_audio = np.concatenate([joined_audio, seg[crossfade_samples:]])
        else:
            joined_audio = np.concatenate([joined_audio, seg])
    return joined_audio

def build_transcript_html(conversation):
    """Build an HTML transcript with speaker labels."""
    html = ""
    for speaker, text in conversation:
        html += f"<p><strong>{speaker}:</strong> {text}</p>\n"
    return html

def generate_line_audio(speaker, text, generation_mode, ref_audio, seed, common_params, progress):
    """
    Generate audio for a single line by calling infer().
    Returns (sample_rate, audio_np).
    """
    result = infer(
        generation_mode,
        ref_audio,
        text,
        common_params["model_version"],
        common_params["hf_api_key"],
        common_params["trim_audio"],
        common_params["max_length"],
        common_params["temperature"],
        common_params["top_p"],
        common_params["whisper_language"],
        seed,
        common_params["random_seed_each_gen"],
        common_params["beam_search_enabled"],
        common_params["auto_optimize_length"],
        prev_history=[],  # Do not update history for per-line synthesis.
        progress=progress
    )
    return result[0]

def infer_podcast(
    conversation_text,
    generation_mode,  # Should be "Podcast"
    model_choice,
    hf_api_key,
    trim_audio,
    max_length,
    temperature,
    top_p,
    whisper_language,
    user_seed,  # Not used in podcast mode
    random_seed_each_gen,
    beam_search_enabled,
    auto_optimize_length,
    prev_history,
    progress=gr.Progress(),
    speaker_config=dict()
):
    """
    Generate podcast audio by synthesizing each line using speaker-specific settings.
    The speaker_config argument is a dictionary mapping speaker names (case-insensitive) to:
       { "ref_audio": <filepath or empty string>, "seed": <number or None> }
    """
    lower_config = {k.lower(): v for k, v in speaker_config.items()}
    conversation, speakers = parse_conversation(conversation_text)
    audio_segments = []
    for speaker, line_text in conversation:
        config = lower_config.get(speaker.lower(), {"ref_audio": "", "seed": None})
        ref_audio = config.get("ref_audio", "")
        seed = config.get("seed", None)
        line_mode = "Reference audio" if ref_audio else "Text only"
        _, line_audio = generate_line_audio(speaker, line_text, line_mode, ref_audio, seed,
            {
                "model_version": model_choice,
                "hf_api_key": hf_api_key,
                "trim_audio": trim_audio,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "whisper_language": whisper_language,
                "user_seed": user_seed,
                "random_seed_each_gen": random_seed_each_gen,
                "beam_search_enabled": beam_search_enabled,
                "auto_optimize_length": auto_optimize_length,
            },
            progress
        )
        audio_segments.append(line_audio)
    final_audio = join_audio_segments(audio_segments, sample_rate=16000, crossfade_duration=0.05)
    transcript_html = build_transcript_html(conversation)
    new_entry = {
        "mode": "Podcast",
        "text": conversation_text,
        "audio_url": generate_audio_data_url(final_audio, sample_rate=16000),
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "seed": "N/A",
    }
    if len(prev_history) >= MAX_HISTORY:
        prev_history.pop(0)
    prev_history.append(new_entry)
    updated_dashboard_html = render_previous_generations(prev_history, is_generating=False)
    # FIX: Return updated_dashboard_html (full info) instead of transcript_html
    return (16000, final_audio), updated_dashboard_html, prev_history

###############################################################################
#                          MAIN INFERENCE FUNCTION                            #
###############################################################################

def set_seed(seed):
    """Set seeds for reproducible generation."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

def infer(
    generation_mode,         # "Text only" or "Reference audio"
    ref_audio_path,          # Reference audio file path (if any)
    target_text,             # Text to synthesize
    model_version,           # "3B" or "8B"
    hf_api_key,              # Hugging Face API key
    trim_audio,              # Trim reference audio to 15s?
    max_length,              # Generation parameter
    temperature,             # Generation parameter
    top_p,                   # Generation parameter
    whisper_language,        # Whisper language
    user_seed,               # User-provided seed
    random_seed_each_gen,    # Whether to use a random seed each generation
    beam_search_enabled,     # Beam search flag
    auto_optimize_length,    # Auto-optimize length flag
    prev_history,            # Dashboard history list
    progress=gr.Progress()
):
    if random_seed_each_gen:
        chosen_seed = random.randint(0, 2**31 - 1)
    else:
        chosen_seed = user_seed
    set_seed(chosen_seed)
    if not hf_api_key or not hf_api_key.strip():
        env_key = os.environ.get(HF_KEY_ENV_VAR, "").strip()
        if env_key:
            hf_api_key = env_key
    tokenizer, model = get_llasa_model(model_version, hf_api_key=hf_api_key)
    if len(target_text) == 0:
        return None, render_previous_generations(prev_history), prev_history
    elif len(target_text) > 1000:
        gr.warning("Text is too long. Truncating to 1000 characters.")
        target_text = target_text[:1000]
    if auto_optimize_length:
        input_len = tokenizer.apply_chat_template(
            [{"role": "user", "content": target_text}],
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True
        ).shape[1]
        margin = 100 if generation_mode == "Reference audio" else 50
        if input_len + margin > max_length:
            old_val = max_length
            max_length = input_len + margin
            print(f"Auto optimizing: input length is {input_len}, raising max_length from {old_val} to {max_length}.")
    speech_ids_prefix = []
    prompt_text = ""
    if generation_mode == "Reference audio" and ref_audio_path:
        progress(0, "Loading & trimming reference audio...")
        waveform, sample_rate = torchaudio.load(ref_audio_path)
        if trim_audio and len(waveform[0]) / sample_rate > 15:
            waveform = waveform[:, :sample_rate * 15]
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True) if waveform.size(0) > 1 else waveform
        prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)
        whisper_args = {}
        if whisper_language != "auto":
            whisper_args["language"] = whisper_language
        prompt_text = whisper_turbo_pipe(
            prompt_wav[0].numpy(),
            generate_kwargs=whisper_args
        )['text'].strip()
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
    num_beams = 2 if beam_search_enabled else 1
    early_stopping_val = (num_beams > 1)
    model_inputs = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        return_tensors="pt",
        continue_final_message=True
    )
    input_ids = model_inputs.to("cuda")
    attention_mask = torch.ones_like(input_ids).to("cuda")
    if auto_optimize_length:
        input_len = input_ids.shape[1]
        margin = 100 if generation_mode == "Reference audio" else 50
        if input_len + margin > max_length:
            old_val = max_length
            max_length = input_len + margin
            print(f"Auto optimizing: input length is {input_len}, raising max_length from {old_val} to {max_length}.")
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
            max_length=int(max_length),
            min_length=int(max_length * 0.5),
            eos_token_id=speech_end_id,
            do_sample=True,
            num_beams=num_beams,
            length_penalty=1.5,
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=1.2,
            early_stopping=early_stopping_val,
            no_repeat_ngram_size=3,
        )
        prefix_len = len(speech_ids_prefix)
        generated_ids = outputs[0][(input_ids.shape[1] - prefix_len): -1]
        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_tokens = extract_speech_ids(speech_tokens)
        speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)
        gen_wav = Codec_model.decode_code(speech_tokens)
        if speech_ids_prefix:
            gen_wav = gen_wav[:, :, prompt_wav.shape[1]:]
    sr = 16000
    out_audio_np = gen_wav[0, 0, :].cpu().numpy()
    progress(0.9, "Finalizing audio...")
    audio_data_url = generate_audio_data_url(out_audio_np, sample_rate=sr)
    new_entry = {
        "mode": generation_mode,
        "text": target_text,
        "audio_url": audio_data_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "seed": chosen_seed,
    }
    if len(prev_history) >= MAX_HISTORY:
        prev_history.pop(0)
    prev_history.append(new_entry)
    updated_dashboard_html = render_previous_generations(prev_history, is_generating=False)
    return (sr, out_audio_np), updated_dashboard_html, prev_history

###############################################################################
#                             NEW DASHBOARD UI                                #
###############################################################################

NEW_CSS = """
/* Remove Gradio branding/footer */
#footer, .gradio-container a[target="_blank"] { display: none; }
/* Simple dark background */
body, .gradio-container { margin: 0; padding: 0; background-color: #1E1E2A; color: #EAEAEA; font-family: 'Segoe UI', sans-serif; }
/* Header styling */
#header { background-color: #2E2F46; padding: 1rem 2rem; text-align: center; }
#header h1 { margin: 0; font-size: 2rem; }
/* Main content row styling */
#content-row { display: flex; flex-direction: row; gap: 1rem; padding: 1rem 2rem; }
/* Synthesis panel */
#synthesis-panel { flex: 2; background-color: #222233; border-radius: 8px; padding: 1.5rem; }
/* History panel */
#history-panel { flex: 1; background-color: #222233; border-radius: 8px; padding: 1.5rem; }
/* Form elements styling */
.gr-textbox input, .gr-textbox textarea, .gr-dropdown select { background-color: #38395A; border: 1px solid #4A4B6F; color: #F1F1F1; border-radius: 4px; padding: 0.5rem; }
/* Audio components */
.audio-input, .audio-output { background-color: #2E2F46 !important; border-radius: 8px !important; padding: 12px !important; margin: 8px 0 !important; }
"""

def build_dashboard():
    """Build the Gradio interface with separate tabs for Standard TTS and Podcast Mode."""
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter")],
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        background_fill_primary="#1E1E2A",
        background_fill_secondary="#222233",
        border_color_primary="#4A4B6F",
        body_text_color="#EAEAEA",
        block_title_text_color="#EAEAEA",
        block_label_text_color="#EAEAEA",
        input_background_fill="#38395A",
    )

    with gr.Blocks(theme=theme, css=NEW_CSS) as demo:
        gr.Markdown("<div id='header'><h1>Llasa TTS Dashboard</h1></div>", elem_id="header")
        # Shared state for previous generations
        prev_history_state = gr.State([])

        with gr.Tabs():
            # --- Standard TTS Tab ---
            with gr.TabItem("Standard TTS"):
                with gr.Row(elem_id="content-row"):
                    with gr.Column(elem_id="synthesis-panel"):
                        gr.Markdown("## Standard TTS")
                        model_choice_std = gr.Dropdown(label="Select llasa Model", choices=["3B", "8B"], value="3B")
                        generation_mode_std = gr.Radio(label="Generation Mode", choices=["Text only", "Reference audio"], value="Text only", type="value")
                        with gr.Group():
                            ref_audio_input = gr.Audio(label="Reference Audio (Optional)", sources=["upload", "microphone"], type="filepath")
                            trim_audio_checkbox_std = gr.Checkbox(label="Trim Reference Audio to 15s?", value=False)
                        gen_text_input = gr.Textbox(label="Text to Generate", lines=4, placeholder="Enter text here...")
                        with gr.Accordion("Advanced Generation Settings", open=False):
                            max_length_slider_std = gr.Slider(minimum=64, maximum=4096, value=1024, step=64, label="Max Length (tokens)")
                            temperature_slider_std = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                            top_p_slider_std = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p")
                            whisper_language_std = gr.Dropdown(label="Whisper Language (for reference audio)",
                                                               choices=["en", "auto", "ja", "zh", "de", "es", "ru", "ko", "fr"],
                                                               value="en", type="value")
                            random_seed_checkbox_std = gr.Checkbox(label="Random seed each generation", value=True)
                            beam_search_checkbox_std = gr.Checkbox(label="Enable beam search", value=False)
                            auto_optimize_checkbox_std = gr.Checkbox(label="[Text Only] Auto Optimize Length", value=True)
                            seed_number_std = gr.Number(label="Seed (if not random)", value=None, precision=0, minimum=0, maximum=2**32-1, step=1)
                        api_key_input_std = gr.Textbox(label="Hugging Face API Key (Optional, required for 8B)", type="password", placeholder="Enter your HF token or leave blank")
                        synthesize_btn_std = gr.Button("Synthesize")
                        with gr.Group():
                            audio_output_std = gr.Audio(label="Synthesized Audio", type="numpy", interactive=False, show_label=True, autoplay=False)
                    with gr.Column(elem_id="history-panel"):
                        gr.Markdown("## Previous Generations")
                        dashboard_html_std = gr.HTML(value="<div style='color: #999; font-style: italic;'>No previous generations yet.</div>", show_label=False)

            # --- Podcast Mode Tab ---
            with gr.TabItem("Podcast Mode"):
                with gr.Row(elem_id="content-row"):
                    with gr.Column(elem_id="synthesis-panel"):
                        gr.Markdown("## Podcast Mode")
                        gr.Markdown("⚠️ **Experimental Feature** ⚠️\nThis mode works best with reference audio. Text-only generations may be unreliable.")
                        model_choice_pod = gr.Dropdown(label="Select llasa Model", choices=["3B", "8B"], value="3B")
                        podcast_transcript = gr.Textbox(label="Podcast Transcript",
                                                        lines=6,
                                                        placeholder="Enter conversation transcript. Each line should be formatted as 'Speaker Name: message'")
                        with gr.Accordion("Speaker Configuration (Add as many as needed)", open=True):
                            gr.Markdown("Fill in the details for each speaker you expect to appear in the transcript.")
                            speaker1_name = gr.Textbox(label="Speaker 1 Name", placeholder="e.g., Alex")
                            ref_audio_speaker1 = gr.Audio(label="Reference Audio for Speaker 1 (Optional)", sources=["upload", "microphone"], type="filepath")
                            seed_speaker1 = gr.Number(label="Seed for Speaker 1 (Optional)", value=None, precision=0)
                            
                            speaker2_name = gr.Textbox(label="Speaker 2 Name", placeholder="e.g., Jamie")
                            ref_audio_speaker2 = gr.Audio(label="Reference Audio for Speaker 2 (Optional)", sources=["upload", "microphone"], type="filepath")
                            seed_speaker2 = gr.Number(label="Seed for Speaker 2 (Optional)", value=None, precision=0)
                            
                            speaker3_name = gr.Textbox(label="Speaker 3 Name (Optional)", placeholder="e.g., Casey")
                            ref_audio_speaker3 = gr.Audio(label="Reference Audio for Speaker 3 (Optional)", sources=["upload", "microphone"], type="filepath")
                            seed_speaker3 = gr.Number(label="Seed for Speaker 3 (Optional)", value=None, precision=0)
                        with gr.Accordion("Advanced Generation Settings", open=False):
                            max_length_slider_pod = gr.Slider(minimum=64, maximum=4096, value=1024, step=64, label="Max Length (tokens)")
                            temperature_slider_pod = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                            top_p_slider_pod = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p")
                            whisper_language_pod = gr.Dropdown(label="Whisper Language (for reference audio)",
                                                               choices=["en", "auto", "ja", "zh", "de", "es", "ru", "ko", "fr"],
                                                               value="en", type="value")
                            random_seed_checkbox_pod = gr.Checkbox(label="Random seed each generation", value=True)
                            beam_search_checkbox_pod = gr.Checkbox(label="Enable beam search", value=False)
                            auto_optimize_checkbox_pod = gr.Checkbox(label="[Text Only] Auto Optimize Length", value=True)
                            seed_number_pod = gr.Number(label="Seed (if not random)", value=None, precision=0, minimum=0, maximum=2**32-1, step=1)
                        api_key_input_pod = gr.Textbox(label="Hugging Face API Key (Optional, required for 8B)", type="password", placeholder="Enter your HF token or leave blank")
                        synthesize_btn_pod = gr.Button("Synthesize Podcast")
                        with gr.Group():
                            audio_output_pod = gr.Audio(label="Synthesized Podcast Audio", type="numpy", interactive=False, show_label=True, autoplay=False)
                    with gr.Column(elem_id="history-panel"):
                        gr.Markdown("## Previous Generations")
                        dashboard_html_pod = gr.HTML(value="<div style='color: #999; font-style: italic;'>No previous generations yet.</div>", show_label=False)
        
        # --- Callback Functions ---
        def synthesize_standard(generation_mode, ref_audio_input, gen_text_input, model_choice, api_key_input,
                                max_length_slider, temperature_slider, top_p_slider, whisper_language,
                                seed_number, random_seed_checkbox, beam_search_checkbox, auto_optimize_checkbox,
                                trim_audio, prev_history):
            common_params = {
                "model_version": model_choice,
                "hf_api_key": api_key_input,
                "trim_audio": trim_audio,
                "max_length": max_length_slider,
                "temperature": temperature_slider,
                "top_p": top_p_slider,
                "whisper_language": whisper_language,
                "user_seed": seed_number,
                "random_seed_each_gen": random_seed_checkbox,
                "beam_search_enabled": beam_search_checkbox,
                "auto_optimize_length": auto_optimize_checkbox,
            }
            return infer(generation_mode, ref_audio_input, gen_text_input, **common_params, prev_history=prev_history)
        
        def synthesize_podcast(podcast_transcript, model_choice, api_key_input,
                               max_length_slider, temperature_slider, top_p_slider, whisper_language,
                               seed_number, random_seed_checkbox, beam_search_checkbox, auto_optimize_checkbox,
                               prev_history,
                               speaker1_name, ref_audio_speaker1, seed_speaker1,
                               speaker2_name, ref_audio_speaker2, seed_speaker2,
                               speaker3_name, ref_audio_speaker3, seed_speaker3):
            speaker_config = {}
            for name, ref, seed in [
                (speaker1_name, ref_audio_speaker1, seed_speaker1),
                (speaker2_name, ref_audio_speaker2, seed_speaker2),
                (speaker3_name, ref_audio_speaker3, seed_speaker3),
            ]:
                if name and name.strip():
                    speaker_config[name.strip()] = {"ref_audio": ref if ref else "", "seed": seed}
            return infer_podcast(
                podcast_transcript, "Podcast", model_choice, api_key_input, False,
                max_length_slider, temperature_slider, top_p_slider, whisper_language,
                seed_number, random_seed_checkbox, beam_search_checkbox, auto_optimize_checkbox,
                prev_history, speaker_config=speaker_config
            )
        
        # --- Wire up Standard TTS Tab ---
        synthesize_btn_std.click(
            lambda history: render_previous_generations(history, is_generating=True),
            inputs=[prev_history_state],
            outputs=[dashboard_html_std]
        ).then(
            synthesize_standard,
            inputs=[generation_mode_std, ref_audio_input, gen_text_input, model_choice_std, api_key_input_std,
                    max_length_slider_std, temperature_slider_std, top_p_slider_std, whisper_language_std,
                    seed_number_std, random_seed_checkbox_std, beam_search_checkbox_std, auto_optimize_checkbox_std,
                    trim_audio_checkbox_std, prev_history_state],
            outputs=[audio_output_std, dashboard_html_std, prev_history_state]
        )
        
        # --- Wire up Podcast Mode Tab ---
        synthesize_btn_pod.click(
            lambda history: render_previous_generations(history, is_generating=True),
            inputs=[prev_history_state],
            outputs=[dashboard_html_pod]
        ).then(
            synthesize_podcast,
            inputs=[podcast_transcript, model_choice_pod, api_key_input_pod,
                    max_length_slider_pod, temperature_slider_pod, top_p_slider_pod, whisper_language_pod,
                    seed_number_pod, random_seed_checkbox_pod, beam_search_checkbox_pod, auto_optimize_checkbox_pod,
                    prev_history_state,
                    speaker1_name, ref_audio_speaker1, seed_speaker1,
                    speaker2_name, ref_audio_speaker2, seed_speaker2,
                    speaker3_name, ref_audio_speaker3, seed_speaker3],
            outputs=[audio_output_pod, dashboard_html_pod, prev_history_state]
        )
    return demo

###############################################################################
#                             MAIN ENTRY POINT                                #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Run the redesigned Llasa TTS dashboard with Podcast mode.")
    parser.add_argument("--share", help="Enable gradio share", action="store_true")
    args = parser.parse_args()
    print("Step 1/3: Loading XCodec2 and Whisper models...", flush=True)
    initialize_models()
    print("\nStep 2/3: Pre-loading Llasa 3B model...", flush=True)
    get_llasa_model("3B")
    print("Llasa 3B model loaded successfully!")
    print("\nStep 3/3: Starting Gradio interface...", flush=True)
    app = build_dashboard()
    app.launch(share=args.share, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    print("\n=== Llasa TTS Dashboard with Podcast Mode ===", flush=True)
    main()
