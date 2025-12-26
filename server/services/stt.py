import os
from io import BytesIO
import tempfile
import wave
import warnings
import sys
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv
import torch
import numpy as np 
from pydub import AudioSegment
import logging

# Load .env early so backend/model env vars are picked up.
load_dotenv()

# Determine device (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.getLogger("voice_assistant").info(f"STT using device: {DEVICE}")

# Backend selection: "whisper" (default) or "funasr"
_DEFAULT_BACKEND = os.getenv(
    "STT_BACKEND_DEFAULT",
    "kyutai",
)
STT_BACKEND = os.getenv("STT_BACKEND", _DEFAULT_BACKEND).lower()
STT_FALLBACK_TO_WHISPER = os.getenv("STT_FALLBACK_TO_WHISPER", "1") == "1"

# Silence the most common transformer warnings/log spam.
warnings.filterwarnings("ignore", message=".*Special tokens have been added.*")
try:
    from transformers.utils import logging as hf_logging  # type: ignore
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
except Exception:
    pass

# Whisper (default) - using transformers library with CUDA support
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa
except Exception as e:
    print(f"Failed to import Whisper dependencies: {e}")
    WhisperProcessor = None
    WhisperForConditionalGeneration = None
    librosa = None

WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-tiny")
_whisper_processor: Optional["WhisperProcessor"] = None
_whisper_model: Optional["WhisperForConditionalGeneration"] = None

# Fun-ASR (optional) - using transformers if available
try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    FUNASR_AVAILABLE = True
except Exception:
    FUNASR_AVAILABLE = False
    AutoProcessor = None
    AutoModelForSpeechSeq2Seq = None

# The previous default FunASR model id was invalid/outdated. Keep env override.
FUNASR_MODEL_ID = os.getenv("FUNASR_MODEL_ID", "mlx-community/Fun-ASR-Nano-2512-fp16")
_funasr_processor: Optional["AutoProcessor"] = None
_funasr_model: Optional["AutoModelForSpeechSeq2Seq"] = None
_funasr_use_fp16 = False

# MLX FunASR (Apple Silicon) - via mlx-audio-plus
MLX_FUNASR_MODEL_ID = os.getenv("MLX_FUNASR_MODEL_ID", "mlx-community/Fun-ASR-Nano-2512-fp16")
MLX_FUNASR_FALLBACK_MODEL_ID = os.getenv("MLX_FUNASR_FALLBACK_MODEL_ID", "mlx-community/Fun-ASR-Nano-2512-8bit")
_mlx_funasr_model = None

# Kyutai (Delayed Streams Modeling) STT on MLX via moshi_mlx
# NOTE: The semantic VAD extra-heads path is available in the *-candle* checkpoints.
# For best turn-taking, default to the candle repo.
KYUTAI_STT_MODEL_ID = os.getenv("KYUTAI_STT_MODEL_ID", "kyutai/stt-1b-en_fr-candle")
_kyutai_model = None
_kyutai_text_tokenizer = None
_kyutai_audio_tokenizer = None
_kyutai_gen = None
_kyutai_other_codebooks = None

# If the user sets STT_BACKEND=funasr on macOS, treat it as an alias for mlx_funasr.
# This avoids trying to load MLX model repos through Transformers.
if sys.platform == "darwin" and STT_BACKEND == "funasr" and os.getenv("PREFER_MLX_FUNASR", "1") == "1":
    STT_BACKEND = "mlx_funasr"
    print("STT: funasr -> mlx_funasr (macOS alias)")

# If STT_BACKEND=funasr on other platforms but mlx-audio-plus is installed, prefer MLX
# (useful for Apple Silicon python environments where sys.platform may still be darwin).
if STT_BACKEND == "funasr" and os.getenv("PREFER_MLX_FUNASR", "1") == "1":
    try:
        import importlib

        importlib.import_module("mlx_audio.stt.models.funasr")
        STT_BACKEND = "mlx_funasr"
        print("STT: switching backend funasr -> mlx_funasr (PREFER_MLX_FUNASR=1)")
    except Exception:
        pass


def _get_mlx_funasr_model():
    global _mlx_funasr_model
    if _mlx_funasr_model is None:
        try:
            from mlx_audio.stt.models.funasr import Model  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "mlx-audio-plus is required for mlx_funasr backend. Install it and restart."
            ) from e
        # Some mlx-audio-plus versions may not support every model packaging variant.
        # Try the configured model first, then a known-good fallback.
        try:
            print(f"Loading MLX FunASR model: {MLX_FUNASR_MODEL_ID}")
            _mlx_funasr_model = Model.from_pretrained(MLX_FUNASR_MODEL_ID)
        except Exception as e1:
            if MLX_FUNASR_FALLBACK_MODEL_ID and MLX_FUNASR_FALLBACK_MODEL_ID != MLX_FUNASR_MODEL_ID:
                print(
                    f"MLX FunASR load failed for {MLX_FUNASR_MODEL_ID}: {e1}. "
                    f"Trying fallback model {MLX_FUNASR_FALLBACK_MODEL_ID}..."
                )
                _mlx_funasr_model = Model.from_pretrained(MLX_FUNASR_FALLBACK_MODEL_ID)
                # Update model id for logging after successful fallback.
                globals()["MLX_FUNASR_MODEL_ID"] = MLX_FUNASR_FALLBACK_MODEL_ID
            else:
                raise
    return _mlx_funasr_model


def _linear_resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    Fast linear resampler for 1D float32 arrays.
    Good enough for STT input conversion 16k->24k.
    """
    if x.size == 0 or src_rate == dst_rate:
        return x.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False).reshape(-1)
    ratio = float(dst_rate) / float(src_rate)
    n_out = int(round(x.shape[0] * ratio))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t = np.linspace(0.0, x.shape[0] - 1, num=n_out, dtype=np.float32)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, x.shape[0] - 1)
    frac = t - i0.astype(np.float32)
    return (x[i0] * (1.0 - frac) + x[i1] * frac).astype(np.float32)


def _get_kyutai_moshi_mlx():
    """
    Load Kyutai moshi_mlx model + tokenizers and return a tuple:
      (gen, text_tokenizer, audio_tokenizer, other_codebooks)
    """
    global _kyutai_model, _kyutai_text_tokenizer, _kyutai_audio_tokenizer, _kyutai_gen, _kyutai_other_codebooks

    if _kyutai_gen is not None:
        return _kyutai_gen, _kyutai_text_tokenizer, _kyutai_audio_tokenizer, _kyutai_other_codebooks

    try:
        import json
        import mlx.core as mx  # type: ignore
        import mlx.nn as nn  # type: ignore
        import sentencepiece  # type: ignore
        import rustymimi  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
        from moshi_mlx import models, utils  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Kyutai STT backend requires moshi_mlx (+ huggingface_hub, rustymimi, sentencepiece)."
        ) from e

    lm_config_path = hf_hub_download(KYUTAI_STT_MODEL_ID, "config.json")
    with open(lm_config_path, "r") as fobj:
        cfg_dict = json.load(fobj)

    mimi_weights = hf_hub_download(KYUTAI_STT_MODEL_ID, cfg_dict["mimi_name"])
    moshi_name = cfg_dict.get("moshi_name", "model.safetensors")
    moshi_weights = hf_hub_download(KYUTAI_STT_MODEL_ID, moshi_name)
    tok_path = hf_hub_download(KYUTAI_STT_MODEL_ID, cfg_dict["tokenizer_name"])

    lm_config = models.LmConfig.from_config_dict(cfg_dict)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if moshi_weights.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif moshi_weights.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)

    print(f"Loading Kyutai moshi weights from {moshi_weights}")
    if KYUTAI_STT_MODEL_ID.endswith("-candle"):
        model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
    else:
        model.load_weights(moshi_weights, strict=True)

    print(f"Loading Kyutai tokenizer from {tok_path}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tok_path)  # type: ignore

    print(f"Loading Kyutai audio tokenizer from {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    other_codebooks = lm_config.other_codebooks
    mimi_codebooks = max(generated_codebooks, other_codebooks)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)  # type: ignore

    model.warmup()
    gen = models.LmGen(
        model=model,
        max_steps=4096,
        text_sampler=utils.Sampler(top_k=25, temp=0),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        check=False,
    )

    _kyutai_model = model
    _kyutai_text_tokenizer = text_tokenizer
    _kyutai_audio_tokenizer = audio_tokenizer
    _kyutai_gen = gen
    _kyutai_other_codebooks = other_codebooks
    return gen, text_tokenizer, audio_tokenizer, other_codebooks


def _transcribe_with_kyutai(audio_data_16k: np.ndarray) -> str:
    """
    Run Kyutai STT (moshi_mlx) on a full utterance.
    Input: float32 mono @16k
    Output: text
    """
    gen, text_tok, audio_tok, other_codebooks = _get_kyutai_moshi_mlx()
    try:
        import mlx.core as mx  # type: ignore
    except Exception as e:
        raise RuntimeError("mlx is required for kyutai backend.") from e

    # IMPORTANT: rustymimi expects NumPy float32 arrays (see Kyutai stt_from_mic_mlx.py).
    # Keep audio blocks as NumPy for encode_step; convert only the resulting tokens to MLX.

    # Resample to 24k and pad with 2s of silence like Kyutai scripts
    audio_24k = _linear_resample(audio_data_16k, 16000, 24000)
    audio_24k = np.concatenate([audio_24k, np.zeros((48000,), dtype=np.float32)], axis=0)
    audio_24k = np.asarray(audio_24k, dtype=np.float32).reshape(-1)

    out_parts: list[str] = []
    # Process in 1920-sample blocks (80ms @24k)
    for start_idx in range(0, (audio_24k.shape[0] // 1920) * 1920, 1920):
        chunk = audio_24k[start_idx : start_idx + 1920]
        # Match Kyutai mic script: block shape (1, 1920) then encode_step(block[None,0:1]) => (1,1,1920)
        block = chunk.reshape(1, 1920).astype(np.float32, copy=False)
        other_audio_tokens = audio_tok.encode_step(block[None, 0:1])
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, :other_codebooks]

        text_token = gen.step(other_audio_tokens[0])
        tid = int(text_token[0].item())
        if tid not in (0, 3):
            piece = text_tok.id_to_piece(tid)  # type: ignore
            out_parts.append(piece.replace("▁", " "))

    return "".join(out_parts).strip()


def _float32_to_wav_path(audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
    audio_f32 = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (audio_f32 * 32767.0).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return tmp_path


def _transcribe_with_mlx_funasr(audio_data: np.ndarray) -> str:
    model = _get_mlx_funasr_model()
    wav_path = None
    try:
        wav_path = _float32_to_wav_path(audio_data, sample_rate=16000)
        result = model.generate(wav_path)
        text = getattr(result, "text", None)
        return (text or "").strip()
    finally:
        if wav_path:
            try:
                os.remove(wav_path)
            except Exception:
                pass


# Track if model uses FP16
_whisper_use_fp16 = False

def _get_whisper_model() -> tuple["WhisperProcessor", "WhisperForConditionalGeneration"]:
    global _whisper_processor, _whisper_model, _whisper_use_fp16
    if _whisper_processor is None or _whisper_model is None:
        if WhisperProcessor is None or WhisperForConditionalGeneration is None:
            raise RuntimeError("Whisper backend unavailable; install transformers and librosa.")
        print(f"Loading Whisper model: {WHISPER_MODEL_ID} on {DEVICE}")
        _whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
        _whisper_model = _whisper_model.to(DEVICE)
        if DEVICE == "cuda":
            _whisper_model = _whisper_model.half()  # Use FP16 for faster inference on GPU
            _whisper_use_fp16 = True
        _whisper_model.eval()
    return _whisper_processor, _whisper_model


def _get_funasr_model() -> tuple["AutoProcessor", "AutoModelForSpeechSeq2Seq"]:
    global _funasr_processor, _funasr_model, _funasr_use_fp16
    if _funasr_processor is None or _funasr_model is None:
        if not FUNASR_AVAILABLE:
            raise RuntimeError("Fun-ASR backend unavailable; install transformers.")
        print(f"Loading Fun-ASR model: {FUNASR_MODEL_ID} on {DEVICE}")
        try:
            _funasr_processor = AutoProcessor.from_pretrained(FUNASR_MODEL_ID)
            _funasr_model = AutoModelForSpeechSeq2Seq.from_pretrained(FUNASR_MODEL_ID)
        except Exception as e:
            raise RuntimeError(f"Failed to load Fun-ASR model '{FUNASR_MODEL_ID}': {e}") from e
        _funasr_model = _funasr_model.to(DEVICE)
        if DEVICE == "cuda":
            _funasr_model = _funasr_model.half()  # Use FP16 for faster inference on GPU
            _funasr_use_fp16 = True
        _funasr_model.eval()
    return _funasr_processor, _funasr_model


def _transcribe_with_whisper(audio_data: np.ndarray) -> str:
    if WhisperProcessor is None or WhisperForConditionalGeneration is None:
        raise RuntimeError("Whisper backend unavailable; install transformers and librosa.")
    
    processor, model = _get_whisper_model()
    
    # Ensure audio is in the right format (16kHz mono, float32)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Take first channel if stereo
    
    # Ensure audio is 1D
    if len(audio_data.shape) > 1:
        audio_data = audio_data.flatten()
    
    # Ensure audio is not empty and has valid values
    if len(audio_data) == 0:
        print("WARNING: Empty audio data provided to Whisper")
        return ""
    
    # Whisper expects 16kHz, which we already have from preprocessing
    # Use the processor to prepare inputs
    processed = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    
    # Move inputs to device and convert to FP16 if model uses FP16
    inputs = {}
    for k, v in processed.items():
        v = v.to(DEVICE)
        if _whisper_use_fp16 and v.dtype == torch.float32:
            v = v.half()
        inputs[k] = v
    
    # Generate transcription with explicit language='en' to avoid language detection
    # Use better generation parameters
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=448,
            language="en",
            task="transcribe",
            num_beams=1,  # Use greedy decoding for speed
            do_sample=False  # Deterministic output
        )
    
    # Decode
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    result = transcription.strip()
    
    if os.getenv("DEBUG_STT", "0") == "1":
        logging.getLogger("voice_assistant").debug(f"Whisper transcription: '{result}'")
    
    return result


def _transcribe_with_funasr(audio_data: np.ndarray) -> str:
    if not FUNASR_AVAILABLE:
        raise RuntimeError("Fun-ASR backend unavailable; install transformers.")
    
    processor, model = _get_funasr_model()
    
    # Ensure audio is in the right format
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Take first channel if stereo
    
    # Process audio
    processed = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    
    # Move inputs to device and convert to FP16 if model uses FP16
    inputs = {}
    for k, v in processed.items():
        v = v.to(DEVICE)
        if _funasr_use_fp16 and v.dtype == torch.float32:
            v = v.half()
        inputs[k] = v
    
    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=448)
    
    # Decode
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def preload_stt() -> None:
    """Load STT model into memory at startup to avoid first-call latency."""
    try:
        if STT_BACKEND == "kyutai":
            _ = _get_kyutai_moshi_mlx()
            # Warm up with dummy audio
            _ = _transcribe_with_kyutai(np.zeros(16000, dtype=np.float32))
            print(f"Kyutai STT backend preloaded ({KYUTAI_STT_MODEL_ID}).")
        elif STT_BACKEND == "mlx_funasr":
            _ = _get_mlx_funasr_model()
            # Warm up with dummy audio
            _ = _transcribe_with_mlx_funasr(np.zeros(16000, dtype=np.float32))
            print(f"MLX FunASR backend preloaded ({MLX_FUNASR_MODEL_ID}).")
        elif STT_BACKEND == "whisper":
            if WhisperProcessor is None or WhisperForConditionalGeneration is None:
                print("Whisper backend not available; install transformers and librosa to use it.")
            else:
                _ = _get_whisper_model()
                # Warm up with dummy audio
                _ = _transcribe_with_whisper(np.zeros(16000, dtype=np.float32))
                print(f"Whisper backend preloaded ({WHISPER_MODEL_ID}) on {DEVICE}.")
        else:
            if not FUNASR_AVAILABLE:
                print("Fun-ASR backend not available; install transformers to use it.")
            else:
                try:
                    _ = _get_funasr_model()
                    _ = _transcribe_with_funasr(np.zeros(16000, dtype=np.float32))
                    print(f"Fun-ASR backend preloaded ({FUNASR_MODEL_ID}) on {DEVICE}.")
                except Exception as e:
                    print(f"Failed to preload Fun-ASR model ({FUNASR_MODEL_ID}): {e}")
                    if STT_FALLBACK_TO_WHISPER:
                        print("Falling back to Whisper backend.")
                        globals()["STT_BACKEND"] = "whisper"
                        _ = _get_whisper_model()
                        _ = _transcribe_with_whisper(np.zeros(16000, dtype=np.float32))
                        print(f"Whisper backend preloaded ({WHISPER_MODEL_ID}) on {DEVICE}.")
    except Exception as e:
        print(f"Failed to preload STT backend '{STT_BACKEND}': {e}")


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribes a byte string of audio into text using the configured backend.

    Args:
        audio_bytes: The audio data in bytes.

    Returns:
        The transcribed text.
    """
    try:
        wall_start = perf_counter()
        # MediaRecorder sends webm/opus chunks; decode, mono, resample.
        audio_io = BytesIO(audio_bytes)
        segment = AudioSegment.from_file(audio_io)
        segment = segment.set_frame_rate(16000).set_channels(1)
        decode_ms = (perf_counter() - wall_start) * 1000

        # Convert to float32 numpy in range [-1, 1]
        # Use raw_audio property for better compatibility
        audio_samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        if segment.sample_width == 1:  # 8-bit PCM
            audio_samples = (audio_samples - 128.0) / 128.0
        elif segment.sample_width == 2:  # 16-bit PCM
            audio_samples /= 32768.0
        elif segment.sample_width == 4:  # 32-bit PCM
            audio_samples /= 2147483648.0
        else:
            # Default normalization
            audio_samples = audio_samples / np.max(np.abs(audio_samples)) if np.max(np.abs(audio_samples)) > 0 else audio_samples
        
        # Ensure audio is in valid range and not all zeros
        audio_data = np.clip(audio_samples, -1.0, 1.0)
        
        if os.getenv("DEBUG_STT", "0") == "1":
            logging.getLogger("voice_assistant").debug(
                f"Audio stats: shape={audio_data.shape}, duration={len(audio_data)/16000:.2f}s, "
                f"max={np.max(np.abs(audio_data)):.4f}, mean={np.mean(np.abs(audio_data)):.4f}"
            )

        # Transcribe with selected backend (auto-fallback to Whisper to avoid repeated load errors)
        asr_start = perf_counter()
        text = ""
        backend_name = ""
        backend_used = STT_BACKEND
        try:
            if STT_BACKEND == "kyutai":
                text = _transcribe_with_kyutai(audio_data)
                backend_name = f"kyutai ({KYUTAI_STT_MODEL_ID})"
            elif STT_BACKEND == "mlx_funasr":
                text = _transcribe_with_mlx_funasr(audio_data)
                backend_name = f"mlx_funasr ({MLX_FUNASR_MODEL_ID})"
            elif STT_BACKEND == "whisper":
                text = _transcribe_with_whisper(audio_data)
                backend_name = f"whisper ({WHISPER_MODEL_ID}) [{DEVICE}]"
            else:
                text = _transcribe_with_funasr(audio_data)
                backend_name = f"funasr ({FUNASR_MODEL_ID}) [{DEVICE}]"
        except Exception as e:
            print(f"Primary STT backend '{STT_BACKEND}' failed: {e}")
            # If we're not already on MLX, try it next (preferred on Apple Silicon).
            if STT_BACKEND != "mlx_funasr":
                try:
                    globals()["STT_BACKEND"] = "mlx_funasr"
                    backend_used = "mlx_funasr"
                    text = _transcribe_with_mlx_funasr(audio_data)
                    backend_name = f"mlx_funasr ({MLX_FUNASR_MODEL_ID})"
                except Exception as e_mlx:
                    print(f"Fallback MLX FunASR failed: {e_mlx}")
                    text = ""

            if not text and STT_FALLBACK_TO_WHISPER and STT_BACKEND != "whisper":
                try:
                    globals()["STT_BACKEND"] = "whisper"  # stop repeated FunASR load spam
                    backend_used = "whisper"
                    text = _transcribe_with_whisper(audio_data)
                    backend_name = f"whisper ({WHISPER_MODEL_ID}) [{DEVICE}]"
                except Exception as e2:
                    print(f"Fallback Whisper STT also failed: {e2}")
                    text = ""
        asr_ms = (perf_counter() - asr_start) * 1000
        total_ms = (perf_counter() - wall_start) * 1000
        logging.getLogger("voice_assistant").info(
            f"STT [{backend_name or backend_used}] ms decode={decode_ms:.0f} asr={asr_ms:.0f} total={total_ms:.0f}"
        )
        
        return text
    except Exception as e:
        logging.getLogger("voice_assistant").exception(f"Error during transcription: {e}")
        return ""

if __name__ == '__main__':
    # This is for testing the module directly
    # You would need a sample audio file named 'test.wav'
    try:
        with open("test.wav", "rb") as f:
            audio_bytes = f.read()
            text = transcribe_audio(audio_bytes)
            print(f"Transcription: {text}")
    except FileNotFoundError:
        print("Create a 'test.wav' file to test the STT module.")

