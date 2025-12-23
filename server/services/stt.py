import os
from io import BytesIO
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv
import torch
import numpy as np 
from pydub import AudioSegment

# Load .env early so backend/model env vars are picked up.
load_dotenv()

# Determine device (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"STT using device: {DEVICE}")

# Backend selection: "whisper" (default) or "funasr"
STT_BACKEND = os.getenv("STT_BACKEND", "whisper").lower()

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

FUNASR_MODEL_ID = os.getenv("FUNASR_MODEL_ID", "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
_funasr_processor: Optional["AutoProcessor"] = None
_funasr_model: Optional["AutoModelForSpeechSeq2Seq"] = None
_funasr_use_fp16 = False


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
        _funasr_processor = AutoProcessor.from_pretrained(FUNASR_MODEL_ID)
        _funasr_model = AutoModelForSpeechSeq2Seq.from_pretrained(FUNASR_MODEL_ID)
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
    
    # Debug output
    if result:
        print(f"Whisper transcription: '{result}'")
    else:
        print("WARNING: Whisper returned empty transcription")
    
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
        if STT_BACKEND == "whisper":
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
                _ = _get_funasr_model()
                # Warm up with dummy audio
                _ = _transcribe_with_funasr(np.zeros(16000, dtype=np.float32))
                print(f"Fun-ASR backend preloaded ({FUNASR_MODEL_ID}) on {DEVICE}.")
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
        
        # Debug: Check audio characteristics
        if np.max(np.abs(audio_data)) < 0.01:
            print(f"WARNING: Audio appears to be very quiet (max amplitude: {np.max(np.abs(audio_data)):.6f})")
        print(f"Audio stats: shape={audio_data.shape}, duration={len(audio_data)/16000:.2f}s, max={np.max(np.abs(audio_data)):.4f}, mean={np.mean(np.abs(audio_data)):.4f}")

        # Transcribe with selected backend
        asr_start = perf_counter()
        if STT_BACKEND == "whisper":
            text = _transcribe_with_whisper(audio_data)
            backend_name = f"whisper ({WHISPER_MODEL_ID}) [{DEVICE}]"
        else:
            text = _transcribe_with_funasr(audio_data)
            backend_name = f"funasr ({FUNASR_MODEL_ID}) [{DEVICE}]"
        asr_ms = (perf_counter() - asr_start) * 1000
        total_ms = (perf_counter() - wall_start) * 1000
        print(
            f"STT [{backend_name}] (ms): decode+resample={decode_ms:.1f}, asr={asr_ms:.1f}, total={total_ms:.1f}"
        )
        
        return text
    except Exception as e:
        print(f"Error during transcription: {e}")
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

