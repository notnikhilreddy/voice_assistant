import os
from io import BytesIO
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv
import mlx.core as mx
import numpy as np
from pydub import AudioSegment

# Load .env early so backend/model env vars are picked up.
load_dotenv()

# Force CPU to avoid Metal issues in headless shells; adjust if you want GPU.
mx.set_default_device(mx.cpu)

# Backend selection: "funasr" (default) or "whisper"
STT_BACKEND = os.getenv("STT_BACKEND", "funasr").lower()

# Fun-ASR (default)
try:
    from mlx_audio.stt.models.funasr import Model as FunASRModel, TASK_TRANSCRIBE
except Exception:
    FunASRModel = None
    TASK_TRANSCRIBE = None
FUNASR_MODEL_ID = os.getenv("FUNASR_MODEL_ID", "mlx-community/Fun-ASR-Nano-2512-fp16")
_funasr_model: Optional["FunASRModel"] = None

# Whisper (optional)
try:
    import mlx_whisper
except Exception:
    mlx_whisper = None
WHISPER_MODEL_ID = os.getenv("WHISPER_MLX_MODEL", "mlx-community/whisper-tiny.en-mlx")


def _get_funasr_model() -> "FunASRModel":
    global _funasr_model
    if _funasr_model is None:
        if FunASRModel is None:
            raise RuntimeError("Fun-ASR backend unavailable; install mlx-audio-plus.")
        _funasr_model = FunASRModel.from_pretrained(FUNASR_MODEL_ID)
    return _funasr_model


def _transcribe_with_funasr(audio_data: np.ndarray) -> str:
    funasr = _get_funasr_model()
    result = funasr.generate(audio_data, task=TASK_TRANSCRIBE, language="en")
    return result.text


def _transcribe_with_whisper(audio_data: np.ndarray) -> str:
    if mlx_whisper is None:
        raise RuntimeError("mlx_whisper is not installed; cannot use whisper backend.")
    result = mlx_whisper.transcribe(audio_data, path_or_hf_repo=WHISPER_MODEL_ID, fp16=True)
    return result["text"]


def preload_stt() -> None:
    """Load STT model into memory at startup to avoid first-call latency."""
    try:
        if STT_BACKEND == "whisper":
            if mlx_whisper is None:
                print("Whisper backend not available; install mlx-whisper to use it.")
            else:
                _ = _transcribe_with_whisper(np.zeros(16000, dtype=np.float32))
                print(f"Whisper backend preloaded ({WHISPER_MODEL_ID}).")
        else:
            _ = _get_funasr_model()
            print(f"Fun-ASR backend preloaded ({FUNASR_MODEL_ID}).")
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
        audio_samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        if segment.sample_width == 2:  # 16-bit PCM
            audio_samples /= 32768.0
        elif segment.sample_width == 4:  # 32-bit PCM
            audio_samples /= 2147483648.0
        audio_data = audio_samples

        # Transcribe with selected backend
        asr_start = perf_counter()
        if STT_BACKEND == "whisper":
            text = _transcribe_with_whisper(audio_data)
            backend_name = f"whisper ({WHISPER_MODEL_ID})"
        else:
            text = _transcribe_with_funasr(audio_data)
            backend_name = f"funasr ({FUNASR_MODEL_ID})"
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

