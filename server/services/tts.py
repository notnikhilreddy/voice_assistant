import io
import os
import logging
import re
import subprocess
import tempfile
import contextlib
from pathlib import Path
from time import perf_counter
from typing import Generator, Optional, Tuple

import numpy as np
import torch
import pyttsx3
from pydub import AudioSegment

try:
    from kokoro import KModel, KPipeline
except Exception as e:
    # Make it clear when Kokoro is not available so we know why we fell back.
    print(f"Kokoro import failed; falling back to 'say' then pyttsx3. Error: {e}")
    KModel = None
    KPipeline = None

# Kokoro is the primary local TTS.
# Note: macOS "say" command removed for Linux compatibility
DEFAULT_KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_bella")
KOKORO_MODEL_PATH = os.getenv(
    "KOKORO_MODEL_PATH",
    # repo_root/models/kokoro-v1_0.pth (repo_root is two levels above server/services)
    str(Path(__file__).resolve().parents[2] / "models" / "kokoro-v1_0.pth"),
)
KOKORO_SAMPLE_RATE = 24000

# Lazy-load Kokoro to keep startup fast and avoid import crashes when the
# dependency or model file is missing.
_kokoro_pipeline: Optional["KPipeline"] = None
_kokoro_model: Optional["KModel"] = None

print("Initializing pyttsx3 fallback TTS engine...")
tts_engine = None
try:
    # Try to initialize with available driver (Linux: espeak, Windows: sapi5)
    tts_engine = pyttsx3.init()
    # Try to set properties, but don't fail if voice setting fails
    try:
        tts_engine.setProperty("rate", 190)
    except Exception:
        pass  # Rate setting is optional
    print("TTS fallback ready.")
except Exception as e:
    # pyttsx3 is just a fallback; Kokoro is the primary TTS
    print(f"pyttsx3 fallback not available (this is OK, Kokoro is primary): {e}")
    tts_engine = None


def _convert_audio_to_wav_bytes(audio_path: str) -> bytes:
    """Convert audio file (16k mono) to WAV bytes for browser playback."""
    speech = AudioSegment.from_file(audio_path)
    speech = speech.set_frame_rate(16000).set_channels(1)

    buffer = io.BytesIO()
    speech.export(buffer, format="wav")
    return buffer.getvalue()


def _load_kokoro() -> Optional["KPipeline"]:
    """
    Initialize Kokoro model + pipeline lazily. Requires the `kokoro`
    package (pip) and a model file; if the file is missing, the package will
    fetch from HuggingFace by default.
    """
    global _kokoro_pipeline, _kokoro_model

    if _kokoro_pipeline or KModel is None or KPipeline is None:
        if KModel is None or KPipeline is None:
            print("Kokoro not available (import failed).")
        return _kokoro_pipeline

    try:
        # Use CUDA if available, otherwise CPU
        import torch
        import warnings
        device = os.getenv("KOKORO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        # If env overrides path incorrectly, but repo-root models/ exists, prefer it.
        fallback_repo_model = str(Path(__file__).resolve().parents[2] / "models" / "kokoro-v1_0.pth")
        if not os.path.exists(KOKORO_MODEL_PATH) and os.path.exists(fallback_repo_model):
            model_path = fallback_repo_model
        else:
            model_path = KOKORO_MODEL_PATH if os.path.exists(KOKORO_MODEL_PATH) else None

        if model_path is None:
            print(
                f"Kokoro model not found at {KOKORO_MODEL_PATH}. "
                "Will try to download from HuggingFace (hexgrad/Kokoro-82M)."
            )
            # Suppress warnings from Kokoro library about repo_id
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.filterwarnings("ignore", message=".*repo_id.*")
                # Explicitly pass repo_id to suppress warning
                _kokoro_model = KModel(repo_id="hexgrad/Kokoro-82M")
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _kokoro_model = KModel(model=model_path)
        if device and device != "cpu":
            _kokoro_model = _kokoro_model.to(device)

        # Suppress warnings when creating pipeline
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.filterwarnings("ignore", message=".*repo_id.*")
            _kokoro_pipeline = KPipeline(lang_code="a", model=_kokoro_model, device=device)
        print(f"Kokoro TTS loaded on {device} with voice '{DEFAULT_KOKORO_VOICE}'.")
    except Exception as e:
        print(f"Failed to initialize Kokoro TTS: {e}")
        _kokoro_pipeline = None
        _kokoro_model = None

    return _kokoro_pipeline


def _split_text_for_fallback(text: str, max_first: int = 80, max_rest: int = 120) -> list[str]:
    """
    Split text into small chunks, biasing the first chunk to be very short.
    Prefer sentence boundaries; further split on commas if needed.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []

    def flush(current: str):
        if current:
            chunks.append(current.strip())

    current = ""
    is_first = True
    for sent in sentences:
        if not sent:
            continue
        if sent[-1] not in ".!?":
            sent += "."

        limit = max_first if is_first else max_rest

        # If the sentence itself is too long, split on commas
        parts = re.split(r",(?:\s+)?", sent)

        for part in parts:
            if not part:
                continue
            candidate = (current + " " + part).strip() if current else part.strip()
            if len(candidate) <= limit:
                current = candidate
            else:
                flush(current)
                current = part.strip()

        flush(current)
        current = ""
        is_first = False

    flush(current)
    return chunks


def preload_tts() -> None:
    """Warm TTS: try Kokoro load so first request is faster."""
    try:
        pipeline = _load_kokoro()
        if pipeline:
            print("Kokoro TTS preloaded.")
        else:
            print("Kokoro TTS not preloaded (pipeline unavailable).")
    except Exception as e:
        print(f"Failed to preload Kokoro TTS: {e}")


def _synthesize_with_kokoro(text: str) -> bytes:
    """Primary TTS using local Kokoro (hexgrad/Kokoro-TTS)."""
    synth_start = perf_counter()
    pipeline = _load_kokoro()
    if not pipeline:
        elapsed = (perf_counter() - synth_start) * 1000
        print(f"Skipping Kokoro; not initialized (took {elapsed:.1f} ms).")
        return b""

    try:
        audio_chunks = []
        # KPipeline returns a generator of Result objects; each has .audio
        for res in pipeline(text, voice=DEFAULT_KOKORO_VOICE, model=_kokoro_model):
            if res.audio is None:
                continue
            audio_np = res.audio.detach().cpu().numpy().astype(np.float32)
            audio_chunks.append(audio_np)

        if not audio_chunks:
            print("Kokoro produced no audio.")
            return b""

        audio_np = np.concatenate(audio_chunks)
        sample_rate = getattr(pipeline.model, "sample_rate", KOKORO_SAMPLE_RATE)

        # Convert to 16-bit PCM bytes
        pcm16 = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)

        seg = AudioSegment(
            pcm16.tobytes(),
            frame_rate=sample_rate or KOKORO_SAMPLE_RATE,
            sample_width=2,
            channels=1,
        )
        seg = seg.set_frame_rate(16000).set_channels(1)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        elapsed = (perf_counter() - synth_start) * 1000
        print(f"Kokoro synthesis took {elapsed:.1f} ms.")
        return buf.getvalue()
    except Exception as e:
        elapsed = (perf_counter() - synth_start) * 1000
        print(f"Kokoro synthesis failed after {elapsed:.1f} ms: {e}")
        return b""


def _stream_with_kokoro(text: str) -> Generator[bytes, None, None]:
    """
    Stream Kokoro audio chunks as they are produced.
    Falls back to empty iterator if Kokoro is unavailable.
    """
    pipeline = _load_kokoro()
    if not pipeline:
        return

    try:
        for res in pipeline(text, voice=DEFAULT_KOKORO_VOICE, model=_kokoro_model):
            if res.audio is None:
                continue
            audio_np = res.audio.detach().cpu().numpy().astype(np.float32)
            pcm16 = np.clip(audio_np, -1.0, 1.0)
            pcm16 = (pcm16 * 32767).astype(np.int16)
            seg = AudioSegment(
                pcm16.tobytes(),
                frame_rate=getattr(pipeline.model, "sample_rate", KOKORO_SAMPLE_RATE),
                sample_width=2,
                channels=1,
            )
            seg = seg.set_frame_rate(16000).set_channels(1)
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            yield buf.getvalue()
    except Exception as e:
        print(f"Kokoro streaming failed: {e}")
        return


def _synthesize_with_say(text: str) -> bytes:
    """Fallback: Not available on Linux (macOS-only)."""
    print("macOS 'say' command not available on Linux.")
    return b""


def _synthesize_with_pyttsx3(text: str) -> bytes:
    """Fallback TTS using pyttsx3."""
    if tts_engine is None:
        print("pyttsx3 engine not available.")
        return b""
    
    synth_start = perf_counter()
    tmp_path = None
    try:
        # Use .wav instead of .aiff for Linux compatibility
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        tts_engine.save_to_file(text, tmp_path)
        tts_engine.runAndWait()

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            print("pyttsx3 produced no audio.")
            return b""

        # Read WAV file directly (no conversion needed)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        
        # Convert to 16kHz mono if needed
        speech = AudioSegment.from_file(tmp_path)
        speech = speech.set_frame_rate(16000).set_channels(1)
        buffer = io.BytesIO()
        speech.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()
        
        elapsed = (perf_counter() - synth_start) * 1000
        print(f"pyttsx3 synthesis took {elapsed:.1f} ms.")
        return audio_bytes
    except Exception as e:
        elapsed = (perf_counter() - synth_start) * 1000
        print(f"Error during pyttsx3 synthesis after {elapsed:.1f} ms: {e}")
        return b""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def synthesize_speech(text: str) -> bytes:
    """
    Synthesizes speech from text.

    Primary: Kokoro (hexgrad/Kokoro-TTS) for fast local neural TTS.
    Fallback: pyttsx3.
    Returns WAV bytes normalized to 16kHz mono for browser playback.
    """
    if not text or not text.strip():
        return b""

    total_start = perf_counter()
    path_used = "kokoro"

    audio_bytes = _synthesize_with_kokoro(text)
    if audio_bytes:
        total_ms = (perf_counter() - total_start) * 1000
        print(f"TTS total (path={path_used}) took {total_ms:.1f} ms.")
        return audio_bytes

    print("Falling back to pyttsx3 TTS...")
    path_used = "pyttsx3"
    audio_bytes = _synthesize_with_pyttsx3(text)
    total_ms = (perf_counter() - total_start) * 1000
    print(f"TTS total (path={path_used}) took {total_ms:.1f} ms.")
    return audio_bytes


def stream_speech(text: str) -> Generator[bytes, None, None]:
    """
    Stream synthesized speech in small chunks.

    Primary: Kokoro streaming. Fallbacks chunk text and synthesize per chunk.
    Yields WAV bytes per chunk, already 16kHz mono.
    """
    if not text or not text.strip():
        return

    any_yielded = False
    parts = _split_text_for_fallback(text)

    for part in parts:
        part_yielded = False

        # Try Kokoro streaming for this part
        kokoro_stream = _stream_with_kokoro(part)
        if kokoro_stream:
            for chunk in kokoro_stream:
                if not chunk:
                    continue
                part_yielded = True
                any_yielded = True
                if os.getenv("DEBUG_AUDIO", "0") == "1":
                    logging.getLogger("voice_assistant").debug(f"Kokoro stream chunk bytes={len(chunk)}")
                yield chunk

        if part_yielded:
            continue

        # Fallback per part
        print("Fallback: trying pyttsx3.")
        audio_bytes = _synthesize_with_pyttsx3(part)
        if audio_bytes:
            any_yielded = True
            print(f"Fallback 'pyttsx3' chunk bytes={len(audio_bytes)}")
            yield audio_bytes

    if not any_yielded:
        return

if __name__ == '__main__':
    # This is for testing the module directly
    text = "Hello, this is a test of the text to speech system."
    print(f"Synthesizing: '{text}'")
    audio_bytes = synthesize_speech(text)
    
    if audio_bytes:
        with open("tts_test.wav", "wb") as f:
            f.write(audio_bytes)
        print("Test audio saved to 'tts_test.wav'")

