import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyttsx3
from pydub import AudioSegment

try:
    from kokoro import TTS as KokoroTTS
except Exception:
    KokoroTTS = None

# macOS built-in "say" produces natural voices and remains a fallback.
# Kokoro is the new primary local TTS.
SAY_BIN = "/usr/bin/say"
DEFAULT_MAC_VOICE = os.getenv("MAC_TTS_VOICE", "Samantha")
DEFAULT_KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_alloy")
KOKORO_MODEL_PATH = os.getenv(
    "KOKORO_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "kokoro-v1_0.pth"),
)
KOKORO_SAMPLE_RATE = 24000

# Lazy-load Kokoro to keep startup fast and avoid import crashes when the
# dependency or model file is missing.
_kokoro_tts: Optional["KokoroTTS"] = None

print("Initializing pyttsx3 fallback TTS engine...")
tts_engine = pyttsx3.init(driverName="nsss")  # macOS driver; adjust if needed
tts_engine.setProperty("rate", 190)
print("TTS fallback ready.")


def _convert_aiff_to_wav_bytes(aiff_path: str) -> bytes:
    """Convert AIFF (16k mono) to WAV bytes for browser playback."""
    speech = AudioSegment.from_file(aiff_path)
    speech = speech.set_frame_rate(16000).set_channels(1)

    buffer = io.BytesIO()
    speech.export(buffer, format="wav")
    return buffer.getvalue()


def _load_kokoro() -> Optional["KokoroTTS"]:
    global _kokoro_tts
    if _kokoro_tts or KokoroTTS is None:
        return _kokoro_tts

    if not os.path.exists(KOKORO_MODEL_PATH):
        print(
            f"Kokoro model not found at {KOKORO_MODEL_PATH}. "
            "Download from https://github.com/hexgrad/Kokoro-TTS/releases "
            "and set KOKORO_MODEL_PATH."
        )
        return None

    try:
        device = os.getenv("KOKORO_DEVICE", "cpu")
        _kokoro_tts = KokoroTTS(KOKORO_MODEL_PATH, device=device)
        print(f"Kokoro TTS loaded on {device} with voice '{DEFAULT_KOKORO_VOICE}'.")
    except Exception as e:
        print(f"Failed to initialize Kokoro TTS: {e}")
        _kokoro_tts = None
    return _kokoro_tts


def _synthesize_with_kokoro(text: str) -> bytes:
    """Primary TTS using local Kokoro (hexgrad/Kokoro-TTS)."""
    tts = _load_kokoro()
    if not tts:
        return b""

    try:
        audio = tts(text, voice=DEFAULT_KOKORO_VOICE)

        # Kokoro may return (audio, sample_rate) or just audio.
        sample_rate = getattr(tts, "sample_rate", KOKORO_SAMPLE_RATE)
        if isinstance(audio, Tuple) or isinstance(audio, list):
            # handle (audio, sr)
            if len(audio) == 2:
                audio, sample_rate = audio

        if audio is None:
            return b""

        # Ensure float32 numpy array
        audio_np = np.array(audio, dtype=np.float32).flatten()
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
        return buf.getvalue()
    except Exception as e:
        print(f"Kokoro synthesis failed: {e}")
        return b""


def _synthesize_with_say(text: str) -> bytes:
    """Use macOS 'say' for more natural, fast synthesis."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            SAY_BIN,
            "-v",
            DEFAULT_MAC_VOICE,
            "-o",
            tmp_path,
            "--data-format=LEI16@16000",  # little-endian 16kHz PCM
            text,
        ]
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        )
        if completed.returncode != 0:
            print(f"'say' failed: {completed.stderr.strip()}")
            return b""

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            print("No audio produced by 'say'.")
            return b""

        return _convert_aiff_to_wav_bytes(tmp_path)
    except Exception as e:
        print(f"Error during 'say' synthesis: {e}")
        return b""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _synthesize_with_pyttsx3(text: str) -> bytes:
    """Fallback TTS using pyttsx3."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            tmp_path = tmp.name

        tts_engine.save_to_file(text, tmp_path)
        tts_engine.runAndWait()

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            print("pyttsx3 produced no audio.")
            return b""

        return _convert_aiff_to_wav_bytes(tmp_path)
    except Exception as e:
        print(f"Error during pyttsx3 synthesis: {e}")
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
    Fallbacks: macOS 'say', then pyttsx3.
    Returns WAV bytes normalized to 16kHz mono for browser playback.
    """
    if not text or not text.strip():
        return b""

    audio_bytes = _synthesize_with_kokoro(text)
    if audio_bytes:
        return audio_bytes

    audio_bytes = _synthesize_with_say(text)
    if audio_bytes:
        return audio_bytes

    print("Falling back to pyttsx3 TTS...")
    return _synthesize_with_pyttsx3(text)

if __name__ == '__main__':
    # This is for testing the module directly
    text = "Hello, this is a test of the text to speech system."
    print(f"Synthesizing: '{text}'")
    audio_bytes = synthesize_speech(text)
    
    if audio_bytes:
        with open("tts_test.wav", "wb") as f:
            f.write(audio_bytes)
        print("Test audio saved to 'tts_test.wav'")

