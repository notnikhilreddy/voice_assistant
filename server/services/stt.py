from io import BytesIO

import mlx_whisper
import numpy as np
from pydub import AudioSegment

# For simplicity, we'll load the model once when the service is imported.
# A more robust application might manage model loading more dynamically.
model = "mlx-community/whisper-tiny.en-mlx" # A small, fast model for real-time use

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribes a byte string of audio into text using Whisper.

    Args:
        audio_bytes: The audio data in bytes.

    Returns:
        The transcribed text.
    """
    try:
        # MediaRecorder sends webm/opus chunks; decode, mono, resample.
        audio_io = BytesIO(audio_bytes)
        segment = AudioSegment.from_file(audio_io)
        segment = segment.set_frame_rate(16000).set_channels(1)

        # Convert to float32 numpy in range [-1, 1]
        audio_samples = np.array(segment.get_array_of_samples()).astype(np.float32)
        if segment.sample_width == 2:  # 16-bit PCM
            audio_samples /= 32768.0
        elif segment.sample_width == 4:  # 32-bit PCM
            audio_samples /= 2147483648.0
        audio_data = audio_samples

        # Transcribe
        print("Transcribing audio...")
        result = mlx_whisper.transcribe(audio_data, path_or_hf_repo=model, fp16=True)
        print("Transcription complete.")
        
        return result["text"]
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

