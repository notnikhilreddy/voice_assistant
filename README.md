# Voice Assistant

This project is a real-time voice assistant that runs a local AI pipeline for speed and privacy, with the ability to query a powerful remote AI for complex questions. It uses a web interface for capturing and playing back audio.

This project is designed to run on Linux with NVIDIA CUDA GPUs and uses PyTorch for local AI model acceleration.

## Features

- **Real-time Voice Interaction:** Hold down a button to speak, and get a spoken response back.
- **Web-based Client:** Simple and accessible client that runs in any modern browser.
- **Remote LLM (Groq):** Streams tokens so speech can start before the full reply is ready.
- **Local First AI:**
  - **STT:** Speech-to-Text via selectable backend (Kyutai / FunASR / Whisper).
  - **TTS:** Text-to-Speech via Kokoro/pyttsx3 locally.
  - **Turn-taking:** End-of-turn detection via Smart Turn v3 (default) or Silero VAD fallback.

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (CUDA 12.x recommended)
- CUDA Toolkit and cuDNN installed
- PyTorch with CUDA support (see installation instructions below)

### 2. Clone the Repository

If you haven't already, clone this project to your local machine.

### 3. Install Dependencies

It is highly recommended to use a Python virtual environment.

```bash
# Navigate to the project directory
cd voice_assistant

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install PyTorch with CUDA support first (adjust CUDA version as needed)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the required packages
pip install -r requirements.txt
```

*Note: The first time you run the application, the AI models (Whisper, etc.) will be downloaded. This may take some time and require a significant amount of disk space. Make sure your GPU has sufficient VRAM (at least 4GB recommended).*

### 4. Configure Environment Variables

The application uses the Groq API for complex questions. You will need an API key from [Groq](https://console.groq.com/keys).

1. Make a copy of the example `.env` file:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file and replace `"YOUR_GROQ_API_KEY_HERE"` with your actual Groq API key.

#### STT + turn-taking configuration

- **`STT_BACKEND`** (default: `kyutai`)
  - `kyutai`: Kyutai delayed-streams-modeling STT on MLX (`moshi_mlx`)
  - `funasr`: MLX Fun-ASR Nano via `mlx-audio-plus`
  - `whisper`: Whisper tiny via Transformers

- **`TURN_TAKING_BACKEND`** (default: `smart_turn`)
  - `smart_turn`: pipecat Smart Turn v3 (audio-native endpoint model)
  - `silero`: Silero VAD + prosody heuristics

- **Smart Turn model selection**
  - `SMART_TURN_REPO_ID` default is `pipecat-ai/smart-turn-v3`
  - `SMART_TURN_ONNX_FILENAME` optional (if the repo file name changes)
  - `SMART_TURN_THRESHOLD` (default `0.5`)
  - `SMART_TURN_MIN_SILENCE_MS` (default `220`) how much trailing silence before running Smart Turn
  - `SMART_TURN_MIN_INTERVAL_MS` (default `220`) rate limit for Smart Turn calls during silence

- **Gatekeeper VAD (speech/silence + barge-in)**
  - Even with `TURN_TAKING_BACKEND=smart_turn`, we still run a lightweight VAD to gate Smart Turn and support barge-in.
  - To avoid loading Silero ONNX and use the lightweight energy fallback instead, set:
    - `SILERO_VAD_FORCE_FALLBACK=1`

#### Manual vs Auto capture

- **Hold-to-speak (manual)**: records until you release the button. **No turn-taking model is used**; the full audio is transcribed after release.
- **Auto voice detection (auto)**: streams audio continuously and uses `TURN_TAKING_BACKEND` to decide **Hold vs Shift**.

#### Streaming partial transcripts (optional)

The server can stream partial `stt_partial` updates (live-updating the “You:” line):

- **`STT_STREAMING`**:
  - `auto` (default): only Kyutai native partials
  - `on`: enable best-effort partials for Whisper/FunASR too (auto-disables per session if too slow)
  - `off`: disable partial STT updates
- **`STT_STREAM_INTERVAL_MS`** (default `600`): how often to send partials (Whisper/FunASR).
- **`STT_STREAM_WINDOW_MS`** (default `1600`): sliding window size used for partial decoding (Whisper/FunASR).
- **`STT_STREAM_MAX_RTF`** (default `1.0`): if decode is slower than realtime, partials are disabled for that session.

## How to Run

1. **Start the Server:**

   Make sure your virtual environment is activated. In the `voice_assistant` directory, run the following command:

   ```bash
   uvicorn server.main:app --reload
   ```

   The server will start on `http://localhost:8000`.
2. **Open the Client:**

   Open your web browser and navigate to:

   [http://localhost:8000/](http://localhost:8000/)

   The web page will load, and the status should indicate that it is "Ready to speak".
3. **Interact with the Assistant:**

   - Click and hold the **"Hold to Speak"** button.
   - Your browser will likely ask for permission to use your microphone. Please allow it.
   - Speak your query.
   - Release the button when you are finished.
   - The server will process your speech, stream the LLM reply, and play audio chunks through your browser as they are synthesized.

## Client-streaming message types

- `user_text`: final transcription.
- `llm_partial`: growing assistant text as tokens stream.
- `llm_chunk`: text chunk that was just sent to TTS (optional for display).
- `audio_chunk`: base64 WAV chunk; play in arrival order.
- `stream_done`: final assistant text; marks end of this turn.
- `error`: message if something went wrong.

Enjoy your private, locally-run voice assistant!
