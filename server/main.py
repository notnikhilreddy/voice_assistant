import torch
# Allow trusted checkpoints that rely on getattr during torch.load unpickling
torch.serialization.add_safe_globals([getattr])

import asyncio
import base64
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .services import stt, tts, llm

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_CLIENT_DIR = BASE_DIR / "web_client"


@app.on_event("startup")
async def preload_models():
    """Load server-side models on startup to avoid first-request latency."""
    try:
        stt.preload_stt()
    except Exception as e:
        print(f"STT preload failed: {e}")

    try:
        tts.preload_tts()
    except Exception as e:
        print(f"TTS preload failed: {e}")


@app.get("/")
async def get_client():
    """Serves the web client's HTML page."""
    return FileResponse(WEB_CLIENT_DIR / "index.html")

@app.get("/main.js")
async def get_client_js():
    """Serves the web client's JavaScript file."""
    return FileResponse(WEB_CLIENT_DIR / "main.js")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    # Maintain conversation history per WebSocket connection so the LLM can stay in context.
    conversation_history = []

    try:
        while True:
            # Receive audio data from the client.
            # We expect the client to send audio chunks as bytes,
            # and a final text message "END_OF_STREAM" to signal completion.
            audio_data = bytearray()
            while True:
                message = await websocket.receive()
                msg_type = message.get("type")

                # Handle client disconnect mid-stream
                if msg_type == "websocket.disconnect":
                    raise WebSocketDisconnect()

                # Text control messages
                text_payload = message.get("text")
                if text_payload is not None:
                    if text_payload == "END_OF_STREAM":
                        break
                    # Ignore any other text frames
                    continue

                # Binary audio chunks
                chunk = message.get("bytes")
                if chunk:
                    audio_data.extend(chunk)
                else:
                    # No useful payload; continue waiting
                    continue

            if not audio_data:
                continue

            overall_start = perf_counter()

            # 1. Transcribe audio to text
            stt_start = perf_counter()
            user_text = stt.transcribe_audio(bytes(audio_data))
            stt_ms = (perf_counter() - stt_start) * 1000
            if not user_text:
                await websocket.send_json({"type": "error", "message": "Sorry, I couldn't understand that."})
                continue
            
            await websocket.send_json({"type": "user_text", "data": user_text})
            # Yield to the event loop so the client sees the user text before we block on LLM/TTS.
            await asyncio.sleep(0)


            # 2. Get a response from the language model
            llm_start = perf_counter()
            llm_response_text = llm.get_llm_response(user_text, history=conversation_history[-12:])
            llm_ms = (perf_counter() - llm_start) * 1000
            await websocket.send_json({"type": "llm_text", "data": llm_response_text})
            # Update history after we have a successful LLM response.
            conversation_history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": llm_response_text},
                ]
            )


            # 3. Synthesize the response to speech (streaming)
            tts_start = perf_counter()
            chunk_count = 0
            first_chunk_ms = None
            for chunk in tts.stream_speech(llm_response_text):
                if not chunk:
                    continue
                chunk_count += 1
                if first_chunk_ms is None:
                    first_chunk_ms = (perf_counter() - overall_start) * 1000
                audio_base64 = base64.b64encode(chunk).decode("utf-8")
                await websocket.send_json(
                    {
                        "type": "audio_chunk",
                        "data": audio_base64,
                        "final": False,
                        "index": chunk_count - 1,
                    }
                )

            if chunk_count == 0:
                await websocket.send_json({"type": "error", "message": "Sorry, I had trouble generating a response."})
                continue

            tts_ms = (perf_counter() - tts_start) * 1000
            await websocket.send_json(
                {
                    "type": "audio_done",
                    "meta": {
                        "tts_ms": round(tts_ms, 1),
                        "chunks": chunk_count,
                        "first_chunk_ms": round(first_chunk_ms or 0, 1),
                    },
                }
            )
            await websocket.send_json(
                {
                    "type": "stream_done",
                    "data": llm_response_text,
                    "meta": {
                        "tts_ms": round(tts_ms, 1),
                        "chunks": chunk_count,
                        "first_chunk_ms": round(first_chunk_ms or 0, 1),
                    },
                }
            )
            total_ms = (perf_counter() - overall_start) * 1000
            llm_to_first_ms = (
                max(0.0, (first_chunk_ms or 0) - stt_ms - llm_ms) if first_chunk_ms is not None else 0.0
            )
            timing_payload = {
                "stt_ms": round(stt_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "tts_ms": round(tts_ms, 1),
                "first_chunk_ms": round(first_chunk_ms or 0, 1),
                "llm_to_first_ms": round(llm_to_first_ms, 1),
                "total_ms": round(total_ms, 1),
            }
            # Concise turn log focused on latency to first audio
            print(
                "[turn] "
                f"stt={timing_payload['stt_ms']}ms, "
                f"llm={timing_payload['llm_ms']}ms, "
                f"first_audio={timing_payload['first_chunk_ms']}ms, "
                f"llm_to_first={timing_payload['llm_to_first_ms']}ms, "
                f"tts_total={timing_payload['tts_ms']}ms, "
                f"total={timing_payload['total_ms']}ms, "
                f"chunks={chunk_count} | "
                f"user=\"{user_text}\" | assistant=\"{llm_response_text}\""
            )
            await websocket.send_json({"type": "timing", "data": timing_payload})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting server. Make sure you have an .env file with your GROQ_API_KEY.")
    uvicorn.run(app, host="0.0.0.0", port=8000)