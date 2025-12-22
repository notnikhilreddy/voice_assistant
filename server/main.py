import torch
# Allow trusted checkpoints that rely on getattr during torch.load unpickling
torch.serialization.add_safe_globals([getattr])

import asyncio
import base64
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .services import stt, tts, llm

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_CLIENT_DIR = BASE_DIR / "web_client"


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

            print("Received audio stream, processing...")

            # 1. Transcribe audio to text
            print("Transcribing...")
            user_text = stt.transcribe_audio(bytes(audio_data))
            if not user_text:
                print("Transcription failed or produced no text.")
                await websocket.send_json({"type": "error", "message": "Sorry, I couldn't understand that."})
                continue
            
            print(f"User said: {user_text}")
            await websocket.send_json({"type": "user_text", "data": user_text})


            # 2. Get a response from the language model
            print("Getting LLM response...")
            llm_response_text = llm.get_llm_response(user_text)
            print(f"LLM responded: {llm_response_text}")
            await websocket.send_json({"type": "llm_text", "data": llm_response_text})


            # 3. Synthesize the response to speech
            print("Synthesizing response...")
            synthesized_audio = tts.synthesize_speech(llm_response_text)
            if not synthesized_audio:
                print("TTS failed.")
                await websocket.send_json({"type": "error", "message": "Sorry, I had trouble generating a response."})
                continue


            # 4. Send the synthesized audio back to the client
            # We send it as a base64 encoded string within a JSON object.
            audio_base64 = base64.b64encode(synthesized_audio).decode('utf-8')
            await websocket.send_json({
                "type": "audio",
                "data": audio_base64
            })
            print("Sent synthesized audio to client.")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting server. Make sure you have an .env file with your GROQ_API_KEY.")
    uvicorn.run(app, host="0.0.0.0", port=8000)