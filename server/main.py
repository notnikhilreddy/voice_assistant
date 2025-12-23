import torch
# Allow trusted checkpoints that rely on getattr during torch.load unpickling
torch.serialization.add_safe_globals([getattr])

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .services import stt, tts, llm
from .services.sentence_splitter import detect_complete_sentences

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_CLIENT_DIR = BASE_DIR / "web_client"

# Thread pool executor for CPU-bound TTS operations
tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")


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
            await asyncio.sleep(0)  # Yield to event loop

            # 2. Stream LLM response with sentence detection and concurrent TTS
            llm_start = perf_counter()
            llm_response_text = ""
            sentence_buffer = ""
            first_audio_ms = None
            first_sentence_ms = None
            audio_chunk_count = 0
            sentence_count = 0
            
            # Queue for sentences ready for TTS
            sentence_queue = asyncio.Queue()
            tts_tasks = []
            
            async def process_sentence_for_tts(sentence: str, sentence_idx: int):
                """Process a complete sentence through TTS and send audio chunks."""
                nonlocal first_audio_ms, audio_chunk_count
                try:
                    # Run TTS in executor to avoid blocking (TTS is CPU/GPU bound)
                    # Synthesize speech for this sentence (run in executor for async compatibility)
                    def synthesize_sentence(s: str):
                        return list(tts.stream_speech(s))
                    
                    audio_chunks = await asyncio.get_event_loop().run_in_executor(
                        tts_executor,
                        synthesize_sentence,
                        sentence
                    )
                    
                    for audio_chunk in audio_chunks:
                        if not audio_chunk:
                            continue
                        audio_chunk_count += 1
                        if first_audio_ms is None:
                            first_audio_ms = (perf_counter() - overall_start) * 1000
                            first_sentence_ms = (perf_counter() - llm_start) * 1000
                        
                        audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "data": audio_base64,
                            "sentence_idx": sentence_idx,
                            "final": False,
                            "index": audio_chunk_count - 1,
                        })
                except Exception as e:
                    print(f"Error in TTS for sentence {sentence_idx}: {e}")
            
            # Stream LLM tokens and detect sentences
            async for token_chunk in llm.stream_llm_response(user_text, history=conversation_history[-12:]):
                llm_response_text += token_chunk
                
                # Send partial text update to client
                await websocket.send_json({
                    "type": "llm_partial",
                    "data": llm_response_text
                })
                
                # Detect complete sentences
                complete_sentences, sentence_buffer = detect_complete_sentences(token_chunk, sentence_buffer)
                
                # Process each complete sentence through TTS concurrently
                for sentence in complete_sentences:
                    sentence_count += 1
                    # Start TTS task for this sentence (non-blocking)
                    task = asyncio.create_task(process_sentence_for_tts(sentence, sentence_count - 1))
                    tts_tasks.append(task)
                    
                    # Send sentence text to client
                    await websocket.send_json({
                        "type": "llm_chunk",
                        "data": sentence,
                        "sentence_idx": sentence_count - 1,
                    })
            
            # Wait for all TTS tasks to complete
            if tts_tasks:
                await asyncio.gather(*tts_tasks, return_exceptions=True)
            
            llm_ms = (perf_counter() - llm_start) * 1000
            
            # Process any remaining text in buffer as final sentence
            if sentence_buffer.strip():
                sentence_count += 1
                final_sentence = sentence_buffer.strip()
                await process_sentence_for_tts(final_sentence, sentence_count - 1)
                await websocket.send_json({
                    "type": "llm_chunk",
                    "data": final_sentence,
                    "sentence_idx": sentence_count - 1,
                })
            
            # Update conversation history
            conversation_history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": llm_response_text},
            ])
            
            # Note: We don't send llm_text here because llm_partial already updated the client
            # with the complete response. Sending llm_text would create a duplicate.
            
            if audio_chunk_count == 0:
                await websocket.send_json({"type": "error", "message": "Sorry, I had trouble generating a response."})
                continue
            
            await websocket.send_json({
                "type": "stream_done",
                "data": llm_response_text,
                "meta": {
                    "sentences": sentence_count,
                    "audio_chunks": audio_chunk_count,
                    "first_audio_ms": round(first_audio_ms or 0, 1),
                    "first_sentence_ms": round(first_sentence_ms or 0, 1),
                },
            })
            
            total_ms = (perf_counter() - overall_start) * 1000
            llm_to_first_ms = (
                max(0.0, (first_audio_ms or 0) - stt_ms - llm_ms) if first_audio_ms is not None else 0.0
            )
            timing_payload = {
                "stt_ms": round(stt_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "first_audio_ms": round(first_audio_ms or 0, 1),
                "first_sentence_ms": round(first_sentence_ms or 0, 1),
                "llm_to_first_ms": round(llm_to_first_ms, 1),
                "total_ms": round(total_ms, 1),
                "sentences": sentence_count,
                "audio_chunks": audio_chunk_count,
            }
            
            print(
                "[turn] "
                f"stt={timing_payload['stt_ms']}ms, "
                f"llm={timing_payload['llm_ms']}ms, "
                f"first_audio={timing_payload['first_audio_ms']}ms, "
                f"first_sentence={timing_payload['first_sentence_ms']}ms, "
                f"llm_to_first={timing_payload['llm_to_first_ms']}ms, "
                f"total={timing_payload['total_ms']}ms, "
                f"sentences={sentence_count}, chunks={audio_chunk_count} | "
                f"user=\"{user_text}\" | assistant=\"{llm_response_text[:50]}...\""
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