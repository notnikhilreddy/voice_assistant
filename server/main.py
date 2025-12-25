import torch
# Allow trusted checkpoints that rely on getattr during torch.load unpickling
torch.serialization.add_safe_globals([getattr])

import asyncio
import base64
import io
import json
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .services import stt, tts, llm
from .services.sentence_splitter import detect_complete_sentences
from .services.turn_taking import TurnTakingEngine, TurnState
from .services.turn_taking.ring_buffer import AudioFrame

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


def _float32_to_wav_bytes(audio_f32: np.ndarray, sample_rate: int = 16000) -> bytes:
    if audio_f32.dtype != np.float32:
        audio_f32 = audio_f32.astype(np.float32)
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (audio_f32 * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def _parse_pcm_frame(payload: bytes) -> tuple[int, bytes]:
    """
    PCM frame wire format (binary):
      - uint32 little-endian: seq
      - rest: PCM16LE mono samples (typically 20ms @ 16k => 640 bytes)
    """
    if len(payload) < 4:
        raise ValueError("PCM frame too short")
    seq = struct.unpack_from("<I", payload, 0)[0]
    pcm16 = payload[4:]
    return int(seq), pcm16


async def _legacy_ws_loop(
    websocket: WebSocket,
    conversation_history: list,
    history_summary: str,
    turn_id_start: int,
    initial_message: dict | None,
) -> tuple[list, str, int]:
    """
    Legacy mode: client uploads a full MediaRecorder blob, then sends END_OF_STREAM.
    Returns updated (conversation_history, history_summary, next_turn_id).
    """
    turn_id = turn_id_start
    try:
        while True:
            audio_data = bytearray()

            # Consume initial message (if provided) as the first chunk of this stream.
            pending = [initial_message] if initial_message else []
            initial_message = None

            while True:
                message = pending.pop(0) if pending else await websocket.receive()
                msg_type = message.get("type")

                if msg_type == "websocket.disconnect":
                    raise WebSocketDisconnect()

                text_payload = message.get("text")
                if text_payload is not None:
                    if text_payload == "END_OF_STREAM":
                        break
                    continue

                chunk = message.get("bytes")
                if chunk:
                    audio_data.extend(chunk)
                else:
                    continue

            if not audio_data:
                continue

            overall_start = perf_counter()

            stt_start = perf_counter()
            user_text = stt.transcribe_audio(bytes(audio_data))
            stt_ms = (perf_counter() - stt_start) * 1000
            if not user_text:
                await websocket.send_json({"type": "error", "message": "Sorry, I couldn't understand that."})
                continue

            await websocket.send_json({"type": "user_text", "data": user_text})
            await asyncio.sleep(0)

            llm_start = perf_counter()
            llm_response_text = ""
            sentence_buffer = ""
            first_audio_ms = None
            first_sentence_ms = None
            audio_chunk_count = 0
            sentence_count = 0

            # Strict ordering for audio delivery
            audio_by_sentence: dict[int, list[bytes]] = {}
            sentence_chunk_counts: dict[int, int] = {}
            send_condition = asyncio.Condition()
            llm_done = False
            tts_done_sentences: set[int] = set()

            tts_tasks: list[asyncio.Task] = []

            async def process_sentence_for_tts(sentence: str, sentence_idx: int):
                nonlocal audio_chunk_count, first_audio_ms, first_sentence_ms
                try:
                    def synthesize_sentence(s: str):
                        return list(tts.stream_speech(s))

                    audio_chunks = await asyncio.get_event_loop().run_in_executor(
                        tts_executor, synthesize_sentence, sentence
                    )
                    filtered = [c for c in audio_chunks if c]
                    async with send_condition:
                        audio_by_sentence[sentence_idx] = filtered
                        sentence_chunk_counts[sentence_idx] = len(filtered)
                        tts_done_sentences.add(sentence_idx)
                        send_condition.notify_all()
                except Exception as e:
                    print(f"Error in TTS for sentence {sentence_idx}: {e}")

            async def ordered_audio_sender():
                nonlocal audio_chunk_count, first_audio_ms, first_sentence_ms
                next_sentence = 0
                while True:
                    async with send_condition:
                        await send_condition.wait_for(
                            lambda: (next_sentence in tts_done_sentences)
                            or (llm_done and next_sentence >= sentence_count)
                        )
                        if llm_done and next_sentence >= sentence_count:
                            return
                        chunks = audio_by_sentence.get(next_sentence, [])
                        chunk_count = sentence_chunk_counts.get(next_sentence, len(chunks))

                    for chunk_idx, audio_chunk in enumerate(chunks):
                        audio_chunk_count += 1
                        if first_audio_ms is None:
                            first_audio_ms = (perf_counter() - overall_start) * 1000
                            first_sentence_ms = (perf_counter() - llm_start) * 1000
                        audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                        await websocket.send_json(
                            {
                                "type": "audio_chunk",
                                "data": audio_base64,
                                "turn_id": turn_id,
                                "sentence_idx": next_sentence,
                                "chunk_idx": chunk_idx,
                                "chunk_count": chunk_count,
                                "final": False,
                                "index": audio_chunk_count - 1,
                            }
                        )
                        await asyncio.sleep(0)

                    await websocket.send_json(
                        {
                            "type": "sentence_done",
                            "turn_id": turn_id,
                            "sentence_idx": next_sentence,
                            "chunk_count": chunk_count,
                        }
                    )
                    next_sentence += 1

            sender_task = asyncio.create_task(ordered_audio_sender())

            async for token_chunk in llm.stream_llm_response(
                user_text, history=conversation_history, history_summary=history_summary
            ):
                llm_response_text += token_chunk
                await websocket.send_json({"type": "llm_partial", "data": llm_response_text})
                complete_sentences, sentence_buffer = detect_complete_sentences(token_chunk, sentence_buffer)
                for sentence in complete_sentences:
                    sentence_count += 1
                    task = asyncio.create_task(process_sentence_for_tts(sentence, sentence_count - 1))
                    tts_tasks.append(task)
                    await websocket.send_json(
                        {"type": "llm_chunk", "data": sentence, "sentence_idx": sentence_count - 1}
                    )

            if tts_tasks:
                await asyncio.gather(*tts_tasks, return_exceptions=True)

            llm_ms = (perf_counter() - llm_start) * 1000

            if sentence_buffer.strip():
                sentence_count += 1
                final_sentence = sentence_buffer.strip()
                await process_sentence_for_tts(final_sentence, sentence_count - 1)
                await websocket.send_json(
                    {"type": "llm_chunk", "data": final_sentence, "sentence_idx": sentence_count - 1}
                )

            async with send_condition:
                llm_done = True
                send_condition.notify_all()
            try:
                await sender_task
            except Exception as e:
                print(f"Audio sender task error: {e}")

            conversation_history.extend(
                [{"role": "user", "content": user_text}, {"role": "assistant", "content": llm_response_text}]
            )
            try:
                if llm.should_summarize(conversation_history):
                    keep_n = getattr(llm, "HISTORY_KEEP_LAST_MESSAGES", 24)
                    older = conversation_history[:-keep_n]
                    if older:
                        history_summary = llm.summarize_history_locally(older, existing_summary=history_summary)
                        conversation_history = conversation_history[-keep_n:]
            except Exception as e:
                print(f"History summarization failed (continuing without summary): {e}")

            if audio_chunk_count == 0:
                await websocket.send_json({"type": "error", "message": "Sorry, I had trouble generating a response."})
                continue

            await websocket.send_json(
                {
                    "type": "stream_done",
                    "data": llm_response_text,
                    "turn_id": turn_id,
                    "meta": {
                        "sentences": sentence_count,
                        "audio_chunks": audio_chunk_count,
                        "first_audio_ms": round(first_audio_ms or 0, 1),
                        "first_sentence_ms": round(first_sentence_ms or 0, 1),
                    },
                }
            )

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
            await websocket.send_json({"type": "timing", "data": timing_payload})
            turn_id += 1
    except WebSocketDisconnect:
        raise


async def _pcm_stream_ws_loop(websocket: WebSocket):
    """
    New mode: client streams 20ms PCM16 frames; server does turn-taking (VAD+prosody)
    and triggers STT+LLM+TTS as soon as USER_FINISHED is detected.
    """
    conversation_history: list = []
    history_summary = ""
    turn_id = 0

    engine = TurnTakingEngine(sample_rate=16000, frame_ms=20, ring_seconds=2.5)
    speaking = False
    barge_in_ms = 0.0

    # To cancel an in-flight response (barge-in)
    current_response_task: asyncio.Task | None = None
    current_cancel_event: asyncio.Event | None = None

    await websocket.send_json({"type": "pcm_stream_ready"})

    while True:
        message = await websocket.receive()
        msg_type = message.get("type")
        if msg_type == "websocket.disconnect":
            raise WebSocketDisconnect()

        text_payload = message.get("text")
        if text_payload is not None:
            # Optional control: allow client to request cancel/stop.
            try:
                obj = json.loads(text_payload)
                if obj.get("type") == "cancel":
                    if current_cancel_event:
                        current_cancel_event.set()
                    if current_response_task:
                        current_response_task.cancel()
                    await websocket.send_json({"type": "cancelled"})
            except Exception:
                pass
            continue

        payload = message.get("bytes")
        if not payload:
            continue

        try:
            seq, pcm16 = _parse_pcm_frame(payload)
        except Exception:
            continue

        audio_frame = AudioFrame(seq=seq, pcm16=pcm16, sample_rate=16000)
        decision, segment = engine.push(audio_frame)

        # Optional: send turn-state telemetry at low rate (every 5 frames).
        if seq % 5 == 0:
            await websocket.send_json(
                {
                    "type": "turn_state",
                    "state": decision.state,
                    "p_speech": round(decision.p_speech, 3),
                    "silence_ms": round(decision.silence_ms, 1),
                }
            )

        # Barge-in: if we are speaking and user starts talking, cancel response immediately.
        if speaking and decision.p_speech >= engine.p_speech_on:
            barge_in_ms += engine.frame_ms
            if barge_in_ms >= engine.barge_in_speech_ms:
                barge_in_ms = 0.0
                if current_cancel_event:
                    current_cancel_event.set()
                if current_response_task:
                    current_response_task.cancel()
                await websocket.send_json({"type": "barge_in", "turn_id": turn_id})
                speaking = False
        else:
            barge_in_ms = 0.0

        if segment is None or decision.state != TurnState.USER_FINISHED:
            continue

        # Convert segment to WAV bytes and run STT -> LLM -> TTS.
        wav_bytes = _float32_to_wav_bytes(segment.audio_f32, sample_rate=16000)
        user_text = stt.transcribe_audio(wav_bytes)
        if not user_text:
            await websocket.send_json({"type": "error", "message": "Sorry, I couldn't understand that."})
            continue

        await websocket.send_json({"type": "user_text", "data": user_text, "turn_id": turn_id})

        cancel_event = asyncio.Event()
        current_cancel_event = cancel_event

        async def run_response():
            nonlocal history_summary, conversation_history, speaking
            speaking = True

            llm_start = perf_counter()
            llm_response_text = ""
            sentence_buffer = ""
            audio_chunk_count = 0
            sentence_count = 0

            # Strict ordering buffers
            audio_by_sentence: dict[int, list[bytes]] = {}
            sentence_chunk_counts: dict[int, int] = {}
            send_condition = asyncio.Condition()
            llm_done = False
            tts_done_sentences: set[int] = set()
            tts_tasks: list[asyncio.Task] = []

            async def process_sentence_for_tts(sentence: str, sentence_idx: int):
                try:
                    def synthesize_sentence(s: str):
                        return list(tts.stream_speech(s))

                    chunks = await asyncio.get_event_loop().run_in_executor(
                        tts_executor, synthesize_sentence, sentence
                    )
                    filtered = [c for c in chunks if c]
                    async with send_condition:
                        audio_by_sentence[sentence_idx] = filtered
                        sentence_chunk_counts[sentence_idx] = len(filtered)
                        tts_done_sentences.add(sentence_idx)
                        send_condition.notify_all()
                except Exception as e:
                    print(f"Error in TTS for sentence {sentence_idx}: {e}")

            async def ordered_audio_sender():
                nonlocal audio_chunk_count
                next_sentence = 0
                while True:
                    if cancel_event.is_set():
                        return
                    async with send_condition:
                        await send_condition.wait_for(
                            lambda: (next_sentence in tts_done_sentences)
                            or (llm_done and next_sentence >= sentence_count)
                            or cancel_event.is_set()
                        )
                        if cancel_event.is_set():
                            return
                        if llm_done and next_sentence >= sentence_count:
                            return
                        chunks = audio_by_sentence.get(next_sentence, [])
                        chunk_count = sentence_chunk_counts.get(next_sentence, len(chunks))

                    for chunk_idx, audio_chunk in enumerate(chunks):
                        if cancel_event.is_set():
                            return
                        audio_chunk_count += 1
                        audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                        await websocket.send_json(
                            {
                                "type": "audio_chunk",
                                "data": audio_base64,
                                "turn_id": turn_id,
                                "sentence_idx": next_sentence,
                                "chunk_idx": chunk_idx,
                                "chunk_count": chunk_count,
                                "final": False,
                                "index": audio_chunk_count - 1,
                            }
                        )
                        await asyncio.sleep(0)

                    await websocket.send_json(
                        {
                            "type": "sentence_done",
                            "turn_id": turn_id,
                            "sentence_idx": next_sentence,
                            "chunk_count": chunk_count,
                        }
                    )
                    next_sentence += 1

            sender_task = asyncio.create_task(ordered_audio_sender())

            try:
                async for token_chunk in llm.stream_llm_response(
                    user_text, history=conversation_history, history_summary=history_summary
                ):
                    if cancel_event.is_set():
                        break
                    llm_response_text += token_chunk
                    await websocket.send_json({"type": "llm_partial", "data": llm_response_text, "turn_id": turn_id})
                    complete_sentences, sentence_buffer = detect_complete_sentences(token_chunk, sentence_buffer)
                    for sentence in complete_sentences:
                        sentence_count += 1
                        task = asyncio.create_task(process_sentence_for_tts(sentence, sentence_count - 1))
                        tts_tasks.append(task)
                        await websocket.send_json(
                            {"type": "llm_chunk", "data": sentence, "sentence_idx": sentence_count - 1, "turn_id": turn_id}
                        )
            finally:
                if tts_tasks:
                    await asyncio.gather(*tts_tasks, return_exceptions=True)

                if not cancel_event.is_set() and sentence_buffer.strip():
                    sentence_count += 1
                    final_sentence = sentence_buffer.strip()
                    await process_sentence_for_tts(final_sentence, sentence_count - 1)
                    await websocket.send_json(
                        {"type": "llm_chunk", "data": final_sentence, "sentence_idx": sentence_count - 1, "turn_id": turn_id}
                    )

                async with send_condition:
                    llm_done = True
                    send_condition.notify_all()

                try:
                    await sender_task
                except Exception as e:
                    print(f"Audio sender task error: {e}")

            if cancel_event.is_set():
                speaking = False
                return

            conversation_history.extend(
                [{"role": "user", "content": user_text}, {"role": "assistant", "content": llm_response_text}]
            )
            try:
                if llm.should_summarize(conversation_history):
                    keep_n = getattr(llm, "HISTORY_KEEP_LAST_MESSAGES", 24)
                    older = conversation_history[:-keep_n]
                    if older:
                        history_summary = llm.summarize_history_locally(older, existing_summary=history_summary)
                        conversation_history = conversation_history[-keep_n:]
            except Exception as e:
                print(f"History summarization failed (continuing without summary): {e}")

            await websocket.send_json({"type": "stream_done", "data": llm_response_text, "turn_id": turn_id})
            speaking = False

        # Cancel any prior response task before starting new one.
        if current_response_task:
            try:
                current_response_task.cancel()
            except Exception:
                pass
        current_response_task = asyncio.create_task(run_response())
        turn_id += 1


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    # Decide protocol based on the first received message.
    initial_message = None
    try:
        initial_message = await websocket.receive()
        if initial_message.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect()

        text_payload = initial_message.get("text")
        if text_payload is not None:
            try:
                obj = json.loads(text_payload)
                if obj.get("type") == "pcm_stream_start":
                    await _pcm_stream_ws_loop(websocket)
                    return
            except Exception:
                pass

        # Otherwise: legacy mode, using initial_message as first chunk/control.
        conversation_history: list = []
        history_summary = ""
        await _legacy_ws_loop(
            websocket,
            conversation_history=conversation_history,
            history_summary=history_summary,
            turn_id_start=0,
            initial_message=initial_message,
        )
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting server. Make sure you have an .env file with your GROQ_API_KEY.")
    uvicorn.run(app, host="0.0.0.0", port=8000)