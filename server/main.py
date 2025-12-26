import torch
# Allow trusted checkpoints that rely on getattr during torch.load unpickling
torch.serialization.add_safe_globals([getattr])

import asyncio
import base64
import io
import json
import struct
import time
import logging
import warnings
import os
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(message)s")
log = logging.getLogger("voice_assistant")
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)

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


def _pack_audio_header(turn_id: int, sentence_idx: int, chunk_idx: int, chunk_count: int) -> bytes:
    # 16-byte little-endian header: turn_id, sentence_idx, chunk_idx, chunk_count
    return struct.pack("<IIII", int(turn_id), int(sentence_idx), int(chunk_idx), int(chunk_count))


def _unpack_audio_header(payload: bytes) -> tuple[int, int, int, int, bytes]:
    if len(payload) < 16:
        raise ValueError("audio payload too short")
    turn_id, sentence_idx, chunk_idx, chunk_count = struct.unpack_from("<IIII", payload, 0)
    return int(turn_id), int(sentence_idx), int(chunk_idx), int(chunk_count), payload[16:]


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

    # Mode: idle | manual | auto
    mode = "idle"
    manual_pcm_parts: list[bytes] = []
    manual_active = False
    manual_turn_id = 0
    manual_partial_text = ""

    # Optional Kyutai token streamer for manual partials (only when STT_BACKEND=kyutai)
    manual_kyutai_streamer = None
    if getattr(stt, "STT_BACKEND", "").lower() == "kyutai":
        try:
            from .services.turn_taking.kyutai_semvad import KyutaiSemanticVAD
            manual_kyutai_streamer = KyutaiSemanticVAD(input_rate=16000)
        except Exception:
            manual_kyutai_streamer = None

    # To cancel an in-flight response (barge-in)
    current_response_task: asyncio.Task | None = None
    current_cancel_event: asyncio.Event | None = None

    await websocket.send_json({"type": "pcm_stream_ready"})

    # Turn timing telemetry (server epoch)
    turn_metrics: dict[int, dict] = {}

    # STT partial streaming policy
    import os
    stt_streaming = os.getenv("STT_STREAMING", "auto").lower()  # off|auto|on
    stt_stream_interval_ms = int(os.getenv("STT_STREAM_INTERVAL_MS", "600"))
    stt_stream_window_ms = int(os.getenv("STT_STREAM_WINDOW_MS", "1600"))
    stt_stream_max_rtf = float(os.getenv("STT_STREAM_MAX_RTF", "1.0"))
    last_partial_ts = 0.0
    partial_enabled = True
    partial_accum = ""

    while True:
        message = await websocket.receive()
        msg_type = message.get("type")
        if msg_type == "websocket.disconnect":
            raise WebSocketDisconnect()

        text_payload = message.get("text")
        if text_payload is not None:
            # Control plane messages (mode, manual start/end, cancel).
            try:
                obj = json.loads(text_payload)
                t = obj.get("type")
                if t == "mode":
                    mode = str(obj.get("value") or "idle")
                    if mode not in ("idle", "manual", "auto"):
                        mode = "idle"
                    await websocket.send_json({"type": "mode_ack", "value": mode})
                    continue
                if t == "manual_start":
                    manual_active = True
                    manual_pcm_parts = []
                    manual_turn_id += 1
                    manual_partial_text = ""
                    if manual_kyutai_streamer is not None:
                        try:
                            manual_kyutai_streamer.reset()
                        except Exception:
                            pass
                    await websocket.send_json({"type": "manual_ack", "value": "started", "turn_id": manual_turn_id})
                    continue
                if t == "manual_end":
                    manual_active = False
                    mode = "idle"
                    await websocket.send_json({"type": "manual_ack", "value": "ended", "turn_id": manual_turn_id})

                    if manual_pcm_parts:
                        turn_end_epoch_ms = time.time() * 1000.0
                        turn_metrics[manual_turn_id] = {
                            "turn_id": manual_turn_id,
                            "mode": "manual",
                            "turn_end_epoch_ms": turn_end_epoch_ms,
                            "stt_done_ms": None,
                            "llm_first_token_ms": None,
                            "llm_done_ms": None,
                            "first_audio_sent_ms": None,
                            "client_audio_start_ms": None,
                            "client_audio_done_ms": None,
                        }
                        # Run STT once on full manual recording; no turn-taking involved.
                        pcm16 = b"".join(manual_pcm_parts)
                        audio_i16 = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
                        audio_f32 = (audio_i16 / 32768.0).clip(-1.0, 1.0)
                        wav_bytes = _float32_to_wav_bytes(audio_f32, sample_rate=16000)
                        user_text = stt.transcribe_audio(wav_bytes)
                        turn_metrics[manual_turn_id]["stt_done_ms"] = time.time() * 1000.0
                        if not user_text:
                            await websocket.send_json({"type": "error", "message": "Sorry, I couldn't understand that."})
                            continue
                        await websocket.send_json({"type": "user_text", "data": user_text, "turn_id": manual_turn_id})

                        # Reuse existing response runner pattern
                        cancel_event = asyncio.Event()
                        current_cancel_event = cancel_event

                        async def run_response_manual():
                            nonlocal speaking, history_summary, conversation_history
                            speaking = True
                            llm_response_text = ""
                            sentence_buffer = ""
                            audio_chunk_count = 0
                            sentence_count = 0

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
                                        if turn_metrics.get(manual_turn_id, {}).get("first_audio_sent_ms") is None:
                                            turn_metrics[manual_turn_id]["first_audio_sent_ms"] = time.time() * 1000.0
                                        payload = _pack_audio_header(manual_turn_id, next_sentence, chunk_idx, chunk_count) + audio_chunk
                                        await websocket.send_bytes(payload)
                                        await asyncio.sleep(0)

                                    await websocket.send_json(
                                        {
                                            "type": "sentence_done",
                                            "turn_id": manual_turn_id,
                                            "sentence_idx": next_sentence,
                                            "chunk_count": chunk_count,
                                        }
                                    )
                                    next_sentence += 1

                            sender_task = asyncio.create_task(ordered_audio_sender())

                            try:
                                # LLM timing: first token and completion
                                async for token_chunk in llm.stream_llm_response(
                                    user_text, history=conversation_history, history_summary=history_summary
                                ):
                                    if cancel_event.is_set():
                                        break
                                    if turn_metrics.get(manual_turn_id, {}).get("llm_first_token_ms") is None:
                                        turn_metrics[manual_turn_id]["llm_first_token_ms"] = time.time() * 1000.0
                                    llm_response_text += token_chunk
                                    await websocket.send_json({"type": "llm_partial", "data": llm_response_text, "turn_id": manual_turn_id})
                                    complete_sentences, sentence_buffer = detect_complete_sentences(token_chunk, sentence_buffer)
                                    for sentence in complete_sentences:
                                        sentence_count += 1
                                        task = asyncio.create_task(process_sentence_for_tts(sentence, sentence_count - 1))
                                        tts_tasks.append(task)
                                        await websocket.send_json(
                                            {"type": "llm_chunk", "data": sentence, "sentence_idx": sentence_count - 1, "turn_id": manual_turn_id}
                                        )
                            finally:
                                if tts_tasks:
                                    await asyncio.gather(*tts_tasks, return_exceptions=True)

                                if not cancel_event.is_set() and sentence_buffer.strip():
                                    sentence_count += 1
                                    final_sentence = sentence_buffer.strip()
                                    await process_sentence_for_tts(final_sentence, sentence_count - 1)
                                    await websocket.send_json(
                                        {"type": "llm_chunk", "data": final_sentence, "sentence_idx": sentence_count - 1, "turn_id": manual_turn_id}
                                    )

                                async with send_condition:
                                    llm_done = True
                                    send_condition.notify_all()

                                try:
                                    await sender_task
                                except Exception as e:
                                    print(f"Audio sender task error: {e}")
                                turn_metrics[manual_turn_id]["llm_done_ms"] = time.time() * 1000.0

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

                            await websocket.send_json({"type": "stream_done", "data": llm_response_text, "turn_id": manual_turn_id})
                            speaking = False

                            m = turn_metrics.get(manual_turn_id, {})
                            if m:
                                te = m.get("turn_end_epoch_ms")
                                log.info(
                "[turn] "
                                    f"id={manual_turn_id} mode=manual "
                                    f"end_to_stt={((m.get('stt_done_ms')-te) if m.get('stt_done_ms') and te else 0):.0f}ms "
                                    f"end_to_llm_first={((m.get('llm_first_token_ms')-te) if m.get('llm_first_token_ms') and te else 0):.0f}ms "
                                    f"end_to_audio_sent={((m.get('first_audio_sent_ms')-te) if m.get('first_audio_sent_ms') and te else 0):.0f}ms "
                                    f"end_to_client_audio_start={((m.get('client_audio_start_ms')-te) if m.get('client_audio_start_ms') and te else 0):.0f}ms "
                                    f"llm_total={((m.get('llm_done_ms')-m.get('llm_first_token_ms')) if m.get('llm_done_ms') and m.get('llm_first_token_ms') else 0):.0f}ms "
                                    f"audio_play_total={((m.get('client_audio_done_ms')-m.get('client_audio_start_ms')) if m.get('client_audio_done_ms') and m.get('client_audio_start_ms') else 0):.0f}ms "
                                    f"turn_total={((m.get('client_audio_done_ms')-te) if m.get('client_audio_done_ms') and te else 0):.0f}ms"
                                )

                        if current_response_task:
                            try:
                                current_response_task.cancel()
                            except Exception:
                                pass
                        current_response_task = asyncio.create_task(run_response_manual())
                    continue
                if t == "client_audio_started":
                    tid = int(obj.get("turn_id", -1))
                    ts = float(obj.get("client_epoch_ms", 0.0))
                    if tid in turn_metrics:
                        turn_metrics[tid]["client_audio_start_ms"] = ts
                    continue
                if t == "client_audio_done":
                    tid = int(obj.get("turn_id", -1))
                    ts = float(obj.get("client_epoch_ms", 0.0))
                    if tid in turn_metrics:
                        turn_metrics[tid]["client_audio_done_ms"] = ts
                    continue
                if obj.get("type") == "cancel":
                    if current_cancel_event:
                        current_cancel_event.set()
                    if current_response_task:
                        current_response_task.cancel()
                    await websocket.send_json({"type": "cancelled"})
            except Exception:
                pass
            continue

        # Client audio telemetry (binary playback timing)
        # Note: telemetry is sent as JSON text frames.

        payload = message.get("bytes")
        if not payload:
            continue

        try:
            seq, pcm16 = _parse_pcm_frame(payload)
        except Exception:
            continue

        audio_frame = AudioFrame(seq=seq, pcm16=pcm16, sample_rate=16000)
        # Manual mode: buffer audio only (no turn-taking). Auto mode: use turn-taking.
        if mode == "manual" and manual_active:
            manual_pcm_parts.append(pcm16)
            # Optional Kyutai partial text (fast, no turn-taking)
            if manual_kyutai_streamer is not None:
                try:
                    step = manual_kyutai_streamer.push_16k(audio_frame.as_float32())
                    if step is not None:
                        # Prefer decoded full text (stable, avoids <0x..> artifacts)
                        if getattr(step, "text", ""):
                            manual_partial_text = str(step.text).strip()
                        elif step.text_delta:
                            manual_partial_text += step.text_delta
                        await websocket.send_json(
                            {"type": "stt_partial", "data": manual_partial_text.strip(), "turn_id": manual_turn_id}
                        )
                except Exception:
                    pass
            continue

        decision, segment = engine.push(audio_frame)

        # Auto mode partials (display should follow STT_BACKEND; kyutai turn-taker text can contain token artifacts)
        if mode == "auto":
            now = perf_counter()
            # Note: turn-taking backend may be smart_turn; STT partials should be driven by STT streaming policy.

            # Whisper/FunASR best-effort partials only if STT_STREAMING=on
            if stt_streaming == "on" and partial_enabled and stt.STT_BACKEND in ("whisper", "mlx_funasr", "funasr"):
                if (now - last_partial_ts) * 1000 >= stt_stream_interval_ms:
                    # decode on a sliding window from the ring buffer
                    window_frames = engine.ring.slice_by_seq(max(0, seq - int(stt_stream_window_ms / engine.frame_ms)), seq)
                    audio_f32 = engine.ring.concat_float32(window_frames)
                    audio_ms = (audio_f32.shape[0] / 16000.0) * 1000.0
                    if audio_ms >= 300:
                        t0 = perf_counter()
                        try:
                            # Use the backend-specific float path directly for speed
                            if stt.STT_BACKEND == "whisper":
                                txt = stt._transcribe_with_whisper(audio_f32)
                            elif stt.STT_BACKEND == "mlx_funasr":
                                txt = stt._transcribe_with_mlx_funasr(audio_f32)
                            else:
                                txt = stt._transcribe_with_funasr(audio_f32)
                        except Exception:
                            txt = ""
                        dt = (perf_counter() - t0)
                        rtf = dt / max(0.001, (audio_f32.shape[0] / 16000.0))
                        if rtf > stt_stream_max_rtf:
                            partial_enabled = False
                        if txt and txt != partial_accum:
                            partial_accum = txt
                            last_partial_ts = now
                            await websocket.send_json({"type": "stt_partial", "data": txt, "turn_id": turn_id})

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

        # Auto turn-end timing
        turn_end_epoch_ms = time.time() * 1000.0
        turn_metrics[turn_id] = {
            "turn_id": turn_id,
            "mode": "auto",
            "turn_end_epoch_ms": turn_end_epoch_ms,
            "stt_done_ms": None,
            "llm_first_token_ms": None,
            "llm_done_ms": None,
            "first_audio_sent_ms": None,
            "client_audio_start_ms": None,
            "client_audio_done_ms": None,
        }
        # Help client compute end->play_start with shared epoch
        await websocket.send_json({"type": "turn_end", "turn_id": turn_id, "turn_end_epoch_ms": turn_end_epoch_ms})

        # Convert segment to WAV bytes and run STT -> LLM -> TTS.
        # If STT_BACKEND=kyutai, still do a final full transcription at turn-end
        # (partial streaming may refine once full context is available).
        wav_bytes = _float32_to_wav_bytes(segment.audio_f32, sample_rate=16000)
        user_text = stt.transcribe_audio(wav_bytes)
        turn_metrics[turn_id]["stt_done_ms"] = time.time() * 1000.0
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
                        if turn_metrics.get(turn_id, {}).get("first_audio_sent_ms") is None:
                            turn_metrics[turn_id]["first_audio_sent_ms"] = time.time() * 1000.0
                        payload = _pack_audio_header(turn_id, next_sentence, chunk_idx, chunk_count) + audio_chunk
                        await websocket.send_bytes(payload)
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
                    if turn_metrics.get(turn_id, {}).get("llm_first_token_ms") is None:
                        turn_metrics[turn_id]["llm_first_token_ms"] = time.time() * 1000.0
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
                turn_metrics[turn_id]["llm_done_ms"] = time.time() * 1000.0

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

            m = turn_metrics.get(turn_id, {})
            if m:
                te = m.get("turn_end_epoch_ms")
                log.info(
                    "[turn] "
                    f"id={turn_id} mode=auto "
                    f"end_to_stt={((m.get('stt_done_ms')-te) if m.get('stt_done_ms') and te else 0):.0f}ms "
                    f"end_to_llm_first={((m.get('llm_first_token_ms')-te) if m.get('llm_first_token_ms') and te else 0):.0f}ms "
                    f"end_to_audio_sent={((m.get('first_audio_sent_ms')-te) if m.get('first_audio_sent_ms') and te else 0):.0f}ms "
                    f"end_to_client_audio_start={((m.get('client_audio_start_ms')-te) if m.get('client_audio_start_ms') and te else 0):.0f}ms "
                    f"llm_total={((m.get('llm_done_ms')-m.get('llm_first_token_ms')) if m.get('llm_done_ms') and m.get('llm_first_token_ms') else 0):.0f}ms "
                    f"audio_play_total={((m.get('client_audio_done_ms')-m.get('client_audio_start_ms')) if m.get('client_audio_done_ms') and m.get('client_audio_start_ms') else 0):.0f}ms "
                    f"turn_total={((m.get('client_audio_done_ms')-te) if m.get('client_audio_done_ms') and te else 0):.0f}ms"
                )

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