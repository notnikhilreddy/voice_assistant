import torch
# Allow trusted checkpoints that rely on getattr during torch.load unpickling
torch.serialization.add_safe_globals([getattr])

import asyncio
import base64
import io
import json
import os
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .services import stt, tts, llm
from .services.metrics_dashboard import get_store, start_dashboard
from .services.sentence_splitter import detect_complete_sentences
from .services.turn_taking import TurnTakingEngine, TurnState
from .services.turn_taking.ring_buffer import AudioFrame

app = FastAPI()

IGNORED_USER_TEXTS = {"you", "Thank you."}

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_CLIENT_DIR = BASE_DIR / "web_client"

# Thread pool executor for CPU-bound TTS operations
tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

async def _stream_tts_chunks(
    text: str,
    out_q: "asyncio.Queue[bytes]",
    done_evt: asyncio.Event,
    cancel_evt: asyncio.Event | None = None,
) -> None:
    """
    Run TTS in executor but stream chunks back to asyncio as they are produced.
    This restores low-latency first-audio while still allowing ordered playback.
    """
    loop = asyncio.get_running_loop()

    def _run():
        try:
            for chunk in tts.stream_speech(text):
                if cancel_evt is not None and cancel_evt.is_set():
                    break
                if not chunk:
                    continue
                loop.call_soon_threadsafe(out_q.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(done_evt.set)

    await loop.run_in_executor(tts_executor, _run)


@app.on_event("startup")
async def preload_models():
    """Load server-side models on startup to avoid first-request latency."""
    start_dashboard()
    try:
        stt.preload_stt()
    except Exception as e:
        print(f"STT preload failed: {e}")

    try:
        tts.preload_tts()
    except Exception as e:
        print(f"TTS preload failed: {e}")

    try:
        # Pre-download and load SmartTurn model if configured
        if os.getenv("TURN_TAKING_BACKEND") == "smart_turn":
            from .services.turn_taking.smart_turn import preload_smart_turn
            preload_smart_turn()
    except Exception as e:
        print(f"SmartTurn preload failed: {e}")


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
            if user_text in IGNORED_USER_TEXTS:
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
            
            # Strict ordering for audio delivery, while still streaming chunks ASAP.
            llm_done = False
            sentence_queues: dict[int, asyncio.Queue[bytes]] = {}
            sentence_done: dict[int, asyncio.Event] = {}
            sentence_ready = asyncio.Event()
            tts_tasks: list[asyncio.Task] = []
            
            async def process_sentence_for_tts(sentence: str, sentence_idx: int):
                nonlocal audio_chunk_count, first_audio_ms, first_sentence_ms
                try:
                    q = sentence_queues.get(sentence_idx)
                    if q is None:
                        q = asyncio.Queue()
                        sentence_queues[sentence_idx] = q
                    done_evt = sentence_done.get(sentence_idx)
                    if done_evt is None:
                        done_evt = asyncio.Event()
                        sentence_done[sentence_idx] = done_evt
                    sentence_ready.set()
                    await _stream_tts_chunks(sentence, q, done_evt, cancel_evt=None)
                except Exception as e:
                    print(f"Error in TTS for sentence {sentence_idx}: {e}")
                    # Mark sentence done so sender can progress.
                    if sentence_idx not in sentence_done:
                        sentence_done[sentence_idx] = asyncio.Event()
                    sentence_done[sentence_idx].set()

            async def ordered_audio_sender():
                nonlocal audio_chunk_count, first_audio_ms, first_sentence_ms
                next_sentence = 0
                while True:
                    if llm_done and next_sentence >= sentence_count and next_sentence not in sentence_queues:
                        return
                    if next_sentence not in sentence_queues:
                        if llm_done and next_sentence >= sentence_count:
                            return
                        await sentence_ready.wait()
                        sentence_ready.clear()
                        continue

                    q = sentence_queues[next_sentence]
                    done_evt = sentence_done.get(next_sentence) or asyncio.Event()
                    sentence_done[next_sentence] = done_evt
                    sent = 0
                    while True:
                        if q.empty() and done_evt.is_set():
                            break
                        get_task = asyncio.create_task(q.get())
                        done_task = asyncio.create_task(done_evt.wait())
                        done_set, _ = await asyncio.wait(
                            {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if get_task in done_set:
                            done_task.cancel()
                            audio_chunk = get_task.result()
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
                                    "chunk_idx": sent,
                                    "chunk_count": None,
                                    "final": False,
                                    "index": audio_chunk_count - 1,
                                }
                            )
                            sent += 1
                            await asyncio.sleep(0)
                        else:
                            get_task.cancel()
                            break

                    await websocket.send_json(
                        {
                            "type": "sentence_done",
                            "turn_id": turn_id,
                            "sentence_idx": next_sentence,
                            "chunk_count": sent,
                        }
                    )
                    next_sentence += 1

            sender_task = asyncio.create_task(ordered_audio_sender())

            pending_llm_delta = ""
            last_llm_send_ts = perf_counter()
            async for token_chunk in llm.stream_llm_response(
                user_text, history=conversation_history, history_summary=history_summary
            ):
                llm_response_text += token_chunk
                pending_llm_delta += token_chunk
                tnow = perf_counter()
                if (tnow - last_llm_send_ts) >= 0.08 or len(pending_llm_delta) >= 48:
                    await websocket.send_json({"type": "llm_token", "data": pending_llm_delta})
                    pending_llm_delta = ""
                    last_llm_send_ts = tnow
                complete_sentences, sentence_buffer = detect_complete_sentences(token_chunk, sentence_buffer)
                for sentence in complete_sentences:
                    sentence_count += 1
                    task = asyncio.create_task(process_sentence_for_tts(sentence, sentence_count - 1))
                    tts_tasks.append(task)
                    await websocket.send_json(
                        {"type": "llm_chunk", "data": sentence, "sentence_idx": sentence_count - 1}
                    )
            if pending_llm_delta:
                await websocket.send_json({"type": "llm_token", "data": pending_llm_delta})
            
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
    # Multi-conversation support (ChatGPT-style UI): keep independent LLM histories
    # keyed by conversation_id within this websocket session.
    from uuid import uuid4

    conversation_id = "default"
    conversations: dict[str, dict[str, object]] = {conversation_id: {"history": [], "summary": ""}}

    def _ensure_conv(cid: str) -> dict[str, object]:
        cid = cid or "default"
        st = conversations.get(cid)
        if st is None:
            st = {"history": [], "summary": ""}
            conversations[cid] = st
        return st

    def _normalize_client_history(raw) -> list[dict]:
        """
        Accept a client-provided history payload and normalize it to:
          [{"role": "user"|"assistant", "content": "..."}...]
        """
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if content == "/sil" or content in IGNORED_USER_TEXTS:
                continue
            out.append({"role": role, "content": content})
        return out

    ws_alive = True

    async def _send_json(payload: dict, *, cid: str | None = None) -> None:
        nonlocal ws_alive
        if not ws_alive:
            return
        use_cid = cid or conversation_id
        if "conversation_id" not in payload:
            payload["conversation_id"] = use_cid
        try:
            await websocket.send_json(payload)
        except Exception:
            # If the client disconnects (or the server reloads), stop all in-flight work.
            ws_alive = False
            if current_cancel_event:
                current_cancel_event.set()
            if current_response_task:
                try:
                    current_response_task.cancel()
                except Exception:
                    pass
            raise WebSocketDisconnect()

    # Active conversation state (updated by select/new conversation control messages)
    _st0 = _ensure_conv(conversation_id)
    conversation_history: list = list(_st0.get("history") or [])
    history_summary: str = str(_st0.get("summary") or "")
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
    manual_user_end_ts: float | None = None

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

    await _send_json({"type": "pcm_stream_ready"})

    # STT partial streaming policy
    import os
    stt_streaming = os.getenv("STT_STREAMING", "on").lower()  # off|auto|on
    stt_stream_interval_ms = int(os.getenv("STT_STREAM_INTERVAL_MS", "600"))
    stt_stream_window_ms = int(os.getenv("STT_STREAM_WINDOW_MS", "1600"))
    stt_stream_max_rtf = float(os.getenv("STT_STREAM_MAX_RTF", "1.0"))
    last_partial_ts = 0.0
    partial_enabled = True
    partial_accum = ""
    last_manual_partial_ts = 0.0
    manual_partial_accum = ""

    # Filters to avoid "garbage transcription" from near-silent/noise-triggered segments.
    stt_min_peak = float(os.getenv("STT_MIN_PEAK", "0.008"))

    def _looks_like_garbage_text(t: str) -> bool:
        s = (t or "").strip()
        if not s:
            return True
        if s == "/sil" or s == "/sil>":
            return True
        # Drop strings that are mostly punctuation / non-alnum noise.
        compact = "".join(ch for ch in s if not ch.isspace())
        if len(compact) >= 8:
            alnum = sum(1 for ch in compact if ch.isalnum())
            if (alnum / max(1, len(compact))) < 0.45:
                return True
        # Drop control/replacement-heavy outputs.
        bad = sum(1 for ch in s if ord(ch) < 32 or ch == "�")
        if len(s) >= 8 and (bad / len(s)) > 0.05:
            return True
        return False

    while True:
        try:
            message = await websocket.receive()
        except RuntimeError:
            # Starlette raises RuntimeError if receive() is called after disconnect was observed.
            raise WebSocketDisconnect()

        msg_type = message.get("type")
        if msg_type == "websocket.disconnect":
            raise WebSocketDisconnect()

        text_payload = message.get("text")
        if text_payload is not None:
            # Control plane messages (mode, manual start/end, cancel).
            try:
                obj = json.loads(text_payload)
                t = obj.get("type")
                if t in ("new_conversation", "select_conversation"):
                    cid = str(obj.get("conversation_id") or "").strip() or str(uuid4())
                    conversation_id = cid
                    st = _ensure_conv(conversation_id)

                    # If the client sends history (e.g., after page refresh), use it
                    # so the LLM continues in the same context.
                    incoming_hist = obj.get("history")
                    if isinstance(incoming_hist, list) and incoming_hist:
                        st["history"] = _normalize_client_history(incoming_hist)

                    conversation_history = list(st.get("history") or [])
                    history_summary = str(st.get("summary") or "")
                    await _send_json({"type": "conversation_ack", "conversation_id": conversation_id})
                    continue
                if t == "mode":
                    mode = str(obj.get("value") or "idle")
                    if mode not in ("idle", "manual", "auto"):
                        mode = "idle"
                    await _send_json({"type": "mode_ack", "value": mode})
                    continue
                if t == "client_audio_started":
                    try:
                        tid = int(obj.get("turn_id"))
                        store = get_store()
                        now = perf_counter()
                        for tm in store.snapshot():
                            if tm.turn_id == tid and tm.total_turn_ms is None:
                                store.set_client_audio_start(tid, (now - tm.end_ts) * 1000.0)
                                break
                    except Exception:
                        pass
                    continue
                if t == "text_input":
                    # Manual text chat input (ChatGPT-like). No STT/turn-taking; just LLM -> TTS streaming.
                    text = str(obj.get("text") or "").strip()
                    if not text or text == "/sil":
                        continue
                    if text in IGNORED_USER_TEXTS:
                        continue

                    # Bind to the requested conversation id (or current one).
                    conv_id_local = str(obj.get("conversation_id") or conversation_id or "default")
                    conversation_id = conv_id_local
                    conv_state_local = _ensure_conv(conv_id_local)
                    conv_history_local: list = list(conv_state_local.get("history") or [])
                    conv_summary_local: str = str(conv_state_local.get("summary") or "")

                    # Cancel any prior response so text feels snappy.
                    if current_cancel_event:
                        current_cancel_event.set()
                    if current_response_task:
                        current_response_task.cancel()

                    store = get_store()
                    user_end_ts = perf_counter()
                    store.start_turn(turn_id, "text", user_end_ts)
                    store.set_stt_done(turn_id, 0.0)

                    # Client already renders the typed user message immediately.
                    # Do NOT echo user_text here, or it will appear twice.

                    cancel_event = asyncio.Event()
                    current_cancel_event = cancel_event

                    async def run_response_text():
                        nonlocal speaking, turn_id, conv_history_local, conv_summary_local
                        speaking = True
                        llm_response_text = ""
                        sentence_buffer = ""
                        audio_chunk_count = 0
                        sentence_count = 0

                        sentence_queues: dict[int, asyncio.Queue[bytes]] = {}
                        sentence_done: dict[int, asyncio.Event] = {}
                        sentence_ready = asyncio.Event()
                        llm_done = False
                        tts_tasks: list[asyncio.Task] = []

                        async def process_sentence_for_tts(sentence: str, sentence_idx: int):
                            try:
                                q = sentence_queues.get(sentence_idx)
                                if q is None:
                                    q = asyncio.Queue()
                                    sentence_queues[sentence_idx] = q
                                done_evt = sentence_done.get(sentence_idx)
                                if done_evt is None:
                                    done_evt = asyncio.Event()
                                    sentence_done[sentence_idx] = done_evt
                                sentence_ready.set()
                                await _stream_tts_chunks(sentence, q, done_evt, cancel_evt=cancel_event)
                            except Exception as e:
                                print(f"Error in TTS for sentence {sentence_idx}: {e}")
                                if sentence_idx not in sentence_done:
                                    sentence_done[sentence_idx] = asyncio.Event()
                                sentence_done[sentence_idx].set()

                        async def ordered_audio_sender():
                            nonlocal audio_chunk_count
                            next_sentence = 0
                            while True:
                                if cancel_event.is_set():
                                    return
                                if llm_done and next_sentence >= sentence_count and next_sentence not in sentence_queues:
                                    return
                                if next_sentence not in sentence_queues:
                                    if llm_done and next_sentence >= sentence_count:
                                        return
                                    await sentence_ready.wait()
                                    sentence_ready.clear()
                                    continue

                                q = sentence_queues[next_sentence]
                                done_evt = sentence_done.get(next_sentence) or asyncio.Event()
                                sentence_done[next_sentence] = done_evt
                                sent = 0
                                while True:
                                    if cancel_event.is_set():
                                        return
                                    if q.empty() and done_evt.is_set():
                                        break
                                    get_task = asyncio.create_task(q.get())
                                    done_task = asyncio.create_task(done_evt.wait())
                                    done_set, _ = await asyncio.wait(
                                        {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                                    )
                                    if get_task in done_set:
                                        done_task.cancel()
                                        audio_chunk = get_task.result()
                                        audio_chunk_count += 1
                                        store.add_audio_chunk(turn_id, len(audio_chunk))
                                        audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                                        await _send_json(
                                            {
                                                "type": "audio_chunk",
                                                "data": audio_base64,
                                                "turn_id": turn_id,
                                                "sentence_idx": next_sentence,
                                                "chunk_idx": sent,
                                                "chunk_count": None,
                                                "final": False,
                                                "index": audio_chunk_count - 1,
                                            },
                                            cid=conv_id_local,
                                        )
                                        sent += 1
                                        await asyncio.sleep(0)
                                    else:
                                        get_task.cancel()
                                        break

                                await _send_json(
                                    {
                                        "type": "sentence_done",
                                        "turn_id": turn_id,
                                        "sentence_idx": next_sentence,
                                        "chunk_count": sent,
                                    },
                                    cid=conv_id_local,
                                )
                                next_sentence += 1

                        sender_task = asyncio.create_task(ordered_audio_sender())

                        try:
                            pending_llm_delta = ""
                            last_llm_send_ts = perf_counter()
                            async for token_chunk in llm.stream_llm_response(
                                text, history=conv_history_local, history_summary=conv_summary_local
                            ):
                                if cancel_event.is_set():
                                    break
                                store.set_llm_first(turn_id, (perf_counter() - user_end_ts) * 1000.0)
                                llm_response_text += token_chunk
                                pending_llm_delta += token_chunk
                                tnow = perf_counter()
                                if (tnow - last_llm_send_ts) >= 0.08 or len(pending_llm_delta) >= 48:
                                    await _send_json(
                                        {"type": "llm_token", "data": pending_llm_delta, "turn_id": turn_id},
                                        cid=conv_id_local,
                                    )
                                    pending_llm_delta = ""
                                    last_llm_send_ts = tnow
                                complete_sentences, sentence_buffer = detect_complete_sentences(token_chunk, sentence_buffer)
                                for sentence in complete_sentences:
                                    sentence_count += 1
                                    task = asyncio.create_task(process_sentence_for_tts(sentence, sentence_count - 1))
                                    tts_tasks.append(task)
                        finally:
                            if pending_llm_delta:
                                await _send_json(
                                    {"type": "llm_token", "data": pending_llm_delta, "turn_id": turn_id},
                                    cid=conv_id_local,
                                )
                            if tts_tasks:
                                await asyncio.gather(*tts_tasks, return_exceptions=True)

                            if not cancel_event.is_set() and sentence_buffer.strip():
                                sentence_count += 1
                                final_sentence = sentence_buffer.strip()
                                await process_sentence_for_tts(final_sentence, sentence_count - 1)

                            llm_done = True
                            sentence_ready.set()

                            try:
                                await sender_task
                            except Exception as e:
                                print(f"Audio sender task error: {e}")

                        if cancel_event.is_set():
                            speaking = False
                            return

                        conv_history_local.extend(
                            [{"role": "user", "content": text}, {"role": "assistant", "content": llm_response_text}]
                        )
                        try:
                            if llm.should_summarize(conv_history_local):
                                keep_n = getattr(llm, "HISTORY_KEEP_LAST_MESSAGES", 24)
                                older = conv_history_local[:-keep_n]
                                if older:
                                    conv_summary_local = llm.summarize_history_locally(
                                        older, existing_summary=conv_summary_local
                                    )
                                    conv_history_local = conv_history_local[-keep_n:]
                        except Exception as e:
                            print(f"History summarization failed (continuing without summary): {e}")

                        conv_state_local["history"] = conv_history_local
                        conv_state_local["summary"] = conv_summary_local

                        await _send_json({"type": "stream_done", "data": llm_response_text, "turn_id": turn_id}, cid=conv_id_local)
                        tm = store.finish_turn(turn_id, (perf_counter() - user_end_ts) * 1000.0)
                        if tm:
                            print(
                                f"[turn_id={tm.turn_id} mode={tm.mode}] "
                                f"end→stt={tm.stt_done_ms or 0:.0f}ms "
                                f"end→llm1={tm.llm_first_ms or 0:.0f}ms "
                                f"end→client_audio={tm.client_audio_start_ms or 0:.0f}ms "
                                f"turn_total={tm.total_turn_ms or 0:.0f}ms "
                                f"chunks={tm.audio_chunks} sizes={tm.chunk_sizes}"
                            )
                        speaking = False

                    current_response_task = asyncio.create_task(run_response_text())
                    turn_id += 1
                    continue
                if t == "manual_start":
                    manual_active = True
                    manual_pcm_parts = []
                    manual_turn_id += 1
                    manual_partial_text = ""
                    manual_partial_accum = ""
                    manual_user_end_ts = None
                    # Start inflight metrics (end_ts updated on manual_end)
                    get_store().start_turn(manual_turn_id, "manual", perf_counter())
                    if manual_kyutai_streamer is not None:
                        try:
                            manual_kyutai_streamer.reset()
                        except Exception:
                            pass
                    await _send_json({"type": "manual_ack", "value": "started", "turn_id": manual_turn_id})
                    continue
                if t == "manual_end":
                    manual_active = False
                    mode = "idle"
                    await _send_json({"type": "manual_ack", "value": "ended", "turn_id": manual_turn_id})

                    if manual_pcm_parts:
                        store = get_store()
                        manual_user_end_ts = perf_counter()
                        store.start_turn(manual_turn_id, "manual", manual_user_end_ts)
                        # Run STT once on full manual recording; no turn-taking involved.
                        pcm16 = b"".join(manual_pcm_parts)
                        audio_i16 = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
                        audio_f32 = (audio_i16 / 32768.0).clip(-1.0, 1.0)
                        wav_bytes = _float32_to_wav_bytes(audio_f32, sample_rate=16000)
                        user_text = stt.transcribe_audio(wav_bytes)
                        if user_text and user_text.strip() == "/sil":
                            continue
                        store.set_stt_done(manual_turn_id, (perf_counter() - manual_user_end_ts) * 1000.0)
                        if not user_text:
                            await _send_json({"type": "error", "message": "Sorry, I couldn't understand that."})
                            continue
                        if user_text in IGNORED_USER_TEXTS:
                            continue
                        conv_id_local = conversation_id
                        conv_state_local = _ensure_conv(conv_id_local)
                        conv_history_local: list = list(conv_state_local.get("history") or [])
                        conv_summary_local: str = str(conv_state_local.get("summary") or "")
                        await _send_json(
                            {"type": "user_text", "data": user_text, "turn_id": manual_turn_id},
                            cid=conv_id_local,
                        )

                        # Reuse existing response runner pattern
                        cancel_event = asyncio.Event()
                        current_cancel_event = cancel_event

                        async def run_response_manual():
                            nonlocal speaking, conv_history_local, conv_summary_local
                            speaking = True
                            llm_response_text = ""
                            sentence_buffer = ""
                            audio_chunk_count = 0
                            sentence_count = 0

                            sentence_queues: dict[int, asyncio.Queue[bytes]] = {}
                            sentence_done: dict[int, asyncio.Event] = {}
                            sentence_ready = asyncio.Event()
                            llm_done = False
                            tts_tasks: list[asyncio.Task] = []

                            async def process_sentence_for_tts(sentence: str, sentence_idx: int):
                                try:
                                    q = sentence_queues.get(sentence_idx)
                                    if q is None:
                                        q = asyncio.Queue()
                                        sentence_queues[sentence_idx] = q
                                    done_evt = sentence_done.get(sentence_idx)
                                    if done_evt is None:
                                        done_evt = asyncio.Event()
                                        sentence_done[sentence_idx] = done_evt
                                    sentence_ready.set()
                                    await _stream_tts_chunks(sentence, q, done_evt, cancel_evt=cancel_event)
                                except Exception as e:
                                    print(f"Error in TTS for sentence {sentence_idx}: {e}")
                                    if sentence_idx not in sentence_done:
                                        sentence_done[sentence_idx] = asyncio.Event()
                                    sentence_done[sentence_idx].set()

                            async def ordered_audio_sender():
                                nonlocal audio_chunk_count
                                next_sentence = 0
                                while True:
                                    if cancel_event.is_set():
                                        return
                                    if llm_done and next_sentence >= sentence_count and next_sentence not in sentence_queues:
                                        return
                                    if next_sentence not in sentence_queues:
                                        if llm_done and next_sentence >= sentence_count:
                                            return
                                        await sentence_ready.wait()
                                        sentence_ready.clear()
                                        continue

                                    q = sentence_queues[next_sentence]
                                    done_evt = sentence_done.get(next_sentence) or asyncio.Event()
                                    sentence_done[next_sentence] = done_evt
                                    sent = 0
                                    while True:
                                        if cancel_event.is_set():
                                            return
                                        if q.empty() and done_evt.is_set():
                                            break
                                        get_task = asyncio.create_task(q.get())
                                        done_task = asyncio.create_task(done_evt.wait())
                                        done_set, _ = await asyncio.wait(
                                            {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                                        )
                                        if get_task in done_set:
                                            done_task.cancel()
                                            audio_chunk = get_task.result()
                                            audio_chunk_count += 1
                                            get_store().add_audio_chunk(manual_turn_id, len(audio_chunk))
                                            audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                                            await _send_json(
                                                {
                                                    "type": "audio_chunk",
                                                    "data": audio_base64,
                                                    "turn_id": manual_turn_id,
                                                    "sentence_idx": next_sentence,
                                                    "chunk_idx": sent,
                                                    "chunk_count": None,
                                                    "final": False,
                                                    "index": audio_chunk_count - 1,
                                                }
                                                , cid=conv_id_local
                                            )
                                            sent += 1
                                            await asyncio.sleep(0)
                                        else:
                                            get_task.cancel()
                                            break

                                    await _send_json(
                                        {
                                            "type": "sentence_done",
                                            "turn_id": manual_turn_id,
                                            "sentence_idx": next_sentence,
                                            "chunk_count": sent,
                                        }
                                        , cid=conv_id_local
                                    )
                                    next_sentence += 1

                            sender_task = asyncio.create_task(ordered_audio_sender())

                            try:
                                pending_llm_delta = ""
                                last_llm_send_ts = perf_counter()
                                async for token_chunk in llm.stream_llm_response(
                                    user_text, history=conv_history_local, history_summary=conv_summary_local
                                ):
                                    if cancel_event.is_set():
                                        break
                                    if manual_user_end_ts is not None:
                                        get_store().set_llm_first(
                                            manual_turn_id, (perf_counter() - manual_user_end_ts) * 1000.0
                                        )
                                    llm_response_text += token_chunk
                                    pending_llm_delta += token_chunk
                                    tnow = perf_counter()
                                    if (tnow - last_llm_send_ts) >= 0.08 or len(pending_llm_delta) >= 48:
                                        await _send_json(
                                            {
                                                "type": "llm_token",
                                                "data": pending_llm_delta,
                                                "turn_id": manual_turn_id,
                                            },
                                            cid=conv_id_local,
                                        )
                                        pending_llm_delta = ""
                                        last_llm_send_ts = tnow
                                    complete_sentences, sentence_buffer = detect_complete_sentences(token_chunk, sentence_buffer)
                                    for sentence in complete_sentences:
                                        sentence_count += 1
                                        task = asyncio.create_task(process_sentence_for_tts(sentence, sentence_count - 1))
                                        tts_tasks.append(task)
                                        await _send_json(
                                            {"type": "llm_chunk", "data": sentence, "sentence_idx": sentence_count - 1, "turn_id": manual_turn_id},
                                            cid=conv_id_local,
                                        )
                            finally:
                                if pending_llm_delta:
                                    await _send_json(
                                        {"type": "llm_token", "data": pending_llm_delta, "turn_id": manual_turn_id},
                                        cid=conv_id_local,
                                    )
                                if tts_tasks:
                                    await asyncio.gather(*tts_tasks, return_exceptions=True)

                                if not cancel_event.is_set() and sentence_buffer.strip():
                                    sentence_count += 1
                                    final_sentence = sentence_buffer.strip()
                                    await process_sentence_for_tts(final_sentence, sentence_count - 1)
                                    await _send_json(
                                        {"type": "llm_chunk", "data": final_sentence, "sentence_idx": sentence_count - 1, "turn_id": manual_turn_id},
                                        cid=conv_id_local,
                                    )

                                llm_done = True
                                sentence_ready.set()

                                try:
                                    await sender_task
                                except Exception as e:
                                    print(f"Audio sender task error: {e}")

                            if cancel_event.is_set():
                                speaking = False
                                return

                            conv_history_local.extend(
                                [{"role": "user", "content": user_text}, {"role": "assistant", "content": llm_response_text}]
                            )
                            try:
                                if llm.should_summarize(conv_history_local):
                                    keep_n = getattr(llm, "HISTORY_KEEP_LAST_MESSAGES", 24)
                                    older = conv_history_local[:-keep_n]
                                    if older:
                                        conv_summary_local = llm.summarize_history_locally(older, existing_summary=conv_summary_local)
                                        conv_history_local = conv_history_local[-keep_n:]
                            except Exception as e:
                                print(f"History summarization failed (continuing without summary): {e}")

                            # Persist the conversation state back to this conversation_id.
                            conv_state_local["history"] = conv_history_local
                            conv_state_local["summary"] = conv_summary_local

                            await _send_json(
                                {"type": "stream_done", "data": llm_response_text, "turn_id": manual_turn_id},
                                cid=conv_id_local,
                            )
                            if manual_user_end_ts is not None:
                                tm = get_store().finish_turn(manual_turn_id, (perf_counter() - manual_user_end_ts) * 1000.0)
                                if tm:
                                    print(
                                        f"[turn_id={tm.turn_id} mode={tm.mode}] "
                                        f"end→stt={tm.stt_done_ms or 0:.0f}ms "
                                        f"end→llm1={tm.llm_first_ms or 0:.0f}ms "
                                        f"end→client_audio={tm.client_audio_start_ms or 0:.0f}ms "
                                        f"turn_total={tm.total_turn_ms or 0:.0f}ms "
                                        f"chunks={tm.audio_chunks} sizes={tm.chunk_sizes}"
                                    )
                            speaking = False

                        if current_response_task:
                            try:
                                current_response_task.cancel()
                            except Exception:
                                pass
                        current_response_task = asyncio.create_task(run_response_manual())
                    continue
                if obj.get("type") == "cancel":
                    if current_cancel_event:
                        current_cancel_event.set()
                    if current_response_task:
                        current_response_task.cancel()
                    await _send_json({"type": "cancelled"})
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
        # Manual mode: buffer audio only (no turn-taking). Auto mode: use turn-taking.
        if mode == "manual" and manual_active:
            manual_pcm_parts.append(pcm16)

            # FunASR streaming partials for manual mode
            if stt_streaming == "on" and partial_enabled and stt.STT_BACKEND == "mlx_funasr":
                now = perf_counter()
                if (now - last_manual_partial_ts) * 1000 >= stt_stream_interval_ms:
                    # transcribe accumulated manual pcm
                    pcm_all = b"".join(manual_pcm_parts)
                    audio_i16 = np.frombuffer(pcm_all, dtype=np.int16).astype(np.float32)
                    audio_f32 = (audio_i16 / 32768.0).clip(-1.0, 1.0)
                    
                    if audio_f32.shape[0] >= 4800:  # at least 300ms
                        try:
                            # Use streaming transcription for token-level updates
                            # Note: stt_token data is the delta, accumulated is full text
                            full_text_now = ""
                            pending_delta = ""
                            last_send_ts = perf_counter()
                            prev_accum = manual_partial_accum or ""

                            for token in stt._transcribe_with_mlx_funasr_streaming(audio_f32):
                                if not token:
                                    continue
                                full_text_now += token

                                # Compute delta vs previous accumulated best-effort.
                                if full_text_now.startswith(prev_accum):
                                    delta = full_text_now[len(prev_accum):]
                                else:
                                    delta = full_text_now
                                pending_delta = delta

                                # Throttle websocket sends (avoid spamming per-token).
                                tnow = perf_counter()
                                if (tnow - last_send_ts) >= 0.12 or len(pending_delta) >= 16:
                                    await _send_json(
                                        {
                                            "type": "stt_token",
                                            "data": pending_delta,
                                            "accumulated": full_text_now,
                                            "turn_id": manual_turn_id,
                                        }
                                    )
                                    last_send_ts = tnow

                            # Flush trailing update, and persist accum for next interval.
                            if full_text_now and full_text_now != prev_accum:
                                await _send_json(
                                    {
                                        "type": "stt_token",
                                        "data": full_text_now[len(prev_accum):] if full_text_now.startswith(prev_accum) else full_text_now,
                                        "accumulated": full_text_now,
                                        "turn_id": manual_turn_id,
                                    }
                                )
                                manual_partial_accum = full_text_now
                            last_manual_partial_ts = now
                        except Exception:
                            pass

            # Optional Kyutai partial text (fast, no turn-taking)
            if manual_kyutai_streamer is not None and stt.STT_BACKEND == "kyutai":
                try:
                    step = manual_kyutai_streamer.push_16k(audio_frame.as_float32())
                    if step is not None:
                        # Prefer decoded full text (stable, avoids <0x..> artifacts)
                        if getattr(step, "text", ""):
                            manual_partial_text = str(step.text).strip()
                        elif step.text_delta:
                            manual_partial_text += step.text_delta
                        await _send_json(
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
                        # If the window is effectively silent, don't run streaming STT (prevents garbage partials).
                        try:
                            if float(np.max(np.abs(audio_f32))) < stt_min_peak:
                                continue
                        except Exception:
                            pass
                        t0 = perf_counter()
                        try:
                            # Use streaming for FunASR MLX to get token-level updates
                            if stt.STT_BACKEND == "mlx_funasr":
                                # Use streaming transcription for token-level updates
                                full_text_now = ""
                                pending_delta = ""
                                last_send_ts = perf_counter()
                                prev_accum = partial_accum or ""
                                try:
                                    for token in stt._transcribe_with_mlx_funasr_streaming(audio_f32):
                                        if not token:
                                            continue
                                        full_text_now += token

                                        if full_text_now.startswith(prev_accum):
                                            delta = full_text_now[len(prev_accum):]
                                        else:
                                            delta = full_text_now
                                        pending_delta = delta

                                        tnow = perf_counter()
                                        if (tnow - last_send_ts) >= 0.12 or len(pending_delta) >= 16:
                                            await _send_json(
                                                {
                                                    "type": "stt_token",
                                                    "data": pending_delta,
                                                    "accumulated": full_text_now,
                                                    "turn_id": turn_id,
                                                }
                                            )
                                            last_send_ts = tnow

                                    # Flush final update for this interval.
                                    if full_text_now and full_text_now != prev_accum:
                                        await _send_json(
                                            {
                                                "type": "stt_token",
                                                "data": full_text_now[len(prev_accum):] if full_text_now.startswith(prev_accum) else full_text_now,
                                                "accumulated": full_text_now,
                                                "turn_id": turn_id,
                                            }
                                        )
                                    txt = full_text_now
                                    partial_accum = full_text_now
                                except Exception as e:
                                    # Fallback to non-streaming if streaming fails
                                    print(f"FunASR streaming failed, falling back to regular transcription: {e}")
                                    txt = stt._transcribe_with_mlx_funasr(audio_f32)
                            elif stt.STT_BACKEND == "whisper":
                                txt = stt._transcribe_with_whisper(audio_f32)
                            else:
                                txt = stt._transcribe_with_funasr(audio_f32)
                        except Exception:
                            txt = ""
                        dt = (perf_counter() - t0)
                        rtf = dt / max(0.001, (audio_f32.shape[0] / 16000.0))
                        if rtf > stt_stream_max_rtf:
                            partial_enabled = False
                        if txt and txt != partial_accum and not _looks_like_garbage_text(txt):
                            partial_accum = txt
                            last_partial_ts = now
                            # Only send full partial if not using token streaming (to avoid duplicate)
                            if stt.STT_BACKEND != "mlx_funasr":
                                await _send_json({"type": "stt_partial", "data": txt, "turn_id": turn_id})

        # Optional: send turn-state telemetry at low rate (every 5 frames).
        if seq % 5 == 0:
            await _send_json(
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
                await _send_json({"type": "barge_in", "turn_id": turn_id})
                speaking = False
        else:
            barge_in_ms = 0.0

        if segment is None or decision.state != TurnState.USER_FINISHED:
            continue

        # Reset partial accumulator for next turn
        partial_accum = ""

        # Bind this turn to the active conversation at the moment the user finishes.
        conv_id_local = conversation_id
        conv_state_local = _ensure_conv(conv_id_local)
        conv_history_local: list = list(conv_state_local.get("history") or [])
        conv_summary_local: str = str(conv_state_local.get("summary") or "")

        # Metrics: user end -> STT/LLM/audio/client playback
        store = get_store()
        user_end_ts = perf_counter()
        store.start_turn(turn_id, "auto", user_end_ts)

        # Convert segment to WAV bytes and run STT -> LLM -> TTS.
        # If STT_BACKEND=kyutai, still do a final full transcription at turn-end
        # (partial streaming may refine once full context is available).
        try:
            if float(np.max(np.abs(segment.audio_f32))) < stt_min_peak:
                # Treat as silence/noise-triggered turn; don't run STT/LLM.
                continue
        except Exception:
            pass
        wav_bytes = _float32_to_wav_bytes(segment.audio_f32, sample_rate=16000)
        user_text = stt.transcribe_audio(wav_bytes)
        if user_text and user_text.strip() == "/sil":
            continue
        store.set_stt_done(turn_id, (perf_counter() - user_end_ts) * 1000.0)
        if not user_text or _looks_like_garbage_text(user_text):
            await _send_json({"type": "error", "message": "Sorry, I couldn't understand that."}, cid=conv_id_local)
            continue
        if user_text in IGNORED_USER_TEXTS:
            continue

        await _send_json({"type": "user_text", "data": user_text, "turn_id": turn_id}, cid=conv_id_local)

        cancel_event = asyncio.Event()
        current_cancel_event = cancel_event

        async def run_response():
            nonlocal speaking, conv_history_local, conv_summary_local
            speaking = True

            llm_start = perf_counter()
            llm_response_text = ""
            sentence_buffer = ""
            audio_chunk_count = 0
            sentence_count = 0

            sentence_queues: dict[int, asyncio.Queue[bytes]] = {}
            sentence_done: dict[int, asyncio.Event] = {}
            sentence_ready = asyncio.Event()
            llm_done = False
            tts_tasks: list[asyncio.Task] = []

            async def process_sentence_for_tts(sentence: str, sentence_idx: int):
                try:
                    q = sentence_queues.get(sentence_idx)
                    if q is None:
                        q = asyncio.Queue()
                        sentence_queues[sentence_idx] = q
                    done_evt = sentence_done.get(sentence_idx)
                    if done_evt is None:
                        done_evt = asyncio.Event()
                        sentence_done[sentence_idx] = done_evt
                    sentence_ready.set()
                    await _stream_tts_chunks(sentence, q, done_evt, cancel_evt=cancel_event)
                except Exception as e:
                    print(f"Error in TTS for sentence {sentence_idx}: {e}")
                    if sentence_idx not in sentence_done:
                        sentence_done[sentence_idx] = asyncio.Event()
                    sentence_done[sentence_idx].set()

            async def ordered_audio_sender():
                nonlocal audio_chunk_count
                next_sentence = 0
                while True:
                    if cancel_event.is_set():
                        return
                    if llm_done and next_sentence >= sentence_count and next_sentence not in sentence_queues:
                        return
                    if next_sentence not in sentence_queues:
                        if llm_done and next_sentence >= sentence_count:
                            return
                        await sentence_ready.wait()
                        sentence_ready.clear()
                        continue

                    q = sentence_queues[next_sentence]
                    done_evt = sentence_done.get(next_sentence) or asyncio.Event()
                    sentence_done[next_sentence] = done_evt
                    sent = 0
                    while True:
                        if cancel_event.is_set():
                            return
                        if q.empty() and done_evt.is_set():
                            break
                        get_task = asyncio.create_task(q.get())
                        done_task = asyncio.create_task(done_evt.wait())
                        done_set, _ = await asyncio.wait(
                            {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if get_task in done_set:
                            done_task.cancel()
                            audio_chunk = get_task.result()
                            audio_chunk_count += 1
                            store.add_audio_chunk(turn_id, len(audio_chunk))
                            audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                            await _send_json(
                                {
                                    "type": "audio_chunk",
                                    "data": audio_base64,
                                    "turn_id": turn_id,
                                    "sentence_idx": next_sentence,
                                    "chunk_idx": sent,
                                    "chunk_count": None,
                                    "final": False,
                                    "index": audio_chunk_count - 1,
                                }
                                , cid=conv_id_local
                            )
                            sent += 1
                            await asyncio.sleep(0)
                        else:
                            get_task.cancel()
                            break

                    await _send_json(
                        {
                            "type": "sentence_done",
                            "turn_id": turn_id,
                            "sentence_idx": next_sentence,
                            "chunk_count": sent,
                        }
                        , cid=conv_id_local
                    )
                    next_sentence += 1

            sender_task = asyncio.create_task(ordered_audio_sender())

            try:
                try:
                    pending_llm_delta = ""
                    last_llm_send_ts = perf_counter()
                    async for token_chunk in llm.stream_llm_response(
                        user_text, history=conv_history_local, history_summary=conv_summary_local
                    ):
                        if cancel_event.is_set():
                            break
                        store.set_llm_first(turn_id, (perf_counter() - user_end_ts) * 1000.0)
                        llm_response_text += token_chunk
                        pending_llm_delta += token_chunk
                        tnow = perf_counter()
                        if (tnow - last_llm_send_ts) >= 0.08 or len(pending_llm_delta) >= 48:
                            await _send_json(
                                {"type": "llm_token", "data": pending_llm_delta, "turn_id": turn_id},
                                cid=conv_id_local,
                            )
                            pending_llm_delta = ""
                            last_llm_send_ts = tnow
                        complete_sentences, sentence_buffer = detect_complete_sentences(token_chunk, sentence_buffer)
                        for sentence in complete_sentences:
                            sentence_count += 1
                            task = asyncio.create_task(process_sentence_for_tts(sentence, sentence_count - 1))
                            tts_tasks.append(task)
                            await _send_json(
                                {"type": "llm_chunk", "data": sentence, "sentence_idx": sentence_count - 1, "turn_id": turn_id},
                                cid=conv_id_local,
                            )
                except Exception as e:
                    # Never let an exception wedge the session; surface it and mark as cancelled.
                    if not cancel_event.is_set():
                        await _send_json({"type": "error", "message": f"LLM error: {e}"}, cid=conv_id_local)
                    cancel_event.set()
                finally:
                    if pending_llm_delta:
                        await _send_json(
                            {"type": "llm_token", "data": pending_llm_delta, "turn_id": turn_id},
                            cid=conv_id_local,
                        )
                    if tts_tasks:
                        await asyncio.gather(*tts_tasks, return_exceptions=True)

                    if not cancel_event.is_set() and sentence_buffer.strip():
                        sentence_count += 1
                        final_sentence = sentence_buffer.strip()
                        await process_sentence_for_tts(final_sentence, sentence_count - 1)
                        await _send_json(
                            {"type": "llm_chunk", "data": final_sentence, "sentence_idx": sentence_count - 1, "turn_id": turn_id},
                            cid=conv_id_local,
                        )

                    llm_done = True
                    sentence_ready.set()

                    try:
                        await sender_task
                    except Exception as e:
                        print(f"Audio sender task error: {e}")
            finally:
                speaking = False

            if cancel_event.is_set():
                return

            conv_history_local.extend(
                [{"role": "user", "content": user_text}, {"role": "assistant", "content": llm_response_text}]
            )
            try:
                if llm.should_summarize(conv_history_local):
                    keep_n = getattr(llm, "HISTORY_KEEP_LAST_MESSAGES", 24)
                    older = conv_history_local[:-keep_n]
                    if older:
                        conv_summary_local = llm.summarize_history_locally(older, existing_summary=conv_summary_local)
                        conv_history_local = conv_history_local[-keep_n:]
            except Exception as e:
                print(f"History summarization failed (continuing without summary): {e}")

            conv_state_local["history"] = conv_history_local
            conv_state_local["summary"] = conv_summary_local

            await _send_json({"type": "stream_done", "data": llm_response_text, "turn_id": turn_id}, cid=conv_id_local)
            tm = store.finish_turn(turn_id, (perf_counter() - user_end_ts) * 1000.0)
            if tm:
                print(
                    f"[turn_id={tm.turn_id} mode={tm.mode}] "
                    f"end→stt={tm.stt_done_ms or 0:.0f}ms "
                    f"end→llm1={tm.llm_first_ms or 0:.0f}ms "
                    f"end→client_audio={tm.client_audio_start_ms or 0:.0f}ms "
                    f"turn_total={tm.total_turn_ms or 0:.0f}ms "
                    f"chunks={tm.audio_chunks} sizes={tm.chunk_sizes}"
                )
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