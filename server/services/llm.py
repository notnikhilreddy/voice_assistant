import os
import time
import asyncio
import json
import math
import subprocess
import sys
from time import perf_counter
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 0.1  # seconds

# Groq/model config
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b").split(",")
    if m.strip()
]
GROQ_TIMEOUT_S = float(os.getenv("GROQ_TIMEOUT_S", "8.0"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "1024")))

# Local MLX fallback (Apple Silicon)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "1024"))
LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.7"))
LOCAL_LLM_TOP_P = float(os.getenv("LOCAL_LLM_TOP_P", "0.9"))

# History/summary policy
HISTORY_KEEP_LAST_MESSAGES = int(os.getenv("HISTORY_KEEP_LAST_MESSAGES", "24"))
HISTORY_SUMMARIZE_AFTER_MESSAGES = int(os.getenv("HISTORY_SUMMARIZE_AFTER_MESSAGES", "48"))

# Persona / style: concise, friendly personal assistant
SYSTEM_PROMPT = (
    "You are a helpful conversational chatbot and personal assistant similar to ChatGPT. Your name is Jarvis."
    "Sound like a real friendly person in a voice chat: warm, natural, and confident. "
    "Do NOT say things like 'As an AI' or 'I cannot' unless absolutely required. "
    "Be friendly, knowledgeable, and energizing. Keep replies concise unless the user asks for detail. "
    "CRITICAL LATENCY RULE: Start EVERY reply with exactly one very short FIRST sentence of 1–5 words, "
    "and end that first sentence immediately with punctuation ('.' or '?' or '!'). "
    "Example formats: 'Sure.' 'Okay.' 'Got it.' 'One moment.' "
    "After that first 1–5 word sentence, continue with the full answer normally. "
    "Avoid long lists or detailed descriptions; use short headings only if needed. "
    "Engage and be proactive to keep the conversation flowing naturally. "
    "DO NOT use emojis, emoticons, asterisks(*), or other odd characters that can't be spoken. Respond like natural speech; you can use punctuation to indicate emotions. "
    "When you need to communicate numbers, math, equations, formulas, or computer code for text-to-speech, rewrite it into natural, speakable text. "
    "Examples: say 'three point one four' instead of '3.14' when appropriate; say 'x squared' or 'x to the power of two'; say 'open paren' and 'close paren'; "
    "say 'slash', 'dash', 'underscore', 'colon', and 'dot' for symbols; say 'equals' for '=' and 'not equal' for '!='. "
    "For code, prefer describing it line by line in plain words and avoid dense punctuation-heavy formatting."
)

# --- Remote LLM (Groq) ---
groq_client = Groq(api_key=GROQ_API_KEY)


def _build_messages(
    user_text: str,
    history: Optional[List[Dict[str, str]]],
    history_summary: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history_summary and history_summary.strip():
        messages.append(
            {
                "role": "system",
                "content": (
                    "Conversation summary (for context, treat as authoritative for prior turns):\n"
                    + history_summary.strip()
                ),
            }
        )

    # Keep recent turns verbatim (older turns should be represented via summary).
    if history:
        messages.extend(history[-HISTORY_KEEP_LAST_MESSAGES:])

    messages.append({"role": "user", "content": user_text})
    return messages


def _iter_groq_models() -> List[str]:
    # Try primary then fallbacks; de-dup preserving order.
    models = [GROQ_MODEL] + list(GROQ_FALLBACK_MODELS)
    seen = set()
    out = []
    for m in models:
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def query_remote_llm(
    user_text: str,
    history: Optional[List[Dict[str, str]]] = None,
    history_summary: Optional[str] = None,
) -> str:
    """
    Generates a response using the Groq API (Llama 3).
    """
    if not GROQ_API_KEY:
        return "Groq API key is not configured. Please set it in the .env file."

    models = _iter_groq_models()
    print(f"Querying remote LLM (Groq models={models})...")
    llm_start = perf_counter()
    last_error = None

    for model in models:
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                # Note: Groq SDK timeout kwarg support varies by version; enforce timeouts
                # in async streaming path and via short retries here.
                chat_completion = groq_client.chat.completions.create(
                    messages=_build_messages(user_text, history, history_summary=history_summary),
                    model=model,
                    max_tokens=GROQ_MAX_TOKENS,
                    temperature=0.7,
                )
                content = (chat_completion.choices[0].message.content or "").strip()
                elapsed = (perf_counter() - llm_start) * 1000
                print(f"LLM call model={model} attempt {attempt} took {elapsed:.1f} ms.")
                if content:
                    return content
                last_error = "Empty LLM response"
            except Exception as e:
                last_error = str(e)
                elapsed = (perf_counter() - llm_start) * 1000
                print(f"Error querying Groq model={model} attempt {attempt} after {elapsed:.1f} ms: {e}")

            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_DELAY * (1.5 ** (attempt - 1)))

    print(f"LLM failed after {LLM_MAX_RETRIES} attempts: {last_error}")
    return ""


def get_llm_response(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    history_summary: Optional[str] = None,
) -> str:
    """
    Main function to get a response from the remote Groq LLM.
    """
    resp = query_remote_llm(prompt, history, history_summary=history_summary)
    if resp:
        return resp
    # Fallback to local if remote failed/empty
    return query_local_llm(prompt, history=history, history_summary=history_summary)


def _format_chat_for_local(
    user_text: str,
    history: Optional[List[Dict[str, str]]],
    history_summary: Optional[str],
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history_summary and history_summary.strip():
        msgs.append({"role": "system", "content": "Conversation summary:\n" + history_summary.strip()})
    if history:
        msgs.extend(history[-HISTORY_KEEP_LAST_MESSAGES:])
    msgs.append({"role": "user", "content": user_text})
    return msgs


def _try_mlx_lm_python(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Try MLX-LM Python API first. If it's not installed or the API differs,
    return None and we will attempt a subprocess fallback.
    """
    try:
        from mlx_lm import load, generate  # type: ignore
    except Exception as e:
        print(f"MLX-LM import failed: {e}")
        return None

    try:
        model, tokenizer = load(LOCAL_LLM_MODEL)
        # HuggingFace tokenizers generally support chat templates for instruct models.
        prompt = None
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Very conservative fallback prompt formatting.
            prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=LOCAL_LLM_MAX_TOKENS,
            temp=LOCAL_LLM_TEMPERATURE,
            top_p=LOCAL_LLM_TOP_P,
            verbose=False,
        )
        if isinstance(text, str):
            return text.strip()
        return str(text).strip()
    except Exception as e:
        print(f"MLX-LM python generation failed: {e}")
        return None


def _try_mlx_lm_subprocess(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Subprocess fallback: run `python -m mlx_lm.generate` if available.
    This is more robust across MLX-LM API changes.
    """
    prompt = json.dumps(messages, ensure_ascii=False)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model",
        LOCAL_LLM_MODEL,
        "--chat",
        prompt,
        "--max-tokens",
        str(LOCAL_LLM_MAX_TOKENS),
        "--temp",
        str(LOCAL_LLM_TEMPERATURE),
        "--top-p",
        str(LOCAL_LLM_TOP_P),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            print(f"MLX-LM subprocess failed rc={proc.returncode}: {err}")
            return None
        # Heuristic: last non-empty line is usually the generation.
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not lines:
            return None
        return lines[-1]
    except Exception as e:
        print(f"MLX-LM subprocess error: {e}")
        return None


def query_local_llm(
    user_text: str,
    history: Optional[List[Dict[str, str]]] = None,
    history_summary: Optional[str] = None,
) -> str:
    messages = _format_chat_for_local(user_text, history, history_summary)
    print(f"Querying local MLX LLM ({LOCAL_LLM_MODEL})...")
    resp = _try_mlx_lm_python(messages)
    if resp:
        return resp
    resp = _try_mlx_lm_subprocess(messages)
    if resp:
        return resp
    return "Sorry, I couldn't generate a response right now."


def should_summarize(history: List[Dict[str, str]]) -> bool:
    return len(history) >= HISTORY_SUMMARIZE_AFTER_MESSAGES


def summarize_history_locally(history: List[Dict[str, str]], existing_summary: str = "") -> str:
    """
    Summarize history using local MLX LLM so Groq availability doesn't impact memory.
    Returns a compact running summary.
    """
    # Keep summary prompt small; summarize only older turns.
    prompt = (
        "Summarize the conversation so far for a voice assistant. "
        "Preserve: user preferences, names, tasks, decisions, and any unresolved questions. "
        "Be concise and factual.\n\n"
    )
    if existing_summary.strip():
        prompt += "Existing summary:\n" + existing_summary.strip() + "\n\n"
    prompt += "Recent conversation messages to incorporate:\n"
    prompt += "\n".join([f"{m['role']}: {m['content']}" for m in history])

    return query_local_llm(prompt, history=None, history_summary=None).strip()


async def stream_llm_response(
    user_text: str,
    history: Optional[List[Dict[str, str]]] = None,
    history_summary: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM tokens as they are generated for low-latency responses.
    Yields text chunks as they arrive from the Groq API.
    """
    print("Streaming LLM response (Groq-first, local fallback)...")
    llm_start = perf_counter()
    last_error = None

    # If Groq key missing, go straight to local.
    if not GROQ_API_KEY:
        local_text = query_local_llm(user_text, history=history, history_summary=history_summary)
        for i in range(0, len(local_text), 24):
            yield local_text[i : i + 24]
            await asyncio.sleep(0)
        return

    models = _iter_groq_models()
    for model in models:
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                stream = groq_client.chat.completions.create(
                    messages=_build_messages(user_text, history, history_summary=history_summary),
                    model=model,
                    max_tokens=GROQ_MAX_TOKENS,
                    temperature=0.7,
                    stream=True,
                )

                accumulated_text = ""
                last_token_ts = perf_counter()
                for chunk in stream:
                    delta = getattr(chunk.choices[0], "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        accumulated_text += content
                        last_token_ts = perf_counter()
                        yield content
                        await asyncio.sleep(0)
                    else:
                        # Idle watchdog: if stream stalls for too long, fail over.
                        if (perf_counter() - last_token_ts) > GROQ_TIMEOUT_S:
                            raise TimeoutError("Groq stream stalled")

                elapsed = (perf_counter() - llm_start) * 1000
                print(f"LLM stream model={model} completed in {elapsed:.1f} ms.")
                if accumulated_text.strip():
                    return
                last_error = "Empty LLM response"
            except Exception as e:
                last_error = str(e)
                elapsed = (perf_counter() - llm_start) * 1000
                print(f"Error streaming Groq model={model} attempt {attempt} after {elapsed:.1f} ms: {e}")

            if attempt < LLM_MAX_RETRIES:
                await asyncio.sleep(LLM_RETRY_DELAY * (1.5 ** (attempt - 1)))

    print(f"Groq streaming failed; falling back to local. last_error={last_error}")
    local_text = query_local_llm(user_text, history=history, history_summary=history_summary)
    for i in range(0, len(local_text), 24):
        yield local_text[i : i + 24]
        await asyncio.sleep(0)


if __name__ == '__main__':
    # This is for testing the module directly
    prompt = "Hello, how are you today?"
    print("--- Testing Groq LLM ---")
    response = get_llm_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

