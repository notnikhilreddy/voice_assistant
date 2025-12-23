import os
import asyncio
from time import perf_counter
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 0.1  # seconds

# Persona / style: concise, friendly personal assistant
SYSTEM_PROMPT = (
    "You are a helpful conversational chatbot and personal assistant. "
    "Be friendly, knowledgeable, and energizing. Keep replies concise—"
    "Always respond with two or three sentences max—while staying clear and practical. "
    "Avoid long lists or detailed descriptions; use short headings or brief bullets only when appropriate. "
    "Engage and be proactive to keep the conversation flowing while staying focused on what the user wants. "
    "Always start with short (under 3 words) full sentence, like a greeting or a filler (eg: That's great!, Sure.). "
    "DO NOT use emojis, emoticons, asterisks(*), or other odd characters. respond like natural speech."
)

# --- Remote LLM (Groq) ---
groq_client = Groq(api_key=GROQ_API_KEY)


def _build_messages(user_text: str, history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep a sliding window of recent turns to preserve context without bloating the prompt.
    if history:
        messages.extend(history[-12:])  # roughly 6 prior turns (user+assistant)

    messages.append({"role": "user", "content": user_text})
    return messages


def query_remote_llm(user_text: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generates a response using the Groq API (Llama 3).
    """
    if not GROQ_API_KEY:
        return "Groq API key is not configured. Please set it in the .env file."

    print("Querying remote LLM (Groq openai/gpt-oss-20b)...")
    llm_start = perf_counter()
    last_error = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=_build_messages(user_text, history),
                model="openai/gpt-oss-20b",
                max_tokens=200,
                temperature=0.7,
            )
            content = (chat_completion.choices[0].message.content or "").strip()
            elapsed = (perf_counter() - llm_start) * 1000
            print(f"LLM call attempt {attempt} took {elapsed:.1f} ms.")
            if content:
                return content
            last_error = "Empty LLM response"
        except Exception as e:
            last_error = str(e)
            elapsed = (perf_counter() - llm_start) * 1000
            print(f"Error querying Groq API attempt {attempt} after {elapsed:.1f} ms: {e}")

        if attempt < LLM_MAX_RETRIES:
            asyncio.sleep(LLM_RETRY_DELAY)

    print(f"LLM failed after {LLM_MAX_RETRIES} attempts: {last_error}")
    return "Sorry, I couldn’t generate a response right now. Can you rephrase or ask something else?"


def get_llm_response(prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Main function to get a response from the remote Groq LLM.
    """
    return query_remote_llm(prompt, history)


if __name__ == '__main__':
    # This is for testing the module directly
    prompt = "Hello, how are you today?"
    print("--- Testing Groq LLM ---")
    response = get_llm_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

