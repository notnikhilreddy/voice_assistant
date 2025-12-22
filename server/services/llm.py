import os

from dotenv import load_dotenv
from groq import Groq

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Remote LLM (Groq) ---
groq_client = Groq(api_key=GROQ_API_KEY)


def query_remote_llm(prompt: str) -> str:
    """
    Generates a response using the Groq API (Llama 3).
    """
    if not GROQ_API_KEY:
        return "Groq API key is not configured. Please set it in the .env file."

    print("Querying remote LLM (Groq llama3-8b-8192)...")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="openai/gpt-oss-20b",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error querying Groq API: {e}")
        return "Sorry, I had trouble connecting to the remote language model."


def get_llm_response(prompt: str) -> str:
    """
    Main function to get a response from the remote Groq LLM.
    """
    return query_remote_llm(prompt)

if __name__ == '__main__':
    # This is for testing the module directly
    prompt = "Hello, how are you today?"
    print("--- Testing Groq LLM ---")
    response = get_llm_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

