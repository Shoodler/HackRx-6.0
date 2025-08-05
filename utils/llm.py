# utils/llm.py
import os
import requests
from dotenv import load_dotenv
from typing import List, Tuple, Optional

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # change if needed

def query_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = GROQ_MODEL_NAME,
    temperature: float = 0.0,
    max_tokens: int = 800,
    timeout: int = 30,
) -> str:
    """
    Low-level call to Groq chat completions.
    Raises RuntimeError on failures.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not found in environment (.env)")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Network error calling Groq: {e}") from e

    if resp.status_code != 200:
        # include body for debugging; don't leak in production logs to users
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

    j = resp.json()
    try:
        return j["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Groq response shape: {j}") from e


def generate_answer_from_chunks(
    question: str,
    chunks: List[Tuple[str, float]],
    instruction: Optional[str] = None,
    use_only_context: bool = True,
) -> str:
    """
    Given a question and a list of (chunk_text, score) pairs, build a prompt and call the LLM.

    - instruction: optional extra system instruction (e.g., "Answer concisely and return JSON")
    - use_only_context: if True, instructs the model to not hallucinate beyond the given context
    """
    if not chunks:
        return "No relevant context found in the document."

    # join chunks with spacing; keep top-to-bottom order
    context = "\n\n".join([c for c, _ in chunks])

    default_instr = (
        "You are an expert policy assistant. Answer the user's question based ONLY on the provided context. "
        "If the context does not contain the information, state that you could not find it."
    )

    system_prompt = instruction.strip() if instruction else default_instr
    # append the context to the system prompt
    system_prompt = f"{system_prompt}\n\nContext:\n{context}"

    user_prompt = question

    return query_llm(system_prompt, user_prompt)
