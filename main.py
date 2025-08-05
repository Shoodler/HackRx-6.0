# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
from utils.retriever import Retriever
from utils.llm import generate_answer_from_chunks

app = FastAPI(title="HackRx RAG API")


class QuestionRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]


class AnswerResponse(BaseModel):
    question: str
    answer: str


def _normalize_retriever_output(raw_results: Any) -> List[tuple]:
    """
    Accepts either:
      - List[ (chunk_str, score_float) ]
      - List[ { 'chunk': str, 'score': float, ... } ]
    Returns List[(chunk_str, score_float)].
    """
    normalized = []

    if raw_results is None:
        return normalized

    # If list of tuples
    if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], tuple):
        return raw_results  # type: ignore

    # If list of dicts
    if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], dict):
        for entry in raw_results:
            # expect 'chunk' and 'score' keys; fallback to first two values
            chunk = entry.get("chunk") or entry.get("text") or entry.get("content")
            score = entry.get("score")
            if chunk is not None and score is not None:
                normalized.append((chunk, float(score)))
            else:
                # fallback: try to infer
                # if dict has an 'index' and you stored chunks elsewhere, you'd handle it here
                pass
        return normalized

    # If empty list or unknown format, just return empty
    return normalized


@app.post("/hackrx/run", response_model=List[AnswerResponse])
async def run_hackrx(request: QuestionRequest):
    """
    Main endpoint required by HackRx.
    Input JSON:
      { "documents": "<url>", "questions": ["q1", "q2", ...] }
    Output:
      [ {"question": "q1", "answer": "..."}, ... ]
    """
    try:
        # Build retriever from the provided document URL (this will download & parse)
        retriever = Retriever.from_url(request.documents)
    except Exception as e:
        # Parsing / download error
        raise HTTPException(status_code=400, detail=f"Failed to load document: {e}")

    results: List[AnswerResponse] = []

    for question in request.questions:
        try:
            raw_matches = retriever.query(question, top_k=3)
        except Exception as e:
            # If search fails, return an error for this question
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

        # Normalize to (chunk, score) pairs expected by generate_answer_from_chunks
        pairs = _normalize_retriever_output(raw_matches)

        # If no context found, generate_answer_from_chunks will handle it (or we can short-circuit)
        try:
            answer = generate_answer_from_chunks(question, pairs)
        except Exception as e:
            # LLM error
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

        results.append(AnswerResponse(question=question, answer=answer))

    return results
