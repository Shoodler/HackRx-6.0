# utils/embedding.py (improved)
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def split_text(
    text: str,
    max_words: int = 200,
    overlap_words: int = 30
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Split text into chunks by words with optional overlap.
    Returns (chunks, metadata_list) where metadata contains start/end word indices.
    """
    words = text.split()
    if not words:
        return [], []

    chunks = []
    metadata = []
    step = max_words - overlap_words if overlap_words < max_words else max_words
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_words]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
            metadata.append({"start_word": i, "end_word": i + len(chunk_words)})
        if i + max_words >= len(words):
            break
    return chunks, metadata


def embed_chunks(
    chunks: List[str],
    batch_size: int = 32
) -> np.ndarray:
    """
    Returns embeddings as a NumPy array (N, D) dtype float32.
    Uses batching to avoid memory issues.
    """
    if not chunks:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb.astype("float32"))
    return np.vstack(embeddings)


def build_faiss_index(embeddings: np.ndarray, use_cosine: bool = True) -> faiss.IndexFlat:
    """
    Builds a FAISS index. If use_cosine=True, normalize vectors and use IndexFlatIP (inner product).
    Returns the index (and modifies embeddings in-place if normalized).
    """
    if embeddings.size == 0:
        raise ValueError("Empty embeddings array passed to build_faiss_index")

    if use_cosine:
        # Normalize rows to unit length for cosine via inner product
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index


def save_faiss_index(index: faiss.Index, path: str = "faiss.index"):
    faiss.write_index(index, path)


def load_faiss_index(path: str = "faiss.index") -> faiss.Index:
    return faiss.read_index(path)


def semantic_search(
    query: str,
    chunks: List[str],
    index: faiss.Index,
    top_k: int = 3,
    use_cosine: bool = True
) -> List[Tuple[str, float]]:
    """
    Returns a list of (chunk_text, score). If use_cosine=True returns cosine scores (0..1),
    otherwise returns L2 distances (lower is better).
    """
    if not chunks:
        return []

    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    if use_cosine:
        faiss.normalize_L2(q_vec)

    D, I = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        score = float(dist)
        # If cosine (inner product), score is similarity (higher better)
        results.append((chunks[idx], score))
    return results
