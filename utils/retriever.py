# utils/retriever.py
import math
from typing import List, Tuple, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from utils.document_parser import parse_document_from_url

# Load the embedding model once at import (single process)
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)

def _split_text_with_overlap(text: str, max_words: int = 200, overlap: int = 30) -> Tuple[List[str], List[Dict[str, int]]]:
    """
    Split text into word-based chunks with overlap.
    Returns (chunks, metadata_list) where metadata has start_word and end_word indices.
    """
    words = text.split()
    if not words:
        return [], []

    if overlap >= max_words:
        overlap = int(max_words * 0.1)

    step = max_words - overlap
    chunks = []
    metadata = []

    for start in range(0, len(words), step):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
            metadata.append({"start_word": start, "end_word": end})
        if end == len(words):
            break

    return chunks, metadata


def _embed_chunks_batched(chunks: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed chunks in batches and return a float32 numpy array of shape (N, D).
    """
    if not chunks:
        return np.zeros((0, _model.get_sentence_embedding_dimension()), dtype=np.float32)

    embs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # convert_to_numpy True is efficient
        arr = _model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(arr.astype("float32"))
    return np.vstack(embs)


class Retriever:
    """
    Retriever encapsulates chunking, embedding, FAISS index build, and query.
    Use Retriever.from_url(url) to instantiate from a document URL.
    """
    def __init__(self, text: str, max_words: int = 200, overlap: int = 30, batch_size: int = 32, use_cosine: bool = True):
        self.text = text
        self.max_words = max_words
        self.overlap = overlap
        self.batch_size = batch_size
        self.use_cosine = use_cosine

        # chunk + metadata
        self.chunks, self.metadata = _split_text_with_overlap(self.text, max_words=self.max_words, overlap=self.overlap)
        if not self.chunks:
            raise ValueError("No text to index after splitting.")

        # embeddings (numpy array)
        self.embeddings = _embed_chunks_batched(self.chunks, batch_size=self.batch_size)

        # build faiss index
        self.index = self._build_index(self.embeddings, use_cosine=self.use_cosine)

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "Retriever":
        """
        Download and parse a document from URL and create a Retriever instance.
        """
        text = parse_document_from_url(url)
        return cls(text, **kwargs)

    def _build_index(self, embeddings: np.ndarray, use_cosine: bool = True):
        """
        Build a FAISS index. If use_cosine=True, normalize vectors and use IndexFlatIP (inner product).
        """
        if embeddings.size == 0:
            raise ValueError("Empty embeddings array passed to build_index")

        if use_cosine:
            # normalize in-place
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

    def query(self, question: str, top_k: int = 3, as_tuples: bool = False) -> List:
        """
        Query the retriever and return top_k results.

        By default returns a list of dicts:
          [
            {"chunk": str, "score": float, "metadata": {...}, "index": int},
            ...
          ]

        If as_tuples=True, returns:
          [
            (chunk_str, score_float),
            ...
          ]
        """
        if top_k <= 0:
            return []

        qvec = _model.encode([question], convert_to_numpy=True).astype("float32")
        if self.use_cosine:
            faiss.normalize_L2(qvec)

        # search
        distances, indices = self.index.search(qvec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if as_tuples:
                results.append((self.chunks[idx], float(dist)))
            else:
                entry = {
                    "index": int(idx),
                    "chunk": self.chunks[idx],
                    "score": float(dist),
                    "metadata": self.metadata[idx],
                }
                results.append(entry)
        return results


    def get_chunk_by_index(self, idx: int) -> Dict[str, Any]:
        """
        Return the chunk text and metadata for a given embedding index.
        """
        if idx < 0 or idx >= len(self.chunks):
            raise IndexError("chunk index out of range")
        return {"chunk": self.chunks[idx], "metadata": self.metadata[idx]}

    def save_index_and_metadata(self, index_path: str, metadata_path: str):
        """
        Save FAISS index and chunk metadata to disk.
        """
        faiss.write_index(self.index, index_path)
        import json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "metadata": self.metadata}, f, ensure_ascii=False, indent=2)
