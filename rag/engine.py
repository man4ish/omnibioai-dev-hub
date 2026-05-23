import os
import requests
import numpy as np
from typing import List, Dict, Any


# =========================================================
# OLLAMA CONFIG
# =========================================================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api")
EMBED_DIM = 768


def ollama_embed(text: str, model: str = "nomic-embed-text"):
    res = requests.post(
        f"{OLLAMA_URL}/embeddings",
        json={
            "model": model,
            "prompt": text
        },
        timeout=60
    )
    res.raise_for_status()

    vec = res.json()["embedding"]

    # =====================================================
    # V6 HARD NORMALIZATION (FAISS SAFE)
    # =====================================================
    vec = np.array(vec, dtype=np.float32)

    # remove batch dimension if present
    if vec.ndim == 2:
        vec = vec[0]

    # enforce correct dimension
    if vec.shape[0] != EMBED_DIM:
        raise ValueError(
            f"Embedding dim mismatch: got {vec.shape[0]} expected {EMBED_DIM}"
        )

    return vec


def ollama_generate(prompt: str, model: str = "llama3"):
    res = requests.post(
        f"{OLLAMA_URL}/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )
    res.raise_for_status()
    return res.json().get("response", "")


# =========================================================
# COSINE (fallback only)
# =========================================================
def cosine(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# =========================================================
# RAG ENGINE (V6 FAISS-NATIVE)
# =========================================================
class RAGEngine:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.embed_model = "nomic-embed-text"

    # =====================================================
    # EMBEDDING (SINGLE SOURCE OF TRUTH)
    # =====================================================
    def _embed(self, text: str):

        if not isinstance(text, str):
            raise ValueError("Query must be a string")

        vec = ollama_embed(text, model=self.embed_model)

        # ensure numpy + correct shape
        vec = np.array(vec, dtype=np.float32).reshape(-1)

        if vec.shape[0] != EMBED_DIM:
            raise ValueError(
                f"Query embedding mismatch: got {vec.shape[0]} expected {EMBED_DIM}"
            )

        return vec

    # =====================================================
    # RETRIEVAL (FAISS ONLY)
    # =====================================================
    def retrieve(self, query: str, top_k: int = 5):

        query_vec = self._embed(query).reshape(1, -1)

        index = getattr(self.vector_store, "index", None)
        metadata = getattr(self.vector_store, "metadata", [])

        if index is None or index.ntotal == 0:
            return []

        scores, indices = index.search(query_vec, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0 or idx >= len(metadata):
                continue

            results.append({
                "score": float(score),
                "text": metadata[idx].get("text", ""),
                "source": metadata[idx].get("source", "unknown")
            })

        return results

    # =====================================================
    # CONTEXT BUILDER
    # =====================================================
    def build_context(self, docs: List[Dict[str, Any]]) -> str:

        if not docs:
            return "No relevant context found."

        return "\n\n".join(
            f"[{d.get('source')}]\n{d.get('text')}"
            for d in docs
        )

    # =====================================================
    # PROMPT BUILDER
    # =====================================================
    def build_prompt(self, query: str, context: str) -> str:

        return f"""
You are OmniBioAI Dev Hub Assistant (V6).

Use ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly, technically, and concisely:
"""

    # =====================================================
    # MAIN PIPELINE
    # =====================================================
    def answer(self, query: str):

        docs = self.retrieve(query)
        context = self.build_context(docs)
        prompt = self.build_prompt(query, context)

        try:
            response = ollama_generate(prompt)
        except Exception as e:
            response = f"[LLM_ERROR] {str(e)}"

        return {
            "query": query,
            "answer": response,
            "sources": [d.get("source") for d in docs],
            "context_used": len(docs),
            "version": "v6-faiss"
        }

    # =====================================================
    # FASTAPI COMPATIBILITY
    # =====================================================
    def query(self, question: str):
        return self.answer(question)