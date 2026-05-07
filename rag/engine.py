import requests
import numpy as np
from typing import List, Dict


# -----------------------------
# OLLAMA CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api"


def ollama_embed(text: str, model: str = "nomic-embed-text"):
    res = requests.post(
        f"{OLLAMA_URL}/embeddings",
        json={"model": model, "prompt": text}
    )
    res.raise_for_status()
    return res.json()["embedding"]


def ollama_generate(prompt: str, model: str = "deepseek-r1"):
    res = requests.post(
        f"{OLLAMA_URL}/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    res.raise_for_status()
    return res.json()["response"]


# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# RAG ENGINE
# -----------------------------
class RAGEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5):
        qvec = ollama_embed(query)

        results = []

        for i, vec in enumerate(self.vector_store.vectors):
            score = cosine(qvec, vec)

            results.append({
                "score": score,
                "text": self.vector_store.metadata[i]["text"],
                "source": self.vector_store.metadata[i]["source"]
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def build_context(self, docs):
        return "\n\n".join(
            [f"[{d['source']}]\n{d['text']}" for d in docs]
        )

    def answer(self, query: str):
        docs = self.retrieve(query)
        context = self.build_context(docs)

        prompt = f"""
You are OmniBioAI Dev Hub Assistant.

Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly and technically:
"""

        response = ollama_generate(prompt)

        return {
            "query": query,
            "answer": response,
            "sources": [d["source"] for d in docs]
        }