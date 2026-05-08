import sys
import os
import hashlib
import numpy as np

sys.path.append(os.path.abspath("."))

from index.vector_store import VectorStore
from rag.engine import ollama_embed   # SINGLE SOURCE OF TRUTH
from ingestion.doc_loader import load_documents
from processing.chunker import chunk_text

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)

    if vec.ndim == 3:
        vec = vec.squeeze()

    if vec.ndim == 2:
        vec = vec.reshape(-1)

    return vec.astype(np.float32)


def build_index():

    print("🚀 Incremental V6 Indexing Starting...")

    vector_store = VectorStore()

    repos = [
        "../omnibioai",
        "../omnibioai-rag",
        "../omnibioai-toolserver",
        "../omnibioai-sdk",
        "../omnibioai-workflow-bundles",
        "../omnibioai-control-center",
        "../omnibioai-lims",
        "../omnibioai-model-registry",
        "../omnibioai-dev-docker"
    ]

    docs = load_documents(repos)

    all_vectors = []
    all_meta = []
    seen_hashes = set()

    stats = {"skipped": 0, "new": 0, "chunks_indexed": 0}

    for doc in docs:

        text = doc.get("text", "")
        if not text:
            continue

        for chunk in chunk_text(text):

            h = hash_text(chunk)

            if h in seen_hashes:
                stats["skipped"] += 1
                continue

            seen_hashes.add(h)
            stats["new"] += 1

            vec = ollama_embed(chunk)   # <<< SINGLE EMBEDDING SOURCE
            vec = normalize_vector(vec)

            all_vectors.append(vec)
            all_meta.append({
                "text": chunk,
                "source": doc.get("source", "unknown"),
                "hash": h
            })

            stats["chunks_indexed"] += 1

    if not all_vectors:
        print("⚠️ No vectors generated")
        return

    all_vectors = np.vstack(all_vectors).astype(np.float32)

    vector_store.add(all_vectors, all_meta)

    print("✅ V6 Index Complete")
    print(stats)


if __name__ == "__main__":
    build_index()