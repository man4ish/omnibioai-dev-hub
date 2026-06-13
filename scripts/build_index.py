import sys
import os
import hashlib
import numpy as np

sys.path.append(os.path.abspath("."))

from index.vector_store import VectorStore
from rag.engine import ollama_embed   # SINGLE SOURCE OF TRUTH
from ingestion.doc_loader import load_documents
from processing.chunker import chunk_text

MIN_CHUNK_CHARS = 10  # discard overflow tails from chunker.py's hard char-slice

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

    # BUG-1 FIX: default to the actual local machine path.
    # In Docker the image sets ENV REPO_BASE=/repos (volume mount).
    # Locally: export REPO_BASE=/home/manish/Desktop/machine (or set in .env).
    REPO_BASE = os.environ.get("REPO_BASE", "/home/manish/Desktop/machine")

    repos = [
        f"{REPO_BASE}/omnibioai",
        f"{REPO_BASE}/omnibioai-rag",
        f"{REPO_BASE}/omnibioai-toolserver",
        f"{REPO_BASE}/omnibioai-sdk",
        f"{REPO_BASE}/omnibioai-workflow-bundles",
        f"{REPO_BASE}/omnibioai-control-center",
        f"{REPO_BASE}/omnibioai-lims",
        f"{REPO_BASE}/omnibioai-model-registry",
        f"{REPO_BASE}/omnibioai-dev-docker",
        f"{REPO_BASE}/omnibioai-api-gateway",
        f"{REPO_BASE}/omnibioai-docs",
        f"{REPO_BASE}/omnibioai-studio",
        f"{REPO_BASE}/omnibioai-auth",
        f"{REPO_BASE}/omnibioai-tool-runtime",
        f"{REPO_BASE}/omnibioai-iam-client",
        f"{REPO_BASE}/omnibioai-policy-engine",
        f"{REPO_BASE}/omnibioai-security-audit",
        f"{REPO_BASE}/omnibioai-security-sdk",
        f"{REPO_BASE}/omnibioai-hpc-policy-engine",
    ]

    # Fail fast: if not a single repo exists, the REPO_BASE is wrong.
    existing = [r for r in repos if os.path.isdir(r)]
    if not existing:
        raise SystemExit(
            f"\n❌  No repos found under REPO_BASE={REPO_BASE!r}\n"
            f"    Set REPO_BASE to the directory that contains the omnibioai-* repos.\n"
            f"    Example:  export REPO_BASE=/home/manish/Desktop/machine\n"
            f"    Docker:   -e REPO_BASE=/repos  (with repos volume mounted at /repos)\n"
        )

    missing = [os.path.basename(r) for r in repos if not os.path.isdir(r)]
    if missing:
        print(f"⚠️  {len(missing)} repos not found, will be skipped: {missing}")

    all_vectors = []
    all_meta = []

    stats = {"too_short": 0, "deduped": 0, "embed_failed": 0, "new": 0, "chunks_indexed": 0}

    # BUG-2 FIX: seen_hashes is reset per repo so cross-repo identical chunks
    # are each indexed under their own source path.  Within a single repo,
    # dedup still fires to avoid storing the same paragraph twice (e.g. a
    # shared boilerplate section that appears in multiple plugin READMEs).
    for repo_path in repos:
        if not os.path.isdir(repo_path):
            continue

        seen_hashes: set = set()
        repo_docs = load_documents([repo_path])

        for doc in repo_docs:

            text = doc.get("text", "")
            if not text:
                continue

            for chunk in chunk_text(text):

                if len(chunk) < MIN_CHUNK_CHARS:
                    stats["too_short"] += 1
                    continue

                h = hash_text(chunk)

                if h in seen_hashes:
                    stats["deduped"] += 1
                    continue

                seen_hashes.add(h)
                stats["new"] += 1

                try:
                    vec = ollama_embed(chunk)   # <<< SINGLE EMBEDDING SOURCE
                    vec = normalize_vector(vec)
                except Exception as e:
                    print(f"⚠️  Skipping chunk from {doc.get('source')}: {e}")
                    stats["embed_failed"] += 1
                    continue

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

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "faiss_index")
    vector_store.save(save_path)
    print(f"💾 Index saved to {save_path} ({vector_store.index.ntotal} vectors)")

    print("✅ V6 Index Complete")
    print(stats)


if __name__ == "__main__":
    build_index()