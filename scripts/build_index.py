import sys
import os

sys.path.append(os.path.abspath("."))

from index.vector_store import VectorStore
from embeddings.embedder import Embedder
from ingestion.doc_loader import load_documents
from processing.chunker import chunk_text


def build_index():
    print("🚀 Starting OmniBioAI indexing pipeline...")

    vector_store = VectorStore()
    embedder = Embedder()

    repos = [
        "../omnibioai",
        "../omnibioai-rag",
        "../omnibioai-tes",
        "../omnibioai-toolserver",
        "../omnibioai-workflow-bundles"
    ]

    docs = load_documents(repos)

    all_vectors = []
    all_meta = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        vectors = embedder.encode(chunks)

        for i, vec in enumerate(vectors):
            all_vectors.append(vec)
            all_meta.append({
                "id": doc.get("id", ""),
                "text": chunks[i],
                "source": doc.get("source", "unknown")
            })

    vector_store.add(all_vectors, all_meta)

    # optional: persist index
    os.makedirs("data/index", exist_ok=True)
    vector_store.save("data/index/vector_store.pkl")

    print(f"✅ Indexing complete")
    print(f"   Docs: {len(docs)}")
    print(f"   Chunks: {len(all_vectors)}")


if __name__ == "__main__":
    build_index()