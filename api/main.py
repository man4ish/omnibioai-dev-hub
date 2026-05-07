from fastapi import FastAPI

from api.routes import rag

from index.vector_store import VectorStore
from embeddings.embedder import Embedder
from ingestion.doc_loader import load_documents
from processing.chunker import chunk_text

from index.graph_store import GraphStore
from index.plugin_index import PluginIndex

import threading
import time

app = FastAPI(title="OmniBioAI Dev Hub RAG Engine")

# -----------------------------
# GLOBAL OBJECTS
# -----------------------------
vector_store = VectorStore()
embedder = Embedder()

graph_store = GraphStore()
plugin_index = PluginIndex()

rag_engine_ready = False

indexing_status = {
    "running": False,
    "docs": 0,
    "chunks": 0,
    "errors": 0
}

# -----------------------------
# REPO CONFIG (CENTRALIZED)
# -----------------------------
REPOS = [
    "../omnibioai",
    "../omnibioai-rag",
    "../omnibioai-tes",
    "../omnibioai-toolserver",
    "../omnibioai-tool-runtime",
    "../omnibioai-workflow-bundles",
    "../omnibioai-model-registry",
    "../omnibioai-control-center",
    "../omnibioai-tool-images",
    "../omnibioai-dev-docker",
    "../omnibioai-lims",
    "../omnibioai_sdk"
]


# -----------------------------
# INDEX BUILD PIPELINE
# -----------------------------
def build_index():
    global indexing_status

    indexing_status["running"] = True

    docs = load_documents(REPOS)

    all_vectors = []
    all_meta = []

    doc_count = 0
    chunk_count = 0
    error_count = 0

    for doc in docs:
        try:
            text = doc.get("text", "")
            if not text:
                continue

            chunks = chunk_text(text)
            chunk_count += len(chunks)

            vectors = embedder.encode(chunks)

            for i in range(len(vectors)):
                all_vectors.append(vectors[i])
                all_meta.append({
                    "id": doc.get("id", ""),
                    "text": chunks[i],
                    "source": doc.get("source", "unknown")
                })

            doc_count += 1

        except Exception as e:
            error_count += 1
            print(f"[Index Error] {doc.get('source', 'unknown')}: {str(e)}")

    vector_store.add(all_vectors, all_meta)

    indexing_status.update({
        "running": False,
        "docs": doc_count,
        "chunks": chunk_count,
        "errors": error_count
    })


# -----------------------------
# OPTIONAL: GRAPH SEED (can extend later)
# -----------------------------
def build_graph_seed():
    graph_store.add_edge("plugin architecture", "workflow engine", "depends_on")
    graph_store.add_edge("RAG system", "vector store", "uses")
    graph_store.add_edge("OmniBioAI", "plugin system", "includes")


# -----------------------------
# PLUGIN INDEX SEED (minimal bootstrap)
# -----------------------------
def build_plugin_index():
    plugin_index.add([
        {
            "text": "workflow_builder plugin manages DAG-based pipelines",
            "plugin": "workflow_builder"
        },
        {
            "text": "plugin_executor runs bioinformatics workflows",
            "plugin": "plugin_executor"
        },
        {
            "text": "qc_plots generates sequencing QC reports",
            "plugin": "qc_plots"
        }
    ])


# -----------------------------
# RAG INIT
# -----------------------------
def init_rag():
    rag.init_engine(
        vector_store,
        graph_store=graph_store,
        plugin_index=plugin_index
    )


# -----------------------------
# BOOTSTRAP PIPELINE
# -----------------------------
def bootstrap():
    global rag_engine_ready

    print("🚀 Starting OmniBioAI RAG v2 bootstrap...")

    build_graph_seed()
    build_plugin_index()

    build_index()

    time.sleep(1)

    init_rag()

    rag_engine_ready = True

    print("✅ OmniBioAI RAG system ready")


# -----------------------------
# STARTUP (NON-BLOCKING SAFE)
# -----------------------------
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=bootstrap, daemon=True)
    thread.start()


# -----------------------------
# ROUTES
# -----------------------------
app.include_router(rag.router, prefix="/rag")


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "omnibioai-dev-hub",
        "rag_ready": rag_engine_ready
    }


# -----------------------------
# SYSTEM STATUS (UI READY)
# -----------------------------
@app.get("/status")
def status():
    return {
        "rag_ready": rag_engine_ready,
        "indexing": indexing_status,
        "repos_loaded": len(REPOS),
        "graph_edges": len(graph_store.edges),
        "plugins_loaded": len(plugin_index.docs)
    }