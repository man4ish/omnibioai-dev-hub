from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import asyncio
import numpy as np

from api.routes import rag

from index.vector_store import VectorStore
from embeddings.embedder import Embedder
from ingestion.doc_loader import load_documents
from processing.chunker import chunk_text

from index.graph_store import GraphStore
from index.plugin_index import PluginIndex

from rag.control_plane import CONTROL_PLANE
from rag.engine import ollama_embed


# =========================================================
# APP INIT
# =========================================================

app = FastAPI(title="OmniBioAI Dev Hub Control Plane V5")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GLOBAL STATE
# =========================================================

vector_store = VectorStore()
embedder = Embedder()
graph_store = GraphStore()
plugin_index = PluginIndex([])

indexing_status = {
    "running": False,
    "docs": 0,
    "chunks": 0,
    "errors": 0
}


# =========================================================
# REPOSITORIES
# =========================================================

REPOS = [
    "/workspace/omnibioai",
    "/workspace/omnibioai-rag",
    "/workspace/omnibioai-toolserver",
    "/workspace/omnibioai-tool-runtime",
    "/workspace/omnibioai-workflow-bundles",
    "/workspace/omnibioai-model-registry",
    "/workspace/omnibioai-control-center",
    "/workspace/omnibioai_sdk",
    "/workspace/omnibioai-dev-docker",
    "/workspace/omnibioai-lims"
]


# =========================================================
# INDEX PIPELINE
# =========================================================

async def build_index():

    indexing_status["running"] = True

    docs = load_documents(REPOS)

    vectors = []
    metadata = []

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

            for chunk in chunks:
                vec = ollama_embed(chunk)
                vectors.append(vec)
                metadata.append({
                    "id": doc.get("id", ""),
                    "text": chunk,
                    "source": doc.get("source", "unknown")
                })

            doc_count += 1

        except Exception as e:
            error_count += 1
            print(f"[Index Error] {doc.get('source', 'unknown')} -> {str(e)}")

    vector_store.add(vectors, metadata)

    indexing_status.update({
        "running": False,
        "docs": doc_count,
        "chunks": chunk_count,
        "errors": error_count
    })


# =========================================================
# GRAPH SEED
# =========================================================

def build_graph_seed():
    graph_store.add_edge("OmniBioAI", "RAG Engine", "powers")
    graph_store.add_edge("RAG Engine", "Vector Store", "uses")
    graph_store.add_edge("RAG Engine", "Graph Store", "augments")
    graph_store.add_edge("Plugin System", "Workflow Engine", "drives")


# =========================================================
# PLUGIN SEED
# =========================================================

def build_plugin_index():
    plugin_index.docs = [
        {"text": "workflow_builder manages DAG pipelines", "plugin": "workflow_builder"},
        {"text": "plugin_executor runs bioinformatics workflows", "plugin": "plugin_executor"},
        {"text": "qc_plots generates sequencing QC reports", "plugin": "qc_plots"}
    ]


# =========================================================
# CONTROL PLANE INIT
# =========================================================

async def init_control_plane():

    build_graph_seed()
    build_plugin_index()
    await build_index()

    CONTROL_PLANE.init(
        vector_store=vector_store,
        graph_store=graph_store,
        plugin_index=plugin_index,
        embedder=embedder
    )


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup_event():
    await init_control_plane()


# =========================================================
# ROUTES
# =========================================================

app.include_router(rag.router, prefix="/rag")


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "omnibioai-dev-hub",
        "control_plane": CONTROL_PLANE.status()
    }


# =========================================================
# STATUS
# =========================================================

@app.get("/status")
def status():
    return {
        "control_plane": CONTROL_PLANE.status(),
        "indexing": indexing_status,
        "repos_loaded": len(REPOS),
        "graph_edges": len(graph_store.edges),
        "plugins_loaded": len(plugin_index.docs)
    }


# =========================================================
# SAFETY MIDDLEWARE (FIXED)
# =========================================================

@app.middleware("http")
async def guard_requests(request: Request, call_next):

    if request.url.path.startswith("/rag"):
        if CONTROL_PLANE.status()["status"] != "READY":
            return JSONResponse(
                status_code=503,
                content={"detail": "Control plane not ready"}
            )

    return await call_next(request)