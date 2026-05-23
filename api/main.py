from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import logging

from api.routes import rag

from index.vector_store import VectorStore
from embeddings.embedder import Embedder

from index.graph_store import GraphStore
from index.plugin_index import PluginIndex

from rag.control_plane import CONTROL_PLANE

logger = logging.getLogger(__name__)


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
loaded = vector_store.load("data/faiss_index")
if not loaded:
    logger.warning("No persisted FAISS index found; will build from scratch on startup")
embedder = Embedder()
graph_store = GraphStore()
plugin_index = PluginIndex([])

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

    # Wire the control plane immediately with an empty index so the server
    # is READY for non-RAG requests while indexing runs in the background.
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
        "index_vectors": vector_store.index.ntotal if vector_store.index else 0,
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