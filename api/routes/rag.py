from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import traceback

from rag.query_router import RAGQueryRouterV2

router = APIRouter()

_ENGINE = None


# -----------------------------
# ENGINE INITIALIZATION
# -----------------------------
def init_engine(vector_store, graph_store=None, plugin_index=None):
    """
    Initialize global RAG engine.
    """
    global _ENGINE
    _ENGINE = RAGQueryRouterV2(
        vector_store=vector_store,
        graph_store=graph_store,
        plugin_index=plugin_index
    )


def get_engine():
    if _ENGINE is None:
        raise RuntimeError("RAG engine not initialized. Call init_engine() first.")
    return _ENGINE


# -----------------------------
# REQUEST MODEL
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    stream: bool = False


# -----------------------------
# NORMAL QUERY ENDPOINT
# -----------------------------
@router.post("/query")
def query(req: QueryRequest):
    try:
        engine = get_engine()

        result = engine.query(req.query)

        return {
            "status": "ok",
            "query": req.query,
            "intent": result.get("intent"),
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "context": result.get("context", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )


# -----------------------------
# STREAMING ENDPOINT (SSE)
# -----------------------------
@router.post("/stream")
def stream(req: QueryRequest):
    engine = get_engine()

    def event_stream():
        try:
            # Step 1: retrieve
            results = engine.hybrid_retrieve(req.query)

            # Step 2: build context
            context = engine.build_context(results)

            # Step 3: stream LLM response
            for chunk in engine.stream_llm(req.query, context):
                payload = {
                    "chunk": chunk
                }
                yield f"data: {json.dumps(payload)}\n\n"

            # end signal
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )


# -----------------------------
# HEALTH CHECK
# -----------------------------
@router.get("/health")
def health():
    return {
        "status": "ok",
        "rag_version": "v2",
        "engine_ready": _ENGINE is not None
    }