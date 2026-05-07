from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import time

from rag.agent_engine_v4 import RAGAgentEngineV4

router = APIRouter()

_ENGINE = None


# -----------------------------
# INIT ENGINE
# -----------------------------
def init_engine(engine: RAGAgentEngineV4):
    global _ENGINE
    _ENGINE = engine


def get_engine():
    if _ENGINE is None:
        raise RuntimeError("RAG Engine not initialized")
    return _ENGINE


# -----------------------------
# REQUEST MODEL
# -----------------------------
class StreamRequest(BaseModel):
    query: str
    stream_mode: str = "sse"   # sse | raw | json


# -----------------------------
# SSE STREAMING ENDPOINT
# -----------------------------
@router.post("/stream")
def stream(req: StreamRequest):

    engine = get_engine()

    try:
        result = engine.run(req.query)

        stream_iter = result["stream"]
        plan = result.get("plan", {})

        # -----------------------------
        # SSE FORMAT (React-friendly)
        # -----------------------------
        def event_stream():

            # 1. send metadata first
            yield f"event: meta\ndata: {json.dumps(plan)}\n\n"

            # 2. stream tokens
            for token in stream_iter:

                if req.stream_mode == "json":
                    payload = {"type": "token", "data": token}
                    yield f"data: {json.dumps(payload)}\n\n"

                else:
                    # default SSE text stream
                    yield f"data: {token}\n\n"

            # 3. done event
            yield f"event: done\ndata: {{\"status\": \"completed\"}}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# SIMPLE NON-STREAM FALLBACK
# -----------------------------
@router.post("/query")
def query(req: StreamRequest):

    try:
        engine = get_engine()
        result = engine.run(req.query)

        full_text = "".join(list(result["stream"]))

        return {
            "query": req.query,
            "response": full_text,
            "plan": result.get("plan", {})
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# HEALTH
# -----------------------------
@router.get("/health")
def health():
    return {
        "status": "ok",
        "streaming": True,
        "engine_ready": _ENGINE is not None
    }