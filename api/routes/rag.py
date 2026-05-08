from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import traceback

from rag.control_plane import CONTROL_PLANE

router = APIRouter()


# =========================================================
# REQUEST MODEL
# =========================================================

class QueryRequest(BaseModel):
    query: str


# =========================================================
# ENGINE ACCESS (V6 SAFE)
# =========================================================

def get_engine():
    try:
        engine = CONTROL_PLANE.get_engine()

        if engine is None:
            raise RuntimeError("RAG engine not initialized")

        # V6 SAFETY CHECKS
        if not hasattr(engine, "query"):
            raise RuntimeError("Engine missing V6 query method")

        return engine

    except Exception as e:
        raise RuntimeError(f"Engine access failed: {str(e)}")


# =========================================================
# QUERY ENDPOINT (V6)
# =========================================================

@router.post("/query")
def query(req: QueryRequest):

    try:
        engine = get_engine()

        # V6 CONTRACT: only query() exists
        result = engine.query(req.query)

        return {
            **result,
            "api_version": "v6"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )


# =========================================================
# STREAMING ENDPOINT (V6 SAFE SSE)
# =========================================================

@router.post("/stream")
def stream(req: QueryRequest):

    def event_stream():

        try:
            engine = get_engine()

            # V6: no hybrid_retrieve dependency anymore
            # fallback-safe: reuse query pipeline structure

            result = engine.retrieve(req.query)
            context = engine.build_context(result)

            # check optional LLM streaming support
            if hasattr(engine, "stream_llm"):
                for token in engine.stream_llm(req.query, context):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            else:
                # fallback: single response
                response = engine.answer(req.query)
                yield f"data: {json.dumps({'type': 'response', 'content': response['answer']})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )