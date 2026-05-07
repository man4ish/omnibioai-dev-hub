import requests
import json
import time
from typing import List, Dict, Any, Optional


# =========================================================
# RAG QUERY ROUTER V2 - OMNIBIOAI
# Hybrid: Vector + Graph + Plugin + LLM (Ollama)
# =========================================================

class RAGQueryRouterV2:
    """
    Production RAG engine:
    - Hybrid retrieval (vector + graph + plugin)
    - Context builder
    - Ollama LLM streaming
    - Lightweight intent routing
    """

    def __init__(self, vector_store, graph_store=None, plugin_index=None):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.plugin_index = plugin_index

        # Ollama config
        self.llm_url = "http://localhost:11434/api/chat"
        self.model = "llama3"

        # retrieval config
        self.top_k = 8

    # =========================================================
    # INTENT ROUTING (lightweight v2)
    # =========================================================
    def detect_intent(self, query: str) -> str:
        q = query.lower()

        if any(x in q for x in ["graph", "relation", "path", "edge"]):
            return "graph"

        if any(x in q for x in ["plugin", "tool", "workflow"]):
            return "plugin"

        if any(x in q for x in ["compare", "difference", "vs"]):
            return "hybrid"

        return "hybrid"

    # =========================================================
    # VECTOR RETRIEVAL
    # =========================================================
    def vector_search(self, query: str) -> List[Dict]:
        if not self.vector_store:
            return []

        try:
            return self.vector_store.search(query, top_k=self.top_k)
        except Exception:
            return []

    # =========================================================
    # GRAPH RETRIEVAL
    # =========================================================
    def graph_search(self, query: str) -> List[Dict]:
        if not self.graph_store:
            return []

        try:
            return self.graph_store.search(query)
        except Exception:
            return []

    # =========================================================
    # PLUGIN RETRIEVAL
    # =========================================================
    def plugin_search(self, query: str) -> List[Dict]:
        if not self.plugin_index:
            return []

        try:
            return self.plugin_index.search(query)
        except Exception:
            return []

    # =========================================================
    # HYBRID RETRIEVAL
    # =========================================================
    def hybrid_retrieve(self, query: str) -> Dict[str, List[Dict]]:
        intent = self.detect_intent(query)

        results = {
            "vector": self.vector_search(query),
            "graph": self.graph_search(query),
            "plugin": self.plugin_search(query),
            "intent": intent
        }

        return results

    # =========================================================
    # SCORE + MERGE CONTEXT
    # =========================================================
    def build_context(self, results: Dict[str, List[Dict]]) -> str:
        """
        Merge + deduplicate + format context for LLM
        """

        context_blocks = []

        def format_block(item, source):
            text = item.get("text", "")
            return f"[{source}] {text}"

        # vector results
        for r in results.get("vector", []):
            context_blocks.append(format_block(r, "vector"))

        # graph results
        for r in results.get("graph", []):
            context_blocks.append(format_block(r, "graph"))

        # plugin results
        for r in results.get("plugin", []):
            context_blocks.append(format_block(r, "plugin"))

        # limit context size
        return "\n".join(context_blocks[:12])

    # =========================================================
    # LLM STREAMING (OLLAMA)
    # =========================================================
    def stream_llm(self, query: str, context: str):
        """
        Streams tokens from Ollama
        """

        prompt = f"""
You are OmniBioAI RAG assistant.

Use the context below to answer the query.

CONTEXT:
{context}

QUERY:
{query}

Answer clearly and technically.
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }

        try:
            response = requests.post(
                self.llm_url,
                json=payload,
                stream=True
            )

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token

                except Exception:
                    continue

        except Exception as e:
            yield f"[LLM_ERROR]: {str(e)}"

    # =========================================================
    # MAIN QUERY ENTRYPOINT
    # =========================================================
    def query(self, query: str) -> Dict[str, Any]:
        """
        Full RAG pipeline (non-streaming)
        """

        results = self.hybrid_retrieve(query)
        context = self.build_context(results)

        # collect full response from stream
        output = ""

        for token in self.stream_llm(query, context):
            output += token

        return {
            "intent": results["intent"],
            "answer": output,
            "context": context,
            "sources": {
                "vector": len(results["vector"]),
                "graph": len(results["graph"]),
                "plugin": len(results["plugin"])
            }
        }