from rag.agent_router import AgentRouter
from rag.tool_executor import ToolExecutor
from rag.memory_store import MemoryStore
from rag.streaming_engine import StreamingEngine


class RAGAgentEngineV3:
    """
    OmniBioAI Agentic RAG V3 Engine
    - tool routing
    - hybrid retrieval
    - memory
    - streaming generation
    """

    def __init__(self, vector_store, graph_store, plugin_index, embedder):
        self.router = AgentRouter()
        self.memory = MemoryStore()

        self.tools = ToolExecutor(
            vector_store=vector_store,
            graph_store=graph_store,
            plugin_index=plugin_index,
            embedder=embedder
        )

        self.streamer = StreamingEngine()

    # ---------------- FULL PIPELINE ----------------
    def run(self, query: str):

        # 1. route intent
        intent = self.router.route(query)

        # 2. tool execution (vector/graph/plugin)
        results = self.tools.run(intent, query)

        # 3. context building
        context = "\n".join(
            [r.get("text", "") for r in results if r.get("text")]
        )

        # 4. memory retrieval
        memory_context = self.memory.get_context()

        # 5. prompt assembly
        prompt = f"""
You are OmniBioAI Agent v3 (advanced reasoning system).

MEMORY:
{memory_context}

CONTEXT:
{context}

QUESTION:
{query}

Rules:
- Use memory + context
- Be precise
- Prefer technical explanation
"""

        # 6. store memory
        self.memory.add("user", query)

        # 7. stream response
        return {
            "intent": intent,
            "context": context,
            "memory": memory_context,
            "stream": self.streamer.stream(prompt)
        }