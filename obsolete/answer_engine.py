from rag.agent_router import AgentRouter
from rag.tool_executor import ToolExecutor
from rag.memory_store import MemoryStore
from rag.streaming_engine import StreamingEngine


class RAGAgentEngineV3:
    def __init__(self, vector_store, graph_store, plugin_index, embedder):
        self.router = AgentRouter()
        self.memory = MemoryStore()
        self.tools = ToolExecutor(vector_store, graph_store, plugin_index, embedder)
        self.streamer = StreamingEngine()

    # ---------------- FULL PIPELINE ----------------
    def run(self, query: str):

        intent = self.router.route(query)

        # tool execution
        results = self.tools.run(intent, query)

        context = "\n".join(
            [r.get("text", "") for r in results]
        )

        memory_context = self.memory.get_context()

        prompt = f"""
You are OmniBioAI Agent v3.

MEMORY:
{memory_context}

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly:
"""

        self.memory.add("user", query)

        return {
            "intent": intent,
            "context": context,
            "stream": self.streamer.stream(prompt),
            "memory": memory_context
        }