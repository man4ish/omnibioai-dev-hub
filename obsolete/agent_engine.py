from rag.agent_router_v4 import AgentPlannerV4
from rag.tool_executor_v4 import ToolExecutorV4
from rag.memory_store_v4 import MemoryStoreV4
from rag.streaming_engine_v4 import StreamingEngineV4


class RAGAgentEngineV4:

    def __init__(self, vector_store, graph_store, plugin_index, embedder):

        self.planner = AgentPlannerV4()
        self.memory = MemoryStoreV4()
        self.tools = ToolExecutorV4(
            vector_store,
            graph_store,
            plugin_index,
            embedder
        )
        self.streamer = StreamingEngineV4()

    # -------------------------
    # FULL AUTONOMOUS LOOP
    # -------------------------
    def run(self, query: str):

        # 1. PLAN
        plan = self.planner.route(query)

        # 2. TOOL EXECUTION
        results = self.tools.run(plan, query)

        # 3. CONTEXT BUILD
        context = "\n".join(
            r.get("text", "") for r in results
        )

        # 4. MEMORY
        memory_context = self.memory.get_context()

        # 5. PROMPT
        prompt = f"""
You are OmniBioAI RAG V4 Autonomous Agent.

MEMORY:
{memory_context}

TOOL CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Reason step by step
- Use tools context if needed
- Be concise and accurate
"""

        # 6. STORE MEMORY
        self.memory.add("user", query)

        # 7. RETURN STREAM
        return {
            "plan": plan,
            "context": context,
            "memory": memory_context,
            "stream": self.streamer.stream(prompt)
        }