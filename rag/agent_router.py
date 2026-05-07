from typing import Dict, Any, List


class AgentPlannerV4:
    """
    Converts query -> execution plan
    (this is what makes it "agentic")
    """

    def route(self, query: str) -> Dict[str, Any]:

        q = query.lower()

        plan = {
            "query": query,
            "steps": []
        }

        # -------------------------
        # Memory-aware queries
        # -------------------------
        if any(x in q for x in ["remember", "last", "previous", "before"]):
            plan["steps"].append("memory_search")

        # -------------------------
        # Graph reasoning queries
        # -------------------------
        if any(x in q for x in ["relation", "connected", "path", "graph"]):
            plan["steps"].append("graph_search")

        # -------------------------
        # Plugin-aware queries
        # -------------------------
        if any(x in q for x in ["plugin", "tool", "module", "pipeline"]):
            plan["steps"].append("plugin_search")

        # -------------------------
        # Default: vector retrieval
        # -------------------------
        plan["steps"].append("vector_search")

        # -------------------------
        # Always allow synthesis
        # -------------------------
        plan["steps"].append("synthesize")

        return plan