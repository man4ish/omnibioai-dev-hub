class ToolExecutorV4:
    """
    Executes retrieval tools from plan
    """

    def __init__(self, vector_store, graph_store, plugin_index, embedder):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.plugin_index = plugin_index
        self.embedder = embedder

    # -------------------------
    # MAIN ENTRY
    # -------------------------
    def run(self, plan: dict, query: str):

        results = []

        for step in plan["steps"]:

            if step == "vector_search":
                results.extend(self._vector(query))

            elif step == "graph_search":
                results.extend(self._graph(query))

            elif step == "plugin_search":
                results.extend(self._plugin(query))

            elif step == "memory_search":
                results.extend(self._memory(query))

        return results

    # -------------------------
    def _vector(self, query):
        emb = self.embedder.encode([query])[0]
        return self.vector_store.search(emb, top_k=5)

    def _graph(self, query):
        return self.graph_store.search(query)

    def _plugin(self, query):
        return self.plugin_index.search(query)

    def _memory(self, query):
        # placeholder (hook to memory store later)
        return [{
            "text": "Memory lookup stub (V4 upgrade needed)",
            "source": "memory"
        }]