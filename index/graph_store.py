class GraphStore:
    def __init__(self):
        # adjacency list: entity -> [(related_node, relation)]
        self.edges = {}

    def add_edge(self, a, b, relation="related"):
        if a not in self.edges:
            self.edges[a] = []

        self.edges[a].append((b, relation))

    def search(self, query: str):
        """
        Simple semantic-ish graph expansion (stub v1)
        Later you can upgrade to Neo4j / NetworkX embeddings.
        """

        results = []

        query = query.lower()

        for node, links in self.edges.items():
            if query in node.lower():
                for target, rel in links:
                    results.append({
                        "text": f"{node} -> {target} ({rel})",
                        "source": "graph"
                    })

        return results