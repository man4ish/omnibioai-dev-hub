import ForceGraph3D from "react-force-graph-3d";
import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8082";

export default function KnowledgeGraph() {
  const [graph, setGraph] = useState<any>({ nodes: [], links: [] });

  useEffect(() => {
    fetch(`${API}/status`)
      .then((r) => r.json())
      .then((data) => {
        setGraph({
          nodes: [
            { id: "RAG", group: 1 },
            { id: "Vector", group: 2 },
            { id: "Graph", group: 2 },
            { id: "Plugin", group: 2 },
          ],
          links: [
            { source: "RAG", target: "Vector" },
            { source: "RAG", target: "Graph" },
            { source: "RAG", target: "Plugin" },
          ],
        });
      });
  }, []);

  return (
    <div style={{ height: "400px" }}>
      <ForceGraph3D
        graphData={graph}
        nodeLabel="id"
        nodeAutoColorBy="group"
      />
    </div>
  );
}