import { useEffect, useState } from "react";
import { getStatus } from "../api/client";

export default function GraphView() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    const load = async () => {
      const res = await getStatus();
      setData(res);
    };

    load();
  }, []);

  return (
    <div className="card">
      <h2>Knowledge Graph</h2>

      {!data ? (
        <p>Loading graph...</p>
      ) : (
        <>
          <p>Edges: {data.graph_edges}</p>
          <p>Plugins: {data.plugins_loaded}</p>
          <p>Repos: {data.repos_loaded}</p>
        </>
      )}
    </div>
  );
}