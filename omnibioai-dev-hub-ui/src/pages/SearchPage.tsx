import { useState } from "react";
import { ragQuery } from "../api/client";

export default function SearchPage() {
  const [q, setQ] = useState("");
  const [result, setResult] = useState<any>(null);

  const search = async () => {
    const res = await ragQuery(q);
    setResult(res.data);
  };

  return (
    <div className="card">
      <h2>Vector Search</h2>

      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search embeddings..."
      />

      <button onClick={search}>Search</button>

      {result && (
        <pre className="output">
          {JSON.stringify(result.context, null, 2)}
        </pre>
      )}
    </div>
  );
}