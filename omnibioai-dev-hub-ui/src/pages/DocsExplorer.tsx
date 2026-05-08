import { useEffect, useState } from "react";
import { getStatus } from "../api/client";

export default function DocsExplorer() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    getStatus().then(setStatus);
  }, []);

  return (
    <div className="card">
      <h2>System Explorer</h2>

      {!status ? (
        <p>Loading...</p>
      ) : (
        <pre>{JSON.stringify(status, null, 2)}</pre>
      )}
    </div>
  );
}