import { useEffect, useState } from "react";
import { getStatus } from "../api/client";

export default function DocsExplorer() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStatus()
      .then((s) => { setStatus(s); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  const refresh = () => {
    setLoading(true);
    setError(null);
    getStatus()
      .then((s) => { setStatus(s); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  };

  return (
    <>
      <div className="page-header">
        <div className="page-title">System Status</div>
        <div className="page-sub">Live status from the RAG V6 backend</div>
      </div>

      {loading && (
        <div className="surface" style={{ textAlign: "center", padding: 40 }}>
          <span className="spinner" />
          <span style={{ color: "var(--text-secondary)" }}>Connecting to backend...</span>
        </div>
      )}

      {error && (
        <div className="surface" style={{ borderColor: "var(--red-dim)" }}>
          <div style={{ color: "var(--red)", marginBottom: 8, fontWeight: 500 }}>Backend unreachable</div>
          <div style={{ color: "var(--text-secondary)", fontSize: 13, fontFamily: "var(--mono)", marginBottom: 12 }}>{error}</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
            Make sure the API server is running: <code style={{ color: "var(--teal)" }}>uvicorn api.main:app --port 8082</code>
          </div>
          <button className="btn-primary" onClick={refresh} style={{ marginTop: 12 }}>Retry</button>
        </div>
      )}

      {status && !loading && (
        <>
          <div className="status-grid">
            {status.total_docs !== undefined && (
              <div className="status-block">
                <div className="status-block-label">Documents</div>
                <div className="status-block-val">{status.total_docs?.toLocaleString() ?? "—"}</div>
              </div>
            )}
            {status.total_vectors !== undefined && (
              <div className="status-block">
                <div className="status-block-label">Vectors</div>
                <div className="status-block-val">{status.total_vectors?.toLocaleString() ?? "—"}</div>
              </div>
            )}
            {status.index_size !== undefined && (
              <div className="status-block">
                <div className="status-block-label">Index Size</div>
                <div className="status-block-val">{status.index_size ?? "—"}</div>
              </div>
            )}
            {status.model !== undefined && (
              <div className="status-block">
                <div className="status-block-label">Embed Model</div>
                <div className="status-block-val" style={{ fontSize: 13 }}>{status.model ?? "—"}</div>
              </div>
            )}
            {status.version !== undefined && (
              <div className="status-block">
                <div className="status-block-label">Version</div>
                <div className="status-block-val">{status.version ?? "—"}</div>
              </div>
            )}
            {status.uptime !== undefined && (
              <div className="status-block">
                <div className="status-block-label">Uptime</div>
                <div className="status-block-val">{status.uptime ?? "—"}</div>
              </div>
            )}
          </div>

          <div className="surface">
            <div className="section-header">
              <div className="section-title">Raw Status Payload</div>
              <button className="section-action" onClick={refresh}>↻ Refresh</button>
            </div>
            <pre className="json-dump">{JSON.stringify(status, null, 2)}</pre>
          </div>
        </>
      )}
    </>
  );
}