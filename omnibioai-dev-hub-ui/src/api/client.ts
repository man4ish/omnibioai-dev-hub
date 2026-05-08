const API_BASE = "http://127.0.0.1:8082";

// ------------------ RAG QUERY ------------------
export const ragQuery = async (query: string) => {
  const res = await fetch(`${API_BASE}/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  return res.json();
};

// ------------------ STREAMING ------------------
export const ragStream = async (
  query: string,
  onToken: (t: string) => void,
  onDone?: () => void,
  onError?: (e: any) => void
) => {
  try {
    const res = await fetch(`${API_BASE}/rag/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const reader = res.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error("No stream");

    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const p of parts) {
        const match = p.match(/data:\s*(.*)/);
        if (match?.[1]) {
          try {
            const json = JSON.parse(match[1]);
            if (json.chunk) onToken(json.chunk);
            if (json.done) onDone?.();
          } catch {
            onToken(match[1]);
          }
        }
      }
    }

    onDone?.();
  } catch (e) {
    onError?.(e);
  }
};

// ------------------ STATUS ------------------
export const getStatus = async () => {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
};