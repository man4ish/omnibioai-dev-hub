import { useState } from "react";
import { ragStream } from "../api/client";

export default function ChatPage() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const send = async () => {
    setAnswer("");
    setLoading(true);

    await ragStream(
      query,
      (token) => setAnswer((prev) => prev + token),
      () => setLoading(false),
      (err) => {
        setAnswer("ERROR: " + err);
        setLoading(false);
      }
    );
  };

  return (
    <div className="card">
      <h2>OmniBioAI Chat</h2>

      <textarea
        placeholder="Ask OmniBioAI..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <button onClick={send} disabled={loading}>
        {loading ? "Streaming..." : "Send"}
      </button>

      <pre className="output">{answer}</pre>
    </div>
  );
}