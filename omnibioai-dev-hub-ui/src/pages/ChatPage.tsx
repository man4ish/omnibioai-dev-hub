import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { ragStream } from "../api/client";

interface Message {
  role: "user" | "bot";
  text: string;
  sources?: string[];
  streaming?: boolean;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "bot",
      text: "Hello! I'm the OmniBioAI Query Assistant powered by RAG V6. Ask me anything about your indexed documents.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || loading) return;
    const query = input.trim();
    setInput("");
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", text: query }]);
    setMessages((prev) => [...prev, { role: "bot", text: "", streaming: true }]);

    await ragStream(
      query,
      (token) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "bot") {
            updated[updated.length - 1] = { ...last, text: last.text + token };
          }
          return updated;
        });
      },
      (fullContent?: string) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "bot") {
            updated[updated.length - 1] = {
              ...last,
              text: fullContent ?? last.text,
              streaming: false,
            };
          }
          return updated;
        });
        setLoading(false);
      },
      (err) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "bot",
            text: `Error: ${err}`,
            streaming: false,
          };
          return updated;
        });
        setLoading(false);
      }
    );
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <>
      <div className="page-header">
        <div className="page-title">Query Assistant</div>
        <div className="page-sub">RAG V6 · nomic-embed-text · FAISS</div>
      </div>

      <div className="surface" style={{ flex: 1, display: "flex", flexDirection: "column", padding: 0, overflow: "hidden", minHeight: 0, height: "calc(100vh - 180px)" }}>
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-bubble ${msg.role}`}>
              {msg.role === "bot" ? (
                <div style={{ fontSize: 13 }}>
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                  {msg.streaming && <span className="cursor-blink" />}
                </div>
              ) : (
                <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "var(--sans)", fontSize: 13 }}>
                  {msg.text}
                </pre>
              )}
              {msg.sources && msg.sources.length > 0 && (
                <div className="chat-sources">
                  Sources: {msg.sources.join(" · ")}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="chat-input-row">
          <textarea
            className="chat-input"
            placeholder="Ask OmniBioAI... (Enter to send, Shift+Enter for newline)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            rows={1}
          />
          <button className="btn-send" onClick={send} disabled={loading}>
            {loading ? <><span className="spinner" />Streaming</> : "Send →"}
          </button>
        </div>
      </div>
    </>
  );
}