import { useState } from "react";
import { streamRAG } from "../../hooks/useRagStream";
import { useAppStore } from "../../store/appStore";

export default function ChatPanel() {
  const [input, setInput] = useState("");
  const answer = useAppStore((s) => s.answer);

  const send = async () => {
    await streamRAG(input);
  };

  return (
    <div>
      <div style={{ height: 300, overflow: "auto" }}>
        {answer}
      </div>

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />

      <button onClick={send}>Send</button>
    </div>
  );
}