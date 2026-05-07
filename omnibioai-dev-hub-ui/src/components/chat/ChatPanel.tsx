import React, { useState } from "react";
import { useRagStream } from "../../hooks/useRagStream";

export default function ChatPanel() {
  const [input, setInput] = useState("");
  const { answer, run, loading } = useRagStream();

  return (
    <div className="flex flex-col h-full p-3">
      
      <div className="flex-1 border p-2 overflow-auto">
        <div className="text-sm whitespace-pre-wrap">{answer}</div>
      </div>

      <div className="mt-2 flex">
        <input
          className="border flex-1 p-2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />

        <button
          className="bg-blue-600 text-white px-3 ml-2"
          onClick={() => run(input)}
        >
          {loading ? "Streaming..." : "Send"}
        </button>
      </div>

    </div>
  );
}