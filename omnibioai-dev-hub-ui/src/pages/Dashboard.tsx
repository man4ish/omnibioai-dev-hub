import ChatPanel from "../components/chat/ChatPanel";
import KnowledgeGraph from "../components/graph/KnowledgeGraph";

export default function Dashboard() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
      <div>
        <h2>OmniBioAI Chat</h2>
        <ChatPanel />
      </div>

      <div>
        <h2>Knowledge Graph</h2>
        <KnowledgeGraph />
      </div>
    </div>
  );
}