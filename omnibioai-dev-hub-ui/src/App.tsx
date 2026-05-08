import MainLayout from "./components/layout/MainLayout";
import ChatPage from "./pages/ChatPage";
import GraphView from "./pages/GraphView";
import SearchPage from "./pages/SearchPage";
import DocsExplorer from "./pages/DocsExplorer";

export default function App() {
  return (
    <MainLayout>
      <div className="panel">
        <ChatPage />
      </div>

      <div className="panel">
        <GraphView />
      </div>

      <div className="panel">
        <SearchPage />
      </div>

      <div className="panel">
        <DocsExplorer />
      </div>
    </MainLayout>
  );
}