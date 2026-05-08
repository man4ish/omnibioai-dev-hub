export default function MainLayout({ children }: any) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>OmniBioAI Dev Hub V3 — Control Center</h1>
      </header>

      <div className="grid">{children}</div>
    </div>
  );
}