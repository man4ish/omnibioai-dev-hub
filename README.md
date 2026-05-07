# OmniBioAI Dev Hub

## AI-Native RAG + Graph + Plugin Intelligence Platform

OmniBioAI Dev Hub is a **multi-repository RAG (Retrieval-Augmented Generation) system** designed for:

* Codebase understanding across 12+ repos
* Plugin-aware semantic search
* Hybrid retrieval (Vector + Keyword + Graph)
* Agentic RAG execution (V2–V4 roadmap)
* Streaming LLM responses via Ollama
* Live visualization UI (Graph + Vector + Chat)

It is part of the larger **OmniBioAI ecosystem**.

---

# 🧠 System Architecture

```
omnibioai-dev-hub/
│
├── ingestion/        → Repo + doc + plugin loaders
├── processing/       → Chunking + metadata + graph builder
├── embeddings/       → Ollama embedding interface
├── index/            → Vector + keyword + graph stores
├── retrieval/        → Retriever + reranker + context builder
├── rag/              → Query routing + answer engine
├── api/              → FastAPI backend (RAG API)
├── scripts/          → Index build pipelines
├── data/             → Persistent storage (chunks, graph, embeddings)
├── cache/            → runtime cache
├── logs/             → system logs
├── utils/            → helpers
├── configs/          → repo + index configs
│
└── omnibioai-dev-hub-ui/
    ├── React + Vite UI
    ├── Chat (Streaming)
    ├── Graph Viewer (Force Graph 3D)
    ├── Docs Explorer
    ├── Vector Search UI
    └── Plugin Sidebar
```

---

# 🚀 Backend (RAG Engine)

## Tech Stack

* FastAPI
* Ollama (LLMs + embeddings)
* Sentence Transformers (fallback)
* In-memory Vector Store (custom)
* Graph Store (knowledge graph)
* Plugin Index (semantic plugin search)

---

## 🔍 RAG Pipeline

### 1. Ingestion Layer

Loads multiple repositories:

```
../omnibioai
../omnibioai-rag
../omnibioai-toolserver
../omnibioai-workflow-bundles
../omnibioai-model-registry
../omnibioai-control-center
../omnibioai-sdk
```

### 2. Processing Layer

* Chunking
* Metadata enrichment
* Graph edge creation

### 3. Embedding Layer

Uses:

* `mxbai-embed-large`
* `nomic-embed-text`

### 4. Index Layer

* Vector Index (semantic search)
* Keyword Index (BM25-style logic)
* Graph Index (relations)
* Plugin Index (tool-aware retrieval)

### 5. Retrieval Layer

Hybrid retrieval:

```
Vector + Keyword + Graph + Plugin
```

### 6. RAG Engine

* Context builder
* Memory injection
* Prompt construction
* Ollama LLM streaming

---

# ⚡ API Layer

## FastAPI Endpoints

### Health

```
GET /health
```

### Status

```
GET /status
```

### RAG Query

```
POST /rag/query
```

### Streaming

```
POST /rag/stream
```

---

# 🧩 Key Features

## 1. Hybrid Retrieval

* Semantic (embeddings)
* Keyword matching
* Graph expansion
* Plugin-aware search

---

## 2. Knowledge Graph

Tracks relationships like:

```
Plugin → Tool → Pipeline → Dataset → Model
```

---

## 3. Plugin Intelligence

* Searches plugin documentation
* Injects plugin context into RAG

---

## 4. Memory System

* Stores conversation context
* Improves multi-turn reasoning

---

## 5. Streaming LLM

Uses Ollama:

* llama3
* deepseek-r1
* mistral

Streaming format:

```
token-by-token SSE stream
```

---

# 🖥️ Frontend (UI)

Location:

```
omnibioai-dev-hub-ui/
```

Built with:

* React
* TypeScript
* Vite

---

## UI Modules

### 💬 Chat Panel

* Streaming responses
* Token rendering
* RAG trace view

---

### 🌐 Graph Viewer

* Knowledge graph visualization
* Force Graph 3D
* Node relations (plugin ↔ repo ↔ tool)

---

### 📄 Docs Explorer

* Repo documentation browser
* Chunk-level view

---

### 🔎 Vector Search UI

* semantic search results
* similarity scoring

---

### 📦 Plugin Sidebar

* available tools
* plugin metadata

---

# 🔗 UI ↔ Backend Integration

Vite proxy:

```
/rag → FastAPI
/health → FastAPI
```

Streaming:

```
POST /rag/stream
```

---

# 📊 Current System Status

| Component      | Status                  |
| -------------- | ----------------------- |
| Repo ingestion | ✅ Working               |
| Chunking       | ✅ Working               |
| Embeddings     | ✅ Ollama integrated     |
| Vector store   | ✅ Active                |
| Graph store    | ⚠️ Basic implementation |
| Plugin index   | ⚠️ Prototype            |
| RAG API        | ✅ Working               |
| Streaming      | ✅ Working               |
| UI (React)     | ⚠️ Scaffold ready       |
| Memory system  | ⚠️ Basic                |
| Agent routing  | ⚠️ V3 prototype         |

---

# 🧪 Current Limitations

* No persistent vector DB (in-memory only)
* Graph store is lightweight
* No distributed indexing
* No async ingestion pipeline
* UI not fully connected to streaming state
* No evaluation layer

---

# 🚀 Future Roadmap

## 🔹 RAG V2 (Hybrid Intelligence)

* Graph-enhanced retrieval
* Plugin-aware routing
* Query decomposition

---

## 🔹 RAG V3 (Agentic System)

* Tool executor
* Memory-based reasoning
* Multi-step planning
* Streaming agent thoughts

---

## 🔹 RAG V4 (Autonomous System)

* Self-routing queries
* Auto tool selection
* Long-term memory
* Feedback learning loop

---

## 🔹 UI Enhancements

* Live streaming chat UI
* Graph interaction (click → expand reasoning)
* Vector similarity explorer
* Plugin execution dashboard

---

## 🔹 Infrastructure Upgrades

* FAISS / Milvus integration
* Redis cache layer
* Async ingestion workers
* Background indexing queue

---

## 🔹 Observability

* Query tracing
* RAG debug view
* Token-level latency tracking

---

# 🧠 Design Philosophy

OmniBioAI Dev Hub is not just a RAG system.

It is evolving into:

> **"A self-explaining biomedical + code intelligence system with graph memory + plugin reasoning + autonomous retrieval agents."**

---

# 📦 Run System

## Backend

```bash
PYTHONPATH=. uvicorn api.main:app --reload --port 8082
```

## UI

```bash
cd omnibioai-dev-hub-ui
npm install
npm run dev
```

---

# ⚠️ Notes

* Ensure Ollama is running:

```bash
ollama serve
```

* Recommended models:

  * llama3
  * mxbai-embed-large

---

# 🔥 Summary

OmniBioAI Dev Hub is a:

✔ Multi-repo intelligence engine
✔ Hybrid RAG system (vector + graph + plugin)
✔ Streaming LLM API
✔ React-based visualization platform
✔ Foundation for autonomous scientific agents

---

If you want next upgrade, I can help you build:

### 🔥 “RAG Debug Console”

* shows retrieval path step-by-step
* graph nodes activated per query
* token-level reasoning trace

or

### 🔥 “Autonomous Agent V4”

* planning loop
* tool selection
* memory-driven reasoning

