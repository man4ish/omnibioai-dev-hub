Here is an updated production-style README for your V6 FAISS-native RAG system.

# OmniBioAI Dev Hub — RAG V6

Production-grade Retrieval-Augmented Generation (RAG) system powering the OmniBioAI ecosystem documentation, architecture search, workflow discovery, and developer assistant APIs.

---

# Features

* FAISS-native vector search
* Incremental indexing
* Ollama local embeddings + local LLM inference
* FastAPI API server
* Streaming responses (SSE)
* Chunk-level document retrieval
* Repository-wide multi-project indexing
* Fully local execution
* No OpenAI dependency
* Production-safe embedding normalization
* Hybrid-ready architecture
* V6 dimension consistency enforcement

---

# Architecture

```text
Repositories
     ↓
Document Loader
     ↓
Chunker
     ↓
Ollama Embeddings (768-d)
     ↓
FAISS Vector Index
     ↓
RAG Engine
     ↓
FastAPI API
     ↓
LLM Answer Generation
```

---

# V6 Major Improvements

## FAISS Native Retrieval

Previous versions used brute-force cosine scanning across vectors.

V6 uses:

```python
faiss.IndexFlatIP
```

Benefits:

* 10–50x faster retrieval
* scalable search
* lower latency
* future ANN support

---

## Embedding Consistency Fix

A major issue in previous builds was embedding mismatch.

### Old Problem

| Stage    | Model            | Dimension |
| -------- | ---------------- | --------- |
| Indexing | all-MiniLM-L6-v2 | 384       |
| Querying | nomic-embed-text | 768       |

This caused FAISS assertion failures:

```python
AssertionError: d == self.d
```

---

## V6 Fix

Now BOTH ingestion and retrieval use:

```text
nomic-embed-text
```

Dimension:

```text
768
```

This guarantees:

* stable retrieval
* no dimension mismatch
* deterministic FAISS behavior

---

# Repository Structure

```text
omnibioai-dev-hub/
│
├── api/
│   ├── main.py
│   └── routes/
│
├── rag/
│   ├── engine.py
│   └── control_plane.py
│
├── index/
│   └── vector_store.py
│
├── embeddings/
│   └── embedder.py
│
├── ingestion/
│   └── doc_loader.py
│
├── processing/
│   └── chunker.py
│
├── scripts/
│   └── build_index.py
│
└── data/
```

---

# Requirements

## Python

Recommended:

```text
Python 3.11
```

---

## Ollama

Install:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## Required Ollama Models

### Embedding Model

```bash
ollama pull nomic-embed-text
```

### Generation Models

Recommended:

```bash
ollama pull mistral
```

Optional:

```bash
ollama pull llama3
ollama pull deepseek-coder
ollama pull deepseek-r1
```

---

# Installation

## Create Environment

```bash
conda create -n chemoinfo python=3.11 -y
conda activate chemoinfo
```

---

## Install Dependencies

```bash
pip install fastapi uvicorn requests numpy faiss-cpu sentence-transformers
```

---

# Build Index

## Clean Existing Data

```bash
rm -rf data/*
```

---

## Build V6 Index

```bash
python scripts/build_index.py
```

Expected output:

```text
🚀 Incremental V6 Indexing Starting...
✅ V6 Index Complete
```

---

# Run API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8082 --reload
```

---

# Test Query API

```bash
curl -X POST http://localhost:8082/rag/query \
-H "Content-Type: application/json" \
-d '{"query":"What is workflow engine in OmniBioAI?"}'
```

Example response:

```json
{
  "query": "What is workflow engine in OmniBioAI?",
  "answer": "According to the provided context...",
  "sources": [
    "../omnibioai-workflow-bundles/README.md"
  ],
  "context_used": 5,
  "version": "v6-faiss",
  "api_version": "v6"
}
```

---

# Streaming API

Endpoint:

```text
POST /rag/stream
```

Uses:

* Server-Sent Events (SSE)
* token streaming
* real-time generation

---

# V6 Retrieval Pipeline

## Step 1 — Chunking

Documents are split into semantic chunks.

---

## Step 2 — Embedding

Each chunk is embedded using:

```text
nomic-embed-text
```

Output dimension:

```text
768
```

---

## Step 3 — FAISS Indexing

Vectors are stored in:

```python
faiss.IndexFlatIP
```

---

## Step 4 — Query Embedding

User query is embedded using the SAME embedding model.

---

## Step 5 — Vector Search

FAISS retrieves nearest chunks.

---

## Step 6 — Prompt Assembly

Retrieved chunks become context.

---

## Step 7 — LLM Generation

Prompt sent to local Ollama model.

---

# Supported Repositories

Current indexing targets:

```python
repos = [
    "../omnibioai",
    "../omnibioai-rag",
    "../omnibioai-toolserver",
    "../omnibioai-sdk",
    "../omnibioai-workflow-bundles",
    "../omnibioai-control-center",
    "../omnibioai-lims",
    "../omnibioai-model-registry",
    "../omnibioai-dev-docker"
]
```

---

# Performance

## Before V6

* brute-force cosine scan
* slow retrieval
* dimension mismatch bugs
* unstable indexing

---

## After V6

* FAISS-native retrieval
* stable dimensions
* fast semantic search
* local-only execution
* scalable architecture

---

# Troubleshooting

## FAISS Dimension Mismatch

Error:

```text
AssertionError: d == self.d
```

Cause:

Different embedding models used during indexing vs querying.

Fix:

Rebuild index using the SAME embedding model.

---

## Ollama Timeout

Error:

```text
Read timed out
```

Fix:

Use a smaller generation model:

```python
model="mistral"
```

instead of:

```python
deepseek-r1
```

---

## Empty Retrieval Results

Check:

```bash
python -c "
from index.vector_store import VectorStore
import numpy as np

vs = VectorStore()
vs.add([np.random.rand(768)], [{'text':'test'}])

print(vs.index.ntotal)
"
```

Expected:

```text
1
```

---

# Future Roadmap

## Planned V7 Features

* IVF indexes
* HNSW search
* metadata filtering
* hybrid BM25 + vector search
* reranking
* cross-encoder scoring
* persistent FAISS storage
* multi-user collections
* distributed indexing
* workflow-aware retrieval
* graph RAG
* plugin-aware retrieval

---

# License

Internal OmniBioAI Development License.

---

# OmniBioAI Ecosystem

RAG V6 powers:

* architecture discovery
* workflow documentation search
* plugin documentation retrieval
* developer assistant APIs
* AI infrastructure exploration
* cross-repository semantic search
* internal engineering copilots
