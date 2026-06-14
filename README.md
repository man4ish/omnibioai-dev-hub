# OmniBioAI Dev Hub — RAG V6

Production-grade Retrieval-Augmented Generation (RAG) system powering the OmniBioAI ecosystem documentation, architecture search, workflow discovery, and developer assistant APIs.

---

# Features

* FAISS-native vector search (IndexFlatIP, 768-dim)
* Incremental per-repo indexing with hash-based dedup
* Ollama local embeddings (`nomic-embed-text`) + local LLM inference (`llama3`)
* FastAPI API server
* Real token-level SSE streaming via Ollama `stream: true`
* Chunk-level document retrieval with source attribution
* Repository-wide multi-project indexing (19 repos)
* Fully local execution — no OpenAI dependency
* Production-safe embedding normalization
* V6 dimension consistency enforcement

---

# Architecture

```text
Repositories  (REPO_BASE/omnibioai-*)
     ↓
Document Loader  (ingestion/doc_loader.py)
     ↓
Chunker  (processing/chunker.py, 500-word windows / 2000-char max)
     ↓
Ollama Embeddings  (nomic-embed-text, 768-d, normalized)
     ↓
FAISS Vector Index  (IndexFlatIP, data/faiss_index/)
     ↓
RAG Engine  (rag/engine.py)
     ↓
FastAPI API  (api/main.py + api/routes/rag.py)
     ↓
LLM Answer Generation  (llama3 via Ollama, blocking or token-streamed)
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

### V6 Fix

Both ingestion (`scripts/build_index.py`) and retrieval (`rag/engine.py`) now call the same `ollama_embed("nomic-embed-text")` function — the single source of truth.

`embeddings/embedder.py` (sentence-transformers, 384-dim) is retained for test coverage only and is not on any live request path.

---

# Repository Structure

```text
omnibioai-dev-hub/
│
├── api/
│   ├── main.py              # FastAPI app, startup, /status, /health
│   └── routes/
│       └── rag.py           # /rag/query and /rag/stream endpoints
│
├── rag/
│   ├── engine.py            # RAGEngine: retrieve, build_context, answer, stream_llm
│   └── control_plane.py     # Singleton lifecycle manager
│
├── index/
│   ├── vector_store.py      # FAISS wrapper (add, search, save, load)
│   ├── graph_store.py       # In-memory knowledge graph (BFS expansion)
│   └── plugin_index.py      # Plugin doc registry
│
├── embeddings/
│   └── embedder.py          # SentenceTransformer wrapper (tests only, not live)
│
├── retrieval/
│   └── retriever.py         # Retriever class (tests only, not live)
│
├── ingestion/
│   └── doc_loader.py        # Markdown document loader with SKIP_DIRS/SKIP_PATH_SEGMENTS
│
├── processing/
│   └── chunker.py           # 500-word / 2000-char chunker
│
├── scripts/
│   └── build_index.py       # Index builder entry point
│
├── data/
│   └── faiss_index/         # index.faiss + metadata.pkl (gitignored)
│
├── configs/
│   └── repos.yaml           # Repo list documentation
│
└── .env.example             # Environment variable template
```

---

# Requirements

## Python

Recommended:

```text
Python 3.11 or 3.12
```

> **Note:** Python 3.13 removes `numpy.distutils`, breaking `faiss-cpu`. Use 3.11 or 3.12. In Docker the provided image uses Python 3.12.

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

### Generation Model

The system uses `llama3` by default (hardcoded in `rag/engine.py`):

```bash
ollama pull llama3
```

Optional alternatives (change `model=` in `engine.py` if needed):

```bash
ollama pull mistral
ollama pull deepseek-coder
```

---

# Installation

## Create Environment

```bash
conda create -n omnibioai-dev-hub python=3.12 -y
conda activate omnibioai-dev-hub
```

---

## Install Dependencies

```bash
pip install fastapi uvicorn requests numpy faiss-cpu
```

> `sentence-transformers` is only required to run the test suite (`tests/test_embeddings.py`, `tests/test_retriever.py`). It is not needed to run the server or build the index:
> ```bash
> pip install sentence-transformers  # tests only
> ```

---

# Configuration

## REPO_BASE

`REPO_BASE` is the only required configuration. It must point to the directory that contains the `omnibioai-*` repos as immediate children.

```bash
export REPO_BASE=/home/manish/Desktop/machine
```

The indexer **exits immediately with a clear error** if no repos are found under `REPO_BASE`:

```text
❌  No repos found under REPO_BASE='/some/wrong/path'
    Set REPO_BASE to the directory that contains the omnibioai-* repos.
    Example:  export REPO_BASE=/home/manish/Desktop/machine
    Docker:   -e REPO_BASE=/repos  (with repos volume mounted at /repos)
```

In Docker the image sets `ENV REPO_BASE=/repos` automatically — no action needed.

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
# edit REPO_BASE as needed
```

---

# Build Index

## Clean Existing Data

```bash
rm -rf data/faiss_index/*
```

---

## Build V6 Index

```bash
python scripts/build_index.py
```

Expected output:

```text
🚀 Incremental V6 Indexing Starting...
⚠️  2 repos not found, will be skipped: ['omnibioai-security-audit', 'omnibioai-hpc-policy-engine']
📄 Loaded N documents
...
💾 Index saved to .../data/faiss_index (10877 vectors)
✅ V6 Index Complete
{'too_short': 4, 'deduped': 31, 'embed_failed': 4, 'new': 10881, 'chunks_indexed': 10877}
```

### How deduplication works

`seen_hashes` is **reset per repo** so cross-repo identical chunks each get their own index entry under their canonical source path. Within a single repo, duplicate chunks (e.g. shared boilerplate across plugin READMEs) are deduplicated.

Chunks shorter than 10 characters are discarded (`MIN_CHUNK_CHARS = 10`) to eliminate overflow tails produced by the hard character-slice in `chunker.py`.

### Excluded paths

`ingestion/doc_loader.py` skips the following during the walk:

**By directory name (`SKIP_DIRS`):**

```python
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache", "obsolete"}
```

Note: both `.venv` and bare `venv` are excluded. `.pytest_cache` is excluded because it contains auto-generated `README.md` stubs (present in 17 of the 19 repos) that would otherwise pollute the index with boilerplate.

**By path segment (`SKIP_PATH_SEGMENTS`):**

```python
SKIP_PATH_SEGMENTS = {"work"}
```

This excludes `omnibioai/work/` which contains UUID-named runtime copies of workflow bundle READMEs. Without this exclusion, those copies would claim index slots before the canonical `omnibioai-workflow-bundles/` paths are processed, causing all 50 bundles to appear indexed under `omnibioai/work/` instead.

---

# Current Index Stats

As of 2026-06-14 (Phase 2 rebuild — markdown-aware chunker):

| Metric | Value |
|--------|-------|
| Total vectors | 10,877 |
| Unique source files | ~965 |
| Repos indexed | 17 of 19 |
| Workflow bundles covered | 50 / 50 |
| Chunks filtered (too short) | 4 |
| Cross-repo deduped | 31 |
| **Recall@5 (eval set)** | **96% (24/25)** |

**Chunking strategy:** Markdown-structure-aware — splits on H1/H2/H3 headers as natural section boundaries, never splits inside fenced code blocks, falls back to paragraph boundaries (`\n\n`) for long sections, and word-boundary splitting for oversized paragraphs. Each chunk is prefixed with its ancestor header breadcrumb (e.g. `# ATACseq Pipeline > ## Parameters`) so retrieval context carries section identity. Previous fixed 500-word window chunker produced 2,067 vectors; the markdown chunker produces 10,877 (5.3× more granular chunks).

Missing repos (not present on disk): `omnibioai-security-audit`, `omnibioai-hpc-policy-engine`. The indexer skips them with a warning and continues.

> **Note:** `data/faiss_index/` is excluded from git (see `.gitignore`). Regenerate with `python scripts/build_index.py`.

---

# Run API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8082 --reload
```

In Docker the server starts automatically as PID 1.

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
    "/repos/omnibioai-workflow-bundles/README.md"
  ],
  "context": [
    {"score": 0.87, "text": "...", "source": "/repos/omnibioai-workflow-bundles/README.md"}
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

Body: `{"query": "your question"}`

### How it works

1. The server retrieves the top-5 chunks from FAISS (same path as `/rag/query`).
2. It builds the prompt from the retrieved context.
3. It calls Ollama's `/api/generate` with `"stream": true`.
4. Ollama returns newline-delimited JSON (NDJSON), one object per token:
   ```json
   {"model":"llama3","response":"Based","done":false}
   {"model":"llama3","response":" on","done":false}
   ...
   {"model":"llama3","response":"","done":true}
   ```
5. Each token is immediately forwarded as an SSE event:
   ```text
   data: {"type": "token", "content": "Based"}

   data: {"type": "token", "content": " on"}

   ...

   data: {"type": "done"}
   ```

### Client-side consumption

```typescript
ragStream(
  "your query",
  (token) => appendToUI(token),      // called per token
  () => markComplete(),               // called on done
  (err) => showError(err)            // called on error
);
```

The UI client (`src/api/client.ts`) parses the `data:` envelope and dispatches `type: "token"` and `type: "done"` events.

---

# Supported Repositories

The indexer targets 19 repositories. All paths are relative to `REPO_BASE`:

```python
repos = [
    "omnibioai",                  # main platform repo
    "omnibioai-rag",
    "omnibioai-toolserver",
    "omnibioai-sdk",
    "omnibioai-workflow-bundles",  # 50 workflow bundle subdirectories
    "omnibioai-control-center",
    "omnibioai-lims",
    "omnibioai-model-registry",
    "omnibioai-dev-docker",
    "omnibioai-api-gateway",
    "omnibioai-docs",
    "omnibioai-studio",
    "omnibioai-auth",
    "omnibioai-tool-runtime",
    "omnibioai-iam-client",
    "omnibioai-policy-engine",
    "omnibioai-security-sdk",
    "omnibioai-security-audit",       # not present on disk — skipped with warning
    "omnibioai-hpc-policy-engine",    # not present on disk — skipped with warning
]
```

The two missing repos are skipped gracefully at build time. When they become available, no code change is needed — just add them to the directory.

---

# V6 Retrieval Pipeline

## Step 1 — Chunking

Documents are split using a 500-word sliding window, hard-capped at 2000 characters per chunk. Chunks shorter than 10 characters are discarded.

> **Known limitation:** The hard character slice at 2000 chars can produce 1–2 word overflow fragments at word boundaries (e.g. "onment", "abases"). These are above the 10-char filter threshold. A future fix will snap the slice to the nearest word boundary. See *Future Work* below.

---

## Step 2 — Embedding

Each chunk is embedded using:

```text
nomic-embed-text  (768-dim, L2-normalized)
```

---

## Step 3 — FAISS Indexing

Vectors are stored in:

```python
faiss.IndexFlatIP
```

Pre-normalized vectors make inner product equivalent to cosine similarity.

---

## Step 4 — Query Embedding

User query is embedded using the same `nomic-embed-text` model at query time.

---

## Step 5 — Vector Search

FAISS retrieves the top-5 nearest chunks (configurable via `top_k`).

---

## Step 6 — Prompt Assembly

Retrieved chunks become the context block in a structured prompt.

---

## Step 7 — LLM Generation

Prompt sent to local Ollama `llama3` model, either blocking (`/rag/query`) or token-streamed (`/rag/stream`).

---

# Performance

## Before V6

* brute-force cosine scan
* slow retrieval
* embedding dimension mismatch
* unstable indexing
* global dedup silencing canonical paths

## After V6

* FAISS-native inner product retrieval
* stable 768-dim pipeline end-to-end
* per-repo dedup with canonical path preservation
* real SSE token streaming
* local-only execution

---

# Troubleshooting

## REPO_BASE not set or wrong path

Symptom:

```text
❌  No repos found under REPO_BASE='/repos'
```

Cause: `REPO_BASE` defaults to `/home/manish/Desktop/machine` locally and `/repos` in Docker. If neither is correct, set it explicitly:

```bash
export REPO_BASE=/path/to/parent/of/omnibioai-repos
python scripts/build_index.py
```

---

## Index loads but all queries return empty results

Cause: The index on disk was built with a different embedding model or dimension. The loaded vectors won't match query vectors.

Fix: Delete and rebuild:

```bash
rm data/faiss_index/index.faiss data/faiss_index/metadata.pkl
python scripts/build_index.py
```

---

## FAISS Dimension Mismatch

Error:

```text
AssertionError: d == self.d
```

Cause: Different embedding models used during indexing vs querying (the pre-V6 problem). Rebuild the index — V6 enforces 768-dim at both stages.

---

## Ollama Timeout

Error:

```text
Read timed out
```

Fix: Use a smaller generation model. Change the `model=` argument in `rag/engine.py`:

```python
model="mistral"  # faster than llama3 on smaller hardware
```

---

## Empty Retrieval Results

Verify the index loaded correctly:

```bash
python -c "
from index.vector_store import VectorStore
vs = VectorStore()
vs.load('data/faiss_index')
print('ntotal:', vs.index.ntotal if vs.index else 'no index')
"
```

Expected: `ntotal: 2067` (or your current count).

---

# Known Limitations / Future Work

## Chunker word-wrap (planned)

`chunker.py` slices at a hard 2000-character boundary without snapping to word boundaries. This produces 1–2 word fragments at the tail of some documents (e.g. "onment", "abases" — 16–18 chars, above the `MIN_CHUNK_CHARS=10` filter). These fragments are harmless but pollute the index with low-information chunks. The fix is to snap the slice to the nearest preceding space.

## Planned V7 Features

* Chunker word-boundary snapping
* IVF or HNSW indexes for million-scale corpora
* BM25 hybrid search
* Metadata filtering (by repo, file type, date)
* Cross-encoder reranking
* Distributed / incremental index updates
* Graph RAG (graph store already seeded)
* Plugin-aware retrieval

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
