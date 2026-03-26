# RAG Implementation Log

Status key: ✅ Done · 🔄 In Progress · ⬜ Not Started

---

## Phase 1 — Infrastructure ✅

### Files created / modified

| File | Description |
|---|---|
| `docker-compose.yml` | Qdrant service — ports 6333 (HTTP) and 6334 (gRPC), named volume `qdrant_data` |
| `app/rag/__init__.py` | RAG package root |
| `app/rag/ingestion/__init__.py` | Ingestion sub-package |
| `app/rag/retrieval/__init__.py` | Retrieval sub-package |
| `app/rag/models.py` | Pydantic models: `Document`, `Chunk`, `QueryResult` |
| `pyproject.toml` | Added `qdrant-client>=1.9`, `docling>=2.0`, `tiktoken>=0.7`, `python-dotenv>=1.0.0` |

### How to test

```bash
# 1. Start Qdrant
docker compose up qdrant -d

# 2. Confirm it is healthy
curl localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}

# 3. Install new dependencies
uv sync
```

---

## Phase 2 — Ingestion Pipeline ✅

### Files created / modified

| File | Description |
|---|---|
| `app/rag/ingestion/parsers.py` | `parse_document(path)` — `.md/.txt` direct read, `.docx` via `python-docx`, PDF/advanced formats via optional `docling` |
| `app/rag/ingestion/chunker.py` | `chunk_document()` — h1/h2/h3 structural split (parents) + recursive ≤256-token children with 50-token overlap |
| `app/rag/ingestion/embedder.py` | `embed_documents()` / `embed_query()` — batched `gemini-embedding-2-preview` with `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types |
| `app/rag/ingestion/indexer.py` | `ensure_collection()` + `upsert_children()` + `upsert_parents()` — Qdrant collection with `dense` (768-dim cosine) + sparse BM25 vectors; parents stored payload-only |
| `app/rag/pipeline.py` | `ingest(path)` orchestrator + CLI entrypoint (`python -m app.rag.pipeline ingest`) |
| `pyproject.toml` | Lean base deps + optional extras: `agent` (ADK stack) and `pdf` (docling) |

### How to test

```bash
# 1. Ingest a PDF (replace with any real PDF)
python -m app.rag.pipeline ingest ./sample.pdf
# Expected: "Indexed N child chunks from 'sample.pdf'."

# 2. Verify points in Qdrant dashboard
open http://localhost:6333/dashboard
# Collection 'rag_docs' should appear with N points, each having
# a 'dense' vector.

# 3. Quick Python smoke test
python - <<'EOF'
from app.rag.ingestion.chunker import chunk_document
from app.rag.models import Document
from datetime import datetime

doc = Document(source_file="test.pdf", doc_type="pdf")
chunks = chunk_document(doc, "# Intro\nHello world.\n\n## Background\nMore text.")
parents = [c for c in chunks if c.is_parent]
children = [c for c in chunks if not c.is_parent]
print(f"Parents: {len(parents)}, Children: {len(children)}")
assert all(c.parent_chunk_id is not None for c in children)
print("OK")
EOF
```

---

## Dependency Profiles ✅

### Files modified

| File | Description |
|---|---|
| `pyproject.toml` | Moved heavy dependencies (`google-adk`, telemetry, `docling`) into optional extras; base install remains RAG-first |
| `app/rag/ingestion/parsers.py` | Added explicit runtime message when `docling` is missing (`uv sync --extra pdf`) |

### How to test

```bash
# 1. Lean install (no ADK/docling)
uv sync

# 2. Add only PDF parsing support
uv sync --extra pdf

# 3. Add full ADK app stack
uv sync --extra agent

# 4. Add both
uv sync --extra pdf --extra agent
```

---

## Phase 3 — Retrieval Pipeline ⬜

### Files to create

| File | Description |
|---|---|
| `app/rag/retrieval/query_processor.py` | HyDE — Gemini generates a hypothetical answer; embed original + hypothetical with `RETRIEVAL_QUERY` |
| `app/rag/retrieval/retriever.py` | Qdrant `query_points` with dense + BM25 sparse prefetch, RRF fusion; merge across both query vectors → top 20 |
| `app/rag/retrieval/reranker.py` | Gemini 1.5 Flash reranker — top 20 `(query, chunk)` pairs → ranked chunk IDs → keep top 7 |
| `app/rag/retrieval/assembler.py` | Parent-child swap, metadata header, dedup by `parent_chunk_id`, token-budget packing |
| `app/rag/pipeline.py` | Add `retrieve(query)` entrypoint |

### How to test (once implemented)

```bash
# Requires an ingested collection from Phase 2
python -m app.rag.pipeline retrieve "what is X"
# Expected: formatted text blocks with [Source: ...] headers

# Debug: set env var to see HyDE expansion
LOGLEVEL=DEBUG python -m app.rag.pipeline retrieve "what is X"
```

---

## Phase 4 — ADK Integration ⬜

### Files to modify

| File | Change |
|---|---|
| `app/agent.py` | Import `pipeline.retrieve`; expose as `retrieve_context(query: str) -> str` tool; update agent instruction |

### How to test

```bash
make playground
# Ask agent a question about an ingested document.
# Expected: response cites "[Source: file.pdf | Section: ...]"
```

---

## Phase 5 — Tests ⬜

### Files to create

| File | Description |
|---|---|
| `tests/unit/test_chunker.py` | Assert parent/child relationships, overlap, token counts |
| `tests/integration/test_rag.py` | Ingest a small PDF, retrieve one query, assert non-empty result with source metadata |

### How to test

```bash
# Unit tests (no Qdrant needed)
pytest tests/unit/ -v

# Integration tests (Qdrant must be running)
pytest tests/integration/test_rag.py -v
```
