# Plan: State-of-the-Art RAG System

**TL;DR** — Build a tool-based RAG pipeline wired into the existing ADK agent. Qdrant (Docker) stores both dense and sparse vectors per chunk. Ingestion uses docling for parsing, hierarchical chunking, and BM25+dense indexing. Retrieval uses multi-query expansion → Qdrant native hybrid search (RRF) → local cross-encoder reranking → parent-child context swap.

---

## Module Layout

```
app/
  agent.py                  ← add retrieve_context tool here
  rag/
    __init__.py
    models.py               ← shared Pydantic models: Chunk, Document, QueryResult
    ingestion/
      parsers.py            ← docling: PDF/DOCX → structured Markdown
      chunker.py            ← hierarchical parent-child splitting
      embedder.py           ← Google text-embedding-004 (batch)
      sparse.py             ← BM25 sparse vectors (rank-bm25)
      indexer.py            ← Qdrant collection setup + upsert
    retrieval/
      query_processor.py    ← multi-query expansion via Gemini
      retriever.py          ← Qdrant hybrid search + RRF fusion
      reranker.py           ← cross-encoder ms-marco-MiniLM-L-6-v2
      assembler.py          ← parent-child swap, dedup, context packing
    pipeline.py             ← ingest() and retrieve() public entrypoints
docker-compose.yml          ← Qdrant service
```

---

## Phase 1 — Infrastructure *(prerequisite for everything)*

1. Add `docker-compose.yml` with `qdrant/qdrant:latest`, ports 6333/6334, persisted volume
2. Create `app/rag/` skeleton with `__init__.py` files
3. Define `models.py` — `Chunk`, `Document`, `QueryResult` Pydantic models
4. Add dependencies to `pyproject.toml`: `qdrant-client>=1.9`, `docling>=2.0`, `rank-bm25>=0.2`, `sentence-transformers>=3.0`, `tiktoken`

---

## Phase 2 — Ingestion Pipeline *(steps 5–9 can run in parallel after phase 1)*

5. **`parsers.py`** — `docling` parses PDF/DOCX into structured Markdown preserving heading hierarchy, tables, and document-level metadata (title, author, page count)
6. **`chunker.py`** — Split first at structural boundaries (h1/h2/h3), then recursively split oversized sections to ~256-token child chunks with 50-token overlap. Each child stores a `parent_chunk_id`
7. **`embedder.py`** — Batch embed all child chunks via `text-embedding-004` (768-dim, 2048-token context; already in your SDK, no extra key)
8. **`sparse.py`** — Fit BM25 on the corpus at build time; produce sparse term-weight vectors per chunk for Qdrant's `SparseVector` format
9. **`indexer.py`** — Create Qdrant collection with two named vectors (`dense: 768`, `sparse: BM25`). Upsert children with both vectors + full metadata payload. Also store parent chunks (only payload, no vector needed)
10. **`pipeline.py` `ingest(path)`** entrypoint that chains steps 5–9

---

## Phase 3 — Retrieval Pipeline *(steps 11–13 depend on step 10)*

11. **`query_processor.py`** — Call Gemini to rephrase the user query into 3 paraphrases; embed each
12. **`retriever.py`** — For each query variant, run Qdrant `query_points` with two `prefetch` branches (dense k=30, sparse k=30) fused via **RRF** natively. Merge and deduplicate across all query variants by `chunk_id` → top 20 candidates
13. **`reranker.py`** — Run `cross-encoder/ms-marco-MiniLM-L-6-v2` on all (query, chunk_text) pairs → sort → keep top 7
14. **`assembler.py`** — Swap each child chunk for its stored parent (larger context window), prepend metadata header `[Source: file.pdf | Section: Intro > Background]`, deduplicate by `parent_chunk_id`, pack into token budget
15. **`pipeline.py` `retrieve(query)`** entrypoint → returns formatted, cited context string

---

## Phase 4 — ADK Integration

16. Import `pipeline.retrieve` in `app/agent.py`, expose it as `retrieve_context(query: str) -> str` tool
17. Update `instruction` to say: *"For any question about documents, always call `retrieve_context` first and cite the source in your answer."*

---

## Phase 5 — Tests

18. Unit tests: `chunker.py` produces parent/child pairs, `assembler.py` deduplicates correctly
19. Integration test: ingest one small PDF, retrieve one query, assert non-empty result with source metadata

---

## Metadata stored per chunk

| Field | Purpose |
|---|---|
| `doc_id`, `source_file`, `doc_type` | Filtering and citation |
| `page_number`, `section_path` | Human-readable source reference |
| `parent_chunk_id`, `is_parent` | Parent-child swap in assembler |
| `chunk_index`, `token_count` | Context window budgeting |
| `language`, `ingested_at` | Ops / auditing |

---

## Key Decisions

- **Qdrant over pgvector** — purpose-built, native hybrid search, simpler ops for a dedicated RAG workload
- **Qdrant native hybrid (RRF)** — no ElasticSearch or separate BM25 service needed; `qdrant-client` v1.9+ handles it in one API call
- **docling over unstructured** — IBM-backed, best-in-class table extraction and heading preservation from PDFs
- **BM25 over SPLADE** — sufficient for < 1,000 docs, zero model download, in-process
- **Local cross-encoder over Cohere** — free, < 5 ms latency at this scale, `sentence-transformers` model is ~80 MB
- **Multi-query expansion (3 paraphrases) over HyDE** — lower LLM cost, comparable recall at small scale, simpler to debug

---

## Implementation Scope

- **Phase 1**: Infrastructure — 1 file (`docker-compose.yml`) + 2 package init files + 1 models file + update `pyproject.toml`
- **Phase 2**: Ingestion — 5 files (`parsers.py`, `chunker.py`, `embedder.py`, `sparse.py`, `indexer.py`, `pipeline.py`)
- **Phase 3**: Retrieval — 4 files (`query_processor.py`, `retriever.py`, `reranker.py`, `assembler.py`) + expand `pipeline.py`
- **Phase 4**: Integration — 1 edit to `app/agent.py`
- **Phase 5**: Tests — 2 files under `tests/`

---

## Verification

1. `docker compose up qdrant -d` → `curl localhost:6333/healthz` returns OK
2. `python -m app.rag.pipeline ingest ./sample.pdf` → Qdrant collection shows N points
3. `python -m app.rag.pipeline retrieve "what is X"` → returns cited text blocks
4. `make playground` → ask agent an in-document question → response cites source file
