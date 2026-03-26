# Plan: State-of-the-Art RAG System

**TL;DR** — Build a tool-based RAG pipeline wired into the existing ADK agent. Qdrant (Docker) stores dense vectors and handles BM25 sparse indexing natively server-side. Ingestion uses docling for parsing, hierarchical chunking, and task-type-aware dense embedding. Retrieval uses HyDE query expansion → Qdrant native hybrid search (RRF) → Gemini 1.5 Flash reranking → parent-child context swap.

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
      embedder.py           ← Google text-embedding-004 (batch, task_type=RETRIEVAL_DOCUMENT)
      indexer.py            ← Qdrant collection setup + upsert (qdrant/bm25 for sparse)
    retrieval/
      query_processor.py    ← HyDE expansion via Gemini
      retriever.py          ← Qdrant hybrid search + RRF fusion (task_type=RETRIEVAL_QUERY)
      reranker.py           ← Gemini 1.5 Flash reranker
      assembler.py          ← parent-child swap, dedup, context packing
    pipeline.py             ← ingest() and retrieve() public entrypoints
docker-compose.yml          ← Qdrant service
```

---

## Phase 1 — Infrastructure *(prerequisite for everything)*

1. Add `docker-compose.yml` with `qdrant/qdrant:latest`, ports 6333/6334, persisted volume
2. Create `app/rag/` skeleton with `__init__.py` files
3. Define `models.py` — `Chunk`, `Document`, `QueryResult` Pydantic models
4. Add dependencies to `pyproject.toml`: `qdrant-client>=1.9`, `docling>=2.0`, `tiktoken`
   - `rank-bm25` and `sentence-transformers` are **not needed** — Qdrant handles BM25 server-side and reranking moves to Gemini 1.5 Flash

---

## Phase 2 — Ingestion Pipeline *(steps 5–8 can run in parallel after phase 1)*

5. **`parsers.py`** — `docling` parses PDF/DOCX into structured Markdown preserving heading hierarchy, tables, and document-level metadata (title, author, page count)
6. **`chunker.py`** — Split first at structural boundaries (h1/h2/h3), then recursively split oversized sections to ~256-token child chunks with 50-token overlap. Each child stores a `parent_chunk_id`
7. **`embedder.py`** — Batch embed all child chunks via `text-embedding-004` (768-dim, 2048-token context) with **`task_type="RETRIEVAL_DOCUMENT"`**. No sparse pre-processing needed.
8. **`indexer.py`** — Create Qdrant collection with two named vectors: `dense` (768-dim) and `sparse` configured to use the **`qdrant/bm25` built-in model** (server-side tokenisation, no local BM25 fitting, always in sync with the dense index). Upsert children with dense vectors + full metadata payload; Qdrant derives sparse vectors automatically at upsert. Also store parent chunks (payload only).
9. **`pipeline.py` `ingest(path)`** entrypoint that chains steps 5–8

---

## Phase 3 — Retrieval Pipeline *(steps 10–12 depend on step 9)*

10. **`query_processor.py`** — **HyDE**: call Gemini to generate one *hypothetical document answer* to the user query, then embed both the hypothetical answer and the original query with **`task_type="RETRIEVAL_QUERY"`**. Two embeddings total — same latency budget as one paraphrase round-trip, better recall than rephrasing.
11. **`retriever.py`** — For each of the two query vectors (original + HyDE), run Qdrant `query_points` with two `prefetch` branches: `dense k=30` (using `text-embedding-004`) and `sparse k=30` (using `qdrant/bm25` — Qdrant tokenises the raw query text server-side, no local BM25 model needed). RRF fusion natively. Merge and deduplicate across both variants by `chunk_id` → top 20 candidates.
12. **`reranker.py`** — **Gemini 1.5 Flash as reranker**: pass top-20 `(query, chunk_text)` pairs in a single prompt asking Flash to return a ranked list of chunk IDs by relevance. No local model, no 512-token limit, reasoning quality exceeds small cross-encoders. Keep top 7.
13. **`assembler.py`** — Swap each child chunk for its stored parent (larger context window), prepend metadata header `[Source: file.pdf | Section: Intro > Background]`, deduplicate by `parent_chunk_id`, pack into token budget
14. **`pipeline.py` `retrieve(query)`** entrypoint → returns formatted, cited context string

---

## Phase 4 — ADK Integration

15. Import `pipeline.retrieve` in `app/agent.py`, expose it as `retrieve_context(query: str) -> str` tool
16. Update `instruction` to say: *"For any question about documents, always call `retrieve_context` first and cite the source in your answer."*

---

## Phase 5 — Tests

17. Unit tests: `chunker.py` produces parent/child pairs, `assembler.py` deduplicates correctly
18. Integration test: ingest one small PDF, retrieve one query, assert non-empty result with source metadata

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
- **Qdrant native BM25 over `rank-bm25`** — server-side sparse indexing keeps dense/sparse indices always in sync; no re-fit required when new documents are added; eliminates `sparse.py` entirely
- **Task-type embeddings** — `RETRIEVAL_DOCUMENT` at index time, `RETRIEVAL_QUERY` at query time; separate embedding spaces in `text-embedding-004` meaningfully improve recall
- **HyDE over 3-paraphrase multi-query** — one LLM call generates a hypothetical answer that shifts the query embedding toward document space; lower latency and better recall than generating and embedding 3 paraphrases
- **Gemini 1.5 Flash reranker over local MiniLM cross-encoder** — no local CPU/GPU overhead, no 512-token limit, reasoning quality exceeds small cross-encoders; reuses the same SDK already in the project

---

## Implementation Scope

- **Phase 1**: Infrastructure — 1 file (`docker-compose.yml`) + 2 package init files + 1 models file + update `pyproject.toml` (2 fewer deps)
- **Phase 2**: Ingestion — 4 files (`parsers.py`, `chunker.py`, `embedder.py`, `indexer.py`, `pipeline.py`) — `sparse.py` eliminated
- **Phase 3**: Retrieval — 4 files (`query_processor.py`, `retriever.py`, `reranker.py`, `assembler.py`) + expand `pipeline.py`
- **Phase 4**: Integration — 1 edit to `app/agent.py`
- **Phase 5**: Tests — 2 files under `tests/`

---

## Verification

1. `docker compose up qdrant -d` → `curl localhost:6333/healthz` returns OK
2. `python -m app.rag.pipeline ingest ./sample.pdf` → Qdrant collection shows N points with both dense and server-side BM25 sparse vectors
3. `python -m app.rag.pipeline retrieve "what is X"` → returns cited text blocks; confirm HyDE expansion in debug log
4. `make playground` → ask agent an in-document question → response cites source file

---

## Further Considerations

- **Context Caching** — If usage patterns show the same large document being queried repeatedly in one session, Gemini Context Caching can cache its tokens server-side, cutting latency and cost on every follow-up call. Straightforward to add to `assembler.py` or `pipeline.py` as a per-session document cache key once the core pipeline is stable.
