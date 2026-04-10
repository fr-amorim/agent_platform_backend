"""Public entrypoints for the RAG pipeline.

Usage (CLI):
    python -m app.rag.pipeline ingest ./path/to/doc.pdf
    python -m app.rag.pipeline retrieve "your query here"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Configure logging BEFORE importing heavy dependencies so first log lines are visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

logger.info("Importing RAG modules...")
from app.rag.ingestion.chunker import chunk_document  # noqa: E402
from app.rag.ingestion.embedder import embed_documents  # noqa: E402
from app.rag.ingestion.indexer import ensure_collection, upsert_children, upsert_parents  # noqa: E402
from app.rag.ingestion.parsers import parse_document  # noqa: E402
logger.info("RAG modules imported.")


def ingest(path: str | Path) -> int:
    """Parse, chunk, embed, and index a document.

    Returns:
        Number of child chunks indexed.
    """
    logger.info("=" * 60)
    logger.info(f"Starting ingestion pipeline for: {path}")
    logger.info("=" * 60)
    
    logger.info("Step 1/5: Parsing document...")
    document, markdown = parse_document(path)
    
    logger.info("Step 2/5: Chunking document...")
    chunks = chunk_document(document, markdown)

    parent_chunks = [c for c in chunks if c.is_parent]
    child_chunks = [c for c in chunks if not c.is_parent]
    logger.info(f"Step 3/5: Created {len(parent_chunks)} parents and {len(child_chunks)} children")

    logger.info("Step 4/5: Embedding child chunks...")
    dense_vectors = embed_documents([c.text for c in child_chunks])

    logger.info("Step 5/5: Upserting to Qdrant...")
    ensure_collection()
    upsert_children(child_chunks, dense_vectors)
    upsert_parents(parent_chunks)

    logger.info("=" * 60)
    logger.info(f"Ingestion complete: {len(child_chunks)} child chunks indexed")
    logger.info("=" * 60)
    return len(child_chunks)


def retrieve(query: str) -> str:
    """Retrieve relevant context for a query.

    Pipeline:
        1. HyDE query expansion — embed original + hypothetical answer
        2. Hybrid search — dense + sparse (BM25) prefetch + RRF fusion over both vectors
        3. Gemini Flash reranking — top 20 → top 7
        4. Context assembly — parent-child swap, dedup, token-budget packing

    Returns:
        Formatted, cited context string (blocks separated by '---').
    """
    from app.rag.retrieval.assembler import assemble_context
    from app.rag.retrieval.query_processor import expand_query
    from app.rag.retrieval.reranker import rerank
    from app.rag.retrieval.retriever import hybrid_search

    logger.info("=" * 60)
    logger.info("Starting retrieval for: %s", query[:100])
    logger.info("=" * 60)

    logger.info("Step 1/4: Query expansion (HyDE)...")
    original_vec, hyde_vec = expand_query(query)

    logger.info("Step 2/4: Hybrid search (dense + sparse, original + HyDE)...")
    candidates = hybrid_search(query, original_vec, hyde_vec)
    logger.info("Step 2/4 complete: %d candidates", len(candidates))

    logger.info("Step 3/4: Reranking with Gemini Flash...")
    ranked = rerank(query, candidates)
    logger.info("Step 3/4 complete: %d results after reranking", len(ranked))

    logger.info("Step 4/4: Assembling context (parent swap + token budget)...")
    context = assemble_context(ranked)

    logger.info("=" * 60)
    logger.info("Retrieval complete.")
    logger.info("=" * 60)
    return context


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m app.rag.pipeline ingest <path>")
        print("  python -m app.rag.pipeline retrieve <query>")
        sys.exit(1)

    command, arg = sys.argv[1], sys.argv[2]

    if command == "ingest":
        try:
            count = ingest(arg)
            print(f"\n✓ Indexed {count} child chunks from '{arg}'.")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            sys.exit(1)
    elif command == "retrieve":
        try:
            result = retrieve(arg)
            print(result if result else "(no results found)")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
