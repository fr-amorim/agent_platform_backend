"""Hybrid retrieval from Qdrant.

For each of two query vectors (original + HyDE), runs a hybrid search with:
  - dense prefetch (cosine, k=30)
  - sparse BM25 prefetch (k=30)
  - RRF fusion (native Qdrant)

Results from both vectors are merged and deduplicated by chunk_id, returning
up to TOP_CANDIDATES candidates sorted by best score.
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_docs"
BM25_MODEL = "Qdrant/bm25"
PREFETCH_K = 30
TOP_CANDIDATES = 20

_qdrant: QdrantClient | None = None
_bm25: SparseTextEmbedding | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    return _qdrant


def _get_bm25() -> SparseTextEmbedding:
    global _bm25
    if _bm25 is None:
        logger.info("Initializing BM25 model for query encoding: %s", BM25_MODEL)
        _bm25 = SparseTextEmbedding(model_name=BM25_MODEL)
    return _bm25


def hybrid_search(
    query_text: str,
    original_vec: list[float],
    hyde_vec: list[float],
) -> list[dict]:
    """Run hybrid search using both original and HyDE query vectors.

    For each vector: dense + sparse prefetch with RRF fusion.
    Results from both are merged and deduplicated by chunk_id (highest score wins).

    Returns:
        List of dicts with keys 'chunk_id', 'score', 'payload',
        sorted by score descending, length <= TOP_CANDIDATES.
    """
    client = _get_qdrant()
    bm25 = _get_bm25()

    # Encode query text once for the sparse branch (shared across both vector runs)
    sparse_emb = next(bm25.embed([query_text]))
    sparse_query = SparseVector(
        indices=list(sparse_emb.indices),
        values=list(sparse_emb.values),
    )

    results: dict[str, dict] = {}

    for label, dense_vec in [("original", original_vec), ("hyde", hyde_vec)]:
        logger.debug("Running hybrid search (%s)...", label)
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=PREFETCH_K),
                Prefetch(query=sparse_query, using="sparse", limit=PREFETCH_K),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=TOP_CANDIDATES,
            with_payload=True,
        ).points

        logger.debug("%s query: %d hits", label, len(hits))
        for hit in hits:
            cid = hit.payload.get("chunk_id", str(hit.id))
            if cid not in results or hit.score > results[cid]["score"]:
                results[cid] = {
                    "chunk_id": cid,
                    "score": hit.score,
                    "payload": hit.payload,
                }

    sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    logger.info(
        "Hybrid search complete: %d unique candidates (original + HyDE)", len(sorted_results)
    )
    return sorted_results[:TOP_CANDIDATES]
