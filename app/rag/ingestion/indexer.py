from __future__ import annotations

import logging
import os
from uuid import UUID

from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.rag.models import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_docs"
DENSE_DIM = 768
BM25_MODEL = "Qdrant/bm25"

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
        logger.info(f"Initializing sparse embedder: {BM25_MODEL}")
        _bm25 = SparseTextEmbedding(model_name=BM25_MODEL)
    return _bm25


def _to_qdrant_id(chunk_id: str) -> str:
    """Convert hex UUID string to hyphenated UUID format expected by Qdrant."""
    return str(UUID(hex=chunk_id))


def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    client = _get_qdrant()
    logger.info(f"Checking Qdrant collection: {COLLECTION_NAME}")
    
    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )
        logger.info(f"Collection created: {COLLECTION_NAME}")
    else:
        logger.info(f"Collection already exists: {COLLECTION_NAME}")
        info = client.get_collection(COLLECTION_NAME)
        existing_dim = info.config.params.vectors["dense"].size
        if existing_dim != DENSE_DIM:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' has dense dim={existing_dim}, expected {DENSE_DIM}. "
                "Delete and recreate it: curl -X DELETE http://localhost:6333/collections/rag_docs"
            )

        sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
        if not sparse_cfg or "sparse" not in sparse_cfg:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' is missing sparse vector config. "
                "Delete and recreate it: curl -X DELETE http://localhost:6333/collections/rag_docs"
            )


def upsert_children(child_chunks: list[Chunk], dense_vectors: list[list[float]]) -> None:
    """Upsert child chunks with dense + sparse vectors and metadata payload."""
    if not child_chunks:
        logger.warning("upsert_children called with empty list")
        return

    logger.info(f"Upserting {len(child_chunks)} child chunks to {COLLECTION_NAME}")
    client = _get_qdrant()
    bm25 = _get_bm25()

    texts = [chunk.text for chunk in child_chunks]
    sparse_embeddings = list(bm25.embed(texts))

    points = [
        PointStruct(
            id=_to_qdrant_id(chunk.chunk_id),
            vector={
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=list(sparse_emb.indices),
                    values=list(sparse_emb.values),
                ),
            },
            payload=chunk.model_dump(mode="json"),
        )
        for chunk, dense_vec, sparse_emb in zip(
            child_chunks, dense_vectors, sparse_embeddings
        )
    ]
    logger.debug(f"Constructed {len(points)} point objects")

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Successfully upserted {len(child_chunks)} child chunks")


def upsert_parents(parent_chunks: list[Chunk]) -> None:
    """Upsert parent chunks as payload-only points (no vectors; retrieved by ID only)."""
    if not parent_chunks:
        logger.warning("upsert_parents called with empty list")
        return

    logger.info(f"Upserting {len(parent_chunks)} parent chunks to {COLLECTION_NAME}")
    client = _get_qdrant()
    points = [
        PointStruct(
            id=_to_qdrant_id(p.chunk_id),
            vector={},
            payload=p.model_dump(mode="json"),
        )
        for p in parent_chunks
    ]
    logger.debug(f"Constructed {len(points)} parent point objects")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Successfully upserted {len(parent_chunks)} parent chunks")
