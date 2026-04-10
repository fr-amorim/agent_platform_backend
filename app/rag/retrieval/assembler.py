"""Context assembler.

Takes reranked child-chunk candidates, swaps each for its stored parent chunk
(larger context), deduplicates by parent_chunk_id (preserving rank order),
and packs them into a token-budgeted, cited context string.
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

import tiktoken
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_docs"
TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "6000"))

_enc = tiktoken.get_encoding("cl100k_base")
_qdrant: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    return _qdrant


def _to_qdrant_id(chunk_id: str) -> str:
    return str(UUID(hex=chunk_id))


def assemble_context(candidates: list[dict]) -> str:
    """Swap child chunks for parents, deduplicate, pack into a cited context string.

    Args:
        candidates: Reranked list of dicts with keys 'chunk_id', 'score', 'payload'.

    Returns:
        Formatted, cited context string ready to inject into an LLM prompt.
        Blocks are separated by '---'. Empty string when candidates is empty.
    """
    if not candidates:
        logger.warning("assemble_context called with empty candidates list")
        return ""

    logger.info("Assembling context from %d reranked candidates...", len(candidates))
    client = _get_qdrant()

    # Collect parent IDs in ranked order — deduplicate while preserving rank.
    # Carry ALL child payloads per parent for fallback when the parent is too large.
    ordered_parent_ids: list[str] = []
    seen_parents: set[str] = set()
    children_by_parent: dict[str, list[dict]] = {}  # parent_id -> ordered child payloads

    for c in candidates:
        payload = c["payload"]
        if payload is None:
            logger.warning("Candidate has None payload — skipping. chunk_id=%s", c.get("chunk_id"))
            continue

        parent_id = payload.get("parent_chunk_id") or payload.get("chunk_id")
        if not parent_id:
            logger.warning(
                "Candidate payload missing parent_chunk_id and chunk_id. Keys present: %s",
                list(payload.keys()),
            )
            continue

        if parent_id not in seen_parents:
            ordered_parent_ids.append(parent_id)
            seen_parents.add(parent_id)
            children_by_parent[parent_id] = []
        children_by_parent[parent_id].append(payload)

    logger.info("Fetching %d unique parent chunks from Qdrant...", len(ordered_parent_ids))
    if not ordered_parent_ids:
        logger.warning("No parent IDs resolved from candidates — returning empty context.")
        return ""

    try:
        qdrant_ids = [_to_qdrant_id(pid) for pid in ordered_parent_ids]
    except Exception as exc:
        logger.warning("Failed to convert parent IDs to Qdrant UUIDs (%s) — falling back to child text.", exc)
        qdrant_ids = []

    parent_lookup: dict[str, dict] = {}
    if qdrant_ids:
        parent_points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=qdrant_ids,
            with_payload=True,
        )
        logger.info("Qdrant returned %d parent points.", len(parent_points))
        for pt in parent_points:
            key = (pt.payload or {}).get("chunk_id") or str(pt.id)
            parent_lookup[key] = pt.payload or {}

    # Build a flat ranked list of (payload, label) to pack.
    # For each unique parent:
    #   - if parent fits budget → use parent (larger context, single block)
    #   - if parent oversized → use individual child chunks (already ~256 tokens each)
    ranked_payloads: list[tuple[dict, str]] = []
    for parent_id in ordered_parent_ids:
        parent_payload = parent_lookup.get(parent_id)
        children = children_by_parent[parent_id]

        if parent_payload:
            parent_tokens = len(_enc.encode(parent_payload.get("text", "")))
            if parent_tokens <= TOKEN_BUDGET:
                ranked_payloads.append((parent_payload, "parent"))
                continue
            # Parent is too large — use children instead (don't duplicate if same child appears twice)
            logger.info(
                "Parent %s is %d tokens (> budget %d) — using %d child chunk(s) instead.",
                parent_id, parent_tokens, TOKEN_BUDGET, len(children),
            )

        # Fall back to child payloads (deduplicated by chunk_id, preserving rank order)
        seen_child_ids: set[str] = set()
        for child in children:
            cid = child.get("chunk_id", "")
            if cid not in seen_child_ids:
                ranked_payloads.append((child, "child"))
                seen_child_ids.add(cid)

    # Pack into token budget
    blocks: list[str] = []
    tokens_used = 0
    seen_block_ids: set[str] = set()

    for payload, _kind in ranked_payloads:
        bid = payload.get("chunk_id", "")
        if bid in seen_block_ids:
            continue
        seen_block_ids.add(bid)

        header = _format_header(payload)
        text = payload.get("text", "")
        block = f"{header}\n{text}"
        block_tokens = len(_enc.encode(block))

        remaining = TOKEN_BUDGET - tokens_used
        if block_tokens > remaining:
            if not blocks:
                # Always emit at least one block — truncate to fit.
                encoded = _enc.encode(block)
                block = _enc.decode(encoded[:TOKEN_BUDGET])
                block_tokens = TOKEN_BUDGET
                logger.warning(
                    "First block (%d tokens) exceeds budget (%d) — truncated.",
                    len(encoded), TOKEN_BUDGET,
                )
            else:
                logger.info(
                    "Token budget (%d) reached after %d blocks (~%d tokens used).",
                    TOKEN_BUDGET, len(blocks), tokens_used,
                )
                break

        blocks.append(block)
        tokens_used += block_tokens

    logger.info("Context assembled: %d blocks, ~%d tokens.", len(blocks), tokens_used)
    return "\n\n---\n\n".join(blocks)


def _format_header(payload: dict) -> str:
    """Format a citation header from chunk payload metadata."""
    source = payload.get("source_file", "unknown")
    section = payload.get("section_path")
    page = payload.get("page_number")

    parts = [f"Source: {source}"]
    if section:
        parts.append(f"Section: {section}")
    if page is not None:
        parts.append(f"Page: {page}")

    return f"[{' | '.join(parts)}]"
