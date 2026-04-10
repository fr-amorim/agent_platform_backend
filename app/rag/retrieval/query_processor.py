"""HyDE query expansion.

Generates a hypothetical document-style answer to the query and embeds both
the original query and the hypothetical answer with task_type=RETRIEVAL_QUERY.
Two embeddings per query, one LLM call — improves recall by shifting the
query vector towards document space.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.rag.ingestion.embedder import embed_query

load_dotenv()

logger = logging.getLogger(__name__)

HYDE_MODEL = os.getenv("HYDE_MODEL", "gemini-2.5-flash")

_HYDE_SYSTEM = (
    "Write a concise, factual passage (2–4 sentences) that directly answers the question below. "
    "Write as if it came from an authoritative document. "
    "Do not say 'the document says' or refer to yourself."
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Configure a Gemini Developer API key in your environment."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def expand_query(query: str) -> tuple[list[float], list[float]]:
    """Return (original_embedding, hyde_embedding) for a query using HyDE.

    Generates a hypothetical document answer and embeds it alongside the
    original query so retrieval benefits from document-space alignment.

    Returns:
        (original_vec, hyde_vec) — both shaped [EMBEDDING_DIM]
    """
    client = _get_client()

    logger.info("HyDE: generating hypothetical answer for query: %s", query[:100])
    response = client.models.generate_content(
        model=HYDE_MODEL,
        contents=f"Question: {query}",
        config=types.GenerateContentConfig(
            system_instruction=_HYDE_SYSTEM,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    hypothetical_answer = (response.text or "").strip()
    if not hypothetical_answer:
        logger.warning("HyDE returned empty answer — falling back to original query embedding only.")
        original_vec = embed_query(query)
        return original_vec, original_vec

    logger.debug("HyDE answer (%d chars): %s", len(hypothetical_answer), hypothetical_answer[:200])

    logger.info("Embedding original query + hypothetical answer...")
    original_vec = embed_query(query)
    hyde_vec = embed_query(hypothetical_answer)
    logger.info("Query expansion complete.")

    return original_vec, hyde_vec
