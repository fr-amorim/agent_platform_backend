from __future__ import annotations

import logging
import os
import random
import re
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 768
_MAX_BATCH_SIZE = 100
_BATCH_SIZE = min(int(os.getenv("EMBED_BATCH_SIZE", "100")), _MAX_BATCH_SIZE)
_MAX_RETRIES = 6
_BASE_BACKOFF_SECONDS = 25.0

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


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed texts for indexing using task_type=RETRIEVAL_DOCUMENT."""
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a single query text using task_type=RETRIEVAL_QUERY."""
    return _embed([text], "RETRIEVAL_QUERY")[0]


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    logger.info(f"Embedding {len(texts)} texts (task_type={task_type})")
    client = _get_client()
    results: list[list[float]] = []
    
    num_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
    logger.debug(f"Processing in {num_batches} batches of {_BATCH_SIZE}")

    for batch_idx, i in enumerate(range(0, len(texts), _BATCH_SIZE)):
        batch = texts[i : i + _BATCH_SIZE]
        logger.debug(f"Batch {batch_idx + 1}/{num_batches}: embedding {len(batch)} texts")

        response = _embed_with_retry(client, batch, task_type)
        results.extend(e.values for e in response.embeddings)
        logger.debug(f"Batch {batch_idx + 1}/{num_batches} complete")

    logger.info(f"Embedding complete: {len(results)} vectors")
    return results


def _embed_with_retry(
    client: genai.Client, batch: list[str], task_type: str
) -> types.EmbedContentResponse:
    for attempt in range(_MAX_RETRIES):
        try:
            return _embed_batch(client, batch, task_type)
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            is_rate_or_transient = (
                "429" in message
                or "quota" in message
                or "rate" in message
                or "resource_exhausted" in message
                or "503" in message
                or "unavailable" in message
            )

            if not is_rate_or_transient or attempt == _MAX_RETRIES - 1:
                raise

            # Honor server-provided retry delay when available (e.g. "retryDelay: '25s'").
            hinted_delay = _extract_retry_delay_seconds(message)
            if hinted_delay is not None:
                sleep_seconds = hinted_delay + random.uniform(0, 0.5)
            else:
                sleep_seconds = _BASE_BACKOFF_SECONDS * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Embedding request throttled/transient failure (attempt %s/%s). Retrying in %.1fs.",
                attempt + 1,
                _MAX_RETRIES,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable retry state in _embed_with_retry")


def _extract_retry_delay_seconds(message: str) -> float | None:
    match = re.search(r"retrydelay[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*s", message)
    if match:
        return float(match.group(1))
    return None


def _embed_batch(
    client: genai.Client, batch: list[str], task_type: str
) -> types.EmbedContentResponse:
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIM,
    )

    # Prefer batch endpoint when available to reduce request count pressure.
    batch_method = getattr(client.models, "batch_embed_contents", None)
    if callable(batch_method):
        return batch_method(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=config,
        )

    return client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=batch,
        config=config,
    )
