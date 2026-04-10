"""Gemini Flash reranker.

Takes up to 20 candidate chunks and a query, asks Gemini Flash to rank them
by relevance in a single prompt (one API call, no 512-token cross-encoder limit),
and returns the top TOP_RERANKED results in ranked order.
"""

from __future__ import annotations

import json
import logging
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

RERANK_MODEL = os.getenv("RERANK_MODEL", "gemini-2.5-flash")
TOP_RERANKED = 7

_RERANK_SYSTEM = """\
You are a relevance ranking assistant.
You will be given a search query and a numbered list of text passages.
Return a JSON array of the passage numbers in order from most to least relevant to the query.
Only return the JSON array and nothing else. Example: [3, 1, 7, 2]"""

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


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Rerank candidates with Gemini Flash and return the top TOP_RERANKED.

    Args:
        query: User query string.
        candidates: List of dicts with keys 'chunk_id', 'score', 'payload'.

    Returns:
        Reranked and trimmed list (up to TOP_RERANKED items).
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates

    logger.info("Reranking %d candidates with %s...", len(candidates), RERANK_MODEL)

    passages = "\n\n".join(
        f"[{idx + 1}] {c['payload'].get('text', '')[:600]}"
        for idx, c in enumerate(candidates)
    )
    prompt = f"Query: {query}\n\nPassages:\n{passages}"

    client = _get_client()
    response = client.models.generate_content(
        model=RERANK_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_RERANK_SYSTEM,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    # response.text can be None when the model returns only thinking parts.
    response_text = response.text or ""
    if not response_text:
        for candidate in response.candidates or []:
            for part in (candidate.content.parts or []):
                if getattr(part, "text", None):
                    response_text = part.text
                    break
            if response_text:
                break

    if not response_text:
        logger.warning("Reranker returned empty response — using original order.")
        return candidates[:TOP_RERANKED]

    ranked_indices = _parse_ranking(response_text.strip(), len(candidates))
    logger.info("Reranker order (0-based): %s", ranked_indices[:TOP_RERANKED])

    reranked = [candidates[i] for i in ranked_indices]
    return reranked[:TOP_RERANKED]


def _parse_ranking(text: str, n_candidates: int) -> list[int]:
    """Parse a JSON array of 1-based indices from reranker response.

    Returns 0-based indices. Any missing indices are appended at the end
    so no candidate is silently dropped.
    """
    try:
        match = re.search(r"\[[\s\d,]+\]", text)
        if match:
            one_based: list[int] = json.loads(match.group(0))
            seen: set[int] = set()
            result: list[int] = []
            for idx in one_based:
                zero_based = int(idx) - 1
                if 0 <= zero_based < n_candidates and zero_based not in seen:
                    result.append(zero_based)
                    seen.add(zero_based)
            # Append any candidate not returned by the reranker (safety net)
            for i in range(n_candidates):
                if i not in seen:
                    result.append(i)
            return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse reranker response (%s) — using original order", exc)

    return list(range(n_candidates))
