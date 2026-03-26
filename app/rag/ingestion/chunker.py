from __future__ import annotations

import logging
import re

import tiktoken

from app.rag.models import Chunk, Document

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")

TARGET_CHILD_TOKENS = 256
OVERLAP_TOKENS = 50

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_by_headings(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (section_path, text) pairs at h1/h2/h3 boundaries."""
    matches = list(_HEADING_RE.finditer(markdown))

    if not matches:
        return [("Document", markdown.strip())]

    sections: list[tuple[str, str]] = []
    breadcrumb: list[str] = []

    # Text before the first heading
    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        sections.append(("Preamble", preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        breadcrumb = breadcrumb[: level - 1] + [title]
        section_path = " > ".join(breadcrumb)

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        # Include heading line itself in the section text
        text = (m.group(0) + "\n" + markdown[start:end]).strip()

        if text:
            sections.append((section_path, text))

    return sections


def _split_into_children(text: str, target: int, overlap: int) -> list[str]:
    """Split text into token-bounded chunks with overlap."""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= target:
        return [text]

    children: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + target, len(tokens))
        children.append(_ENCODER.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap

    return children


def chunk_document(document: Document, markdown: str) -> list[Chunk]:
    """Produce parent + child Chunk objects from a document's markdown.

    Each section becomes one parent chunk. Sections exceeding TARGET_CHILD_TOKENS
    are also split into overlapping child chunks. Sections that fit within the
    target still get a single child so every parent has at least one child.
    """
    logger.info(f"Starting chunking for {document.source_file}")
    sections = _split_by_headings(markdown)
    logger.info(f"Found {len(sections)} sections")
    
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for i, (section_path, section_text) in enumerate(sections):
        logger.debug(f"Processing section {i+1}/{len(sections)}: {section_path}")
        
        parent = Chunk(
            doc_id=document.doc_id,
            source_file=document.source_file,
            doc_type=document.doc_type,
            is_parent=True,
            parent_chunk_id=None,
            chunk_index=chunk_index,
            text=section_text,
            token_count=_count_tokens(section_text),
            section_path=section_path,
            language=document.language,
            ingested_at=document.ingested_at,
        )
        all_chunks.append(parent)
        chunk_index += 1

        child_texts = _split_into_children(section_text, TARGET_CHILD_TOKENS, OVERLAP_TOKENS)
        logger.debug(f"Section split into {len(child_texts)} child chunks")
        
        for j, child_text in enumerate(child_texts):
            child = Chunk(
                doc_id=document.doc_id,
                source_file=document.source_file,
                doc_type=document.doc_type,
                is_parent=False,
                parent_chunk_id=parent.chunk_id,
                chunk_index=chunk_index,
                text=child_text,
                token_count=_count_tokens(child_text),
                section_path=section_path,
                language=document.language,
                ingested_at=document.ingested_at,
            )
            all_chunks.append(child)
            chunk_index += 1

    logger.info(f"Chunking complete: {len(all_chunks)} total chunks (parents + children)")
    return all_chunks
