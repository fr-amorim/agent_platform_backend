from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: uuid4().hex)
    source_file: str
    doc_type: str  # "pdf" | "docx" | "md" | ...
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: Optional[int] = None
    language: str = "en"
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    doc_id: str
    source_file: str
    doc_type: str
    is_parent: bool = False
    parent_chunk_id: Optional[str] = None  # None when is_parent=True
    chunk_index: int
    text: str
    token_count: int
    page_number: Optional[int] = None
    section_path: Optional[str] = None  # e.g. "Intro > Background"
    language: str = "en"
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class QueryResult(BaseModel):
    chunk_id: str
    parent_chunk_id: Optional[str]
    source_file: str
    section_path: Optional[str]
    page_number: Optional[int]
    text: str
    score: float
