"""Storage interface shared by the SQLite (zero-infra) and Postgres/pgvector backends."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    id: str
    doc_id: str
    ord: int
    section: str
    text: str
    doc_title: str = ""
    source: str = ""
    page: int | None = None
    score: float = 0.0
    scores: dict = field(default_factory=dict)  # per-stage scores, for tracing

    def to_public(self) -> dict:
        return {
            "chunk_id": self.id, "doc_id": self.doc_id, "doc_title": self.doc_title,
            "section": self.section, "page": self.page, "source": self.source,
            "text": self.text, "score": round(self.score, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


_TOKEN = re.compile(r"[a-z0-9]+")


def keyword_terms(query: str) -> list[str]:
    """Tokens safe to hand to FTS5 / to_tsquery (no operators, no injection)."""
    from app.retrieval.text import STOPWORDS
    toks = [t for t in _TOKEN.findall(query.lower()) if t not in STOPWORDS and len(t) > 1]
    return list(dict.fromkeys(toks))[:32]


class Store(ABC):
    dim: int

    # --- documents -----------------------------------------------------------
    @abstractmethod
    def add_document(self, doc: dict, chunks: list[dict], embeddings: np.ndarray) -> None: ...
    @abstractmethod
    def find_document_by_hash(self, sha256: str) -> dict | None: ...
    @abstractmethod
    def list_documents(self, roles: list[str]) -> list[dict]: ...
    @abstractmethod
    def get_document(self, doc_id: str, roles: list[str]) -> dict | None: ...
    @abstractmethod
    def get_chunks(self, doc_id: str, roles: list[str]) -> list[Chunk]: ...
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool: ...
    @abstractmethod
    def count_chunks(self) -> int: ...

    # --- retrieval (roles are ALWAYS applied inside the query) ---------------
    @abstractmethod
    def dense_search(self, vec: np.ndarray, roles: list[str], k: int) -> list[Chunk]: ...
    @abstractmethod
    def keyword_search(self, query: str, roles: list[str], k: int) -> list[Chunk]: ...

    # --- observability -------------------------------------------------------
    @abstractmethod
    def save_trace(self, trace: dict) -> None: ...
    @abstractmethod
    def list_traces(self, limit: int = 50) -> list[dict]: ...
    @abstractmethod
    def get_trace(self, trace_id: str) -> dict | None: ...
    @abstractmethod
    def save_eval_run(self, run: dict) -> None: ...
    @abstractmethod
    def list_eval_runs(self) -> list[dict]: ...


def make_store(url: str, dim: int) -> Store:
    if url.startswith("sqlite"):
        from app.db.sqlite_store import SQLiteStore
        return SQLiteStore(url.split("sqlite:///", 1)[1], dim)
    if url.startswith(("postgres://", "postgresql://")):
        from app.db.pg_store import PgStore
        return PgStore(url, dim)
    raise ValueError(f"unsupported DATABASE_URL: {url}")
