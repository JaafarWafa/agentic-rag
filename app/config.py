"""Central configuration. Everything is overridable through environment variables
so the same image runs locally (SQLite, offline LLM) and in the cloud (pgvector, Claude)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Storage: "sqlite:///path.db" or "postgresql://user:pw@host/db" (pgvector)
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'rag.db'}"))

    # Embeddings: "hashing" (offline, zero-download) or "sentence-transformers:<model>"
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "hashing"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))

    # Reranker: "lexical" (offline) or "cross-encoder:<model>"
    reranker: str = field(default_factory=lambda: os.getenv("RERANKER", "lexical"))

    # LLM: "auto" -> Claude when credentials exist, else offline extractive. "claude" | "offline"
    llm: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-5-5"))
    claude_effort: str = field(default_factory=lambda: os.getenv("CLAUDE_EFFORT", "medium"))

    # Chunking
    chunk_target_words: int = field(default_factory=lambda: int(os.getenv("CHUNK_TARGET_WORDS", "140")))
    chunk_max_words: int = field(default_factory=lambda: int(os.getenv("CHUNK_MAX_WORDS", "220")))

    # Retrieval / agent knobs (these are what the eval harness ablates)
    hybrid: bool = field(default_factory=lambda: _bool("HYBRID", True))
    use_rerank: bool = field(default_factory=lambda: _bool("USE_RERANK", True))
    allow_refine: bool = field(default_factory=lambda: _bool("ALLOW_REFINE", True))
    candidate_k: int = field(default_factory=lambda: int(os.getenv("CANDIDATE_K", "20")))
    final_k: int = field(default_factory=lambda: int(os.getenv("FINAL_K", "5")))
    max_search_rounds: int = field(default_factory=lambda: int(os.getenv("MAX_SEARCH_ROUNDS", "3")))
    # Evidence threshold below which the agent refines or refuses (rerank score in [0,1])
    evidence_threshold: float = field(default_factory=lambda: float(os.getenv("EVIDENCE_THRESHOLD", "0.50")))

    seed_corpus: bool = field(default_factory=lambda: _bool("SEED_CORPUS", True))
    corpus_dir: Path = field(default_factory=lambda: Path(os.getenv("CORPUS_DIR", str(ROOT / "corpus" / "raw"))))


settings = Settings()
