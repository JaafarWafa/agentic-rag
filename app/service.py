"""Wires the components together. Used by the API, the CLI and the eval harness so they
all exercise exactly the same code path."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

from app.agent.agent import Agent, make_policy
from app.config import Settings
from app.db.base import make_store
from app.ingest.pipeline import ingest_bytes, ingest_directory
from app.retrieval.embeddings import make_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import make_reranker

log = logging.getLogger("rag.service")


class RAGService:
    def __init__(self, settings: Settings, llm_client=None):
        self.settings = settings
        self.embedder = make_embedder(settings.embedder, settings.embedding_dim)
        self.store = make_store(settings.database_url, self.embedder.dim)
        self.reranker = make_reranker(settings.reranker)
        self.retriever = HybridRetriever(self.store, self.embedder, self.reranker, hybrid=settings.hybrid,
                                         use_rerank=settings.use_rerank, candidate_k=settings.candidate_k,
                                         final_k=settings.final_k)
        self.policy = make_policy(settings, llm_client)
        self.agent = Agent(self.retriever, self.policy)

    def describe(self) -> dict:
        s = self.settings
        return {"llm": self.policy.name, "embedder": self.embedder.name, "reranker": self.reranker.name,
                "store": type(self.store).__name__, "hybrid": s.hybrid, "use_rerank": s.use_rerank,
                "allow_refine": s.allow_refine, "evidence_threshold": s.evidence_threshold,
                "candidate_k": s.candidate_k, "final_k": s.final_k, "max_search_rounds": s.max_search_rounds,
                "chunk_target_words": s.chunk_target_words}

    def seed(self) -> list[dict]:
        return ingest_directory(self.store, self.embedder, self.settings.corpus_dir,
                                target_words=self.settings.chunk_target_words,
                                max_words=self.settings.chunk_max_words)

    def ingest(self, filename: str, data: bytes, roles: list[str], title: str | None = None) -> dict:
        return ingest_bytes(self.store, self.embedder, filename, data, roles, title,
                            target_words=self.settings.chunk_target_words, max_words=self.settings.chunk_max_words)

    def ask(self, question: str, user: str, roles: list[str], *, save_trace: bool = True) -> dict:
        t0 = time.perf_counter()
        res = self.agent.ask(question, roles)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        trace = {
            "id": uuid.uuid4().hex[:16], "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "user": user, "roles": roles, "question": question, "answer": res.answer, "refused": res.refused,
            "provider": res.provider, "latency_ms": latency, "input_tokens": res.usage.get("input_tokens", 0),
            "output_tokens": res.usage.get("output_tokens", 0),
            "citations": [{"marker": c["marker"], "chunk_id": c["chunk_id"], "doc_title": c["doc_title"]}
                          for c in res.citations],
            "steps": res.steps,
        }
        if save_trace:
            try:
                self.store.save_trace(trace)
            except Exception:  # tracing must never break answering
                log.exception("failed to save trace")
        log.info('ask user=%s roles=%s route=%s refused=%s cites=%d latency_ms=%.0f trace=%s',
                 user, roles, res.route, res.refused, len(res.citations), latency, trace["id"])
        out = asdict(res)
        out.update(trace_id=trace["id"], latency_ms=latency)
        return out
