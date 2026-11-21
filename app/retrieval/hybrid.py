"""Hybrid retrieval: dense + keyword -> Reciprocal Rank Fusion -> re-rank."""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.db.base import Chunk, Store
from app.retrieval.embeddings import Embedder
from app.retrieval.rerank import LexicalReranker, Reranker

RRF_K = 60


@dataclass
class RetrievalResult:
    query: str
    chunks: list[Chunk]
    best_score: float
    timings_ms: dict

    def summary(self) -> dict:
        return {
            "query": self.query,
            "best_score": round(self.best_score, 4),
            "timings_ms": self.timings_ms,
            "results": [{"chunk_id": c.id, "doc": c.doc_title, "section": c.section,
                         "score": round(c.score, 4), "scores": {k: round(v, 4) for k, v in c.scores.items()}}
                        for c in self.chunks],
        }


class HybridRetriever:
    def __init__(self, store: Store, embedder: Embedder, reranker: Reranker, *,
                 hybrid: bool = True, use_rerank: bool = True, candidate_k: int = 20, final_k: int = 5):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.hybrid, self.use_rerank = hybrid, use_rerank
        self.candidate_k, self.final_k = candidate_k, final_k
        # Evidence is always measured with a calibrated scorer, even when re-ranking
        # is disabled for ordering, so ablations change *one* thing at a time.
        self._evidence = reranker if use_rerank else LexicalReranker()

    def search(self, query: str, roles: list[str], k: int | None = None,
               grade_against: str | None = None) -> RetrievalResult:
        """`grade_against`: score evidence against this text (the user's original question)
        instead of the possibly-rewritten search query, so query refinement can widen
        recall without being able to drift onto an easier question."""
        k = k or self.final_k
        t0 = time.perf_counter()
        qvec = self.embedder.embed([query])[0]
        dense = self.store.dense_search(qvec, roles, self.candidate_k)
        t1 = time.perf_counter()
        sparse = self.store.keyword_search(query, roles, self.candidate_k) if self.hybrid else []
        t2 = time.perf_counter()

        fused: dict[str, Chunk] = {}
        for name, hits in (("dense", dense), ("bm25", sparse)):
            for rank, c in enumerate(hits):
                cur = fused.setdefault(c.id, c)
                cur.scores[name] = c.score
                cur.scores[f"{name}_rank"] = rank + 1
                cur.scores["rrf"] = cur.scores.get("rrf", 0.0) + 1.0 / (RRF_K + rank + 1)
        cands = sorted(fused.values(), key=lambda c: -c.scores["rrf"])[: self.candidate_k]

        ev = self._evidence.score(grade_against or query, cands)
        for c, s in zip(cands, ev):
            c.scores["rerank"] = s
        if self.use_rerank:
            # Re-ranker decides the order; RRF breaks ties.
            cands.sort(key=lambda c: (-c.scores["rerank"], -c.scores["rrf"]))
            for c in cands:
                c.score = c.scores["rerank"]
        else:
            for c in cands:
                c.score = c.scores["rrf"]
        t3 = time.perf_counter()
        top = cands[:k]
        best = max((c.scores["rerank"] for c in top), default=0.0)
        return RetrievalResult(query, top, best, {
            "dense": round((t1 - t0) * 1000, 2), "keyword": round((t2 - t1) * 1000, 2),
            "fuse_rerank": round((t3 - t2) * 1000, 2)})
