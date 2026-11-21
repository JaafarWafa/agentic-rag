"""Embedding backends.

`HashingEmbedder` is the zero-download default: signed feature hashing of stemmed
unigrams, bigrams and character 4-grams, L2-normalised. It is stateless (no corpus
fit), so documents can be added incrementally, and it runs anywhere - CI included.
Character n-grams give it robustness to typos/morphology that BM25 lacks, which is
why hybrid still beats either retriever alone in the eval.

For real semantic recall set EMBEDDER=sentence-transformers:BAAI/bge-small-en-v1.5
(or plug in Voyage/OpenAI/etc. behind the same two-method interface).
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from app.retrieval.text import content_terms


class Embedder(Protocol):
    dim: int
    name: str
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class HashingEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        terms = content_terms(text)
        feats: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in terms]
        feats += [(f"b:{a}_{b}", 1.5) for a, b in zip(terms, terms[1:])]
        for t in terms:
            p = f"#{t}#"
            feats += [(f"c:{p[i:i + 4]}", 0.35) for i in range(max(1, len(p) - 3))]
        for f, w in feats:
            h = _h(f)
            v[h % self.dim] += w if (h >> 63) & 1 else -w
        v = np.sign(v) * np.log1p(np.abs(v))  # sublinear tf
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts):
        return np.vstack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model

    def embed(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def make_embedder(spec: str, dim: int) -> Embedder:
    if spec == "hashing":
        return HashingEmbedder(dim)
    if spec.startswith("sentence-transformers:"):
        return SentenceTransformerEmbedder(spec.split(":", 1)[1])
    raise ValueError(f"unknown EMBEDDER {spec}")
