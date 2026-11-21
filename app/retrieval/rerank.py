"""Re-rankers. Both return scores in [0, 1] so the agent's evidence threshold means the
same thing regardless of backend."""
from __future__ import annotations

import math
from typing import Protocol

from app.db.base import Chunk
from app.retrieval.text import SOFT_TERMS, content_terms, satisfied, sentences, term_alternatives


class Reranker(Protocol):
    name: str
    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


SOFT_WEIGHT = 0.3


class LexicalReranker:
    """Answerability-oriented lexical re-ranker.

    score = 0.40 * chunk coverage      (IDF-weighted share of question terms in the chunk)
          + 0.40 * best-window coverage (same, but inside the best 2-sentence window
                                          + section title: the terms must co-occur)
          + 0.10 * phrase match (question bigrams found in order)
          + 0.10 * section-title match

    * A question term is satisfied by the term itself or one of its thesaurus synonyms.
    * Question-intent words ("how long", "minimum", "deadline") and numbers get a small
      fixed weight instead of IDF.
    * IDF comes from the candidate pool with smoothing, so a term that appears in no
      candidate keeps a high weight and pulls the score down. That is what lets
      the agent tell that the corpus doesn't cover a question.
    * Window coverage is what stops "parking" (sentence 4) + "Tangier" (sentence 1)
      from looking like an answer to "parking policy at the Tangier hub".
    """
    name = "lexical"

    def score(self, query, chunks):
        q = list(dict.fromkeys(content_terms(query)))
        if not q or not chunks:
            return [0.0] * len(chunks)
        alts = {t: term_alternatives(t) for t in q}
        bodies = [set(content_terms(c.section + " " + c.text)) for c in chunks]
        n = len(chunks)
        w = {}
        for t in q:
            df = sum(satisfied(alts[t], b) for b in bodies)
            soft = t in SOFT_TERMS or t.replace(".", "").isdigit()  # numbers in a question are scenario parameters
            # soft terms get a small FIXED weight: they're absent from most passages, so
            # IDF would otherwise hand them the maximum weight
            w[t] = SOFT_WEIGHT if soft else math.log(1 + (n + 1) / (df + 0.5))
        total = sum(w.values())

        def cov(terms: set[str]) -> float:
            return sum(w[t] for t in q if satisfied(alts[t], terms)) / total

        hard = [t for t in q if w[t] != SOFT_WEIGHT]
        q_bigrams = set(zip(hard, hard[1:]))
        out = []
        for c, body in zip(chunks, bodies):
            sec = set(content_terms(c.section))
            sents = [set(content_terms(s)) for s in sentences(c.text)] or [set()]
            windows = [sec | a | b for a, b in zip(sents, sents[1:] + [set()])]
            seq = content_terms(c.text)
            bg = (len(q_bigrams & set(zip(seq, seq[1:]))) / len(q_bigrams)) if q_bigrams else 0.0
            sec_match = (sum(satisfied(alts[t], sec) for t in hard) / max(1, len(hard))) if sec else 0.0
            out.append(round(0.40 * cov(body) + 0.40 * max(cov(x) for x in windows)
                             + 0.10 * bg + 0.10 * sec_match, 4))
        return out


class CrossEncoderReranker:  # pragma: no cover - optional dependency
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)
        self.name = model

    def score(self, query, chunks):
        logits = self.model.predict([(query, c.section + "\n" + c.text) for c in chunks])
        return [1 / (1 + math.exp(-float(x))) for x in logits]


def make_reranker(spec: str) -> Reranker:
    if spec == "lexical":
        return LexicalReranker()
    if spec.startswith("cross-encoder:"):
        return CrossEncoderReranker(spec.split(":", 1)[1])
    raise ValueError(f"unknown RERANKER {spec}")
