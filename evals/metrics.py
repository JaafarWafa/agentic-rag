"""Scoring functions for the eval harness.

Two judges are available:
* `HeuristicJudge` (default, free, deterministic): claim-level groundedness by
  term-overlap + exact number matching against the *cited* passages; relevance by
  coverage of the (expanded) question terms. Strict on numbers, which is where
  policy-QA hallucinations actually hurt.
* `ClaudeJudge` (`--judge claude`): an LLM grader returning per-claim support and a
  1-5 relevance score through structured outputs. Use it when the answerer is an
  LLM that paraphrases.
"""
from __future__ import annotations

import json
import re

from app.retrieval.text import content_terms, expand_query, sentences

_MARK = re.compile(r"\[(c\d+)\]")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def norm(s: str) -> str:
    s = s.lower().replace(" ", " ")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 1,200 -> 1200
    return " ".join(s.split())


def fact_recall(answer: str, must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    a = norm(answer)
    return sum(any(norm(alt) in a for alt in fact.split("|")) for fact in must_include) / len(must_include)


def _numbers(s: str) -> set[str]:
    return {re.sub(r"[.,]0+$", "", n.replace(",", "")) for n in _NUM.findall(norm(s))}


def split_claims(answer: str) -> list[tuple[str, list[str]]]:
    out = []
    for sent in sentences(answer):
        markers = _MARK.findall(sent)
        text = _MARK.sub("", sent).strip(" -")
        if len(content_terms(text)) >= 2:
            out.append((text, markers))
    return out


class HeuristicJudge:
    name = "heuristic"

    def grade(self, question: str, answer: str, citations: list[dict]) -> dict:
        by_marker = {c["marker"]: c["text"] for c in citations}
        claims = []
        for text, markers in split_claims(answer):
            evidence = " ".join(by_marker[m] for m in markers if m in by_marker) or " ".join(by_marker.values())
            ev_terms = set(content_terms(evidence))
            terms = content_terms(text)
            overlap = sum(t in ev_terms for t in terms) / max(1, len(terms))
            nums_ok = _numbers(text) <= _numbers(evidence)
            claims.append({"claim": text[:200], "supported": bool(overlap >= 0.8 and nums_ok),
                           "overlap": round(overlap, 3), "numbers_ok": nums_ok})
        q_terms = set(content_terms(question))
        exp_terms = set(content_terms(expand_query(question)))
        a_terms = set(content_terms(answer))
        relevance = min(1.0, len(exp_terms & a_terms) / max(1, len(q_terms))) if q_terms else 0.0
        return {"claims": claims, "relevance": round(relevance, 3)}


class ClaudeJudge:  # pragma: no cover - needs credentials
    def __init__(self, model: str = "claude-opus-5-5", client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model = client, model
        self.name = f"claude-judge:{model}"

    SCHEMA = {
        "type": "object",
        "properties": {
            "claims": {"type": "array", "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
                "required": ["claim", "supported"], "additionalProperties": False}},
            "relevance": {"type": "integer", "description": "1-5: how directly the answer addresses the question"},
        },
        "required": ["claims", "relevance"],
        "additionalProperties": False,
    }

    def grade(self, question, answer, citations):
        passages = "\n\n".join(f"[{c['marker']}] {c['text']}" for c in citations) or "(none)"
        prompt = (
            "You are grading a retrieval-augmented answer. Split the ANSWER into atomic factual claims. "
            "For each claim decide whether it is fully supported by the PASSAGES alone (no outside knowledge; "
            "numbers must match exactly). Then rate relevance 1-5: does the answer directly address the QUESTION?\n\n"
            f"QUESTION:\n{question}\n\nPASSAGES:\n{passages}\n\nANSWER:\n{answer}")
        resp = self.client.messages.create(
            model=self.model, max_tokens=4000, output_config={"effort": "low", "format": {
                "type": "json_schema", "schema": self.SCHEMA}},
            messages=[{"role": "user", "content": prompt}])
        data = json.loads(next(b.text for b in resp.content if b.type == "text"))
        data["relevance"] = round((max(1, min(5, int(data["relevance"]))) - 1) / 4, 3)
        return data


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[i]
