"""The agent layer: decides whether to search, re-searches with a refined query when the
evidence is weak, answers with citations, or refuses.

Two interchangeable policies share the same tools, citation registry and guardrails:

* `ClaudePolicy`  - Claude drives the loop through tool use (`search_corpus`,
  `submit_answer`). The model chooses queries and when to stop.
* `OfflinePolicy` - a deterministic state machine (route -> search -> grade ->
  refine -> answer|refuse) with an extractive answerer. No API key needed; it is what
  CI and the default eval run use, and it is a useful baseline to beat.

Guardrails applied to BOTH (so the eval measures the system, not just the model):
citations must reference passages actually retrieved for this user in this request;
an answer with no valid citation is converted into a refusal.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

from app.agent.prompts import ANSWER_TOOL, REFUSAL_TEXT, SEARCH_TOOL, SYSTEM_PROMPT
from app.db.base import Chunk
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.text import (SOFT_TERMS, STOPWORDS, content_terms, expand_query, sentences, stem,
                                satisfied, term_alternatives, words)

log = logging.getLogger("rag.agent")

SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|salam|bonjour|thanks?|thank you|good (morning|afternoon|evening)|"
                        r"who are you|what can you do)[\s!.?]*$", re.I)
_MARKER = re.compile(r"\[(c\d+)\]")


@dataclass
class AgentResult:
    answer: str
    refused: bool
    citations: list[dict]
    route: str
    steps: list[dict]
    provider: str
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    retrieved: list[dict] = field(default_factory=list)  # every passage seen (for eval / tracing)


class Session:
    """Per-request state: the citation registry and the step log (= the trace)."""

    def __init__(self, retriever: HybridRetriever, roles: list[str]):
        self.retriever, self.roles = retriever, roles
        self.registry: dict[str, Chunk] = {}
        self._by_chunk: dict[str, str] = {}
        self.steps: list[dict] = []
        self.rounds = 0
        self._t0 = time.perf_counter()

    def log(self, kind: str, **data):
        self.steps.append({"step": len(self.steps) + 1, "type": kind,
                           "t_ms": round((time.perf_counter() - self._t0) * 1000, 1), **data})

    def search(self, query: str, k: int = 5, grade_against: str | None = None):
        self.rounds += 1
        res = self.retriever.search(query, self.roles, max(1, min(int(k or 5), 8)), grade_against)
        for c in res.chunks:
            if c.id not in self._by_chunk:
                m = f"c{len(self.registry) + 1}"
                self.registry[m], self._by_chunk[c.id] = c, m
        self.log("search", round=self.rounds, **res.summary())
        return res

    def marker(self, chunk: Chunk) -> str:
        return self._by_chunk[chunk.id]

    def render(self, res) -> str:
        if not res.chunks:
            return "No passages found."
        return "\n\n".join(
            f"[{self.marker(c)}] (score {c.scores.get('rerank', c.score):.2f}) {c.doc_title} > {c.section}\n{c.text}"
            for c in res.chunks)

    def finalize(self, answer: str, markers: list[str], refused: bool, route: str, provider: str,
                 usage: dict | None = None) -> AgentResult:
        valid = [m for m in dict.fromkeys(markers + _MARKER.findall(answer)) if m in self.registry]
        dropped = sorted(set(markers + _MARKER.findall(answer)) - set(valid))
        if dropped:
            self.log("guardrail", rule="unknown_citation_removed", markers=dropped)
            answer = _MARKER.sub(lambda m: m.group(0) if m.group(1) in self.registry else "", answer)
        if not refused and route != "direct" and not valid:
            self.log("guardrail", rule="uncited_answer_blocked", original=answer[:300])
            answer, refused = REFUSAL_TEXT, True
        if refused:
            valid = []
        cites = [{"marker": m, **self.registry[m].to_public()} for m in valid]
        self.log("answer", refused=refused, citations=valid)
        return AgentResult(answer=answer.strip(), refused=refused, citations=cites, route=route, steps=self.steps,
                           provider=provider, usage=usage or {"input_tokens": 0, "output_tokens": 0},
                           retrieved=[{"marker": m, "chunk_id": c.id, "doc_id": c.doc_id, "text": c.text}
                                      for m, c in self.registry.items()])


# --------------------------------------------------------------------------- offline
class OfflinePolicy:
    name = "offline-extractive"

    def __init__(self, threshold: float, max_rounds: int, allow_refine: bool):
        self.threshold, self.max_rounds, self.allow_refine = threshold, max_rounds, allow_refine

    @staticmethod
    def _focus(q: str) -> str:
        # keep the specific terms, drop stopwords and question-intent words
        return " ".join(dict.fromkeys(w for w in words(q) if w not in STOPWORDS and stem(w) not in SOFT_TERMS))

    def run(self, s: Session, question: str) -> AgentResult:
        if SMALL_TALK.match(question):
            s.log("route", decision="direct", reason="small talk; no retrieval needed")
            return s.finalize("Hello! Ask me a question about the documents you can access and I'll answer "
                              "with citations.", [], False, "direct", self.name)
        s.log("route", decision="search", reason="information-seeking question")
        plans = [question]
        if self.allow_refine:
            plans += [expand_query(question), self._focus(expand_query(question))]
        best = None
        for q in plans[: self.max_rounds]:
            if best and q == best.query:
                continue
            res = s.search(q, grade_against=question)
            if best is None or res.best_score > best.best_score:
                best = res
            if res.best_score >= self.threshold:
                s.log("grade", verdict="sufficient", best_score=round(res.best_score, 3), threshold=self.threshold)
                break
            s.log("grade", verdict="insufficient", best_score=round(res.best_score, 3), threshold=self.threshold,
                  next="refine" if self.allow_refine else "stop")
        if not best or best.best_score < self.threshold:
            return s.finalize(REFUSAL_TEXT, [], True, "refused", self.name)

        qalts = [term_alternatives(t) for t in content_terms(question)]
        cands = []
        for rank, c in enumerate(best.chunks[:3]):
            if c.scores.get("rerank", 0) < self.threshold * 0.85:
                continue
            for i, sent in enumerate(sentences(c.text)):
                st = set(content_terms(sent))
                ov = sum(satisfied(a, st) for a in qalts)
                # numbers in a sentence are usually the payload of policy answers
                has_num = bool(re.search(r"\d", sent))
                cands.append((ov + 0.5 * has_num - 0.3 * rank, rank, i, sent, c))
        cands = [x for x in cands if x[0] > 0.6]
        cands.sort(key=lambda x: -x[0])
        picked = sorted(cands[:3], key=lambda x: (x[1], x[2]))
        if not picked:
            return s.finalize(REFUSAL_TEXT, [], True, "refused", self.name)
        parts, markers = [], []
        for _, _, _, sent, c in picked:
            m = s.marker(c)
            parts.append(f"{sent} [{m}]")
            markers.append(m)
        return s.finalize(" ".join(parts), markers, False, "answered", self.name)


# --------------------------------------------------------------------------- Claude
class ClaudePolicy:
    def __init__(self, model: str, effort: str, max_rounds: int, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.effort, self.max_rounds = client, model, effort, max_rounds
        self.name = f"claude:{model}"

    def _create(self, system: str, messages: list):
        # Server-side refusal fallback is opt-in; "default" routes by refusal category.
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=8000,
            system=system,
            tools=[SEARCH_TOOL, ANSWER_TOOL],
            tool_choice={"type": "auto"},
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        )

    def run(self, s: Session, question: str) -> AgentResult:
        system = SYSTEM_PROMPT.format(max_rounds=self.max_rounds)
        messages: list = [{"role": "user", "content": question}]
        usage = {"input_tokens": 0, "output_tokens": 0}
        for _turn in range(self.max_rounds + 3):
            resp = self._create(system, messages)
            usage["input_tokens"] += getattr(resp.usage, "input_tokens", 0) or 0
            usage["output_tokens"] += getattr(resp.usage, "output_tokens", 0) or 0
            s.log("llm", stop_reason=resp.stop_reason, model=getattr(resp, "model", self.model),
                  input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
            if resp.stop_reason == "refusal":
                return s.finalize(REFUSAL_TEXT, [], True, "refused", self.name, usage)
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                text = "".join(b.text for b in resp.content if b.type == "text")
                if not s.rounds and text:
                    s.log("route", decision="direct", reason="model answered without searching")
                    return s.finalize(text, [], False, "direct", self.name, usage)
                return s.finalize(text or REFUSAL_TEXT, [], not text, "answered", self.name, usage)

            results = []
            for tu in tool_uses:
                if tu.name == "submit_answer":
                    inp = tu.input if isinstance(tu.input, dict) else {}
                    if not isinstance(inp.get("answer"), str):
                        results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                        "content": "submit_answer needs answer (string), citations, refused"})
                        continue
                    if s.rounds == 0:
                        s.log("route", decision="direct", reason="model answered without searching")
                    route = "refused" if inp.get("refused") else ("direct" if s.rounds == 0 else "answered")
                    return s.finalize(inp["answer"], [str(m).strip("[]") for m in inp.get("citations", [])],
                                      bool(inp.get("refused")), route, self.name, usage)
                if tu.name == "search_corpus":
                    if s.rounds >= self.max_rounds:
                        content = "Search budget exhausted. Call submit_answer now (refuse if evidence is missing)."
                    else:
                        if s.rounds:
                            s.log("refine", query=tu.input.get("query", ""))
                        content = s.render(s.search(str(tu.input.get("query", question)), tu.input.get("k", 5)))
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content})
                else:
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                    "content": f"unknown tool {tu.name}"})
            messages.append({"role": "user", "content": results})
        s.log("guardrail", rule="turn_limit")
        return s.finalize(REFUSAL_TEXT, [], True, "refused", self.name, usage)


# --------------------------------------------------------------------------- facade
class Agent:
    def __init__(self, retriever: HybridRetriever, policy):
        self.retriever, self.policy = retriever, policy

    def ask(self, question: str, roles: list[str]) -> AgentResult:
        s = Session(self.retriever, roles)
        try:
            return self.policy.run(s, question)
        except Exception as e:  # never 500 on a model hiccup - degrade to a logged refusal
            log.exception("agent failure")
            s.log("error", error=f"{type(e).__name__}: {e}")
            return s.finalize(REFUSAL_TEXT, [], True, "error", getattr(self.policy, "name", "?"))


def has_claude_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def make_policy(settings, client=None):
    choice = settings.llm
    if choice == "auto":
        choice = "claude" if (client or has_claude_credentials()) else "offline"
    if choice == "claude":
        return ClaudePolicy(settings.claude_model, settings.claude_effort, settings.max_search_rounds, client)
    return OfflinePolicy(settings.evidence_threshold, settings.max_search_rounds, settings.allow_refine)


__all__ = ["Agent", "AgentResult", "ClaudePolicy", "OfflinePolicy", "make_policy", "stem"]
