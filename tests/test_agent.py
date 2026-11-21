from types import SimpleNamespace

from app.agent.agent import Agent, ClaudePolicy, OfflinePolicy
from app.auth import expand_roles

EMP = expand_roles(["employee"])


def test_small_talk_skips_retrieval(svc):
    res = svc.ask("hello!", "t", EMP, save_trace=False)
    assert res["route"] == "direct"
    assert not any(s["type"] == "search" for s in res["steps"])


def test_answer_has_valid_citations(svc):
    res = svc.ask("What is the nightly hotel cap in Morocco?", "t", EMP, save_trace=False)
    assert not res["refused"] and "1,200 MAD" in res["answer"]
    assert res["citations"] and all(f"[{c['marker']}]" in res["answer"] for c in res["citations"])
    assert res["citations"][0]["doc_title"] == "Travel and Expense Policy"


def test_out_of_corpus_question_is_refused(svc):
    res = svc.ask("Does the company match pension contributions?", "t", expand_roles(["admin"]), save_trace=False)
    assert res["refused"] and res["citations"] == []


def test_weak_evidence_triggers_refinement(svc):
    # the first search finds nothing about pregnancy; the refined query adds "maternity leave"
    res = svc.ask("I'm pregnant, how much time off do I get?", "t", EMP, save_trace=False)
    searches = [s for s in res["steps"] if s["type"] == "search"]
    assert len(searches) >= 2, "agent should re-search with a refined query"
    assert "maternity leave" in searches[1]["query"]
    assert not res["refused"] and "14 weeks" in res["answer"]


def test_refinement_cannot_drift_to_an_easier_question(svc):
    # the refined query "name ceo" would match "names of the attendees"; grading against
    # the original question must still refuse
    res = svc.ask("What is the name of the CEO?", "t", expand_roles(["admin"]), save_trace=False)
    assert res["refused"]


# ---------------------------------------------------------------- Claude policy
class FakeClient:
    """Scripted stand-in for anthropic.Anthropic - lets us test the tool loop offline."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def _resp(*blocks, stop="tool_use"):
    return SimpleNamespace(stop_reason=stop, content=list(blocks), model="fake",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def _tool(name, inp, id_="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=id_)


def test_claude_loop_search_then_answer(svc):
    client = FakeClient([
        _resp(_tool("search_corpus", {"query": "hotel nightly cap Morocco"})),
        _resp(_tool("submit_answer", {"answer": "The cap is 1,200 MAD per night [c1].",
                                      "citations": ["c1"], "refused": False}, "t2")),
    ])
    agent = Agent(svc.retriever, ClaudePolicy("claude-opus-5-5", "medium", 3, client))
    res = agent.ask("What is the hotel cap in Morocco?", EMP)
    assert not res.refused and res.citations[0]["doc_title"] == "Travel and Expense Policy"
    assert res.usage == {"input_tokens": 200, "output_tokens": 40}
    # tool result went back to the model with citation markers
    tool_result_msg = client.calls[1]["messages"][2]
    assert tool_result_msg["role"] == "user" and "[c1]" in tool_result_msg["content"][0]["content"]
    assert client.calls[0]["model"] == "claude-opus-5-5"


def test_claude_hallucinated_citation_is_blocked(svc):
    client = FakeClient([
        _resp(_tool("search_corpus", {"query": "pension matching"})),
        _resp(_tool("submit_answer", {"answer": "We match 5% of pension contributions [c9].",
                                      "citations": ["c9"], "refused": False}, "t2")),
    ])
    res = Agent(svc.retriever, ClaudePolicy("m", "low", 3, client)).ask("Pension match?", EMP)
    assert res.refused, "an answer citing a passage that was never retrieved must be blocked"
    assert any(s["type"] == "guardrail" for s in res.steps)


def test_claude_search_budget_enforced(svc):
    client = FakeClient([_resp(_tool("search_corpus", {"query": f"q{i}"}, f"t{i}")) for i in range(5)] + [
        _resp(_tool("submit_answer", {"answer": "x", "citations": [], "refused": True}, "tz"))])
    res = Agent(svc.retriever, ClaudePolicy("m", "low", 2, client)).ask("anything", EMP)
    assert sum(s["type"] == "search" for s in res.steps) == 2
    assert res.refused


def test_offline_policy_direct_name():
    assert OfflinePolicy(0.5, 3, True).name == "offline-extractive"
