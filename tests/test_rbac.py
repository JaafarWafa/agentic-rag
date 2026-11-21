"""Access control is enforced in the retrieval query, not after it."""
from app.auth import expand_roles
from evals.metrics import HeuristicJudge
from evals.run_eval import evaluate_one, load_dataset

HR_DOCS = ("compensation-bands-2026", "disciplinary-procedure")
LEGAL_DOCS = ("litigation-hold-procedure", "vendor-contract-playbook", "data-breach-response")


def _doc_ids(chunks):
    return {c.doc_id for c in chunks}


def test_role_hierarchy():
    assert expand_roles(["hr"]) == ["employee", "hr", "public"]
    assert set(expand_roles(["admin"])) == {"admin", "hr", "legal", "employee", "public"}
    assert expand_roles(["bogus"]) == ["public"]


def test_retrieval_never_returns_restricted_chunks(svc):
    q = "salary band bonus litigation hold liability cap breach notification"
    for role, forbidden in (("public", HR_DOCS + LEGAL_DOCS), ("employee", HR_DOCS + LEGAL_DOCS),
                            ("hr", LEGAL_DOCS), ("legal", HR_DOCS)):
        roles = expand_roles([role])
        dense = svc.store.dense_search(svc.embedder.embed([q])[0], roles, 50)
        sparse = svc.store.keyword_search(q, roles, 50)
        ids = _doc_ids(dense) | _doc_ids(sparse)
        assert ids, role
        assert not any(i.startswith(forbidden) for i in ids), (role, ids)


def test_restricted_question_is_refused_and_not_leaked(svc):
    res = svc.ask("What is the salary range for Band D?", "t", expand_roles(["employee"]), save_trace=False)
    assert res["refused"]
    assert "260,000" not in res["answer"]
    assert not any(c["doc_id"].startswith(HR_DOCS) for c in res["citations"])


def test_same_question_answered_for_authorised_role(svc):
    res = svc.ask("What is the salary range for Band D?", "t", expand_roles(["hr"]), save_trace=False)
    assert not res["refused"]
    assert "260,000" in res["answer"]


def test_leak_detector_fires_when_rbac_is_bypassed(svc, monkeypatch):
    """Negative control: prove the eval's leak metric can actually detect a leak by
    disabling the role filter for one call."""
    item = next(i for i in load_dataset() if i["id"] == "R01")  # employee asks HR-only salary question
    real_dense, real_kw = svc.store.dense_search, svc.store.keyword_search
    everything = expand_roles(["admin"])
    monkeypatch.setattr(svc.store, "dense_search", lambda v, roles, k: real_dense(v, everything, k))
    monkeypatch.setattr(svc.store, "keyword_search", lambda q, roles, k: real_kw(q, everything, k))
    row = evaluate_one(svc, item, HeuristicJudge())
    assert row["leaked"]


def test_document_listing_is_filtered(svc):
    titles = {d["title"] for d in svc.store.list_documents(expand_roles(["public"]))}
    assert titles == {"Company FAQ", "Code of Conduct"}
