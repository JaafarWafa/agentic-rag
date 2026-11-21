from evals.metrics import HeuristicJudge, fact_recall
from evals.run_eval import check_gates, load_dataset, run_eval

ADMIN = {"X-API-Key": "admin-demo-key"}
EMP = {"X-API-Key": "employee-demo-key"}


def test_health_and_seed(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["chunks"] > 50 and h["llm"] == "offline-extractive"


def test_query_and_trace_roundtrip(client):
    r = client.post("/query", json={"question": "How many days per week can I work remotely?"}, headers=EMP).json()
    assert not r["refused"] and "2 days" in r["answer"]
    t = client.get(f"/traces/{r['trace_id']}", headers=ADMIN).json()
    assert t["question"].startswith("How many days") and any(s["type"] == "search" for s in t["steps"])
    assert client.get("/traces", headers=EMP).status_code == 403


def test_bad_key_rejected(client):
    assert client.post("/query", json={"question": "x"}, headers={"X-API-Key": "nope"}).status_code == 401


def test_upload_requires_admin_and_respects_roles(client):
    doc = b"# Parking Policy\n\n## Tangier hub\n\nStaff at the Tangier hub may park in zone B, free of charge."
    files = {"file": ("parking.md", doc, "text/markdown")}
    assert client.post("/documents", files=files, data={"allowed_roles": "legal"}, headers=EMP).status_code == 403
    r = client.post("/documents", files=files, data={"allowed_roles": "legal"}, headers=ADMIN)
    assert r.status_code == 201, r.text
    doc_id = r.json()["id"]
    # employee can't see it, and gets 404 (not 403) for its chunks
    assert client.get(f"/documents/{doc_id}/chunks", headers=EMP).status_code == 404
    q = {"question": "Where can staff park at the Tangier hub?"}
    assert client.post("/query", json=q, headers=EMP).json()["refused"]
    legal = client.post("/query", json=q, headers={"X-API-Key": "legal-demo-key"}).json()
    assert not legal["refused"] and "zone B" in legal["answer"]
    assert client.delete(f"/documents/{doc_id}", headers=ADMIN).status_code == 204


def test_upload_rejects_bad_roles(client):
    files = {"file": ("x.md", b"# T\n\nSome text that is long enough to chunk.", "text/markdown")}
    assert client.post("/documents", files=files, data={"allowed_roles": "superuser"}, headers=ADMIN).status_code == 422


def test_fact_recall_normalises_numbers():
    assert fact_recall("The cap is 1200 MAD", ["1,200|1200"]) == 1.0
    assert fact_recall("Tuesday only", ["Tuesday", "Thursday"]) == 0.5


def test_heuristic_judge_flags_wrong_number():
    cites = [{"marker": "c1", "text": "Hotel costs are reimbursed up to a nightly cap of 1,200 MAD in Morocco."}]
    good = HeuristicJudge().grade("q", "Hotel costs are reimbursed up to a nightly cap of 1,200 MAD [c1].", cites)
    bad = HeuristicJudge().grade("q", "Hotel costs are reimbursed up to a nightly cap of 1,500 MAD [c1].", cites)
    assert good["claims"][0]["supported"] and not bad["claims"][0]["supported"]


def test_dataset_shape():
    data = load_dataset()
    assert len(data) >= 50
    cats = {d["category"] for d in data}
    assert {"answerable", "unanswerable", "restricted", "paraphrase"} <= cats
    assert {d["split"] for d in data} == {"dev", "test"}


def test_full_eval_passes_quality_gates(svc):
    run = run_eval(svc, "pytest", write=False)
    assert not check_gates(run["metrics"]), run["metrics"]
