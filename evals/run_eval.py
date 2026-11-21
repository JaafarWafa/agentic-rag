"""Evaluation harness.

    python -m evals.run_eval --label baseline
    python -m evals.run_eval --label no-rerank --set USE_RERANK=0
    python -m evals.run_eval --ablation            # runs the standard ablation ladder
    python -m evals.run_eval --gate                 # CI: exit 1 if quality gates fail

Each run writes evals/results/<run_id>.json (per-question detail), appends a row to
evals/history.csv, stores the summary in the eval_runs table (shown in the UI), and
prints the delta against the previous run with the same LLM.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.auth import expand_roles
from app.config import ROOT, Settings
from evals.metrics import ClaudeJudge, HeuristicJudge, fact_recall, percentile

DATASET = ROOT / "evals" / "dataset.jsonl"
RESULTS = ROOT / "evals" / "results"
HISTORY = ROOT / "evals" / "history.csv"

HEADLINE = ["answer_accuracy", "retrieval_hit_rate", "mrr", "citation_precision", "groundedness",
            "answer_relevance", "hallucination_rate", "false_refusal_rate", "correct_refusal_rate",
            "rbac_leak_rate", "latency_p50_ms", "latency_p95_ms"]
# Lower is better for these
LOWER_BETTER = {"hallucination_rate", "false_refusal_rate", "rbac_leak_rate", "latency_p50_ms", "latency_p95_ms"}

GATES = {  # what CI enforces on the offline configuration
    "rbac_leak_rate": ("max", 0.0),
    "hallucination_rate": ("max", 0.05),
    "answer_accuracy": ("min", 0.80),
    "correct_refusal_rate": ("min", 0.90),
}


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _doc_matches(doc_id: str, slugs: list[str]) -> bool:
    return any(doc_id.startswith(s + "-") for s in slugs)


def evaluate_one(svc, item: dict, judge) -> dict:
    roles = expand_roles([item["role"]])
    t0 = time.perf_counter()
    out = svc.ask(item["question"], f"eval:{item['id']}", roles, save_trace=False)
    latency = (time.perf_counter() - t0) * 1000
    cat = item["category"]
    should_answer = cat in ("answerable", "paraphrase")
    allowed_docs = {d["id"] for d in svc.store.list_documents(roles)}

    searches = [s for s in out["steps"] if s["type"] == "search"]
    retrieved_docs = [r["chunk_id"].split("#")[0] for s in searches for r in s["results"]]
    first = [r["chunk_id"].split("#")[0] for r in searches[0]["results"]] if searches else []
    hit = any(_doc_matches(d, item["expected_docs"]) for d in retrieved_docs) if should_answer else None
    rr = next((1 / (i + 1) for i, d in enumerate(first) if _doc_matches(d, item["expected_docs"])), 0.0) \
        if should_answer else None

    cited_docs = [c["doc_id"] for c in out["citations"]]
    cit_prec = (sum(_doc_matches(d, item["expected_docs"]) for d in cited_docs) / len(cited_docs)) \
        if (cited_docs and should_answer) else None

    answered = not out["refused"]
    grade = judge.grade(item["question"], out["answer"], out["citations"]) if answered else {"claims": [], "relevance": None}
    claims = grade["claims"]
    groundedness = (sum(c["supported"] for c in claims) / len(claims)) if claims else (1.0 if answered else None)
    recall = fact_recall(out["answer"], item["must_include"]) if should_answer and answered else 0.0

    # Leakage: anything outside the caller's roles reaching retrieval/citations, or the
    # restricted fact showing up in the answer.
    out_of_scope = [d for d in retrieved_docs + cited_docs if d not in allowed_docs]
    leaked = bool(out_of_scope) or (cat == "restricted" and answered and
                                    fact_recall(out["answer"], item["must_include"]) > 0)
    hallucinated = answered and (not should_answer or (groundedness is not None and groundedness < 1.0))

    return {
        "id": item["id"], "category": cat, "role": item["role"], "question": item["question"],
        "answer": out["answer"], "refused": out["refused"], "route": out["route"],
        "citations": [{"marker": c["marker"], "doc_id": c["doc_id"], "section": c["section"]} for c in out["citations"]],
        "n_searches": len(searches), "queries": [s["query"] for s in searches],
        "retrieval_hit": hit, "reciprocal_rank": rr, "citation_precision": cit_prec,
        "fact_recall": recall if should_answer else None, "groundedness": groundedness,
        "relevance": grade["relevance"], "claims": claims, "hallucinated": hallucinated, "leaked": leaked,
        "latency_ms": round(latency, 1), "tokens": out["usage"],
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def aggregate(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["category"] in ("answerable", "paraphrase")]
    neg = [r for r in rows if r["category"] in ("unanswerable", "restricted")]
    m = {
        "n": len(rows),
        "answer_accuracy": _mean([r["fact_recall"] for r in ans]),
        "strict_accuracy": _mean([float(r["fact_recall"] == 1.0) for r in ans]),
        "retrieval_hit_rate": _mean([float(r["retrieval_hit"]) for r in ans]),
        "mrr": _mean([r["reciprocal_rank"] for r in ans]),
        "citation_precision": _mean([r["citation_precision"] for r in ans]),
        "groundedness": _mean([r["groundedness"] for r in rows if not r["refused"]]),
        "answer_relevance": _mean([r["relevance"] for r in ans if not r["refused"]]),
        "hallucination_rate": _mean([float(r["hallucinated"]) for r in rows]),
        "unsupported_claim_rate": _mean([1 - r["groundedness"] for r in rows if r["groundedness"] is not None]),
        "false_refusal_rate": _mean([float(r["refused"]) for r in ans]),
        "correct_refusal_rate": _mean([float(r["refused"]) for r in neg]),
        "rbac_leak_rate": _mean([float(r["leaked"]) for r in rows]),
        "avg_searches": _mean([r["n_searches"] for r in rows]),
        "latency_p50_ms": round(percentile([r["latency_ms"] for r in rows], 50), 1),
        "latency_p95_ms": round(percentile([r["latency_ms"] for r in rows], 95), 1),
        "input_tokens": sum(r["tokens"].get("input_tokens", 0) for r in rows),
        "output_tokens": sum(r["tokens"].get("output_tokens", 0) for r in rows),
    }
    m["by_category"] = {}
    for cat in sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["category"] == cat]
        m["by_category"][cat] = {"n": len(rs), "refused": _mean([float(r["refused"]) for r in rs]),
                                 "fact_recall": _mean([r["fact_recall"] for r in rs]),
                                 "hallucinated": _mean([float(r["hallucinated"]) for r in rs])}
    return m


def run_eval(svc, label: str = "run", judge=None, dataset: list[dict] | None = None, write: bool = True) -> dict:
    judge = judge or HeuristicJudge()
    items = dataset or load_dataset()
    rows = [evaluate_one(svc, it, judge) for it in items]
    metrics = aggregate(rows)
    split_of = {it["id"]: it.get("split", "dev") for it in items}
    test_rows = [r for r in rows if split_of[r["id"]] == "test"]
    if test_rows:
        t = aggregate(test_rows)
        metrics["test_split"] = {k: t[k] for k in ("n", "answer_accuracy", "hallucination_rate",
                                                   "false_refusal_rate", "correct_refusal_rate", "rbac_leak_rate")}
    run = {"id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:4]}",
           "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"), "label": label,
           "config": {**svc.describe(), "judge": judge.name, "dataset_size": len(items)},
           "metrics": metrics, "n": len(rows)}
    if write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{run['id']}.json").write_text(json.dumps({**run, "rows": rows}, indent=2))
        new = not HISTORY.exists()
        with HISTORY.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["run_id", "created_at", "label", "llm", "hybrid", "use_rerank", "allow_refine",
                            "threshold", *HEADLINE])
            c = run["config"]
            w.writerow([run["id"], run["created_at"], label, c["llm"], c["hybrid"], c["use_rerank"],
                        c["allow_refine"], c["evidence_threshold"], *[metrics.get(k) for k in HEADLINE]])
        try:
            svc.store.save_eval_run(run)
        except Exception:
            pass
    return {**run, "rows": rows}


def previous_run(llm: str, before: str) -> dict | None:
    if not HISTORY.exists():
        return None
    rows = [r for r in csv.DictReader(HISTORY.open()) if r["llm"] == llm and r["run_id"] < before]
    return rows[-1] if rows else None


def print_report(run: dict, prev: dict | None = None):
    m = run["metrics"]
    print(f"\n=== {run['label']}  ({run['config']['llm']}, judge={run['config']['judge']}, n={run['n']}) ===")
    for k in HEADLINE:
        v = m.get(k)
        line = f"  {k:<22} {v if v is not None else '-':>8}"
        if prev and prev.get(k) not in (None, "", "None") and v is not None:
            d = v - float(prev[k])
            better = (d < 0) if k in LOWER_BETTER else (d > 0)
            if abs(d) > 1e-9:
                line += f"   {'+' if d > 0 else ''}{d:.4f} {'better' if better else 'worse'} vs {prev['label']}"
        print(line)
    print("  by category:", json.dumps(m["by_category"]))
    if "test_split" in m:
        print("  held-out test split:", json.dumps(m["test_split"]))
    fails = [r for r in run["rows"] if r["hallucinated"] or r["leaked"] or
             (r["category"] in ("answerable", "paraphrase") and (r["fact_recall"] or 0) < 1)]
    if fails:
        print(f"\n  {len(fails)} questions to look at:")
        for r in fails[:15]:
            why = "LEAK" if r["leaked"] else "HALLUCINATION" if r["hallucinated"] else \
                "FALSE-REFUSAL" if r["refused"] else "MISSING-FACT"
            print(f"   [{why}] {r['id']} ({r['role']}) {r['question'][:70]}")


def check_gates(m: dict) -> list[str]:
    errs = []
    for k, (kind, thr) in GATES.items():
        v = m.get(k)
        if v is None or (kind == "max" and v > thr) or (kind == "min" and v < thr):
            errs.append(f"{k}={v} violates {kind} {thr}")
    return errs


ABLATION = [  # each step changes one thing
    ("v1-dense-only", {"HYBRID": "0", "USE_RERANK": "0", "ALLOW_REFINE": "0", "EVIDENCE_THRESHOLD": "0"}),
    ("v2-+refusal-threshold", {"HYBRID": "0", "USE_RERANK": "0", "ALLOW_REFINE": "0"}),
    ("v3-+bm25-hybrid", {"USE_RERANK": "0", "ALLOW_REFINE": "0"}),
    ("v4-+rerank", {"ALLOW_REFINE": "0"}),
    ("v5-+query-refinement", {}),
]


def build_service(overrides: dict[str, str], db_url: str | None = None):
    from app.service import RAGService
    old = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        s = Settings()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    s = dataclasses.replace(s, database_url=db_url or "sqlite:///:memory:")
    svc = RAGService(s)
    if svc.store.count_chunks() == 0:
        svc.seed()
    return svc


def sweep(judge, data, values=None, cost_hallucination: float = 3.0):
    """Pick EVIDENCE_THRESHOLD on the dev split only. Cost = 3 x hallucinations + false refusals
    (an invented policy answer is worse than "I don't know"). The test split is never used."""
    dev = [d for d in data if d.get("split", "dev") == "dev"]
    values = values or [round(0.35 + 0.025 * i, 3) for i in range(11)]
    print(f"\nthreshold sweep on dev (n={len(dev)}), cost = {cost_hallucination} x halluc + false refusals")
    print(f"  {'thr':>6} {'acc':>7} {'halluc':>7} {'false_ref':>9} {'corr_ref':>8} {'cost':>6}")
    best = None
    for v in values:
        svc = build_service({"EVIDENCE_THRESHOLD": str(v)})
        rows = [evaluate_one(svc, it, judge) for it in dev]
        m = aggregate(rows)
        cost = cost_hallucination * sum(r["hallucinated"] for r in rows) + sum(
            r["refused"] for r in rows if r["category"] in ("answerable", "paraphrase"))
        print(f"  {v:>6} {m['answer_accuracy']:>7} {m['hallucination_rate']:>7} {m['false_refusal_rate']:>9} "
              f"{m['correct_refusal_rate']:>8} {cost:>6}")
        if best is None or cost < best[1]:
            best = (v, cost)
    print(f"  -> best threshold {best[0]} (cost {best[1]})")
    return best[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="run")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE settings override (repeatable)")
    ap.add_argument("--judge", choices=["heuristic", "claude"], default="heuristic")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--gate", action="store_true", help="exit non-zero if quality gates fail")
    ap.add_argument("--db", default=None, help="DATABASE_URL to evaluate against (default: fresh in-memory index)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sweep", action="store_true", help="tune EVIDENCE_THRESHOLD on the dev split")
    args = ap.parse_args(argv)
    judge = ClaudeJudge() if args.judge == "claude" else HeuristicJudge()
    data = load_dataset()[: args.limit] if args.limit else load_dataset()

    if args.sweep:
        sweep(judge, data)
        return
    plans = ABLATION if args.ablation else [(args.label, dict(kv.split("=", 1) for kv in args.set))]
    last = None
    for label, ov in plans:
        svc = build_service(ov, args.db)
        run = run_eval(svc, label, judge, data)
        print_report(run, previous_run(run["config"]["llm"], run["id"]))
        last = run
    if args.gate:
        errs = check_gates(last["metrics"])
        if errs:
            print("\nQUALITY GATES FAILED:\n  " + "\n  ".join(errs))
            sys.exit(1)
        print("\nquality gates passed")


if __name__ == "__main__":
    main()
