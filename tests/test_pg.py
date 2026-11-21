"""Runs the same RBAC + tracing checks against Postgres/pgvector when TEST_PG_URL is set
(CI starts a pgvector service container)."""
import dataclasses
import os

import pytest

from app.auth import expand_roles
from app.config import Settings

PG = os.getenv("TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG, reason="set TEST_PG_URL to run pgvector tests")


@pytest.fixture(scope="module")
def pg_svc():
    import psycopg
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS chunks, documents, traces, eval_runs CASCADE")
    from app.service import RAGService
    s = RAGService(dataclasses.replace(Settings(), database_url=PG, llm="offline"))
    s.seed()
    return s


def test_pg_rbac_prefilter(pg_svc):
    q = "salary band bonus litigation hold liability cap"
    roles = expand_roles(["employee"])
    hits = pg_svc.store.dense_search(pg_svc.embedder.embed([q])[0], roles, 50) + \
        pg_svc.store.keyword_search(q, roles, 50)
    assert hits and not any(h.doc_id.startswith(("compensation", "litigation", "vendor")) for h in hits)


def test_pg_answer_and_trace(pg_svc):
    out = pg_svc.ask("What is the per diem when travelling abroad?", "t", expand_roles(["employee"]))
    assert not out["refused"] and "60 EUR" in out["answer"]
    t = pg_svc.store.get_trace(out["trace_id"])
    assert t["question"].startswith("What is the per diem") and t["steps"]
    assert pg_svc.store.list_traces(5)
