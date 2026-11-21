"""PostgreSQL + pgvector backend.

* dense: HNSW index on `vector_cosine_ops`, `<=>` cosine distance
* keyword: generated `tsvector` column + GIN index, ranked with ts_rank_cd
* RBAC: `documents.allowed_roles && $roles` (array overlap) in the WHERE clause
"""
from __future__ import annotations

import json

import numpy as np
import psycopg
from psycopg.rows import dict_row

from app.db.base import Chunk, Store, keyword_terms


def _schema(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, title TEXT, source TEXT, mime TEXT, sha256 TEXT UNIQUE,
  allowed_roles TEXT[] NOT NULL, n_chunks INT, n_chars INT, created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_documents_roles ON documents USING GIN (allowed_roles);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, doc_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
  ord INT, section TEXT, text TEXT, page INT, embedding vector({dim}),
  tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(section,'')), 'A') || to_tsvector('english', text)
  ) STORED
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS ix_chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY, created_at TIMESTAMPTZ, "user" TEXT, roles JSONB, question TEXT, answer TEXT,
  refused BOOLEAN, provider TEXT, latency_ms REAL, input_tokens INT, output_tokens INT,
  citations JSONB, steps JSONB
);
CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY, created_at TIMESTAMPTZ, label TEXT, config JSONB, metrics JSONB, n INT
);
"""


def _vec(v) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in np.asarray(v, dtype=np.float32)) + "]"


_CHUNK_COLS = "c.id, c.doc_id, c.ord, c.section, c.text, c.page, d.title, d.source"


class PgStore(Store):
    def __init__(self, url: str, dim: int):
        self.dim = dim
        self.conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
        self.conn.execute(_schema(dim))

    def add_document(self, doc, chunks, embeddings):
        with self.conn.transaction():
            self.conn.execute(
                "INSERT INTO documents(id,title,source,mime,sha256,allowed_roles,n_chunks,n_chars) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (doc["id"], doc["title"], doc["source"], doc["mime"], doc["sha256"], doc["allowed_roles"],
                 len(chunks), doc.get("n_chars", 0)))
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks(id,doc_id,ord,section,text,page,embedding) VALUES (%s,%s,%s,%s,%s,%s,%s::vector)",
                    [(c["id"], doc["id"], c["ord"], c["section"], c["text"], c.get("page"), _vec(v))
                     for c, v in zip(chunks, embeddings)])

    def find_document_by_hash(self, sha256):
        return self.conn.execute("SELECT * FROM documents WHERE sha256=%s", (sha256,)).fetchone()

    def list_documents(self, roles):
        return self.conn.execute("SELECT * FROM documents WHERE allowed_roles && %s ORDER BY title", (roles,)).fetchall()

    def get_document(self, doc_id, roles):
        return self.conn.execute("SELECT * FROM documents WHERE id=%s AND allowed_roles && %s", (doc_id, roles)).fetchone()

    def get_chunks(self, doc_id, roles):
        rows = self.conn.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks c JOIN documents d ON d.id=c.doc_id "
            "WHERE c.doc_id=%s AND d.allowed_roles && %s ORDER BY c.ord", (doc_id, roles)).fetchall()
        return [self._chunk(r) for r in rows]

    def delete_document(self, doc_id):
        return self.conn.execute("DELETE FROM documents WHERE id=%s", (doc_id,)).rowcount > 0

    def count_chunks(self):
        return self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    @staticmethod
    def _chunk(r, score=0.0):
        return Chunk(id=r["id"], doc_id=r["doc_id"], ord=r["ord"], section=r["section"] or "", text=r["text"],
                     page=r["page"], doc_title=r["title"], source=r["source"], score=float(score))

    def dense_search(self, vec, roles, k):
        # Raise ef_search so the role filter doesn't starve the HNSW candidate list.
        with self.conn.transaction():
            self.conn.execute("SET LOCAL hnsw.ef_search = 200")
            rows = self.conn.execute(
                f"SELECT {_CHUNK_COLS}, 1 - (c.embedding <=> %s::vector) AS s FROM chunks c "
                "JOIN documents d ON d.id=c.doc_id WHERE d.allowed_roles && %s "
                "ORDER BY c.embedding <=> %s::vector LIMIT %s", (_vec(vec), roles, _vec(vec), k)).fetchall()
        return [self._chunk(r, r["s"]) for r in rows]

    def keyword_search(self, query, roles, k):
        terms = keyword_terms(query)
        if not terms:
            return []
        tsq = " | ".join(terms)
        rows = self.conn.execute(
            f"SELECT {_CHUNK_COLS}, ts_rank_cd(c.tsv, q, 32) AS s FROM chunks c "
            "JOIN documents d ON d.id=c.doc_id, to_tsquery('english', %s) q "
            "WHERE c.tsv @@ q AND d.allowed_roles && %s ORDER BY s DESC LIMIT %s", (tsq, roles, k)).fetchall()
        return [self._chunk(r, r["s"]) for r in rows]

    def save_trace(self, t):
        self.conn.execute(
            'INSERT INTO traces VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
            (t["id"], t["created_at"], t["user"], json.dumps(t["roles"]), t["question"], t["answer"], t["refused"],
             t["provider"], t["latency_ms"], t.get("input_tokens", 0), t.get("output_tokens", 0),
             json.dumps(t["citations"]), json.dumps(t["steps"])))

    def list_traces(self, limit=50):
        rows = self.conn.execute(
            'SELECT id, created_at::text, "user", roles, question, answer, refused, provider, latency_ms, '
            "input_tokens, output_tokens, citations FROM traces ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
        return rows

    def get_trace(self, trace_id):
        return self.conn.execute(
            'SELECT id, created_at::text AS created_at, "user", roles, question, answer, refused, provider, '
            "latency_ms, input_tokens, output_tokens, citations, steps FROM traces WHERE id=%s", (trace_id,)).fetchone()

    def save_eval_run(self, run):
        self.conn.execute(
            "INSERT INTO eval_runs VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET metrics=EXCLUDED.metrics",
            (run["id"], run["created_at"], run["label"], json.dumps(run["config"]), json.dumps(run["metrics"]), run["n"]))

    def list_eval_runs(self):
        return self.conn.execute("SELECT id, created_at::text, label, config, metrics, n FROM eval_runs ORDER BY created_at").fetchall()
