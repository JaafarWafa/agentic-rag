"""SQLite backend: FTS5 gives real BM25; dense vectors live in a BLOB column and are
scored with numpy over the role-filtered candidate set. Good to tens of thousands of
chunks, needs no services - ideal for local dev, CI and the eval harness."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from app.db.base import Chunk, Store, keyword_terms

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, title TEXT, source TEXT, mime TEXT, sha256 TEXT UNIQUE,
  allowed_roles TEXT, n_chunks INTEGER, n_chars INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS doc_roles (doc_id TEXT, role TEXT, PRIMARY KEY (doc_id, role));
CREATE INDEX IF NOT EXISTS ix_doc_roles_role ON doc_roles(role);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, doc_id TEXT, ord INTEGER, section TEXT, text TEXT, page INTEGER, embedding BLOB
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED, section, text, tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY, created_at TEXT, user TEXT, roles TEXT, question TEXT, answer TEXT,
  refused INTEGER, provider TEXT, latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER,
  citations TEXT, steps TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY, created_at TEXT, label TEXT, config TEXT, metrics TEXT, n INTEGER
);
"""


class SQLiteStore(Store):
    def __init__(self, path: str, dim: int):
        self.dim = dim
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # ------------------------------------------------------------------ docs
    def add_document(self, doc, chunks, embeddings):
        with self.lock, self.conn:
            c = self.conn
            c.execute(
                "INSERT INTO documents(id,title,source,mime,sha256,allowed_roles,n_chunks,n_chars) VALUES (?,?,?,?,?,?,?,?)",
                (doc["id"], doc["title"], doc["source"], doc["mime"], doc["sha256"],
                 json.dumps(doc["allowed_roles"]), len(chunks), doc.get("n_chars", 0)),
            )
            c.executemany("INSERT INTO doc_roles VALUES (?,?)", [(doc["id"], r) for r in doc["allowed_roles"]])
            for ch, vec in zip(chunks, embeddings):
                c.execute(
                    "INSERT INTO chunks(id,doc_id,ord,section,text,page,embedding) VALUES (?,?,?,?,?,?,?)",
                    (ch["id"], doc["id"], ch["ord"], ch["section"], ch["text"], ch.get("page"),
                     np.asarray(vec, dtype=np.float32).tobytes()),
                )
                c.execute("INSERT INTO chunks_fts(chunk_id,section,text) VALUES (?,?,?)",
                          (ch["id"], ch["section"], ch["text"]))

    def find_document_by_hash(self, sha256):
        with self.lock:
            r = self.conn.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
        return self._doc(r) if r else None

    @staticmethod
    def _doc(r) -> dict:
        d = dict(r)
        d["allowed_roles"] = json.loads(d["allowed_roles"])
        return d

    def _role_clause(self, roles):
        ph = ",".join("?" * len(roles))
        return f"EXISTS (SELECT 1 FROM doc_roles dr WHERE dr.doc_id = d.id AND dr.role IN ({ph}))", list(roles)

    def list_documents(self, roles):
        clause, params = self._role_clause(roles)
        with self.lock:
            rows = self.conn.execute(f"SELECT * FROM documents d WHERE {clause} ORDER BY title", params).fetchall()
        return [self._doc(r) for r in rows]

    def get_document(self, doc_id, roles):
        clause, params = self._role_clause(roles)
        with self.lock:
            r = self.conn.execute(f"SELECT * FROM documents d WHERE d.id=? AND {clause}", [doc_id, *params]).fetchone()
        return self._doc(r) if r else None

    def get_chunks(self, doc_id, roles):
        clause, params = self._role_clause(roles)
        with self.lock:
            rows = self.conn.execute(
                f"""SELECT c.id,c.doc_id,c.ord,c.section,c.text,c.page,d.title,d.source
                    FROM chunks c JOIN documents d ON d.id=c.doc_id
                    WHERE c.doc_id=? AND {clause} ORDER BY c.ord""", [doc_id, *params]).fetchall()
        return [self._chunk(r) for r in rows]

    def delete_document(self, doc_id):
        with self.lock, self.conn:
            ids = [r[0] for r in self.conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            self.conn.executemany("DELETE FROM chunks_fts WHERE chunk_id=?", [(i,) for i in ids])
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM doc_roles WHERE doc_id=?", (doc_id,))
            n = self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount
        return n > 0

    def count_chunks(self):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    @staticmethod
    def _chunk(r, score: float = 0.0) -> Chunk:
        return Chunk(id=r["id"], doc_id=r["doc_id"], ord=r["ord"], section=r["section"] or "",
                     text=r["text"], page=r["page"], doc_title=r["title"], source=r["source"], score=score)

    # ------------------------------------------------------------- retrieval
    def dense_search(self, vec, roles, k):
        clause, params = self._role_clause(roles)
        with self.lock:
            rows = self.conn.execute(
                f"""SELECT c.id,c.doc_id,c.ord,c.section,c.text,c.page,c.embedding,d.title,d.source
                    FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE {clause}""", params).fetchall()
        if not rows:
            return []
        mat = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), self.dim)
        sims = mat @ np.asarray(vec, dtype=np.float32)
        top = np.argsort(-sims)[:k]
        return [self._chunk(rows[i], float(sims[i])) for i in top]

    def keyword_search(self, query, roles, k):
        terms = keyword_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        clause, params = self._role_clause(roles)
        with self.lock:
            rows = self.conn.execute(
                f"""SELECT c.id,c.doc_id,c.ord,c.section,c.text,c.page,d.title,d.source,
                           -bm25(chunks_fts, 0.0, 2.0, 1.0) AS s
                    FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id JOIN documents d ON d.id=c.doc_id
                    WHERE chunks_fts MATCH ? AND {clause}
                    ORDER BY s DESC LIMIT ?""", [match, *params, k]).fetchall()
        return [self._chunk(r, float(r["s"])) for r in rows]

    # --------------------------------------------------------- observability
    def save_trace(self, t):
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t["id"], t["created_at"], t["user"], json.dumps(t["roles"]), t["question"], t["answer"],
                 int(t["refused"]), t["provider"], t["latency_ms"], t.get("input_tokens", 0),
                 t.get("output_tokens", 0), json.dumps(t["citations"]), json.dumps(t["steps"])),
            )

    @staticmethod
    def _trace(r) -> dict:
        d = dict(r)
        for key in ("roles", "citations", "steps"):
            d[key] = json.loads(d[key])
        d["refused"] = bool(d["refused"])
        return d

    def list_traces(self, limit=50):
        with self.lock:
            rows = self.conn.execute("SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = self._trace(r)
            d.pop("steps")
            out.append(d)
        return out

    def get_trace(self, trace_id):
        with self.lock:
            r = self.conn.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
        return self._trace(r) if r else None

    def save_eval_run(self, run):
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO eval_runs VALUES (?,?,?,?,?,?)",
                              (run["id"], run["created_at"], run["label"], json.dumps(run["config"]),
                               json.dumps(run["metrics"]), run["n"]))

    def list_eval_runs(self):
        with self.lock:
            rows = self.conn.execute("SELECT * FROM eval_runs ORDER BY created_at").fetchall()
        return [{**dict(r), "config": json.loads(r["config"]), "metrics": json.loads(r["metrics"])} for r in rows]
