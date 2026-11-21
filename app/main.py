"""FastAPI application."""
from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.auth import ALL_ROLES, Principal, get_principal, require_admin
from app.config import settings
from app.service import RAGService


class _JsonFormatter(logging.Formatter):
    def format(self, r):
        return json.dumps({"ts": self.formatTime(r), "level": r.levelname, "logger": r.name, "msg": r.getMessage()})


_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(_JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_h], force=True)

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 15 * 1024 * 1024
svc: RAGService | None = None


def service() -> RAGService:
    global svc
    if svc is None:
        svc = RAGService(settings)
    return svc


@asynccontextmanager
async def lifespan(_app):
    s = service()
    if settings.seed_corpus and s.store.count_chunks() == 0 and settings.corpus_dir.exists():
        s.seed()
    yield


app = FastAPI(title="Agentic RAG", version="1.0.0", lifespan=lifespan,
              description="Hybrid-search, role-aware, self-evaluating document Q&A.")


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    s = service()
    return {"status": "ok", "chunks": s.store.count_chunks(), **s.describe()}


@app.get("/me")
def me(p: Principal = Depends(get_principal)):
    return {"user": p.user, "roles": p.roles, "effective_roles": p.effective_roles}


@app.post("/query")
def query(body: AskIn, p: Principal = Depends(get_principal)):
    return service().ask(body.question.strip(), p.user, p.effective_roles)


# ------------------------------------------------------------------ documents
@app.post("/documents", status_code=201)
async def upload(file: UploadFile = File(...), allowed_roles: str = Form("employee"),
                 title: str | None = Form(None), p: Principal = Depends(get_principal)):
    require_admin(p)
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (15 MB max)")
    roles = [r.strip() for r in allowed_roles.split(",") if r.strip()]
    try:
        return service().ingest(file.filename or "upload.txt", data, roles, title)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/documents")
def list_documents(p: Principal = Depends(get_principal)):
    return service().store.list_documents(p.effective_roles)


@app.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, p: Principal = Depends(get_principal)):
    s = service()
    if not s.store.get_document(doc_id, p.effective_roles):
        raise HTTPException(404, "document not found")  # 404, not 403: don't confirm existence
    return [c.to_public() for c in s.store.get_chunks(doc_id, p.effective_roles)]


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, p: Principal = Depends(get_principal)):
    require_admin(p)
    if not service().store.delete_document(doc_id):
        raise HTTPException(404, "document not found")


@app.get("/roles")
def roles():
    return sorted(ALL_ROLES)


# --------------------------------------------------------------- observability
@app.get("/traces")
def traces(limit: int = 50, p: Principal = Depends(get_principal)):
    require_admin(p)
    return service().store.list_traces(min(limit, 500))


@app.get("/traces/{trace_id}")
def trace(trace_id: str, p: Principal = Depends(get_principal)):
    require_admin(p)
    t = service().store.get_trace(trace_id)
    if not t:
        raise HTTPException(404, "trace not found")
    return t


@app.get("/eval/runs")
def eval_runs():
    """Runs stored in this deployment's DB plus the run history committed in evals/results/."""
    runs = {r["id"]: r for r in service().store.list_eval_runs()}
    for f in sorted((Path(__file__).parent.parent / "evals" / "results").glob("*.json")):
        try:
            d = json.loads(f.read_text())
            d.pop("rows", None)
            runs.setdefault(d["id"], d)
        except (OSError, ValueError, KeyError):
            continue
    return sorted(runs.values(), key=lambda r: str(r["created_at"]))


@app.post("/eval/run", status_code=202)
def eval_run(bg: BackgroundTasks, label: str = "api", p: Principal = Depends(get_principal)):
    require_admin(p)
    from evals.run_eval import run_eval
    bg.add_task(run_eval, service(), label=label)
    return {"status": "started", "label": label}


# ------------------------------------------------------------------- frontend
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
