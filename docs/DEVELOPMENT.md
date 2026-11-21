# Development guide

Setup, configuration, API and deployment details. For an overview of the project, see the [README](../README.md).

## Running locally

```bash
make install && make test          # 31 tests (+2 pgvector tests when TEST_PG_URL is set)
make run                           # http://localhost:8000 - seeds the sample corpus on first boot
```

Use Claude as the agent with `export ANTHROPIC_API_KEY=...`. With `LLM_PROVIDER=auto` (the default) it switches automatically. The default model is `claude-opus-5-5` (`CLAUDE_MODEL`, `CLAUDE_EFFORT=medium`), with server-side refusal fallback enabled. Without a key, everything runs offline.

| Demo key (`X-API-Key`) | Roles |
|---|---|
| `public-demo-key` (or none) | public |
| `employee-demo-key` | employee |
| `hr-demo-key` / `legal-demo-key` | hr / legal (+ employee) |
| `admin-demo-key` | everything; can upload, delete, view traces and start eval runs |

In production, replace these with `API_KEYS='{"key": {"user": "...", "roles": [...]}}'` or swap `get_principal` for your OIDC/JWT validator.

**API:** `POST /query`, `POST /documents` (multipart: `file`, `allowed_roles`), `GET /documents`, `GET /documents/{id}/chunks`, `DELETE /documents/{id}`, `GET /traces[/{id}]`, `GET /eval/runs`, `POST /eval/run`, `GET /health`. OpenAPI docs are at `/docs`.

### Deploy

- **Docker + pgvector:** `docker compose up --build`
- **Render (free tier):** `render.yaml` is a blueprint. Its disk is ephemeral, so the SQLite index rebuilds from `corpus/` on boot. For persistence, set `DATABASE_URL` to a free Neon or Supabase Postgres with pgvector.
- Any container host (Fly.io, Railway, Cloud Run) works with the `Dockerfile` and `PORT`.

| Env var | Default | |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/rag.db` | or `postgresql://...` |
| `LLM_PROVIDER` | `auto` | `claude` / `offline` |
| `EMBEDDER` / `RERANKER` | `hashing` / `lexical` | sentence-transformers / cross-encoder options |
| `HYBRID`, `USE_RERANK`, `ALLOW_REFINE` | on | ablation switches |
| `EVIDENCE_THRESHOLD` | `0.50` | chosen by `--sweep` |

## Layout

```
app/   main.py (API) · service.py · auth.py (RBAC) · config.py
       ingest/ loaders, cleaning, chunking, pipeline   retrieval/ embeddings, hybrid, rerank, text
       agent/ agent.py (Claude + offline policies, guardrails), prompts.py   db/ sqlite_store, pg_store
       static/index.html (UI: ask, documents, traces, eval history)
evals/ dataset.jsonl · run_eval.py · metrics.py · history.csv · results/
corpus/raw/ sample documents + manifest.json (roles)     scripts/build_corpus.py (messy PDFs)
tests/ ingestion, RBAC (incl. leak negative control), agent (incl. scripted Claude tool loop), API, eval gates, pgvector
```

## Next steps

- Run `--judge claude` against the Claude policy and add it to the history (needs an API key).
- Use a semantic embedder and cross-encoder (one env var each) and re-run the ablation.
- Load a real public corpus (labour code, arXiv) and write a second question set with a domain expert.
- Add OpenTelemetry or LangSmith export of the trace table.
