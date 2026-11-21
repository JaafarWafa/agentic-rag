# Agentic RAG with an evaluation harness

A document Q&A system that **decides when to search**, **re-searches with a refined query** when the evidence is weak, **re-ranks**, **cites every claim**, **refuses** when the answer isn't in the corpus, and enforces **role-based access control** inside the retrieval query. It ships with an **82-question evaluation harness** that measures its own hallucination rate on every commit.

![Demo: grounded answer, role-based refusal, same question answered for HR](docs/demo.gif)

*Employee asks about maternity leave and gets a cited answer after one query refinement. The same employee asks about salary bands and is refused, because HR documents are invisible to that role. Switched to HR, the same question is answered. ([MP4 version](docs/demo.mp4))*

## Results

The offline configuration needs no API key. It was measured on 82 questions: 54 answerable, 5 paraphrased, 12 unanswerable and 11 RBAC-restricted.

Each row of the ablation ladder changes one component (`python -m evals.run_eval --ablation`):

| Run | Accuracy | MRR | Citation precision | **Hallucination rate** | False refusals | Correct refusals | RBAC leaks |
|---|---|---|---|---|---|---|---|
| v1 dense only, always answers | 94.9% | 0.953 | 85.6% | **19.5%** | 1.7% | 30.4% | 0 |
| v2 + calibrated refusal threshold | 93.2% | 0.953 | 94.6% | **0.0%** | 5.1% | 100% | 0 |
| v3 + BM25 (hybrid, RRF) | 94.9% | 0.952 | 95.2% | 0.0% | 5.1% | 100% | 0 |
| v4 + re-ranking | 94.9% | **0.989** | 96.2% | 0.0% | 3.4% | 100% | 0 |
| **v5 + agentic query refinement** | **96.6%** | 0.989 | 96.3% | **0.0%** | **1.7%** | 100% | 0 |

On the **held-out test split** (27 questions never used for tuning), v5 scores 100% accuracy, 0% hallucination and 0 false refusals. Median latency is about 5 ms per question offline.

What these numbers show:
- **Naive RAG hallucinates through neighbouring documents.** In v1, an employee asking about HR-only salary bands didn't see the HR document (RBAC held). Instead they got a confident answer stitched together from the travel policy. RBAC alone doesn't prevent hallucination, but the refusal gate does.
- **The refusal threshold costs some recall,** and the later components win it back.
- **The threshold was not hand-picked.** `--sweep` chooses it on the dev split by minimising `3 × hallucinations + false refusals`. An invented policy answer costs more than "I don't know".

### What the eval caught during development

These were real bugs, found because the harness listed the failing questions:

| Symptom in the eval | Root cause | Fix |
|---|---|---|
| "What is the name of the CEO?" answered with a sentence about *names of attendees* | The refined query (`"what name"`) was graded instead of the user's question | Evidence is always graded against the **original question** (`grade_against`) |
| "Parking policy at the Tangier hub" answered from a travel paragraph | "parking" and "Tangier" were in different sentences, but the score was per chunk | **Co-occurrence window** scoring (2 sentences + section title) |
| "How much PTO do I get?" matched *flying time* | The synonym phrase "paid time off" was satisfied by the single word "time" | Synonym phrases must match **as a whole** |
| "Salary range for Band **C**" returned Band B's bonus | Single-letter tokens were dropped | Single-character terms kept |
| 10% false refusals | "how far *in advance*" and "*minimum* length" had full weight and no counterpart in the passages | Intent words get a small fixed weight |

The figures went from 88.1% accuracy and 3.7% hallucination on the first run to the table above.

## Architecture

```
            upload (pdf/html/md/txt)                      POST /query  (X-API-Key -> roles)
                     │                                          │
   load → clean → structure-aware chunk → embed          ┌──────▼───────┐
   (headers/footers, de-hyphenation, boilerplate,         │  Agent loop  │ route: small talk → answer directly
    dedupe)          │                                    │              │ search → grade evidence ─┐
                     ▼                                    │              │   weak? refine query ◄───┘ (≤3 rounds)
   ┌───────────── Store (SQLite FTS5 | Postgres+pgvector) ┤              │   still weak? → refuse
   │ documents(allowed_roles) · chunks(embedding, tsv)    │              │ answer with [cN] citations
   │ traces · eval_runs                                   └──────┬───────┘
   └──────────────────────────────────────────────────────────── │ guardrails: citations must reference passages
        dense top-20 ─┐  role filter INSIDE both queries          │ retrieved for THIS user; uncited → refusal
        BM25 top-20 ──┴─► RRF fusion ─► re-rank ─► top-5         ▼
                                                           trace (every step) → traces table → /traces UI
```

| Layer | Implementation |
|---|---|
| Ingestion | `app/ingest/`: PDF (pypdf), HTML (strips nav, footer, cookie banners, scripts), Markdown, TXT. Removes running headers/footers, page numbers, line-break hyphens and duplicated paragraphs. Chunks along heading paths (`Handbook > 3. Sick Leave`) with whole sentences and 1-sentence overlap. Idempotent by SHA-256. |
| Dense retrieval | `EMBEDDER=hashing` (default) is a stateless, zero-download hashed word/bigram/char-4-gram embedding. `EMBEDDER=sentence-transformers:BAAI/bge-small-en-v1.5` gives real semantic recall. |
| Keyword retrieval | SQLite FTS5 `bm25()` with the section title weighted 2×, or Postgres `tsvector` + GIN + `ts_rank_cd`. |
| Fusion + rerank | Reciprocal Rank Fusion (k=60), then a re-ranker scoring in [0,1]: `lexical` (IDF coverage + co-occurrence window + phrase + section match) or `cross-encoder:<model>`. |
| Agent | `app/agent/agent.py`. **ClaudePolicy** uses Claude tool use (`search_corpus`, `submit_answer`, strict schemas): the model writes queries, judges evidence and decides when to stop, within a search budget. **OfflinePolicy** is a deterministic route → search → grade → refine → answer/refuse state machine with an extractive answerer. Both share the guardrails. |
| Access control | Roles `public < employee < {hr, legal} < admin`. The filter sits in the SQL `WHERE` of both retrievers (pre-filter), so restricted chunks are never scored, never reach the LLM and can't be cited. Out-of-scope documents return 404, not 403. |
| Observability | Every request writes a trace (route decision, each query, per-stage scores, grade verdicts, guardrail hits, token usage, latency) to the `traces` table. Logs are JSON. |
| Evaluation | `evals/`: 82 questions, dev/test split, metrics, threshold sweep, ablations, CI quality gates, and a run history in `evals/history.csv`, `evals/results/*.json` and the UI chart. |

## Evaluation harness

```bash
python -m evals.run_eval --label my-change          # run, compare against the previous run, save history
python -m evals.run_eval --ablation                 # the 5-step ladder above
python -m evals.run_eval --sweep                    # tune the refusal threshold on dev only
python -m evals.run_eval --gate                     # exit 1 if a quality gate fails (used in CI)
python -m evals.run_eval --set HYBRID=0 --label x   # any setting can be overridden
python -m evals.run_eval --judge claude             # LLM-graded groundedness/relevance
```

Each question in `evals/dataset.jsonl` has a `role`, `category`, `expected_docs`, `must_include` facts (with `|` alternatives) and a `split`. The harness computes:

| Metric | Definition |
|---|---|
| answer_accuracy | Share of required facts present in the answer (answerable + paraphrase). |
| retrieval_hit_rate, mrr | Whether an expected document was retrieved at all, and how high it ranked in the first search. |
| citation_precision | Share of cited passages that come from an expected document. |
| groundedness | Share of answer claims supported by their **cited** passage: ≥80% term overlap and **every number must match**. `--judge claude` replaces this with per-claim LLM grading. |
| **hallucination_rate** | Share of all questions where the system answered and either the question had no answer in the user's corpus, or a claim was unsupported. |
| false / correct refusal rate | Refusals on answerable questions vs on unanswerable and restricted ones. |
| rbac_leak_rate | Any out-of-role document in retrieval or citations, or the restricted fact appearing in the answer. A negative-control test disables the role filter to prove the detector fires. |

CI gates: leak rate = 0, hallucination ≤ 5%, accuracy ≥ 80%, correct refusals ≥ 90%.

**Honest caveats.** The corpus (11 documents for a fictional company, `corpus/`) and the questions were written for this project. The thesaurus used for query refinement is hand-built, and its entries were partly motivated by eval failures. That's why the threshold was tuned on dev only and the test split is reported separately. The heuristic judge suits the extractive answerer. For Claude's paraphrased answers, use `--judge claude`. To make the benchmark harder, swap in a real corpus (for example the Moroccan Labour Code or your own policy PDFs) with `corpus/raw/manifest.json` and write questions for it.

## Running it

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
