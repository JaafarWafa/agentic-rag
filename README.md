# Agentic RAG with self-evaluation

[![ci](https://github.com/JaafarWafa/agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/JaafarWafa/agentic-rag/actions)

A document Q&A assistant for company policies. It **searches, checks its evidence, re-searches when needed, answers with cited sources**, and **says "I don't know"** when the answer isn't in the documents. Each user only sees the documents their role allows (employee, HR, legal…).

It comes with an **evaluation harness** of 82 test questions that measures the system's own hallucination rate on every commit.

## Demo

![Demo](docs/demo.gif)

1. An employee asks about maternity leave. The first search is weak, so the agent rewrites the query and answers with a source.
2. The same employee asks about salary bands and is refused: HR documents are invisible to that role.
3. Switched to the HR role, the same question is answered.

## Results

Each step adds one component and is measured on the same 82 questions:

| Version | Accuracy | Hallucination rate | False refusals |
|---|---|---|---|
| Basic RAG (vector search, always answers) | 94.9% | **19.5%** | 1.7% |
| + refuses when evidence is weak | 93.2% | **0%** | 5.1% |
| + keyword search (hybrid) | 94.9% | 0% | 5.1% |
| + re-ranking | 94.9% | 0% | 3.4% |
| **+ query refinement (final)** | **96.6%** | **0%** | **1.7%** |

- On 27 **held-out** questions never used for tuning: 100% accuracy, 0% hallucination.
- **0 access-control leaks**: restricted documents never reach the model.

![Eval history](docs/evals.png)

## Architecture

```mermaid
flowchart LR
  subgraph ING["① Ingestion"]
    direction TB
    U["📄 Upload<br/>PDF · HTML · MD · TXT"] --> CL["🧹 Clean<br/>headers, footers, boilerplate"]
    CL --> CH["✂️ Chunk by section"]
    CH --> EM["🔢 Embed"]
  end

  EM --> DB[("🗄️ Vector + keyword store<br/>SQLite or Postgres/pgvector")]

  subgraph AG["② Agent loop"]
    direction TB
    REQ["💬 Question<br/>+ user role"] --> S["🔍 Hybrid search<br/>dense + BM25 → re-rank"]
    S --> G{"Enough<br/>evidence?"}
    G -- "no" --> RF["✏️ Refine query<br/>(max 3 rounds)"]
    RF --> S
    G -- "yes" --> A["✅ Answer<br/>with citations"]
    G -- "still no" --> X["🚫 Refuse"]
  end

  DB -. "🔒 only documents<br/>this role may see" .-> S
  A --> GR["🛡️ Guardrail<br/>citations verified"]
  GR --> T["📊 Trace + eval<br/>dashboard"]
  X --> T

  classDef ing fill:#e8f0fe,stroke:#4a6fd1,color:#1c2b4a
  classDef store fill:#fff4e0,stroke:#d99a2b,color:#4a3510
  classDef agent fill:#eaf7ef,stroke:#2f9e5e,color:#173a26
  classDef good fill:#d8f3e3,stroke:#2f9e5e,color:#173a26
  classDef bad fill:#fde2e2,stroke:#d04a4a,color:#4a1717
  classDef obs fill:#f1ecfb,stroke:#7a5cc9,color:#2c1f4d
  class U,CL,CH,EM ing
  class DB store
  class REQ,S,G,RF agent
  class A,GR good
  class X bad
  class T obs
  style ING fill:#f7f9ff,stroke:#c9d6f5
  style AG fill:#f6fcf8,stroke:#bfe6cf
```

## Tech stack

| Area | Tools |
|---|---|
| Backend | Python, FastAPI |
| Retrieval | Hybrid search (vector + BM25), Reciprocal Rank Fusion, re-ranking |
| Storage | PostgreSQL + pgvector, or SQLite (FTS5) |
| LLM agent | Claude via tool use, or an offline mode that needs no API key |
| Access control | Role-based filtering inside the database query |
| Evaluation | Custom harness: groundedness, hallucination rate, refusals, retrieval MRR, leak detection |
| Ops | Docker, docker-compose, GitHub Actions CI with quality gates, Render deployment |

## How it's evaluated

The test set has 82 questions in four types: **answerable**, **reworded** (different words from the documents), **unanswerable**, and **restricted** (the answer exists but the user's role can't see it).

Every run measures accuracy, hallucination rate, correct and false refusals, citation precision and access-control leaks, and is saved so progress can be tracked over time. CI fails the build if the hallucination rate goes above 5% or any leak appears.

## Run it

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000      # open http://localhost:8000
python -m evals.run_eval              # run the evaluation
```

Setup, configuration, API and deployment details: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
