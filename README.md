# Neat-RAG

A production-ready Retrieval-Augmented Generation (RAG) system built with Pydantic AI, FastAPI, and PostgreSQL. Supports multi-provider LLMs and embeddings, hybrid search, automatic citations, multi-turn conversations, and an interactive Streamlit UI.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Directory Structure](#directory-structure)
- [Tech Stack](#tech-stack)
- [Development Phases](#development-phases)
- [Self-Service API Key Issuance](#self-service-api-key-issuance)
- [Development & Testing](#development--testing)
---

## Features

- **Document Ingestion** — PDF, DOCX, Markdown, HTML, TXT with semantic or recursive chunking
- **Multi-Provider LLMs** — OpenAI, Google Gemini, Anthropic, DeepSeek, Ollama
- **Multi-Provider Embeddings** — OpenAI, Gemini, Ollama, or any custom OpenAI-compatible endpoint
- **Hybrid Search** — Semantic vector search + BM25 full-text search with configurable weights
- **Advanced Retrieval** — HyDE query rewriting, Multi-Query decomposition, Reciprocal Rank Fusion
- **Smart Reranking** — BGE CrossEncoder (local) or Cohere API
- **Automatic Citations** — Responses include `[1][2]` references linked to source chunks
- **Multi-turn Conversations** — Session-based memory with auto-generated titles
- **Streaming Responses** — Server-Sent Events (SSE) for real-time output
- **Background Ingestion** — Async job queue with progress tracking via Redis + arq
- **API Key Auth** — Optional authentication with scoped keys and rate limiting
- **Self-Service Key Issuance** — Invite-token flow: admin generates a one-time code, user redeems it in the UI to receive their own API key — no account management system required
- **Web UI** — Interactive Streamlit interface for chat, document management, feedback, and invite management
- **Evaluation** — Built-in RAGAS metrics for retrieval and generation quality

---

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│  Streamlit  │────▶│                   FastAPI                         │
│    UI       │     │  /chat  /documents  /sessions  /jobs  /admin  /auth│
└─────────────┘     └───────────────┬──────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │          Agent / LLM           │
                    │  orchestrator → tools → memory │
                    └───────────────┬───────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
┌─────────▼──────────┐  ┌──────────▼──────────┐  ┌──────────▼──────────┐
│  Ingestion Pipeline │  │  Retrieval System   │  │  Providers          │
│  extract→chunk→     │  │  vector / hybrid /  │  │  LLM / Embedding /  │
│  embed→store        │  │  rerank / rewrite   │  │  Reranker           │
└─────────┬──────────┘  └──────────┬──────────┘  └─────────────────────┘
          │                         │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   PostgreSQL + pgvector  │
          │   HNSW index + FTS      │
          └─────────────────────────┘
          ┌─────────────────────────┐
          │   Redis + arq worker    │
          │   (background jobs)     │
          └─────────────────────────┘
```

---

## Quick Start

### Docker Compose (Recommended)

```bash
git clone <repo-url>
cd neat_rag

cp .env.example .env
# Edit .env — at minimum set your LLM and embedding provider keys

# unset DOCKER_CONTENT_TRUST  # On Mac: avoids "No such image" pull bug

# docker compose up -d --build  # first start
docker compose up -d          # follow-up start command

# Run database migrations
docker exec -it neat-rag-api alembic upgrade head
```


### First Steps

1. **Open the UI**: Navigate to `http://localhost:8501`.
2. **Upload documents**.
3. **start your chat**.

---

## Installation

### Prerequisites

- Python 3.12
- Docker & Docker Compose (for PostgreSQL, Redis)
- An API key for your chosen LLM / embedding provider

### Local Development Setup

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd neat_rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e .

# 3. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# 4. Start infrastructure services
docker compose up postgres redis -d

# 5. Run database migrations
alembic upgrade head

# 6. Start the API server
uvicorn neat_rag.api:app --reload --port 8058

# 7. Start the UI (separate terminal)
streamlit run src/neat_rag/ui.py
```

---

## Configuration

All settings are loaded from environment variables (or a `.env` file). See `.env.example` for a full reference.

### Core Settings

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8058` | API port |

### Database

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `rag` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `postgres` | Database password |

### LLM Provider

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | `openai` \| `gemini` \| `anthropic` \| `deepseek` \| `ollama` |
| `LLM_MODEL` | `deepseek-chat` | Model name (e.g. `gpt-4o-mini`, `gemini-2.5-pro`) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |

### Embedding Provider

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `gemini` \| `ollama` \| `custom` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_DIM` | `1536` | Vector dimension — must match the model |
| `EMBEDDING_BASE_URL` | — | Custom OpenAI-compatible endpoint |

> **Note:** `EMBEDDING_DIM` must be consistent. OpenAI/Gemini default to 1536; Ollama `nomic-embed-text` uses 768.

### Retrieval & Reranking

| Variable | Default | Description |
|---|---|---|
| `ENABLE_RERANK` | `true` | Enable reranking |
| `RERANKER_PROVIDER` | `cohere` | `bge` (local) \| `cohere` |
| `RERANKER_MODEL` | `rerank-multilingual-v3.0` | Reranker model name |
| `COHERE_API_KEY` | — | Cohere API key |
| `RERANK_TOP_K` | `5` | Top-K results after reranking |
| `RETRIEVE_CANDIDATE_K` | `25` | Candidate pool size before reranking |
| `ENABLE_HYDE` | `false` | Enable HyDE query expansion |
| `ENABLE_MULTI_QUERY` | `false` | Enable Multi-Query decomposition |
| `MULTI_QUERY_COUNT` | `3` | Number of sub-queries to generate |

### Security

| Variable | Default | Description |
|---|---|---|
| `ENABLE_AUTH` | `false` | Require API key authentication |
| `RATE_LIMIT_DEFAULT` | `60/minute` | Default rate limit |
| `RATE_LIMIT_CHAT` | `10/minute` | Rate limit for chat endpoints |
| `CORS_ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |

### Example Configurations

<details>
<summary>OpenAI (LLM + Embeddings)</summary>

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```
</details>

<details>
<summary>Google Gemini</summary>

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-pro
GEMINI_API_KEY=...

EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=1536
```
</details>

<details>
<summary>Fully Local (Ollama)</summary>

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
EMBEDDING_BASE_URL=http://localhost:11434
```
</details>

---

## API Reference

Full interactive documentation is available at `http://localhost:8058/docs`.

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Blocking chat — returns full response |
| `POST` | `/chat/stream` | Streaming SSE chat |

**Request body:**
```json
{
  "message": "What is RAG?",
  "session_id": "your-session-id",
  "search_type": "hybrid"
}
```

> **Note:** `session_id` is required. Create one first with `POST /sessions`, then reuse it across turns.

**Response:**
```json
{
  "content": "RAG stands for ...",
  "session_id": "abc-123",
  "citations": [
    {
      "citation_number": 1,
      "document_title": "RAG Overview",
      "content_snippet": "..."
    }
  ]
}
```

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload a file for ingestion |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}` | Get document metadata |
| `PATCH` | `/documents/{id}` | Update document metadata |
| `DELETE` | `/documents/{id}` | Delete document and its chunks |

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/jobs` | List ingestion jobs |
| `GET` | `/jobs/{id}` | Get job status and progress |

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Create a new session |
| `GET` | `/sessions` | List sessions |
| `GET` | `/sessions/{id}` | Get session metadata |
| `GET` | `/sessions/{id}/messages` | Get all messages in a session |
| `DELETE` | `/sessions/{id}` | Delete a session |

### Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/feedback` | Submit thumbs up/down on a message |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/admin/keys` | Create API key |
| `GET` | `/admin/keys` | List API keys |
| `DELETE` | `/admin/keys/{id}` | Revoke API key |
| `POST` | `/admin/invites` | Generate a one-time invite token |
| `GET` | `/admin/invites` | List all invite tokens |
| `DELETE` | `/admin/invites/{id}` | Revoke an invite token |

**Create invite request body:**
```json
{
  "owner": "alice",
  "expires_in_days": 7
}
```

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/redeem` | Exchange an invite token for a new API key (public, no auth required) |

**Redeem request body:**
```json
{ "token": "abc123..." }
```

**Response** (raw key shown once, never stored):
```json
{ "owner": "alice", "raw_key": "nrag_..." }
```

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe (checks DB) |

---

## Directory Structure

```
neat_rag/
├── src/neat_rag/
│   ├── config.py               # Settings via pydantic-settings
│   ├── models.py               # Pydantic domain models
│   ├── exceptions.py           # Custom exceptions
│   ├── logger.py               # structlog setup
│   ├── eval.py                 # RAGAS evaluation helpers
│   ├── ui.py                   # Streamlit frontend
│   │
│   ├── providers/              # External service factories
│   │   ├── llm.py              # LLM provider (OpenAI, Gemini, …)
│   │   ├── embedding.py        # Embedding provider
│   │   └── reranker.py         # Reranker (BGE / Cohere)
│   │
│   ├── ingestion/              # Document processing pipeline
│   │   ├── extractors.py       # PDF / DOCX / HTML / TXT handlers
│   │   ├── chunkers.py         # Recursive & semantic chunking
│   │   └── pipeline.py         # extract → chunk → embed → store
│   │
│   ├── retrieval/              # Information retrieval
│   │   ├── retrievers.py       # VectorRetriever & HybridRetriever
│   │   ├── rerank.py           # Reranking logic
│   │   ├── rewrite.py          # HyDE & Multi-Query rewriting
│   │   └── citation.py         # Citation extraction from responses
│   │
│   ├── agent/                  # Agentic RAG orchestration
│   │   ├── orchestrator.py     # Main run_query() entry point
│   │   ├── tools.py            # Agent tools (search, get_document, …)
│   │   ├── memory.py           # Conversation history management
│   │   ├── prompts.py          # System prompt templates
│   │   └── title.py            # Auto session title generation
│   │
│   ├── db/                     # Database layer
│   │   ├── pool.py             # asyncpg connection pool
│   │   ├── documents.py        # Document & chunk CRUD
│   │   ├── sessions.py         # Session & message CRUD
│   │   ├── jobs.py             # Ingestion job tracking
│   │   ├── feedback.py         # User feedback CRUD
│   │   ├── api_keys.py         # API key management
│   │   └── invites.py          # Invite token CRUD
│   │
│   └── api/                    # FastAPI application
│       ├── __init__.py         # App factory
│       ├── schemas.py          # Request / response DTOs
│       ├── deps.py             # FastAPI dependencies
│       ├── middleware.py       # Auth, rate limiting, request tracing
│       ├── health.py           # Health check endpoints
│       ├── documents.py        # Document & job routes
│       ├── chat.py             # Chat routes (blocking & streaming)
│       ├── sessions.py         # Session routes
│       ├── feedback.py         # Feedback routes
│       ├── admin.py            # Admin routes (keys + invites)
│       └── auth.py             # Public auth routes (redeem invite)
│
├── migrations/                 # Alembic migrations
│   ├── env.py
│   ├── versions/
│   │   ├── 001_initial_schema.py   # Tables, HNSW index, FTS
│   │   └── 002_invite_tokens.py    # invite_tokens table
│   └── script.py.mako
│
├── docker/
│   └── Dockerfile              # Multi-stage build (api / worker / ui)
│
├── docker-compose.yml
├── alembic.ini
├── pyproject.toml
├── .env.example
├── test_ingestion.py
├── test_api.py
├── test_agent.py
└── test_phase5.py
```

---

## Tech Stack

| Category | Library / Tool |
|---|---|
| **API Framework** | FastAPI 0.116, Uvicorn 0.34 |
| **AI / Agent** | Pydantic AI 0.7.6, LangChain 0.3 |
| **LLM Integrations** | langchain-openai, langchain-google-genai, Cohere |
| **Document Parsing** | Docling 2.48 (PDF), python-docx, trafilatura |
| **Embeddings** | OpenAI / Gemini / Ollama via OpenAI-compatible API |
| **Reranking** | sentence-transformers (BGE), Cohere Rerank |
| **Database** | PostgreSQL 17 + pgvector, asyncpg, Alembic |
| **Background Jobs** | Redis, arq |
| **Web UI** | Streamlit 1.49 |
| **Validation** | Pydantic 2.7, Pydantic Settings |
| **Rate Limiting** | slowapi |
| **Logging** | structlog |
| **Testing** | pytest, pytest-asyncio |
| **Evaluation** | RAGAS, datasets |
| **Containerization** | Docker, Docker Compose |

---

## Development Phases

The project was built bottom-up across 9 phases, each with a clear exit criterion:

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4
 Core         Ingest      Converse    REST API    Async jobs
 infra        pipeline    in terminal via HTTP    + progress

Phase 5 ──▶ Phase 6 ──▶ Phase 7 ──▶ Phase 8
 Advanced     Security    Infra       UI +
 retrieval    + auth      + Docker    evaluation
```

### Phase 0 — Core Infrastructure

**Goal:** A solid foundation with no external dependencies.

| File | Purpose |
|---|---|
| `config.py` | Centralized settings via `pydantic-settings` — replaces scattered `os.getenv()` calls |
| `exceptions.py` + `logger.py` | Custom exception hierarchy and `structlog` configuration |
| `models.py` | Core Pydantic DTOs: `Document`, `Chunk`, `Session`, `Message`, `SearchResult`, `Citation` |
| `db/pool.py` | asyncpg connection pool with lifecycle management |
| `db/documents.py`, `db/sessions.py`, `db/jobs.py` | Repository-pattern CRUD split by aggregate root |

### Phase 1 — Ingestion Pipeline

**Goal:** Drop a PDF in, see chunks and vectors appear in the database.

| File | Purpose |
|---|---|
| `providers/embedding.py` | Embedding provider abstraction (OpenAI / Gemini / Ollama / custom) |
| `ingestion/extractors.py` | Per-format text extraction (Docling for PDF, python-docx, trafilatura for HTML) |
| `ingestion/chunkers.py` | Recursive and semantic chunking strategies |
| `ingestion/pipeline.py` | Orchestrates extract → chunk → embed → store with atomic DB writes |

**Exit check:** `pipeline.run("doc.pdf")` → `SELECT COUNT(*) FROM chunks` returns rows.

### Phase 2 — Conversation Layer

**Goal:** Ask a question in the terminal; the agent finds the answer from ingested documents.

| File | Purpose |
|---|---|
| `retrieval/retrievers.py` | `VectorRetriever` (cosine HNSW) and `HybridRetriever` (70% semantic + 30% BM25) |
| `providers/llm.py` | LLM factory supporting OpenAI, Gemini, Anthropic, DeepSeek, Ollama |
| `agent/prompts.py` | Templated system prompt with citation formatting rules |
| `agent/tools.py` | `AgentContext` + 4 tool functions: `hybrid_search`, `vector_search`, `get_document`, `list_documents` |
| `agent/memory.py` | Bidirectional conversion between DB `Message` records and pydantic-ai `ModelMessage` objects |
| `agent/orchestrator.py` | Lazy-loaded singleton `get_agent()` + `run_query(question, session_id)` entry point |

**Key design decisions:** no circular imports (AgentContext lives in `tools.py`); lazy init so API keys aren't required at import time; search tool failures return empty lists (LLM-visible) while document tool failures raise `ToolExecutionError`.

**Exit check:** Terminal script calling `run_query()` returns a grounded answer.

### Phase 3 — REST API Layer

**Goal:** `curl` can upload a file and stream a chat response over HTTP.

| File | Purpose |
|---|---|
| `api/schemas.py` | Request / response DTOs (no business logic) |
| `api/deps.py` | FastAPI `Depends` factories for DB connection, embedder, pipeline |
| `api/health.py` | `/health/live` + `/health/ready` (real DB ping) |
| `api/documents.py` | Upload, list, get, patch, delete documents; list/get jobs |
| `api/chat.py` | Blocking `/chat` and SSE streaming `/chat/stream` |
| `api/sessions.py` | Session CRUD + message history endpoint |
| `api/feedback.py` | `POST /feedback` for thumbs up / down |
| `api/__init__.py` | `create_app()` factory — lifespan, CORS, request-id middleware, exception handlers, router registration |

**Exit check:** `curl -X POST /documents/upload -F file=@doc.pdf` succeeds; streaming chat returns SSE chunks.

### Phase 4 — Async Job Progress

**Goal:** Upload a large file and watch the progress field increment from 0 to 1.

| Change | Detail |
|---|---|
| `ingestion/pipeline.py` | Calls `job_repo.update_progress()` at each pipeline stage |
| `api/documents.py` | Runs pipeline via FastAPI `BackgroundTasks`; returns `job_id` immediately |

> **Note:** The interface for job_id had been pre-wired in Phase 1–3. Phase 4 was mainly filling in the progress update calls — the "upload → poll progress" loop was already functionally closed.

**Exit check:** Upload PDF → receive `job_id` → poll `GET /jobs/{id}` and observe `progress` increasing.

### Phase 5 — Advanced Retrieval

**Goal:** Answers improve noticeably; responses contain `[1][2]` citation markers.

| File | Purpose |
|---|---|
| `providers/reranker.py` | `CrossEncoderReranker` (local BGE) and `CohereReranker` with shared factory |
| `retrieval/rerank.py` | `rerank_hits()` — scores and filters the candidate pool |
| `retrieval/rewrite.py` | `hyde_rewrite()`, `multi_query_rewrite()`, `rrf_merge()` (Reciprocal Rank Fusion) |
| `retrieval/citation.py` | `build_citation_context()` + `extract_citations()` — matches `[N]` markers to source chunks |
| Updated `agent/tools.py` | Advanced search pipeline: rewrite → multi-retrieve → RRF merge → rerank → citation format |
| Updated `agent/prompts.py` | Strict citation rules: every factual claim must have an inline `[N]` tag |
| Updated `agent/orchestrator.py` | Injects reranker; `run_query()` now returns a `citations` list |
| Updated `api/chat.py` + `schemas.py` | Both `/chat` and `/chat/stream` return a `citations` array in the response |

**Exit check:** Compare answer quality before/after; `[1][2]` references appear and map to real chunks.

### Phase 6 — Security & Middleware

**Goal:** Unauthenticated requests return 401; rate-exceeded requests return 429; every request carries a trace ID.

| File | Purpose |
|---|---|
| `api/middleware.py` | `slowapi` rate limiter, `APIKeyHeader` scheme, `verify_api_key()` dependency, key hashing utilities |
| `api/admin.py` | `POST/GET/DELETE /admin/keys` — full API key lifecycle |
| Updated `api/__init__.py` | Registers `SlowAPIMiddleware` and `RateLimitExceeded` handler |
| Updated `api/chat.py` | Chat endpoints get `@limiter.limit()` + `Depends(verify_api_key)` |
| `db/api_keys.py` | `ApiKeyRepository`: create, lookup by hash, touch `last_used_at`, delete, list |

**Modes:** `ENABLE_AUTH=false` (development) — all requests pass through. `ENABLE_AUTH=true` (production) — `X-API-Key: nrag_...` header required.

**Exit check:** Request without key → 401; exceed rate limit → 429.

### Phase 7 — Infrastructure

**Goal:** Fully containerized; database schema is version-controlled and reproducible.

| File | Purpose |
|---|---|
| `migrations/versions/001_initial_schema.py` | All 7 tables + HNSW index (replaces IVFFlat) + GIN full-text index; `EMBEDDING_DIM` read from settings |
| `migrations/env.py` | Connects pydantic Settings → SQLAlchemy; supports offline and online migration modes |
| `docker/Dockerfile` | Multi-stage build: `builder` (uv install) → `runtime` → `api` / `worker` / `ui` targets |
| `docker-compose.yml` | Five services (postgres, redis, api, worker, ui) each with `healthcheck` and correct `depends_on` ordering |
| `.dockerignore` | Excludes `.venv/`, `documents/`, `.env`, test caches |

**Exit check:** `docker compose up` starts all services; `alembic upgrade head` applies cleanly to a fresh database.

### Phase 8 — UI & Evaluation

**Goal:** An interactive web interface and quantitative quality metrics.

| File | Purpose |
|---|---|
| `ui.py` | Streamlit frontend — dark theme, session sidebar, document library, job monitor, streaming chat, citations expander, thumbs up/down feedback |
| `eval.py` | RAGAS evaluation: Faithfulness, Answer Relevancy, Context Precision |

**UI highlights vs. a plain Streamlit app:**

---

## Self-Service API Key Issuance

Users can obtain their own API keys through an invite-token flow — no account management system or email service required.

### How it works

| Step | Who | Action |
|------|-----|--------|
| 1 | Admin | Opens the **Admin** panel in the UI (or calls `POST /admin/invites`) → generates a one-time token with an owner name and expiry |
| 2 | Admin | Shares the token with the user via any channel (Slack, email, etc.) |
| 3 | User | Opens the UI → expands **Redeem invite code** in the Settings sidebar → pastes the token → clicks **Get API Key** |
| 4 | System | Validates the token, issues a new API key, invalidates the token immediately |
| 5 | User | Copies the key (shown once), pastes it into the **API Key** field — done |

### Bootstrapping your first admin key

When `ENABLE_AUTH=false` (the development default), the `/admin/*` endpoints require no authentication. Create your first key with:

```bash
curl -X POST http://localhost:8058/admin/keys \
  -H "Content-Type: application/json" \
  -d '{"owner": "admin"}'
# → { "raw_key": "nrag_...", ... }  — save this, it won't be shown again
```

Then set `ENABLE_AUTH=true` in `.env` before going to production.

### New files

| File | Purpose |
|------|---------|
| `db/invites.py` | `InviteTokenRepository` — create, lookup, mark used, delete, list |
| `api/auth.py` | `POST /auth/redeem` — public endpoint, no auth required |
| `api/admin.py` | Extended with `POST/GET/DELETE /admin/invites` |
| `migrations/002_invite_tokens.py` | Adds the `invite_tokens` table |

---

## Development & Testing

### Running Tests

```bash
# Ingestion pipeline
pytest test_ingestion.py -v

# API endpoints
pytest test_api.py -v

# Agent orchestration
pytest test_agent.py -v

# Advanced retrieval (rerank, citations)
pytest test_phase5.py -v
```

### Database Migrations

```bash
# Apply all migrations
alembic upgrade head

# Create a new migration
alembic revision --autogenerate -m "description"

# Rollback one step
alembic downgrade -1
```

### Evaluation

```bash
# Run with built-in demo questions (requires indexed documents)
python -m neat_rag.eval --demo

# Run against a custom question file
python -m neat_rag.eval --questions eval/questions.json --output report.json

# Against a deployed instance with auth
python -m neat_rag.eval --questions qa.json --api-url http://prod:8058 --api-key nrag_...
```

Metrics reported: **Faithfulness**, **Answer Relevancy**, **Context Precision** (via RAGAS).

### Docker Build Targets

The `docker/Dockerfile` uses build targets to produce separate images:

```bash
# API server
docker build -f docker/Dockerfile --target api -t neat-rag-api .

# Background worker
docker build -f docker/Dockerfile --target worker -t neat-rag-worker .

# Streamlit UI
docker build -f docker/Dockerfile --target ui -t neat-rag-ui .
```
