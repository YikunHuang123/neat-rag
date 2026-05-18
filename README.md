# 🔍 Neat-RAG

**A production-ready Retrieval-Augmented Generation (RAG) backend** that turns private document collections — including **PDF, DOCX, Markdown, HTML, TXT, and images** — into a queryable knowledge base. It offers ultimate flexibility by supporting both **high-performance cloud LLMs** (OpenAI, Gemini, Anthropic, DeepSeek) and **fully private, offline local models** via Ollama. Features include hybrid search, agentic tool routing, automatic inline citations, and a self-service API-key onboarding flow.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-Agent-E92063)](https://ai.pydantic.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo & Visuals](#-demo--visuals)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License & Contact](#-license--contact)

---

## ✨ Features

- **Multi-Format Document Ingestion** — PDF, DOCX, Markdown, HTML, TXT, and images (with Tesseract OCR + VLM descriptions); documents are processed asynchronously via a Redis/arq background worker
- **Pluggable LLM Providers** — OpenAI, Google Gemini, Anthropic, DeepSeek, and Ollama, all switchable via a single environment variable
- **Pluggable Embedding Providers** — OpenAI, Gemini, Ollama, or any OpenAI-compatible endpoint; embedding dimension is fully configurable
- **Hybrid Search** — Combines dense vector retrieval (cosine similarity) with sparse BM25 full-text search; weights are tunable at runtime
- **Agentic RAG Orchestration** — A Pydantic AI agent selects among `hybrid_search`, `vector_search`, `get_document`, and `list_documents` tools to answer multi-hop questions
- **Retrieval Techniques** — HyDE query rewriting, Multi-Query decomposition, and Reciprocal Rank Fusion (RRF) for improved recall
- **Smart Reranking** — Local BGE CrossEncoder or Cohere API reranking to boost precision before the LLM call
- **Automatic Inline Citations** — Every answer includes `[1][2]` references with direct links back to source chunks and documents
- **Streaming & Blocking Responses** — SSE streaming (`/chat/stream`) and synchronous (`/chat`) endpoints; both work with conversation memory
- **Multi-Turn Session Memory** — Session-scoped conversation history with auto-generated titles; full CRUD management via REST API
- **Dual Vector Store Backends** — Switch between an integrated **pgvector** (PostgreSQL) backend and a dedicated **Qdrant** instance with zero code changes
- **Background Ingestion Jobs** — Job status and progress tracking via `/jobs/{id}`; the worker runs independently for non-blocking uploads
- **Self-Service API Key Issuance** — Admins generate single-use invite tokens; users redeem them for personal API keys without manual provisioning
- **Role-Based Access Control** — `admin`-scoped keys unlock `/admin/*` routes; regular keys are rate-limited via slowapi
- **RAGAS Evaluation** — Built-in evaluation helpers for Faithfulness, Answer Relevancy, and Context Precision metrics
- **Streamlit UI** — Chat interface with a document library. Admin interface, user management interface, job monitor, and session sidebar

---

## 🎬 Demo & Visuals

### System Architecture

```
┌─────────────────┐     REST / SSE      ┌──────────────────────────────────────────┐
│   Streamlit UI  │ ─────────────────▶  │              FastAPI (port 8058)          │
│   (port 8501)   │                      │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
└─────────────────┘                      │  │  /chat   │  │  /docs   │  │ /admin │ │
                                         │  └────┬─────┘  └────┬─────┘  └────┬───┘ │
                                         │       │              │              │      │
                                         │  ┌────▼──────────────▼──────────────▼───┐ │
                                         │  │          Agent Orchestrator           │ │
                                         │  │  (Pydantic AI + tool routing)         │ │
                                         │  └──────┬─────────────────────┬──────────┘ │
                                         │  ┌──────▼──────┐    ┌─────────▼──────────┐ │
                                         │  │  Retrieval   │    │   LLM Provider     │ │
                                         │  │  · Hybrid    │    │  OpenAI / Gemini   │ │
                                         │  │  · HyDE/MQ   │    │  Anthropic / Ollama│ │
                                         │  │  · Reranker  │    └────────────────────┘ │
                                         │  └──────┬───────┘                           │
                                         └─────────│─────────────────────────────────  ┘
                                                   │
                          ┌────────────────────────┼──────────────────────────────┐
                          │                         │                              │
                   ┌──────▼──────┐          ┌───────▼──────┐             ┌────────▼──────┐
                   │  PostgreSQL  │          │    Qdrant     │             │     Redis      │
                   │  (pgvector)  │          │ (vector store)│             │  (job queue)   │
                   └─────────────┘          └──────────────┘             └───────────────┘
```

### API Key Onboarding Flow

```
Admin                          New User
  │                               │
  │── POST /admin/invites ──▶ [invite token]
  │                               │
  │              ◀── POST /auth/redeem ──── {token}
  │                               │
  │              ──── {api_key} ──▶
  │                               │
  │                        Uses X-API-Key header
```

> **Screenshots coming soon.** The `docs/` folder will contain terminal session recordings and UI screenshots once uploaded.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **API Framework** | FastAPI 0.115 + Uvicorn |
| **Agent Orchestration** | Pydantic AI (tool-calling agent) |
| **LLM Providers** | OpenAI, Google Gemini, Anthropic, DeepSeek, Ollama |
| **Embedding Providers** | OpenAI, Gemini, Ollama, custom OpenAI-compatible |
| **Vector Store (integrated)** | PostgreSQL 17 + pgvector (HNSW index) |
| **Vector Store (dedicated)** | Qdrant |
| **Relational Database** | PostgreSQL 17 via asyncpg |
| **Full-Text Search** | PostgreSQL `tsvector` / BM25 |
| **Schema Validation** | Pydantic v2 + pydantic-settings |
| **DB Migrations** | Alembic |
| **Background Jobs** | arq + Redis |
| **Document Parsing** | docling (PDF/DOCX/HTML), python-docx, BeautifulSoup |
| **OCR** | Tesseract (10+ languages) + OpenCV |
| **Reranking** | sentence-transformers (BGE CrossEncoder), Cohere API |
| **Chunking** | LangChain RecursiveCharacterTextSplitter + semantic chunker |
| **Rate Limiting** | slowapi |
| **Evaluation** | RAGAS |
| **UI** | Streamlit |
| **Containerisation** | Docker + Docker Compose (multi-stage build) |

---

## 🚀 Installation

### Option A — Docker Compose (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine + Compose v2.

**1. Clone the repository**

```bash
git clone https://github.com/YikunHuang123/neat_rag.git
cd neat_rag
```

**2. Create your environment file**

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum — choose **one** of the following LLM setups:

**Cloud provider (OpenAI / Gemini / Anthropic / DeepSeek)**

```bash
# Pick one key that matches your chosen LLM_MODEL
GEMINI_API_KEY=...
# DEEPSEEK_API_KEY=...
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
# ANTHROPIC_API_KEY=...

LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash

EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001

# Admin bootstrap key — any secret string you choose
ADMIN_BOOTSTRAP_KEY=your-very-secret-admin-key
```

**Local models via Ollama (no API key required)**

```bash
# Point the app at your local Ollama instance
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2          # or any model you have pulled
LLM_BASE_URL=http://host.docker.internal:11434   # from inside Docker
# LLM_BASE_URL=http://localhost:11434            # for local dev (Option B)

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
# EMBEDDING_BASE_URL=http://host.docker.internal:11434

# Admin bootstrap key — any secret string you choose
ADMIN_BOOTSTRAP_KEY=your-very-secret-admin-key
```

> Make sure Ollama is running and you have pulled the required models:
> ```bash
> ollama pull llama3.2
> ollama pull nomic-embed-text
> ```

**3. Build and start all services**

```bash
docker compose up --build
```

This starts six containers: `postgres`, `qdrant`, `redis`, `api` (port **8058**), `worker`, and `ui` (port **8501**).

**4. Run database migrations**

```bash
docker compose exec api alembic upgrade head
```

**5. Open the UI**

Navigate to `http://localhost:8501` in your browser.
The interactive API docs are at `http://localhost:8058/docs`.

---

### Option B — Local Development

**Prerequisites:** [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda / Anaconda), PostgreSQL 17 with pgvector, Redis, (optionally) Qdrant.

**1. Clone the repository**

```bash
git clone https://github.com/YikunHuang123/neat_rag.git
cd neat_rag
```

**2. Create and activate a Conda environment**

```bash
conda create -n neat_rag python=3.12 -y
conda activate neat_rag
```

**3. Install dependencies**

```bash
pip install -e ".[dev]"
```

**4. Configure environment**

```bash
cp .env.example .env
# Edit .env — see the LLM setup options in Option A above
# At minimum set: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, ADMIN_BOOTSTRAP_KEY
```

**5. Apply database migrations**

```bash
alembic upgrade head
```

**6. Start the API server**

```bash
uvicorn neat_rag.api:create_app --factory --host 0.0.0.0 --port 8058 --reload
```

**7. Start the background worker** (separate terminal)

```bash
arq neat_rag.worker.WorkerSettings
```

**8. Start the Streamlit UI** (separate terminal, optional)

```bash
streamlit run src/neat_rag/ui.py
```

---

## 💡 Usage

### Uploading a document

```bash
curl -X POST http://localhost:8058/documents/upload \
  -H "X-API-Key: admin" \   # Defined as ADMIN_BOOTSTRAP_KEY in .env
  -F "file=@/path/to/report.pdf"
```

```json
{
  "job_id": "3f2a1b...",
  "document_id": "d9e8c7...",
  "status": "pending"
}
```

### Checking ingestion progress

```bash
curl http://localhost:8058/jobs/3f2a1b... \
  -H "X-API-Key: admin"   # Defined as ADMIN_BOOTSTRAP_KEY in .env
```

```json
{
  "job_id": "3f2a1b...",
  "status": "completed",
  "progress": 100,
  "chunks_created": 42
}
```

### Asking a question (blocking)

```bash
curl -X POST http://localhost:8058/chat \
  -H "X-API-Key: admin" \   # Defined as ADMIN_BOOTSTRAP_KEY in .env
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings in the report?", "session_id": null}'
```

```json
{
  "answer": "The report highlights three main findings: ... [1][2]",
  "citations": [
    {"index": 1, "document_id": "d9e8c7...", "chunk_id": "...", "excerpt": "Revenue grew 12%..."},
    {"index": 2, "document_id": "d9e8c7...", "chunk_id": "...", "excerpt": "Operating margin..."}
  ],
  "session_id": "s1a2b3..."
}
```

### Streaming response (SSE)

```bash
curl -N -X POST http://localhost:8058/chat/stream \
  -H "X-API-Key: admin" \   # Defined as ADMIN_BOOTSTRAP_KEY in .env
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the methodology section.", "session_id": "s1a2b3..."}'
```

```
data: {"delta": "The methodology"}
data: {"delta": " section describes"}
data: {"delta": " a two-stage process..."}
data: {"done": true, "citations": [...]}
```

### Issuing an invite token (admin)

```bash
# 1. Admin creates a single-use invite
curl -X POST http://localhost:8058/admin/invites \
  -H "X-API-Key: admin"    # Defined as ADMIN_BOOTSTRAP_KEY in .env

# {"token": "inv_abc123...", "expires_at": "2026-06-18T00:00:00Z"}

# 2. New user redeems it for their own API key
curl -X POST http://localhost:8058/auth/redeem \
  -H "Content-Type: application/json" \
  -d '{"token": "inv_abc123..."}'

# {"api_key": "rak_xyz789...", "created_at": "..."}
```

### Running the test suite

```bash
pytest test_ingestion.py test_agent.py test_phase5.py -v
```

---

## 🗂 Project Structure

```
neat_rag/
├── docker-compose.yml          # Six-service stack (postgres, qdrant, redis, api, worker, ui)
├── docker/
│   └── Dockerfile              # Multi-stage build — targets: api | worker | ui
├── alembic.ini                 # Alembic migration configuration
├── migrations/
│   └── versions/
│       ├── 001_initial_schema.py      # Core tables + HNSW index
│       ├── 002_add_user_id.py
│       ├── 003_invite_tokens.py
│       ├── 004_migrate_to_qdrant.py
│       └── 005_add_user_permissions.py
├── src/neat_rag/
│   ├── config.py               # Pydantic Settings — all env vars with defaults
│   ├── models.py               # Domain models: Document, Chunk, Session, Message, Job…
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── eval.py                 # RAGAS evaluation helpers
│   ├── ui.py                   # Streamlit frontend
│   ├── providers/
│   │   ├── llm.py              # LLM factory (OpenAI / Gemini / Anthropic / DeepSeek / Ollama)
│   │   ├── embedding.py        # Embedding provider factory
│   │   └── reranker.py         # BGE CrossEncoder + Cohere reranker
│   ├── ingestion/
│   │   ├── extractors.py       # PDF / DOCX / HTML / image text extraction
│   │   ├── chunkers.py         # Recursive and semantic chunking strategies
│   │   └── pipeline.py         # Extract → chunk → embed → store orchestration
│   ├── retrieval/
│   │   ├── retrievers.py       # VectorRetriever and HybridRetriever
│   │   ├── rerank.py           # Post-retrieval reranking
│   │   ├── rewrite.py          # HyDE, Multi-Query, RRF fusion
│   │   └── citation.py         # Inline citation extraction
│   ├── agent/
│   │   ├── orchestrator.py     # run_query() — main public entry point
│   │   ├── tools.py            # Agent tools + AgentContext (dependency injection)
│   │   ├── memory.py           # Conversation history management
│   │   ├── prompts.py          # System prompt templates
│   │   └── title.py            # Auto-generated session titles
│   ├── db/
│   │   ├── pool.py             # asyncpg connection pool
│   │   ├── vector_store.py     # Abstract VectorStoreBase + backend factory
│   │   ├── pgvector_store.py   # PostgreSQL + pgvector implementation
│   │   ├── qdrant.py           # Qdrant implementation
│   │   ├── documents.py        # Document & chunk CRUD
│   │   ├── sessions.py         # Session & message CRUD
│   │   ├── jobs.py             # Ingestion job tracking
│   │   ├── feedback.py         # User feedback CRUD
│   │   ├── api_keys.py         # API key management
│   │   └── invites.py          # Invite token CRUD
│   └── api/
│       ├── __init__.py         # FastAPI app factory with lifespan management
│       ├── schemas.py          # Request / response DTOs
│       ├── deps.py             # FastAPI Depends factories
│       ├── middleware.py       # Auth, rate limiting, request tracing
│       ├── health.py           # /health/live  &  /health/ready
│       ├── documents.py        # Document & job routes
│       ├── chat.py             # Blocking and SSE streaming chat
│       ├── sessions.py         # Session CRUD + message history
│       ├── feedback.py         # User feedback routes
│       ├── admin.py            # API key + invite management
│       └── auth.py             # Public redeem-invite endpoint
├── test_ingestion.py           # Phase 1 — ingestion pipeline tests
├── test_agent.py               # Phase 2 — agent orchestration tests
└── test_phase5.py              # Phase 5 — advanced retrieval tests
```

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome.

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** — follow the existing code style (Ruff / Black formatting).

3. **Add or update tests** for any new behaviour:
   ```bash
   pytest -v
   ```

4. **Commit** with a descriptive message:
   ```bash
   git commit -m "feat: add XYZ retrieval strategy"
   ```

5. **Open a Pull Request** against `main`. Include:
   - A clear description of the problem solved
   - Steps to reproduce (for bug fixes)
   - Any relevant environment or config changes

**Reporting bugs:** Please open a GitHub Issue with the label `bug`, your Python version, and a minimal reproduction snippet.

---

## 📄 License & Contact

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Author:** Yikun Huang
**Email:** yikun.huang@hotmail.com
**GitHub:** [@YikunHuang123](https://github.com/YikunHuang123)

> Built as a full-stack RAG engineering showcase — covering async API design, agentic LLM orchestration, hybrid retrieval, and production-grade containerisation.
