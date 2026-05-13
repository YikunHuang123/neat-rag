#!/usr/bin/env python3
"""
Phase 1 Ingestion Pipeline Test
================================
Tests the full extract → chunk → embed → store pipeline.

Usage:
    python test_ingestion.py --provider openai    # default
    python test_ingestion.py --provider gemini
    python test_ingestion.py --provider ollama

Prerequisites:
    1. PostgreSQL running (docker-compose up postgres)
    2. .env file configured (copy .env.example, fill in keys)
    3. For ollama: run setup instructions below

Ollama setup (one-time):
    # Mac
    brew install ollama
    ollama pull nomic-embed-text   # 274MB, 768-dim, recommended
    ollama serve                   # keep running in background

    # Or via Docker
    docker run -d -p 11434:11434 ollama/ollama
    docker exec <container> ollama pull nomic-embed-text
"""

import argparse
import asyncio
import sys
import os
import tempfile
from pathlib import Path

# ── path setup (src layout) ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ── test document (plain text, no PDF/docling needed for quick test) ─────────
SAMPLE_TEXT = """\
Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that enhances large language
models by providing them with relevant context retrieved from an external knowledge
base. Instead of relying solely on the model's training data, RAG systems first
retrieve relevant document chunks from a vector database, then pass those chunks
alongside the user's query to the language model.

How RAG Works

The RAG pipeline consists of two main phases. In the ingestion phase, documents
are extracted from various file formats (PDF, DOCX, Markdown, etc.), split into
smaller chunks, converted into vector embeddings using an embedding model, and
stored in a PostgreSQL database with the pgvector extension.

In the retrieval phase, a user's query is converted into a vector embedding using
the same embedding model. The system then performs a similarity search against
the stored chunk embeddings, retrieves the most relevant chunks, and provides
them as context to the language model to generate an accurate answer.

Benefits of RAG

RAG systems offer several advantages over traditional approaches. They can access
up-to-date information without retraining the model. They reduce hallucinations by
grounding answers in retrieved evidence. They allow for domain-specific knowledge
to be added incrementally. Citations can be traced back to specific source documents.

Vector Databases

Vector databases like PostgreSQL with pgvector store high-dimensional float vectors
and support efficient approximate nearest-neighbor search. The cosine similarity
metric is commonly used to measure how semantically similar two text chunks are.
HNSW (Hierarchical Navigable Small World) indexing provides fast retrieval even
for millions of vectors.
"""

# ── new schema matching our code (differs from original schema.sql) ──────────
# Key differences:
#   - documents: added mime_type, removed content (we don't store full text)
#   - chunks: added start_char/end_char, removed chunk_index NOT NULL / token_count
#   - embedding uses vector (no fixed dim) so tests work for any provider dimension
TEST_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    mime_type   TEXT NOT NULL DEFAULT 'application/octet-stream',
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    embedding   vector,
    start_char  INTEGER,
    end_char    INTEGER,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops);

CREATE TABLE IF NOT EXISTS sessions (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id    TEXT,
    title      TEXT NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'agent')),
    content    TEXT NOT NULL,
    metadata   JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename   TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    progress   FLOAT NOT NULL DEFAULT 0.0,
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    user_id    TEXT,
    rating     INTEGER NOT NULL CHECK (rating IN (1, -1)),
    comment    TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (message_id, user_id)
);
"""

def _ok(msg: str):  print(f"  \033[32m✓\033[0m {msg}")
def _fail(msg: str, exc=None):
    print(f"  \033[31m✗\033[0m {msg}")
    if exc:
        print(f"    {type(exc).__name__}: {exc}")
    sys.exit(1)

def _section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

# ── tests ─────────────────────────────────────────────────────────────────────

async def test_db_connection():
    _section("Test 1 — Database connectivity")
    try:
        from neat_rag.db.pool import pg_pool
        await pg_pool.connect()
        async with pg_pool.get_connection() as conn:
            result = await conn.fetchval("SELECT version()")
        _ok(f"Connected: {result[:60]}...")
        return pg_pool
    except Exception as e:
        _fail("Cannot connect to PostgreSQL. Is docker-compose up?", e)


async def apply_schema(pg_pool):
    _section("Test 2 — Schema setup")
    try:
        async with pg_pool.get_connection() as conn:
            await conn.execute(TEST_SCHEMA)
        _ok("Schema ready (all tables created / already exist)")
    except Exception as e:
        _fail("Schema setup failed", e)


async def test_ingestion(pg_pool, provider: str, chunking_strategy: str, chunker_kwargs: dict):
    _section(f"Test 3 — Ingestion pipeline  (provider: {provider}, chunking: {chunking_strategy})")

    from neat_rag.ingestion.pipeline import IngestionPipeline
    from neat_rag.providers.embedding import get_embedder

    # Use the application's native factory function. It will automatically
    # read from neat_rag.config.settings, which has already loaded .env
    try:
        embedder = get_embedder(provider=provider)
    except Exception as e:
        _fail(f"Failed to initialize embedder for {provider}", e)

    pipeline = IngestionPipeline(
        pg_pool=pg_pool,
        embedder=embedder,
        chunking_strategy=chunking_strategy,
        chunker_kwargs=chunker_kwargs,
    )

    # Write sample text to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8")
    tmp.write(SAMPLE_TEXT)
    tmp.close()
    tmp_path = Path(tmp.name)

    document = None
    try:
        print(f"  Ingesting: {tmp_path.name}  ({len(SAMPLE_TEXT)} chars)")
        document = await pipeline.run(tmp_path)
        _ok(f"Document created  id={document.id}")
        _ok(f"Chunks created    count={document.chunk_count}")

        # Verify in DB
        async with pg_pool.get_connection() as conn:
            db_count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE document_id = $1::uuid",
                document.id,
            )
            sample = await conn.fetchrow(
                "SELECT content, start_char, end_char FROM chunks "
                "WHERE document_id = $1::uuid ORDER BY start_char LIMIT 1",
                document.id,
            )
            has_embedding = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks "
                "WHERE document_id = $1::uuid AND embedding IS NOT NULL",
                document.id,
            )

        _ok(f"DB verification   chunks_in_db={db_count}, with_embeddings={has_embedding}")
        if sample:
            snippet = sample["content"][:80].replace("\n", " ")
            _ok(f"First chunk       start={sample['start_char']} end={sample['end_char']}")
            _ok(f"Content preview   \"{snippet}...\"")

        if db_count != document.chunk_count:
            _fail(f"Chunk count mismatch: pipeline said {document.chunk_count}, DB has {db_count}")
        if has_embedding < db_count:
            _fail(f"Only {has_embedding}/{db_count} chunks have embeddings — embedding failed?")

    except Exception as e:
        _fail("Ingestion failed", e)
    finally:
        tmp_path.unlink(missing_ok=True)
        # Clean up test data from DB
        if document:
            try:
                async with pg_pool.get_connection() as conn:
                    await conn.execute("DELETE FROM documents WHERE id = $1::uuid", document.id)
                    # pass
                _ok(f"Cleanup done      (document {document.id} removed)")
            except Exception:
                pass  # cleanup failure is not a test failure


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    # Because our app uses pydantic-settings, it automatically loads .env
    # upon importing neat_rag.config. We don't need a custom _load_dotenv anymore.
    from neat_rag.config import settings

    parser = argparse.ArgumentParser(description="Neat-RAG Phase 1 ingestion test")
    parser.add_argument(
        "--provider",
        default=settings.EMBEDDING_PROVIDER, # Default to whatever is in .env
        choices=["openai", "gemini", "ollama"],
        help="Embedding provider to use (overrides .env EMBEDDING_PROVIDER for this test)",
    )
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Neat-RAG — Phase 1 Ingestion Test")
    print(f"  Provider: {args.provider}")
    print(f"  Model:    {settings.EMBEDDING_MODEL}")
    print(f"{'='*50}")

    pg_pool = await test_db_connection()
    await apply_schema(pg_pool)
    
    # Run test 1: Recursive Chunking
    await test_ingestion(
        pg_pool, 
        args.provider, 
        "recursive", 
        {"chunk_size": 500, "chunk_overlap": 50}
    )
    
    # Run test 2: Semantic Chunking
    await test_ingestion(
        pg_pool, 
        args.provider, 
        "semantic", 
        {"breakpoint_threshold_type": "percentile", "min_chunk_size": 50}
    )

    # Run test 3: Token Chunking
    await test_ingestion(
        pg_pool, 
        args.provider, 
        "token", 
        {"max_tokens": 100, "overlap_tokens": 15}
    )

    await pg_pool.disconnect()

    print(f"\n{'='*50}")
    print(f"  \033[32mAll tests passed!\033[0m")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    asyncio.run(main())
