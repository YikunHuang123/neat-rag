#!/usr/bin/env python3
"""
Phase 3 API Layer Test
======================
Tests all HTTP endpoints via httpx.AsyncClient + ASGITransport
against a real PostgreSQL database.

Usage:
    python test_api.py               # health + sessions + feedback only
    python test_api.py --with-llm    # + document upload (embedding) + chat (LLM)

Prerequisites:
    1. PostgreSQL running (docker-compose up postgres)
    2. .env configured
    3. For --with-llm: valid API key for EMBEDDING_PROVIDER and LLM_PROVIDER
    4. python-multipart installed (pip install python-multipart)
"""

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ── schema identical to test_ingestion.py ────────────────────────────────────
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

-- ── RAG functions ────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector,
    match_count int
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    document_title TEXT,
    document_source TEXT,
    metadata JSONB,
    similarity FLOAT
)
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.content,
        d.title AS document_title,
        d.source AS document_source,
        c.metadata,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector,
    query_text text,
    match_count int,
    text_weight float
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    document_title TEXT,
    document_source TEXT,
    metadata JSONB,
    combined_score FLOAT
)
AS $$
BEGIN
    RETURN QUERY
    WITH vector_matches AS (
        SELECT
            c.id,
            1 - (c.embedding <=> query_embedding) AS similarity
        FROM chunks c
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    text_matches AS (
        SELECT
            c.id,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text)) AS rank
        FROM chunks c
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
        ORDER BY rank DESC
        LIMIT match_count * 2
    )
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.content,
        d.title AS document_title,
        d.source AS document_source,
        c.metadata,
        (COALESCE(v.similarity, 0) * (1 - text_weight) + COALESCE(t.rank, 0) * text_weight) AS combined_score
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    LEFT JOIN vector_matches v ON c.id = v.id
    LEFT JOIN text_matches t ON c.id = t.id
    WHERE v.id IS NOT NULL OR t.id IS NOT NULL
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
"""

SAMPLE_TEXT = """\
Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that enhances large language
models by providing them with relevant context retrieved from an external knowledge
base. Instead of relying solely on the model's training data, RAG systems first
retrieve relevant document chunks from a vector database, then pass those chunks
alongside the user's query to the language model.

Benefits of RAG

RAG systems offer several advantages: they can access up-to-date information
without retraining, reduce hallucinations by grounding answers in evidence, and
allow domain-specific knowledge to be added incrementally. Citations can be traced
back to specific source documents.
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def _section(title: str):
    print(f"\n{'─' * 52}")
    print(f"  {title}")
    print(f"{'─' * 52}")

def _ok(msg: str):
    print(f"  \033[32m✓\033[0m {msg}")

def _skip(msg: str):
    print(f"  \033[33m○\033[0m {msg}")

def _fail(msg: str, exc=None):
    print(f"  \033[31m✗\033[0m {msg}")
    if exc:
        print(f"    {type(exc).__name__}: {exc}")
    sys.exit(1)

def _assert(condition: bool, msg: str):
    if not condition:
        _fail(f"Assertion failed: {msg}")

def _assert_status(response: httpx.Response, expected: int):
    if response.status_code != expected:
        _fail(
            f"Expected HTTP {expected}, got {response.status_code}",
            Exception(response.text[:300]),
        )


# ── test sections ─────────────────────────────────────────────────────────────

async def setup_db():
    _section("Setup — DB connection + schema")
    from neat_rag.db.pool import pg_pool
    try:
        await pg_pool.connect()
        async with pg_pool.get_connection() as conn:
            ver = await conn.fetchval("SELECT version()")
            _ok(f"Connected: {str(ver)[:60]}...")
            await conn.execute(TEST_SCHEMA)
            _ok("Schema ready")
        return pg_pool
    except Exception as e:
        _fail("Cannot connect to PostgreSQL. Is docker-compose up?", e)


async def test_health(client: httpx.AsyncClient):
    _section("Test 1 — Health endpoints")

    r = await client.get("/health/live")
    _assert_status(r, 200)
    _assert(r.json()["status"] == "ok", "liveness status == ok")
    _ok("GET /health/live  → 200  status=ok")

    r = await client.get("/health/ready")
    _assert_status(r, 200)
    body = r.json()
    _assert("db" in body, "readiness has 'db' field")
    _assert("timestamp" in body, "readiness has 'timestamp' field")
    status_str = "ok" if body["db"] else "degraded"
    _ok(f"GET /health/ready → 200  db={body['db']}  status={status_str}")


async def test_sessions(client: httpx.AsyncClient):
    _section("Test 2 — Session CRUD")

    # Create
    r = await client.post("/sessions", json={"user_id": "test_user", "title": "API Test Session"})
    _assert_status(r, 201)
    session = r.json()
    _assert("id" in session, "session has id")
    _assert(session["title"] == "API Test Session", "title matches")
    session_id = session["id"]
    _ok(f"POST /sessions → 201  id={session_id}")

    # List
    r = await client.get("/sessions", params={"user_id": "test_user"})
    _assert_status(r, 200)
    items = r.json()["items"]
    _assert(any(s["id"] == session_id for s in items), "created session appears in list")
    _ok(f"GET  /sessions  → 200  count={len(items)}")

    # Get
    r = await client.get(f"/sessions/{session_id}")
    _assert_status(r, 200)
    _assert(r.json()["id"] == session_id, "get returns correct session")
    _ok(f"GET  /sessions/{{id}} → 200")

    # Get messages (initially empty)
    r = await client.get(f"/sessions/{session_id}/messages")
    _assert_status(r, 200)
    _assert(r.json()["items"] == [], "no messages yet")
    _ok(f"GET  /sessions/{{id}}/messages → 200  count=0")

    # Update title
    r = await client.patch(f"/sessions/{session_id}", json={"title": "Updated Title"})
    _assert_status(r, 200)
    _assert(r.json()["title"] == "Updated Title", "title updated")
    _ok(f"PATCH /sessions/{{id}} → 200  new_title='Updated Title'")

    # 404 for unknown session
    r = await client.get("/sessions/00000000-0000-0000-0000-000000000000")
    _assert_status(r, 404)
    _ok(f"GET  /sessions/unknown → 404 (correct)")

    # Delete
    r = await client.delete(f"/sessions/{session_id}")
    _assert_status(r, 204)
    _ok(f"DELETE /sessions/{{id}} → 204")

    # Confirm deleted
    r = await client.get(f"/sessions/{session_id}")
    _assert_status(r, 404)
    _ok(f"GET  /sessions/{{id}} after delete → 404 (correct)")

    return session_id  # already deleted, but returned for reference


async def test_feedback_direct(client: httpx.AsyncClient):
    """Create a message directly via DB, then POST /feedback against it."""
    _section("Test 3 — Feedback endpoint")
    from neat_rag.db.pool import pg_pool
    from neat_rag.db.sessions import SessionRepository
    from neat_rag.models import MessageRole

    # Create a session + message directly (no LLM needed)
    async with pg_pool.get_connection() as conn:
        repo = SessionRepository(conn)
        session = await repo.create_session(user_id="feedback_test", title="Feedback Test")
        msg = await repo.add_message(session.id, MessageRole.AGENT, "Test answer for feedback.")

    # Submit feedback
    r = await client.post("/feedback", json={
        "message_id": msg.id,
        "rating": 1,
        "user_id": "feedback_test",
        "comment": "Good answer",
    })
    _assert_status(r, 201)
    fb = r.json()
    _assert(fb["rating"] == 1, "rating == 1 (thumbs up)")
    _assert(fb["message_id"] == msg.id, "message_id matches")
    _ok(f"POST /feedback → 201  rating=+1  message_id={msg.id}")

    # Idempotent update (same message_id + user_id → upsert)
    r = await client.post("/feedback", json={
        "message_id": msg.id,
        "rating": -1,
        "user_id": "feedback_test",
        "comment": "Actually, bad answer",
    })
    _assert_status(r, 201)
    _assert(r.json()["rating"] == -1, "rating updated to -1")
    _ok(f"POST /feedback (upsert) → 201  rating=-1")

    # Cleanup
    async with pg_pool.get_connection() as conn:
        await conn.execute("DELETE FROM sessions WHERE id = $1::uuid", session.id)
    _ok("Cleanup done")


async def test_documents_upload(client: httpx.AsyncClient):
    _section("Test 4 — Document upload + job polling")

    # Write sample text to a temp .txt file
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8")
    tmp.write(SAMPLE_TEXT)
    tmp.close()
    tmp_path = Path(tmp.name)

    doc_id = None
    job_id = None
    try:
        with open(tmp_path, "rb") as fh:
            r = await client.post(
                "/documents/upload",
                files={"file": ("rag_intro.txt", fh, "text/plain")},
            )
        _assert_status(r, 202)
        upload = r.json()
        job_id = upload["job_id"]
        _assert(upload["filename"] == "rag_intro.txt", "filename matches")
        _ok(f"POST /documents/upload → 202  job_id={job_id}")

        # Poll job until done (timeout 120 s)
        _ok("Polling job status...")
        deadline = time.time() + 120
        final_status = None
        while time.time() < deadline:
            r = await client.get(f"/jobs/{job_id}")
            _assert_status(r, 200)
            job = r.json()
            final_status = job["status"]
            progress_pct = int(job["progress"] * 100)
            print(f"      status={final_status}  progress={progress_pct}%", end="\r", flush=True)
            if final_status in ("completed", "failed"):
                break
            await asyncio.sleep(2)
        print()  # newline after progress line

        if final_status != "completed":
            _fail(f"Ingestion job ended with status='{final_status}'  (error: {job.get('error')})")
        _ok(f"GET /jobs/{{id}} → completed  progress=100%")

        # GET /jobs list
        r = await client.get("/jobs")
        _assert_status(r, 200)
        job_list = r.json()["items"]
        _assert(any(j["id"] == job_id for j in job_list), "job appears in list")
        _ok(f"GET /jobs → 200  count={len(job_list)}")

        # List documents — find the one we just uploaded
        r = await client.get("/documents")
        _assert_status(r, 200)
        docs = r.json()["items"]
        _assert(len(docs) > 0, "at least one document in list")
        doc_id = docs[0]["id"]
        _ok(f"GET /documents → 200  count={len(docs)}  first_id={doc_id}")

        # Get single document
        r = await client.get(f"/documents/{doc_id}")
        _assert_status(r, 200)
        doc = r.json()
        _assert(doc["chunk_count"] > 0, "document has chunks")
        _ok(f"GET /documents/{{id}} → 200  chunks={doc['chunk_count']}")

        # Patch metadata
        r = await client.patch(f"/documents/{doc_id}", json={"metadata": {"tag": "test"}})
        _assert_status(r, 200)
        _assert(r.json()["metadata"].get("tag") == "test", "metadata updated")
        _ok(f"PATCH /documents/{{id}} → 200  metadata.tag=test")

    finally:
        tmp_path.unlink(missing_ok=True)
        # Cleanup document if created
        if doc_id:
            r = await client.delete(f"/documents/{doc_id}")
            if r.status_code == 204:
                _ok(f"DELETE /documents/{{id}} → 204  (cleanup)")


async def test_chat_sync(client: httpx.AsyncClient):
    _section("Test 5 — Sync chat  POST /chat")

    # Create a session first
    r = await client.post("/sessions", json={"title": "Chat Test"})
    _assert_status(r, 201)
    session_id = r.json()["id"]
    _ok(f"Session created  id={session_id}")

    # Ask a question
    r = await client.post("/chat", json={
        "session_id": session_id,
        "message": "What is RAG and how does it work?",
    }, timeout=120.0)
    _assert_status(r, 200)
    body = r.json()
    _assert(body["session_id"] == session_id, "session_id echoed")
    _assert(len(body["content"]) > 10, "non-empty answer")
    preview = body["content"][:120].replace("\n", " ")
    _ok(f"POST /chat → 200  answer='{preview}...'")

    # Messages should now contain both user + agent turns
    r = await client.get(f"/sessions/{session_id}/messages")
    _assert_status(r, 200)
    messages = r.json()["items"]
    _assert(len(messages) == 2, f"expected 2 messages, got {len(messages)}")
    roles = [m["role"] for m in messages]
    _assert("user" in roles and "agent" in roles, "both user and agent messages stored")
    _ok(f"GET /sessions/{{id}}/messages → {len(messages)} messages (user + agent)")

    # Cleanup
    await client.delete(f"/sessions/{session_id}")


async def test_chat_stream(client: httpx.AsyncClient):
    _section("Test 6 — Streaming chat  POST /chat/stream")

    r = await client.post("/sessions", json={"title": "Stream Test"})
    _assert_status(r, 201)
    session_id = r.json()["id"]

    chunks_received: list[str] = []
    done_event: dict = {}

    async with client.stream(
        "POST",
        "/chat/stream",
        json={
            "session_id": session_id,
            "message": "Give me a one-sentence summary of RAG.",
        },
        timeout=120.0,
    ) as response:
        _assert(response.status_code == 200, f"stream returned {response.status_code}")
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "delta":
                chunks_received.append(event["content"])
            elif event.get("type") == "done":
                done_event = event
            elif event.get("type") == "error":
                _fail(f"Stream error event: {event['content']}")

    _assert(len(chunks_received) > 0, "received at least one delta")
    full_text = "".join(chunks_received)
    _assert(len(full_text) > 5, "accumulated text is non-empty")
    preview = full_text[:100].replace("\n", " ")
    _ok(f"POST /chat/stream → {len(chunks_received)} delta(s)  text='{preview}...'")
    _assert("message_id" in done_event, "done event contains message_id")
    _ok(f"Stream 'done' event  message_id={done_event.get('message_id')}")

    # Cleanup
    await client.delete(f"/sessions/{session_id}")


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    from neat_rag.config import settings

    parser = argparse.ArgumentParser(description="Neat-RAG Phase 3 API test")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run document upload (embedding) + chat (LLM) tests",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 52}")
    print(f"  Neat-RAG — Phase 3 API Test")
    print(f"  DB:      {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    if args.with_llm:
        print(f"  LLM:     {settings.LLM_PROVIDER} / {settings.LLM_MODEL}")
        print(f"  Embed:   {settings.EMBEDDING_PROVIDER} / {settings.EMBEDDING_MODEL}")
    print(f"{'=' * 52}")

    from neat_rag.db.pool import pg_pool
    await setup_db()

    # Build the FastAPI app and connect to it via in-process ASGI transport.
    # pg_pool is already connected above; the app's lifespan connect() is a no-op.
    from neat_rag.api import create_app
    app = create_app()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await test_health(client)
        await test_sessions(client)
        await test_feedback_direct(client)

        if args.with_llm:
            await test_documents_upload(client)
            await test_chat_sync(client)
            await test_chat_stream(client)
        else:
            _section("Skipped — document upload + chat (run with --with-llm)")
            _skip("POST /documents/upload")
            _skip("POST /chat")
            _skip("POST /chat/stream")

    await pg_pool.disconnect()

    print(f"\n{'=' * 52}")
    print(f"  \033[32mAll tests passed!\033[0m")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    asyncio.run(main())
