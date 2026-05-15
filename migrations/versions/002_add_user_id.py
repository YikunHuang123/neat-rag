"""Add user_id to documents and ingest_jobs; update search functions for per-user filtering.

Revision ID: 002
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

from alembic import op

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from neat_rag.config import settings  # noqa: E402

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

EMBEDDING_DIM = settings.EMBEDDING_DIM


def upgrade() -> None:
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id TEXT")
    op.execute(dedent("""
        CREATE INDEX IF NOT EXISTS idx_documents_user_id
        ON documents (user_id)
        WHERE user_id IS NOT NULL
    """))

    op.execute("ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS user_id TEXT")
    op.execute(dedent("""
        CREATE INDEX IF NOT EXISTS idx_ingest_jobs_user_id
        ON ingest_jobs (user_id)
        WHERE user_id IS NOT NULL
    """))

    # match_chunks: add optional p_user_id filter
    op.execute(dedent(f"""
        CREATE OR REPLACE FUNCTION match_chunks(
            query_embedding vector({EMBEDDING_DIM}),
            match_count     INT DEFAULT 10,
            p_user_id       TEXT DEFAULT NULL
        )
        RETURNS TABLE (
            chunk_id        UUID,
            document_id     UUID,
            content         TEXT,
            similarity      FLOAT,
            metadata        JSONB,
            document_title  TEXT,
            document_source TEXT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                (1 - (c.embedding <=> query_embedding))::double precision AS similarity,
                c.metadata,
                d.title  AS document_title,
                d.source AS document_source
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
              AND (p_user_id IS NULL OR d.user_id = p_user_id)
            ORDER BY c.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
    """))

    # hybrid_search: add optional p_user_id filter
    op.execute(dedent(f"""
        CREATE OR REPLACE FUNCTION hybrid_search(
            query_embedding vector({EMBEDDING_DIM}),
            query_text      TEXT,
            match_count     INT DEFAULT 10,
            text_weight     FLOAT DEFAULT 0.3,
            p_user_id       TEXT DEFAULT NULL
        )
        RETURNS TABLE (
            chunk_id          UUID,
            document_id       UUID,
            content           TEXT,
            combined_score    FLOAT,
            vector_similarity FLOAT,
            text_similarity   FLOAT,
            metadata          JSONB,
            document_title    TEXT,
            document_source   TEXT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            WITH vector_results AS (
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    (1 - (c.embedding <=> query_embedding))::double precision AS vector_sim,
                    c.metadata,
                    d.title  AS doc_title,
                    d.source AS doc_source
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                  AND (p_user_id IS NULL OR d.user_id = p_user_id)
            ),
            text_results AS (
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    ts_rank_cd(
                        to_tsvector('english', c.content),
                        plainto_tsquery('english', query_text)
                    )::double precision AS text_sim,
                    c.metadata,
                    d.title  AS doc_title,
                    d.source AS doc_source
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
                  AND (p_user_id IS NULL OR d.user_id = p_user_id)
            )
            SELECT
                COALESCE(v.chunk_id,     t.chunk_id)     AS chunk_id,
                COALESCE(v.document_id,  t.document_id)  AS document_id,
                COALESCE(v.content,      t.content)      AS content,
                (
                    COALESCE(v.vector_sim, 0)::double precision * (1 - text_weight) +
                    COALESCE(t.text_sim,   0)::double precision * text_weight
                )                                        AS combined_score,
                COALESCE(v.vector_sim, 0)::double precision AS vector_similarity,
                COALESCE(t.text_sim,   0)::double precision AS text_similarity,
                COALESCE(v.metadata,     t.metadata)     AS metadata,
                COALESCE(v.doc_title,    t.doc_title)    AS document_title,
                COALESCE(v.doc_source,   t.doc_source)   AS document_source
            FROM vector_results v
            FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
            ORDER BY combined_score DESC
            LIMIT match_count;
        END;
        $$;
    """))


def downgrade() -> None:
    # Restore original functions without user_id parameter
    op.execute(dedent(f"""
        CREATE OR REPLACE FUNCTION match_chunks(
            query_embedding vector({EMBEDDING_DIM}),
            match_count     INT DEFAULT 10
        )
        RETURNS TABLE (
            chunk_id        UUID,
            document_id     UUID,
            content         TEXT,
            similarity      FLOAT,
            metadata        JSONB,
            document_title  TEXT,
            document_source TEXT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.content,
                (1 - (c.embedding <=> query_embedding))::double precision AS similarity,
                c.metadata,
                d.title  AS document_title,
                d.source AS document_source
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
    """))

    op.execute(dedent(f"""
        CREATE OR REPLACE FUNCTION hybrid_search(
            query_embedding vector({EMBEDDING_DIM}),
            query_text      TEXT,
            match_count     INT DEFAULT 10,
            text_weight     FLOAT DEFAULT 0.3
        )
        RETURNS TABLE (
            chunk_id          UUID,
            document_id       UUID,
            content           TEXT,
            combined_score    FLOAT,
            vector_similarity FLOAT,
            text_similarity   FLOAT,
            metadata          JSONB,
            document_title    TEXT,
            document_source   TEXT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            WITH vector_results AS (
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    (1 - (c.embedding <=> query_embedding))::double precision AS vector_sim,
                    c.metadata,
                    d.title  AS doc_title,
                    d.source AS doc_source
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
            ),
            text_results AS (
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    ts_rank_cd(
                        to_tsvector('english', c.content),
                        plainto_tsquery('english', query_text)
                    )::double precision AS text_sim,
                    c.metadata,
                    d.title  AS doc_title,
                    d.source AS doc_source
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
            )
            SELECT
                COALESCE(v.chunk_id,     t.chunk_id)     AS chunk_id,
                COALESCE(v.document_id,  t.document_id)  AS document_id,
                COALESCE(v.content,      t.content)      AS content,
                (
                    COALESCE(v.vector_sim, 0)::double precision * (1 - text_weight) +
                    COALESCE(t.text_sim,   0)::double precision * text_weight
                )                                        AS combined_score,
                COALESCE(v.vector_sim, 0)::double precision AS vector_similarity,
                COALESCE(t.text_sim,   0)::double precision AS text_similarity,
                COALESCE(v.metadata,     t.metadata)     AS metadata,
                COALESCE(v.doc_title,    t.doc_title)    AS document_title,
                COALESCE(v.doc_source,   t.doc_source)   AS document_source
            FROM vector_results v
            FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
            ORDER BY combined_score DESC
            LIMIT match_count;
        END;
        $$;
    """))

    op.execute("DROP INDEX IF EXISTS idx_ingest_jobs_user_id")
    op.execute("ALTER TABLE ingest_jobs DROP COLUMN IF EXISTS user_id")
    op.execute("DROP INDEX IF EXISTS idx_documents_user_id")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS user_id")
