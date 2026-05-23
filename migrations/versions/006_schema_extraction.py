"""Add document_ids filter to pgvector search functions.

Revision ID: 006

Updates the match_chunks and hybrid_search PostgreSQL functions to accept an
optional p_document_ids TEXT[] parameter.  When NULL (the default), behaviour
is identical to the previous version — no existing callers need updating.
When provided, only chunks belonging to the listed document IDs are returned,
enabling schema-field-filtered hybrid search from the agent tools layer.

Note: the parameter is TEXT[] rather than UUID[] so that asyncpg can pass
Python list[str] without an explicit type cast.  The WHERE clause compares
c.document_id::TEXT so the HNSW index on the embedding column is still used
for vector similarity; the document_id index handles the extra filter.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

from alembic import op

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from neat_rag.config import settings  # noqa: E402

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None

EMBEDDING_DIM = settings.EMBEDDING_DIM


def upgrade() -> None:
    # match_chunks: add optional p_document_ids TEXT[] filter
    op.execute(dedent(f"""
        CREATE OR REPLACE FUNCTION match_chunks(
            query_embedding  vector({EMBEDDING_DIM}),
            match_count      INT     DEFAULT 10,
            p_user_id        TEXT    DEFAULT NULL,
            p_document_ids   TEXT[]  DEFAULT NULL
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
              AND (p_user_id       IS NULL OR d.user_id          = p_user_id)
              AND (p_document_ids  IS NULL OR c.document_id::TEXT = ANY(p_document_ids))
            ORDER BY c.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
    """))

    # hybrid_search: add optional p_document_ids TEXT[] filter
    op.execute(dedent(f"""
        CREATE OR REPLACE FUNCTION hybrid_search(
            query_embedding  vector({EMBEDDING_DIM}),
            query_text       TEXT,
            match_count      INT    DEFAULT 10,
            text_weight      FLOAT  DEFAULT 0.3,
            p_user_id        TEXT   DEFAULT NULL,
            p_document_ids   TEXT[] DEFAULT NULL
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
                  AND (p_user_id      IS NULL OR d.user_id          = p_user_id)
                  AND (p_document_ids IS NULL OR c.document_id::TEXT = ANY(p_document_ids))
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
                  AND (p_user_id      IS NULL OR d.user_id          = p_user_id)
                  AND (p_document_ids IS NULL OR c.document_id::TEXT = ANY(p_document_ids))
            )
            SELECT
                COALESCE(v.chunk_id,    t.chunk_id)    AS chunk_id,
                COALESCE(v.document_id, t.document_id) AS document_id,
                COALESCE(v.content,     t.content)     AS content,
                (
                    COALESCE(v.vector_sim, 0)::double precision * (1 - text_weight) +
                    COALESCE(t.text_sim,   0)::double precision * text_weight
                )                                      AS combined_score,
                COALESCE(v.vector_sim, 0)::double precision AS vector_similarity,
                COALESCE(t.text_sim,   0)::double precision AS text_similarity,
                COALESCE(v.metadata,    t.metadata)    AS metadata,
                COALESCE(v.doc_title,   t.doc_title)   AS document_title,
                COALESCE(v.doc_source,  t.doc_source)  AS document_source
            FROM vector_results v
            FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
            ORDER BY combined_score DESC
            LIMIT match_count;
        END;
        $$;
    """))


def downgrade() -> None:
    # Restore functions without p_document_ids — identical to migration 002 upgrade.
    op.execute(dedent(f"""
        CREATE OR REPLACE FUNCTION match_chunks(
            query_embedding vector({EMBEDDING_DIM}),
            match_count     INT  DEFAULT 10,
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

    op.execute(dedent(f"""
        CREATE OR REPLACE FUNCTION hybrid_search(
            query_embedding vector({EMBEDDING_DIM}),
            query_text      TEXT,
            match_count     INT   DEFAULT 10,
            text_weight     FLOAT DEFAULT 0.3,
            p_user_id       TEXT  DEFAULT NULL
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
                COALESCE(v.chunk_id,    t.chunk_id)    AS chunk_id,
                COALESCE(v.document_id, t.document_id) AS document_id,
                COALESCE(v.content,     t.content)     AS content,
                (
                    COALESCE(v.vector_sim, 0)::double precision * (1 - text_weight) +
                    COALESCE(t.text_sim,   0)::double precision * text_weight
                )                                      AS combined_score,
                COALESCE(v.vector_sim, 0)::double precision AS vector_similarity,
                COALESCE(t.text_sim,   0)::double precision AS text_similarity,
                COALESCE(v.metadata,    t.metadata)    AS metadata,
                COALESCE(v.doc_title,   t.doc_title)   AS document_title,
                COALESCE(v.doc_source,  t.doc_source)  AS document_source
            FROM vector_results v
            FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
            ORDER BY combined_score DESC
            LIMIT match_count;
        END;
        $$;
    """))
