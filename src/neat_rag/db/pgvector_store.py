"""
PgVectorStore — wraps the original pgvector SQL logic.

This is an exact preservation of the retrieval and chunk-persistence code that
previously lived in retrievers.py and DocumentRepository.  Nothing here is new
behaviour; it is simply reorganised behind the VectorStoreBase interface so
the rest of the app can be backend-agnostic.

SQL functions relied upon (created by migrations 001/002):
  match_chunks(query_embedding, match_count, p_user_id)
  hybrid_search(query_embedding, query_text, match_count, text_weight, p_user_id)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from neat_rag.db.pool import PgPool
from neat_rag.db.vector_store import VectorStoreBase
from neat_rag.exceptions import DatabaseError
from neat_rag.logger import get_logger
from neat_rag.models import Chunk, Document, SearchHit

logger = get_logger(__name__)


def _vec_str(embedding: List[float]) -> str:
    """Format a float list as the pgvector literal '[x,y,z,...]'."""
    return f"[{','.join(map(str, embedding))}]"


def _parse_vector_rows(rows) -> List[SearchHit]:
    return [
        SearchHit(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            content=row["content"],
            score=float(row["similarity"]),
            document_title=row["document_title"],
            document_source=row["document_source"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
        for row in rows
    ]


def _parse_hybrid_rows(rows) -> List[SearchHit]:
    return [
        SearchHit(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            content=row["content"],
            score=float(row["combined_score"]),
            document_title=row["document_title"],
            document_source=row["document_source"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
        for row in rows
    ]


class PgVectorStore(VectorStoreBase):
    """
    Vector store backed by PostgreSQL + pgvector.

    Uses the same asyncpg pool as the rest of the application.
    connect() / disconnect() are no-ops because the pool lifecycle is managed
    by PgPool (called from the FastAPI lifespan).
    """

    def __init__(self, pg_pool: PgPool) -> None:
        self._pg_pool = pg_pool

    # ── Lifecycle (delegated to PgPool) ──────────────────────────────────────

    async def connect(self) -> None:
        pass  # pool is already managed by PgPool.connect() in app lifespan

    async def disconnect(self) -> None:
        pass  # pool is already managed by PgPool.disconnect() in app lifespan

    # ── Write ────────────────────────────────────────────────────────────────

    async def upsert_chunks(self, document: Document, chunks: List[Chunk]) -> None:
        """Insert chunk rows into the PostgreSQL chunks table."""
        if not chunks:
            return
        chunk_records = [
            (
                c.id,
                c.document_id,
                c.content,
                f"[{','.join(map(str, c.embedding))}]" if c.embedding else None,
                c.start_char,
                c.end_char,
                json.dumps(c.metadata),
                c.created_at,
            )
            for c in chunks
        ]
        try:
            async with self._pg_pool.get_connection() as conn:
                await conn.executemany(
                    """
                    INSERT INTO chunks
                        (id, document_id, content, embedding,
                         start_char, end_char, metadata, created_at)
                    VALUES ($1::uuid, $2::uuid, $3, $4::vector,
                            $5, $6, $7::jsonb, $8)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk_records,
                )
            logger.info("Chunks upserted to pgvector", document_id=document.id, count=len(chunks))
        except Exception as e:
            logger.error("Failed to upsert chunks", document_id=document.id, error=str(e))
            raise DatabaseError(f"Failed to upsert chunks: {e}") from e

    async def delete_by_document(self, document_id: str) -> None:
        """No-op: the chunks table has ON DELETE CASCADE on document_id."""
        pass

    # ── Read ─────────────────────────────────────────────────────────────────

    async def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        try:
            async with self._pg_pool.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id::text, document_id::text, content,
                           start_char, end_char, metadata, created_at
                    FROM chunks
                    WHERE document_id = $1::uuid
                    ORDER BY start_char ASC NULLS LAST, created_at ASC
                    """,
                    document_id,
                )
            return [
                Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    start_char=row.get("start_char"),
                    end_char=row.get("end_char"),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to fetch chunks", document_id=document_id, error=str(e))
            raise DatabaseError(f"Failed to fetch chunks: {e}") from e

    async def count_by_document(self, document_id: str) -> int:
        try:
            async with self._pg_pool.get_connection() as conn:
                return await conn.fetchval(
                    "SELECT COUNT(id) FROM chunks WHERE document_id = $1::uuid",
                    document_id,
                ) or 0
        except Exception as e:
            raise DatabaseError(f"Failed to count chunks: {e}") from e

    # ── Search ───────────────────────────────────────────────────────────────

    async def vector_search(
        self,
        embedding: List[float],
        top_k: int,
        user_id: Optional[str],
    ) -> List[SearchHit]:
        vec = _vec_str(embedding)
        async with self._pg_pool.get_connection() as conn:
            rows = await conn.fetch(
                "SELECT * FROM match_chunks($1::vector, $2, $3)",
                vec,
                top_k,
                user_id,
            )
        return _parse_vector_rows(rows)

    async def hybrid_search(
        self,
        embedding: List[float],
        query: str,
        top_k: int,
        user_id: Optional[str],
        text_weight: float = 0.3,
    ) -> List[SearchHit]:
        vec = _vec_str(embedding)
        async with self._pg_pool.get_connection() as conn:
            rows = await conn.fetch(
                "SELECT * FROM hybrid_search($1::vector, $2::text, $3, $4, $5)",
                vec,
                query,
                top_k,
                text_weight,
                user_id,
            )
        return _parse_hybrid_rows(rows)
