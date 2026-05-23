"""
Abstract vector store interface + factory.

All vector/chunk operations go through VectorStoreBase so the rest of the
codebase stays backend-agnostic. Switch backends by setting VECTOR_STORE_BACKEND
in .env — no other code needs to change.

Supported backends:
  pgvector  (default) — PostgreSQL + pgvector extension, original behaviour
  qdrant              — Qdrant dedicated vector database
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from neat_rag.models import Chunk, Document, SearchHit


class VectorStoreBase(ABC):
    """
    Common interface for all vector store backends.

    Implementations must be safe to call concurrently from asyncio tasks.
    """

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Initialise connections / verify configuration at startup."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Release resources at shutdown."""

    # ── Write ────────────────────────────────────────────────────────────────

    @abstractmethod
    async def upsert_chunks(self, document: Document, chunks: List[Chunk]) -> None:
        """Persist chunk vectors (and payload) for a document."""

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None:
        """Remove all vectors associated with a document."""

    # ── Read ─────────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Return all chunks for a document, ordered by position."""

    @abstractmethod
    async def count_by_document(self, document_id: str) -> int:
        """Return the number of chunks stored for a document."""

    # ── Search ───────────────────────────────────────────────────────────────

    @abstractmethod
    async def vector_search(
        self,
        embedding: List[float],
        top_k: int,
        user_id: Optional[str],
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """Pure dense-vector similarity search.

        Args:
            document_ids: When provided, restricts results to chunks belonging
                          to these document IDs.  Pass IDs returned by the
                          search_by_schema_field agent tool to scope a semantic
                          search to a filtered document set.
        """

    @abstractmethod
    async def hybrid_search(
        self,
        embedding: List[float],
        query: str,
        top_k: int,
        user_id: Optional[str],
        text_weight: float = 0.3,
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchHit]:
        """Hybrid search: dense vector + keyword recall, fused by score.

        Args:
            document_ids: When provided, restricts results to chunks belonging
                          to these document IDs.  Pass IDs returned by the
                          search_by_schema_field agent tool to scope a hybrid
                          search to a filtered document set.
        """


# ── Factory ──────────────────────────────────────────────────────────────────

_instance: Optional[VectorStoreBase] = None


def get_vector_store() -> VectorStoreBase:
    """
    Return the singleton vector store chosen by VECTOR_STORE_BACKEND.

    Lazy-initialised on first call; safe to call multiple times.
    """
    global _instance
    if _instance is not None:
        return _instance

    from neat_rag.config import settings

    backend = settings.VECTOR_STORE_BACKEND.lower()
    if backend == "qdrant":
        from neat_rag.db.qdrant import QdrantStore
        _instance = QdrantStore()
    elif backend == "pgvector":
        from neat_rag.db.pgvector_store import PgVectorStore
        from neat_rag.db.pool import pg_pool
        _instance = PgVectorStore(pg_pool)
    else:
        raise ValueError(
            f"Unknown VECTOR_STORE_BACKEND '{backend}'. Choose 'pgvector' or 'qdrant'."
        )

    return _instance
