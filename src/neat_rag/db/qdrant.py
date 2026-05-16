"""
QdrantStore — VectorStoreBase implementation backed by Qdrant.

Key features vs pgvector backend:
  - Named vectors: each embedding model gets its own named vector slot.
    Switching models adds a new slot without touching existing data, which
    eliminates the "different vector dimensions" error at search time.
  - Sparse vectors: hash-based BM25-lite for keyword recall, any language.
  - Hybrid search: dense + sparse fused by Reciprocal Rank Fusion (built-in).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from neat_rag.config import settings
from neat_rag.db.vector_store import VectorStoreBase
from neat_rag.exceptions import DatabaseError
from neat_rag.logger import get_logger
from neat_rag.models import Chunk, Document, SearchHit

logger = get_logger(__name__)

# ── Sparse-vector helpers ─────────────────────────────────────────────────────

_VOCAB_SIZE = 131072          # 2^17 — low collision rate for typical text
_SPARSE_SLOT = "sparse"

# Matches either a sequence of alphanumeric chars OR a single CJK/J/K character.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[一-鿿぀-ゟ゠-ヿ가-힯]")


def _sparse_embed(text: str) -> SparseVector:
    """
    Feature-hashing BM25-lite with sqrt(TF) damping. Language-agnostic.
    Uses MD5 for stable hashing across process restarts.
    Handles hash collisions by aggregating weights.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return SparseVector(indices=[], values=[])

    tf = Counter(tokens)
    # Map index -> total weight (sqrt(freq))
    aggregated: dict[int, float] = {}

    for token, freq in tf.items():
        h_hex = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(h_hex, 16) % _VOCAB_SIZE

        weight = math.sqrt(float(freq))
        if idx in aggregated:
            aggregated[idx] += weight
        else:
            aggregated[idx] = weight

    # Qdrant requires sorted indices and unique entries
    sorted_indices = sorted(aggregated.keys())
    return SparseVector(
        indices=sorted_indices,
        values=[aggregated[idx] for idx in sorted_indices]
    )


def _vector_name(model: str) -> str:
    """Sanitize model name → valid Qdrant vector name (alphanumeric + _)."""
    return re.sub(r"[^a-zA-Z0-9]", "_", model).strip("_").lower()


# ── Store ─────────────────────────────────────────────────────────────────────

class QdrantStore(VectorStoreBase):
    """
    Qdrant-backed vector store.

    One Qdrant collection holds all chunks. Each embedding model gets its own
    named dense vector slot; a single sparse slot handles keyword matching.
    Switching EMBEDDING_MODEL automatically registers a new slot — no data loss,
    no dimension conflicts.
    """

    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        await self._ensure_collection()
        await self._ensure_named_vector()
        logger.info(
            "QdrantStore connected",
            host=settings.QDRANT_HOST,
            collection=settings.QDRANT_COLLECTION,
            vector=self._vec_name,
        )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def _client_safe(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantStore not connected. Call connect() first.")
        return self._client

    @property
    def _vec_name(self) -> str:
        return _vector_name(settings.EMBEDDING_MODEL)

    @property
    def _collection(self) -> str:
        return settings.QDRANT_COLLECTION

    # ── Collection bootstrap ─────────────────────────────────────────────────

    async def _ensure_collection(self) -> None:
        if await self._client_safe.collection_exists(self._collection):
            return
        await self._client_safe.create_collection(
            collection_name=self._collection,
            vectors_config={
                self._vec_name: VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                _SPARSE_SLOT: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        for field in ("document_id", "user_id"):
            await self._client_safe.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema="keyword",
            )
        logger.info("Qdrant collection created", collection=self._collection)

    async def _ensure_named_vector(self) -> None:
        """
        Add the current model's dense vector slot if it does not exist yet.
        This is what makes model-switching non-destructive: old vectors are
        untouched, the new model just gets its own slot in the same collection.
        """
        info = await self._client_safe.get_collection(self._collection)
        existing = set((info.config.params.vectors or {}).keys())

        if self._vec_name in existing:
            stored_dim = info.config.params.vectors[self._vec_name].size
            if stored_dim != settings.EMBEDDING_DIM:
                raise RuntimeError(
                    f"Dimension mismatch for model '{settings.EMBEDDING_MODEL}': "
                    f"stored={stored_dim}, configured={settings.EMBEDDING_DIM}. "
                    f"Check EMBEDDING_DIM in .env."
                )
            return

        await self._client_safe.update_collection(
            collection_name=self._collection,
            vectors_config={
                self._vec_name: VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            },
        )
        logger.info(
            "Named vector slot added for new model",
            vector=self._vec_name,
            dim=settings.EMBEDDING_DIM,
        )

    # ── Write ────────────────────────────────────────────────────────────────

    async def upsert_chunks(self, document: Document, chunks: List[Chunk]) -> None:
        points = [
            PointStruct(
                id=chunk.id,
                vector={
                    self._vec_name: chunk.embedding,
                    _SPARSE_SLOT: _sparse_embed(chunk.content),
                },
                payload={
                    "document_id": document.id,
                    "user_id": document.user_id,
                    "document_title": document.title,
                    "document_source": document.source,
                    "content": chunk.content,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": chunk.metadata,
                    "created_at": chunk.created_at.isoformat(),
                },
            )
            for chunk in chunks
            if chunk.embedding
        ]
        if not points:
            return
        await self._client_safe.upsert(collection_name=self._collection, points=points)
        logger.info("Chunks upserted to Qdrant", document_id=document.id, count=len(points))

    async def delete_by_document(self, document_id: str) -> None:
        await self._client_safe.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            ),
        )
        logger.info("Chunks deleted from Qdrant", document_id=document_id)

    # ── Read ─────────────────────────────────────────────────────────────────

    async def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        results, _ = await self._client_safe.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            with_payload=True,
            with_vectors=False,
            limit=2000,
        )
        chunks = [_point_to_chunk(p) for p in results]
        chunks.sort(key=lambda c: (c.start_char or 0, c.created_at))
        return chunks

    async def count_by_document(self, document_id: str) -> int:
        result = await self._client_safe.count(
            collection_name=self._collection,
            count_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            exact=True,
        )
        return result.count

    # ── Search ───────────────────────────────────────────────────────────────

    async def vector_search(
        self,
        embedding: List[float],
        top_k: int,
        user_id: Optional[str],
    ) -> List[SearchHit]:
        results = await self._client_safe.search(
            collection_name=self._collection,
            query_vector=NamedVector(name=self._vec_name, vector=embedding),
            query_filter=_build_filter(user_id),
            limit=top_k,
            with_payload=True,
        )
        return [_scored_to_hit(r) for r in results]

    async def hybrid_search(
        self,
        embedding: List[float],
        query: str,
        top_k: int,
        user_id: Optional[str],
        text_weight: float = 0.3,  # kept for interface compatibility; RRF auto-weights
    ) -> List[SearchHit]:
        user_filter = _build_filter(user_id)
        prefetch_limit = top_k * 3
        results = await self._client_safe.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(
                    query=embedding,  # Pass raw list[float] since 'using' is specified
                    using=self._vec_name,
                    limit=prefetch_limit,
                    filter=user_filter,
                ),
                Prefetch(
                    query=_sparse_embed(query),  # Pass raw SparseVector since 'using' is specified
                    using=_SPARSE_SLOT,
                    limit=prefetch_limit,
                    filter=user_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [_scored_to_hit(r) for r in results.points]


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_filter(user_id: Optional[str]) -> Optional[Filter]:
    if user_id is None:
        return None
    return Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])


def _point_to_chunk(point) -> Chunk:
    p = point.payload or {}
    raw_ts = p.get("created_at")
    try:
        created_at = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
    except (ValueError, TypeError):
        created_at = datetime.now(timezone.utc)
    return Chunk(
        id=str(point.id),
        document_id=p.get("document_id", ""),
        content=p.get("content", ""),
        start_char=p.get("start_char"),
        end_char=p.get("end_char"),
        metadata=p.get("metadata") or {},
        created_at=created_at,
    )


def _scored_to_hit(point) -> SearchHit:
    p = point.payload or {}
    return SearchHit(
        chunk_id=str(point.id),
        document_id=p.get("document_id", ""),
        content=p.get("content", ""),
        score=float(point.score),
        document_title=p.get("document_title", ""),
        document_source=p.get("document_source", ""),
        metadata=p.get("metadata") or {},
    )
