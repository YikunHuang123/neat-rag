import time

from neat_rag.db.vector_store import VectorStoreBase
from neat_rag.exceptions import RetrievalError
from neat_rag.logger import get_logger
from neat_rag.models import SearchResult, SearchType
from neat_rag.providers.embedding import OpenAIEmbedder

logger = get_logger(__name__)


class VectorRetriever:
    """
    iRetrieves chunks via cosine-similarty search.
    Delegates to whatever VectorStoreBase backend is configured.
    """

    def __init__(self, embedder: OpenAIEmbedder, vector_store: VectorStoreBase) -> None:
        self._embedder = embedder
        self._store = vector_store

    async def search(
        self, query: str, top_k: int = 10, user_id: str | None = None
    ) -> SearchResult:
        t0 = time.monotonic()
        try:
            embedding = await self._embedder.embed_one(query)
            hits = await self._store.vector_search(embedding, top_k, user_id)
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("Vector search done", query=query[:60], hits=len(hits), ms=round(elapsed, 1))
            return SearchResult(
                hits=hits,
                total_hits=len(hits),
                search_type=SearchType.VECTOR,
                query_time_ms=elapsed,
            )
        except RetrievalError:
            raise
        except Exception as e:
            logger.error("Vector search failed", query=query[:60], error=str(e))
            raise RetrievalError(f"Vector search failed: {e}") from e


class HybridRetriever:
    """
    Retrieves chunks via weighted combination of vector similarity and keyword search.
    Delegates to whatever VectorStoreBase backend is configured.

    For pgvector: uses PostgreSQL FTS + HNSW weighted score (SQL function).
    For Qdrant:   uses dense + sparse RRF fusion.

    The text_weight parameter is forwarded to backends that support it.
    """

    def __init__(self, embedder: OpenAIEmbedder, vector_store: VectorStoreBase) -> None:
        self._embedder = embedder
        self._store = vector_store

    async def search(
        self,
        query: str,
        top_k: int = 10,
        text_weight: float = 0.3,
        user_id: str | None = None,
    ) -> SearchResult:
        t0 = time.monotonic()
        try:
            embedding = await self._embedder.embed_one(query)
            hits = await self._store.hybrid_search(embedding, query, top_k, user_id)
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("Hybrid search done", query=query[:60], hits=len(hits), ms=round(elapsed, 1))
            return SearchResult(
                hits=hits,
                total_hits=len(hits),
                search_type=SearchType.HYBRID,
                query_time_ms=elapsed,
            )
        except RetrievalError:
            raise
        except Exception as e:
            logger.error("Hybrid search failed", query=query[:60], error=str(e))
            raise RetrievalError(f"Hybrid search failed: {e}") from e
