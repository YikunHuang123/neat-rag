import json
import time

from neat_rag.db.pool import PgPool
from neat_rag.exceptions import RetrievalError
from neat_rag.logger import get_logger
from neat_rag.models import SearchHit, SearchResult, SearchType
from neat_rag.providers.embedding import OpenAIEmbedder

logger = get_logger(__name__)


def _vec_str(embedding: list[float]) -> str:
    """Format a float list as the pgvector literal '[x,y,z,...]'."""
    return f"[{','.join(map(str, embedding))}]"


def _parse_hits_vector(rows: list) -> list[SearchHit]:
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


def _parse_hits_hybrid(rows: list) -> list[SearchHit]:
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


class VectorRetriever:
    """
    Retrieves chunks via cosine-similarity search using the `match_chunks`
    PostgreSQL function (backed by an HNSW index on the embedding column).
    """

    def __init__(self, embedder: OpenAIEmbedder, pg_pool: PgPool) -> None:
        self._embedder = embedder
        self._pg_pool = pg_pool

    async def search(self, query: str, top_k: int = 10) -> SearchResult:
        t0 = time.monotonic()
        try:
            embedding = await self._embedder.embed_one(query)
            vec = _vec_str(embedding)

            async with self._pg_pool.get_connection() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM match_chunks($1::vector, $2)",
                    vec,
                    top_k,
                )

            hits = _parse_hits_vector(rows)
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
    Retrieves chunks via weighted combination of vector similarity and PostgreSQL full-text search, using the `hybrid_search` SQL function.

    The default text_weight=0.3 gives 70 % weight to semantic similarity and 30 % to keyword matching — a good general-purpose balance.
    """

    def __init__(self, embedder: OpenAIEmbedder, pg_pool: PgPool) -> None:
        self._embedder = embedder
        self._pg_pool = pg_pool

    async def search(
        self,
        query: str,
        top_k: int = 10,
        text_weight: float = 0.3,
    ) -> SearchResult:
        t0 = time.monotonic()
        try:
            embedding = await self._embedder.embed_one(query)
            vec = _vec_str(embedding)

            async with self._pg_pool.get_connection() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)",
                    vec,
                    query,
                    top_k,
                    text_weight,
                )

            hits = _parse_hits_hybrid(rows)
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
