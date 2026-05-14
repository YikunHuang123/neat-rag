from typing import AsyncIterator

import asyncpg
from fastapi import Depends

from neat_rag.db.pool import pg_pool
from neat_rag.ingestion.pipeline import IngestionPipeline
from neat_rag.providers.embedding import get_embedder as _build_embedder


# --- DB connection (one per request, shared by all repos in that request) ---

async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    async with pg_pool.get_connection() as conn:
        yield conn


# --- Lazy embedder singleton (created once, reused across requests) ---

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = _build_embedder()
    return _embedder


# --- Pipeline factory (new instance per request, embedder is shared) ---

def get_pipeline(embedder=Depends(get_embedder)) -> IngestionPipeline:
    return IngestionPipeline(pg_pool=pg_pool, embedder=embedder)
