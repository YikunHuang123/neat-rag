from datetime import datetime, timezone

import asyncpg
from fastapi import APIRouter, Depends

from neat_rag.api.deps import get_connection
from neat_rag.api.schemas import HealthLive, HealthReady
from neat_rag.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthLive)
async def liveness():
    """K8s liveness probe — always returns 200 if the process is running."""
    return HealthLive()


@router.get("/health/ready", response_model=HealthReady)
async def readiness(conn: asyncpg.Connection = Depends(get_connection)):
    """K8s readiness probe — checks that the database is reachable."""
    db_ok = False
    try:
        await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception as exc:
        logger.warning("DB health check failed", error=str(exc))

    return HealthReady(
        status="ok" if db_ok else "degraded",
        db=db_ok,
        timestamp=datetime.now(timezone.utc),
    )
