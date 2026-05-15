"""Security middleware components for Neat-RAG API.

Three responsibilities:
1. API Key authentication  — verify X-API-Key header against hashed DB records
2. Rate limiting           — slowapi, configurable limits per endpoint group
3. Helper utilities        — key generation, hashing

Request-ID injection lives in api/__init__.py as an HTTP middleware since it
must wrap every request including rate-limited/auth-rejected ones.
"""
from __future__ import annotations

import hashlib
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

from neat_rag.api.schemas import ErrorResponse
from neat_rag.config import settings
from neat_rag.db.jobs import ApiKeyRepository
from neat_rag.db.pool import pg_pool
from neat_rag.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter — module-level singleton shared by all routers via app.state
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
)


# ---------------------------------------------------------------------------
# API Key scheme
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key — the value stored in the database."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """
    Generate a cryptographically-random API key.
    Returns (raw_key, hashed_key).  Only raw_key is shown to the user once;
    only hashed_key is persisted.
    """
    raw = "nrag_" + secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


async def verify_api_key(
    raw_key: Optional[str] = Security(_api_key_header),
) -> Optional[str]:
    """
    FastAPI dependency — verify the X-API-Key header.

    - If ENABLE_AUTH is False, always passes (returns None).
    - If the header is missing or the key is unknown, raises 401.
    - On success returns the key owner string.

    Usage::

        @router.get("/protected")
        async def endpoint(owner: str = Depends(verify_api_key)):
            ...
    """
    if not settings.ENABLE_AUTH:
        return None

    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    hashed = hash_api_key(raw_key)

    async with pg_pool.get_connection() as conn:
        repo = ApiKeyRepository(conn)
        key_record = await repo.get_key_by_hash(hashed)

    if key_record is None:
        logger.warning("Invalid API key attempt", hashed_prefix=hashed[:8])
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Touch last_used_at asynchronously — best-effort, don't fail the request
    try:
        async with pg_pool.get_connection() as conn:
            repo = ApiKeyRepository(conn)
            await repo.touch_last_used(hashed)
    except Exception:
        pass

    return key_record.owner


# ---------------------------------------------------------------------------
# Rate-limit exceeded response handler
# ---------------------------------------------------------------------------

async def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a consistent 429 JSON body instead of slowapi's plain-text default."""
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error="Rate limit exceeded",
            detail=f"Too many requests. Limit: {exc.detail}",
        ).model_dump(),
        headers={"Retry-After": "60"},
    )
