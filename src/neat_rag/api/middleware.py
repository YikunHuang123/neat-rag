"""Security middleware components for Neat-RAG API.

Three responsibilities:
1. API Key authentication  — verify X-API-Key header against hashed DB records
2. Rate limiting           — slowapi, configurable limits per endpoint group
3. Helper utilities        — key generation, hashing

Auth model (no on/off flag — always active):
  - No key supplied  → anonymous access (user_id=None)
  - Regular key      → authenticated user; owner string always returned
  - ADMIN_BOOTSTRAP_KEY or key with scopes=["admin"] → admin endpoints allowed

DATABASE_SHARED only affects document/vector queries, NOT sessions or jobs.
Use doc_scope(owner) at the call site for document reads and vector searches.

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
from neat_rag.db.api_keys import ApiKeyRepository
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


def doc_scope(owner: Optional[str]) -> Optional[str]:
    """
    Return the user_id to use for document/vector read queries.

    When DATABASE_SHARED=True, returns None so queries span all users' documents.
    When False, returns the real owner for strict isolation.

    Apply this at the call site for document reads and vector searches.
    Do NOT use for sessions, jobs, or write operations — those always use the real owner.
    """
    return None if settings.DATABASE_SHARED else owner


async def verify_api_key(
    raw_key: Optional[str] = Security(_api_key_header),
) -> Optional[str]:
    """
    FastAPI dependency for regular endpoints.

    - No key supplied → 401 (key required).
    - ADMIN_BOOTSTRAP_KEY → returns "__bootstrap__" (admin identity, no DB lookup).
    - Valid DB key        → returns the key owner string.
    - Invalid key         → 401.

    Sessions and jobs use this value directly for strict per-user isolation.
    Document/vector queries should wrap the result with doc_scope(owner).
    """
    if not raw_key:
        raise HTTPException(status_code=401, detail="API key required")

    # Operator bootstrap key — fast path, no DB roundtrip
    if settings.ADMIN_BOOTSTRAP_KEY and raw_key == settings.ADMIN_BOOTSTRAP_KEY:
        return "__bootstrap__"

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


async def verify_admin(
    raw_key: Optional[str] = Security(_api_key_header),
) -> Optional[str]:
    """
    FastAPI dependency for admin-only endpoints.

    - No key        → 403 (admin key required).
    - Bootstrap key → pass (operator identity).
    - Invalid key   → 401 (lets UI distinguish invalid from non-admin).
    - Valid non-admin key → 403.
    - Valid admin key     → returns owner.
    """
    if not raw_key:
        raise HTTPException(status_code=403, detail="Admin key required")

    # Operator bootstrap key — fast path, no DB roundtrip
    if settings.ADMIN_BOOTSTRAP_KEY and raw_key == settings.ADMIN_BOOTSTRAP_KEY:
        return "__bootstrap__"

    hashed = hash_api_key(raw_key)

    async with pg_pool.get_connection() as conn:
        repo = ApiKeyRepository(conn)
        key_record = await repo.get_key_by_hash(hashed)

    if key_record is None:
        logger.warning("Unauthorized admin access attempt (invalid key)", hashed_prefix=hashed[:8])
        raise HTTPException(status_code=401, detail="Invalid API key")

    if "admin" not in key_record.scopes:
        raise HTTPException(status_code=403, detail="Admin access required")

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
