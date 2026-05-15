"""Admin endpoints for API key management.

All endpoints under /admin require a valid API key when ENABLE_AUTH=True.
In development (ENABLE_AUTH=False) these endpoints are freely accessible so
you can bootstrap your first key without a chicken-and-egg problem.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from neat_rag.api.deps import get_connection
from neat_rag.api.middleware import generate_api_key, verify_api_key
from neat_rag.db.jobs import ApiKeyRepository
from neat_rag.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


# ── DTOs ─────────────────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    owner: str
    scopes: List[str] = []


class CreateKeyResponse(BaseModel):
    """Returned once on creation — raw_key is never stored and cannot be recovered."""
    id: str
    owner: str
    scopes: List[str]
    raw_key: str


class KeySummary(BaseModel):
    id: str
    owner: str
    scopes: List[str]
    created_at: str
    last_used_at: Optional[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/keys", response_model=CreateKeyResponse, status_code=201)
async def create_api_key(
    body: CreateKeyRequest,
    conn=Depends(get_connection),
    _owner: Optional[str] = Depends(verify_api_key),
):
    """Create a new API key. The raw key is returned once and cannot be retrieved again."""
    raw_key, hashed_key = generate_api_key()
    repo = ApiKeyRepository(conn)
    key = await repo.create_key(hashed_key=hashed_key, owner=body.owner, scopes=body.scopes)
    logger.info("API key created", owner=body.owner, key_id=key.id)
    return CreateKeyResponse(
        id=key.id,
        owner=key.owner,
        scopes=key.scopes,
        raw_key=raw_key,
    )


@router.get("/keys", response_model=List[KeySummary])
async def list_api_keys(
    owner: Optional[str] = None,
    conn=Depends(get_connection),
    _owner: Optional[str] = Depends(verify_api_key),
):
    """List all API keys (hashed — raw keys are never returned after creation)."""
    repo = ApiKeyRepository(conn)
    keys = await repo.list_keys(owner=owner)
    return [
        KeySummary(
            id=k.id,
            owner=k.owner,
            scopes=k.scopes,
            created_at=k.created_at.isoformat(),
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}", status_code=204)
async def delete_api_key(
    key_id: str,
    conn=Depends(get_connection),
    _owner: Optional[str] = Depends(verify_api_key),
):
    """Revoke an API key by its ID."""
    repo = ApiKeyRepository(conn)
    await repo.delete_key(key_id)
    logger.info("API key revoked", key_id=key_id)
