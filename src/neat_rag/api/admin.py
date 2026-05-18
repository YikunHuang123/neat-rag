"""Admin endpoints for API key and invite token management.

All endpoints under /admin require either:
- ADMIN_BOOTSTRAP_KEY (operator key set in .env)
- A regular API key with scopes=["admin"]
"""
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from neat_rag.api.deps import get_connection
from neat_rag.api.middleware import generate_api_key, verify_admin
from neat_rag.db.api_keys import ApiKeyRepository
from neat_rag.db.invites import InviteTokenRepository
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
    _owner: Optional[str] = Depends(verify_admin),
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
    _owner: Optional[str] = Depends(verify_admin),
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
    _owner: Optional[str] = Depends(verify_admin),
):
    """Revoke an API key by its ID."""
    repo = ApiKeyRepository(conn)
    await repo.delete_key(key_id)
    logger.info("API key revoked", key_id=key_id)


# ── Invite token endpoints ────────────────────────────────────────────────────

class CreateInviteRequest(BaseModel):
    owner: str
    expires_in_days: int = 7


class InviteSummary(BaseModel):
    id: str
    token: str
    owner: str
    used: bool
    used_at: Optional[str]
    expires_at: str
    created_at: str


@router.post("/invites", response_model=InviteSummary, status_code=201)
async def create_invite(
    body: CreateInviteRequest,
    conn=Depends(get_connection),
    _owner: Optional[str] = Depends(verify_admin),
):
    """Generate a one-time invite token for a user to self-redeem an API key."""
    if body.expires_in_days < 1 or body.expires_in_days > 90:
        raise HTTPException(status_code=422, detail="expires_in_days must be between 1 and 90.")
    token = secrets.token_urlsafe(24)
    expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)
    repo = InviteTokenRepository(conn)
    invite = await repo.create(token=token, owner=body.owner, expires_at=expires_at)
    logger.info("Invite token created", owner=body.owner, token_id=invite.id)
    return InviteSummary(
        id=invite.id,
        token=invite.token,
        owner=invite.owner,
        used=invite.used,
        used_at=invite.used_at.isoformat() if invite.used_at else None,
        expires_at=invite.expires_at.isoformat(),
        created_at=invite.created_at.isoformat(),
    )


@router.get("/invites", response_model=List[InviteSummary])
async def list_invites(
    conn=Depends(get_connection),
    _owner: Optional[str] = Depends(verify_admin),
):
    """List all invite tokens."""
    repo = InviteTokenRepository(conn)
    invites = await repo.list_all()
    return [
        InviteSummary(
            id=i.id,
            token=i.token,
            owner=i.owner,
            used=i.used,
            used_at=i.used_at.isoformat() if i.used_at else None,
            expires_at=i.expires_at.isoformat(),
            created_at=i.created_at.isoformat(),
        )
        for i in invites
    ]


@router.delete("/invites/{token_id}", status_code=204)
async def delete_invite(
    token_id: str,
    conn=Depends(get_connection),
    _owner: Optional[str] = Depends(verify_admin),
):
    """Revoke an invite token by its ID."""
    repo = InviteTokenRepository(conn)
    await repo.delete(token_id)
    logger.info("Invite token revoked", token_id=token_id)
