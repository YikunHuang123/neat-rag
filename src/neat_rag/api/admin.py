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
from neat_rag.exceptions import RecordNotFoundError
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
    is_active: bool
    can_upload: bool
    can_delete: bool
    can_chat: bool


class UpdatePermissionsRequest(BaseModel):
    is_active: bool = True
    can_upload: bool = True
    can_delete: bool = True
    can_chat: bool = True


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
    return [_to_key_summary(k) for k in keys]


@router.delete("/keys/{key_id}", status_code=204)
async def delete_api_key(
    key_id: str,
    conn=Depends(get_connection),
    _owner: Optional[str] = Depends(verify_admin),
):
    """Revoke an API key by its ID."""
    repo = ApiKeyRepository(conn)
    try:
        await repo.delete_key(key_id)
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found.")
    logger.info("API key revoked", key_id=key_id)


@router.patch("/keys/{key_id}/permissions", response_model=KeySummary)
async def update_key_permissions(
    key_id: str,
    body: UpdatePermissionsRequest,
    conn=Depends(get_connection),
    _owner: Optional[str] = Depends(verify_admin),
):
    """Update the permission flags for a user's API key."""
    logger.debug("Updating key permissions", key_id=key_id, body=body.model_dump())
    repo = ApiKeyRepository(conn)
    try:
        key = await repo.update_permissions(
            key_id=key_id,
            is_active=body.is_active,
            can_upload=body.can_upload,
            can_delete=body.can_delete,
            can_chat=body.can_chat,
        )
    except RecordNotFoundError:
        logger.warning("Update failed: Key not found", key_id=key_id)
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found.")
    logger.info(
        "Key permissions updated",
        key_id=key_id,
        is_active=body.is_active,
        can_upload=body.can_upload,
        can_delete=body.can_delete,
        can_chat=body.can_chat,
    )
    return _to_key_summary(key)


# ── Invite token endpoints ────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_key_summary(k) -> KeySummary:
    return KeySummary(
        id=k.id,
        owner=k.owner,
        scopes=k.scopes,
        created_at=k.created_at.isoformat(),
        last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
        is_active=k.is_active,
        can_upload=k.can_upload,
        can_delete=k.can_delete,
        can_chat=k.can_chat,
    )
