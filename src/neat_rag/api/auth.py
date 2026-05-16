"""Public auth endpoint — redeem an invite token to obtain an API key.

No authentication required on POST /auth/redeem. The invite token itself
acts as the authorization credential.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from neat_rag.api.deps import get_connection
from neat_rag.api.middleware import generate_api_key
from neat_rag.db.api_keys import ApiKeyRepository
from neat_rag.db.invites import InviteTokenRepository
from neat_rag.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class RedeemRequest(BaseModel):
    token: str


class RedeemResponse(BaseModel):
    owner: str
    raw_key: str


@router.post("/redeem", response_model=RedeemResponse, status_code=201)
async def redeem_invite(body: RedeemRequest, conn=Depends(get_connection)):
    """Exchange a valid invite token for a new API key.

    The raw key is returned exactly once. The invite token is consumed
    immediately and cannot be reused.
    """
    invite_repo = InviteTokenRepository(conn)
    invite = await invite_repo.get_by_token(body.token)

    if invite is None:
        raise HTTPException(status_code=404, detail="Invite token not found.")

    if invite.used:
        raise HTTPException(status_code=410, detail="Invite token has already been used.")

    now = datetime.now(timezone.utc)
    expires = invite.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if now > expires:
        raise HTTPException(status_code=410, detail="Invite token has expired.")

    raw_key, hashed_key = generate_api_key()
    key_repo = ApiKeyRepository(conn)
    await key_repo.create_key(hashed_key=hashed_key, owner=invite.owner)
    await invite_repo.mark_used(body.token)

    logger.info("Invite redeemed, API key issued", owner=invite.owner)
    return RedeemResponse(owner=invite.owner, raw_key=raw_key)
