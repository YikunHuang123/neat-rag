from typing import Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from neat_rag.api.deps import get_connection
from neat_rag.api.middleware import verify_api_key
from neat_rag.api.schemas import (
    MessageListResponse,
    MessageResponse,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    SessionUpdateRequest,
)
from neat_rag.db.sessions import SessionRepository
from neat_rag.exceptions import RecordNotFoundError
from neat_rag.models import Message, Session

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreateRequest,
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    repo = SessionRepository(conn)
    # Always use the authenticated owner; ignore body.user_id to prevent spoofing
    session = await repo.create_session(user_id=owner, title=body.title)
    return _to_session_response(session)


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    user_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    repo = SessionRepository(conn)
    # When authenticated, always scope to the token owner regardless of query param
    effective_user_id = owner if owner is not None else user_id
    sessions = await repo.list_sessions(user_id=effective_user_id, limit=limit, offset=offset)
    return SessionListResponse(items=[_to_session_response(s) for s in sessions])


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    repo = SessionRepository(conn)
    session = await repo.get_session(session_id, user_id=owner)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return _to_session_response(session)


@router.get("/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    limit: Optional[int] = Query(None, ge=1, le=500),
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    repo = SessionRepository(conn)
    session = await repo.get_session(session_id, user_id=owner)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    messages = await repo.get_session_messages(session_id, limit=limit)
    return MessageListResponse(items=[_to_message_response(m) for m in messages])


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    body: SessionUpdateRequest,
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    repo = SessionRepository(conn)
    try:
        await repo.update_session_title(session_id, body.title, user_id=owner)
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    session = await repo.get_session(session_id, user_id=owner)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return _to_session_response(session)  # type: ignore[arg-type]


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    repo = SessionRepository(conn)
    try:
        await repo.delete_session(session_id, user_id=owner)
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_session_response(s: Session) -> SessionResponse:
    return SessionResponse(
        id=s.id,
        user_id=s.user_id,
        title=s.title,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


def _to_message_response(m: Message) -> MessageResponse:
    return MessageResponse(
        id=m.id,
        session_id=m.session_id,
        role=m.role,
        content=m.content,
        created_at=m.created_at,
        metadata=m.metadata,
    )
