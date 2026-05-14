import json
from typing import AsyncIterator

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from neat_rag.agent.memory import load_history
from neat_rag.agent.orchestrator import build_agent_context, get_agent, run_query
from neat_rag.api.deps import get_connection
from neat_rag.api.schemas import ChatRequest, ChatResponse
from neat_rag.db.pool import pg_pool
from neat_rag.db.sessions import SessionRepository
from neat_rag.logger import get_logger
from neat_rag.models import MessageRole

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    conn: asyncpg.Connection = Depends(get_connection),
):
    """Blocking chat — waits for the full agent response before returning."""
    session_repo = SessionRepository(conn)
    if await session_repo.get_session(request.session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    answer = await run_query(
        request.message,
        request.session_id,
        search_type=request.search_type,
    )
    return ChatResponse(session_id=request.session_id, content=answer)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    conn: asyncpg.Connection = Depends(get_connection),
):
    """
    Streaming chat via Server-Sent Events.

    Event types:
      {"type": "delta",  "content": "<text chunk>"}
      {"type": "done",   "message_id": "<uuid>"}
      {"type": "error",  "content": "<message>"}
    Followed by a final "data: [DONE]" sentinel.
    """
    session_repo = SessionRepository(conn)
    if await session_repo.get_session(request.session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    # Load history while the request connection is still open.
    history = await load_history(session_repo, request.session_id)

    async def generate() -> AsyncIterator[str]:
        agent = get_agent()  # ensures lazy retrievers are initialised
        ctx = build_agent_context(request.session_id, search_type=request.search_type)

        chunks: list[str] = []
        try:
            async with agent.run_stream(
                request.message,
                deps=ctx,
                message_history=history,
            ) as result:
                async for chunk in result.stream_text(delta=True):
                    chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'delta', 'content': chunk})}\n\n"
        except Exception as exc:
            logger.error("Streaming agent error", session_id=request.session_id, error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        full_answer = "".join(chunks)

        # Persist both sides of the exchange using a fresh connection.
        async with pg_pool.get_connection() as persist_conn:
            repo = SessionRepository(persist_conn)
            await repo.add_message(request.session_id, MessageRole.USER, request.message)
            msg = await repo.add_message(request.session_id, MessageRole.AGENT, full_answer)

        yield f"data: {json.dumps({'type': 'done', 'message_id': msg.id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
