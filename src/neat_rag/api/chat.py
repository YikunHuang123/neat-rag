import json
from typing import AsyncIterator, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from neat_rag.agent.memory import load_history
from neat_rag.agent.orchestrator import build_agent_context, get_agent, run_query
from neat_rag.api.deps import get_connection
from neat_rag.api.middleware import limiter, verify_api_key
from neat_rag.api.schemas import ChatRequest, ChatResponse, CitationResponse
from neat_rag.config import settings
from neat_rag.db.pool import pg_pool
from neat_rag.db.sessions import SessionRepository
from neat_rag.logger import get_logger
from neat_rag.models import MessageRole
from neat_rag.retrieval.citation import extract_citations

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    body: ChatRequest,
    conn: asyncpg.Connection = Depends(get_connection),
    _owner: Optional[str] = Depends(verify_api_key),
):
    """Blocking chat — waits for the full agent response before returning."""
    session_repo = SessionRepository(conn)
    if await session_repo.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{body.session_id}' not found.")

    answer, citations = await run_query(
        body.message,
        body.session_id,
        search_type=body.search_type,
    )
    return ChatResponse(
        session_id=body.session_id,
        content=answer,
        citations=[
            CitationResponse(
                citation_number=c.citation_number,
                document_title=c.document_title,
                document_source=c.document_source,
                content_snippet=c.content_snippet,
            )
            for c in citations
        ]
    )


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    conn: asyncpg.Connection = Depends(get_connection),
    _owner: Optional[str] = Depends(verify_api_key),
):
    """
    Streaming chat via Server-Sent Events.

    Event types:
      {"type": "delta",  "content": "<text chunk>"}
      {"type": "done",   "message_id": "<uuid>", "citations": [...]}
      {"type": "error",  "content": "<message>"}
    Followed by a final "data: [DONE]" sentinel.
    """
    session_repo = SessionRepository(conn)
    if await session_repo.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail=f"Session '{body.session_id}' not found.")

    # Load history while the request connection is still open.
    history = await load_history(session_repo, body.session_id)

    async def generate() -> AsyncIterator[str]:
        agent = get_agent()  # ensures lazy retrievers are initialised
        ctx = build_agent_context(body.session_id, search_type=body.search_type)

        chunks: list[str] = []
        try:
            # Fallback for providers whose streaming tool-call JSON deltas are incompatible
            # with pydantic-ai's stream reconstruction logic.
            if settings.LLM_PROVIDER.lower() in ["deepseek", "ollama"]:
                import asyncio
                result = await agent.run(
                    body.message,
                    deps=ctx,
                    message_history=history,
                )
                
                # Simulate streaming output so the UI still animates
                text = result.output
                chunk_size = 15
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'delta', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.01)
            else:
                async with agent.run_stream(
                    body.message,
                    deps=ctx,
                    message_history=history,
                ) as result:
                    async for chunk in result.stream_text(delta=True):
                        chunks.append(chunk)
                        yield f"data: {json.dumps({'type': 'delta', 'content': chunk})}\n\n"
        except Exception as exc:
            logger.error("Streaming agent error", session_id=body.session_id, error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        full_answer = "".join(chunks)
        citations = extract_citations(full_answer, ctx.citations)

        # Persist both sides of the exchange using a fresh connection.
        async with pg_pool.get_connection() as persist_conn:
            repo = SessionRepository(persist_conn)
            await repo.add_message(body.session_id, MessageRole.USER, body.message)
            msg = await repo.add_message(body.session_id, MessageRole.AGENT, full_answer)

        # Prepare citations for the frontend
        citation_data = [
            {
                "citation_number": c.citation_number,
                "document_title": c.document_title,
                "document_source": c.document_source,
                "content_snippet": c.content_snippet,
            }
            for c in citations
        ]

        yield f"data: {json.dumps({'type': 'done', 'message_id': msg.id, 'citations': citation_data})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
