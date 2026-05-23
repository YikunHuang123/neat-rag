import asyncio
import json
from typing import Any, AsyncIterator, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent

from neat_rag.agent.memory import load_history
from neat_rag.agent.orchestrator import (
    build_agent_context,
    get_agent,
    run_query,
    inject_language_directive,
    load_relevant_images,
)
from neat_rag.agent.prompts import build_system_prompt
from neat_rag.agent.title import generate_session_title
from neat_rag.agent.tools import AGENT_TOOLS, AgentContext
from neat_rag.api.deps import get_connection
from neat_rag.api.middleware import doc_scope, limiter, require_chat_permission
from neat_rag.api.schemas import ChatRequest, ChatResponse, CitationResponse
from neat_rag.config import settings
from neat_rag.db.pool import pg_pool
from neat_rag.db.sessions import SessionRepository
from neat_rag.logger import get_logger
from neat_rag.models import MessageRole
from neat_rag.providers.llm import get_llm_with_override
from neat_rag.retrieval.citation import extract_citations, sanitize_answer

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])


def _make_override_agent(
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
) -> tuple[Agent, Any]:
    """Build a one-off agent with a custom LLM; reuses the singleton's retrievers.

    Returns (agent, llm) so the caller can reuse the same llm for title generation.
    """
    llm = get_llm_with_override(provider, model, api_key, base_url)
    ag: Agent = Agent(llm, deps_type=AgentContext, system_prompt=build_system_prompt())
    for tool in AGENT_TOOLS:
        ag.tool(tool)
    return ag, llm


async def _update_title(session_id: str, user_message: str, llm: Any = None) -> str:
    """Generate a session title, persist it, and return it."""
    title = await generate_session_title(user_message, llm=llm)
    async with pg_pool.get_connection() as conn:
        await SessionRepository(conn).update_session_title(session_id, title)
    logger.info("Session title updated", session_id=session_id, title=title)
    return title


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    body: ChatRequest,
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(require_chat_permission),
):
    """Blocking chat — waits for the full agent response before returning."""
    session_repo = SessionRepository(conn)
    session = await session_repo.get_session(body.session_id, user_id=owner)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{body.session_id}' not found.")

    override_agent = None
    override_llm = None
    if body.llm_provider:
        try:
            override_agent, override_llm = _make_override_agent(
                body.llm_provider,
                body.llm_model or "",
                body.llm_api_key or "",
                body.llm_base_url or "",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"LLM override error: {exc}")

    answer, citations = await run_query(
        body.message,
        body.session_id,
        user_id=doc_scope(owner),
        search_type=body.search_type,
        agent_override=override_agent,
    )

    if session.title == "New Chat":
        task = asyncio.create_task(_update_title(body.session_id, body.message, llm=override_llm))
        task.add_done_callback(
            lambda t: logger.error("title update failed", error=str(t.exception())) if t.exception() else None
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
    owner: Optional[str] = Depends(require_chat_permission),
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
    session = await session_repo.get_session(body.session_id, user_id=owner)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{body.session_id}' not found.")

    # Load history while the request connection is still open.
    history = await load_history(session_repo, body.session_id)

    async def generate() -> AsyncIterator[str]:
        # Always call get_agent() to ensure retrievers are initialised.
        _default_agent = get_agent()

        stream_override_llm = None
        if body.llm_provider:
            try:
                agent, stream_override_llm = _make_override_agent(
                    body.llm_provider,
                    body.llm_model or "",
                    body.llm_api_key or "",
                    body.llm_base_url or "",
                )
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
                yield "data: [DONE]\n\n"
                return
        else:
            agent = _default_agent

        # Sanitize and prepare the prompt, matching run_query behavior
        sanitized_message = body.message.encode("utf-8", "ignore").decode("utf-8")
        query_message = inject_language_directive(sanitized_message)

        ctx = build_agent_context(body.session_id, user_id=doc_scope(owner), search_type=body.search_type)

        # Handle multimodal input if enabled
        user_content: Any = query_message
        if settings.LLM_MULTIMODAL:
            image_parts = await load_relevant_images(sanitized_message, doc_scope(owner))
            if image_parts:
                user_content = [query_message] + image_parts
                logger.info("Multimodal (stream): images attached", count=len(image_parts))

        # Determine the effective provider to decide whether streaming is supported.
        effective_provider = (body.llm_provider or settings.LLM_PROVIDER).lower()

        chunks: list[str] = []
        try:
            # Fallback for providers whose streaming tool-call JSON deltas are
            # incompatible with pydantic-ai's stream reconstruction logic.
            if effective_provider in ["deepseek", "ollama"]:
                result = await agent.run(
                    user_content,
                    deps=ctx,
                    message_history=history,
                )
                # Simulate streaming output so the UI still animates.
                text = result.output
                chunk_size = 15
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'delta', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.01)
            else:
                async with agent.run_stream(
                    user_content,
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

        # Sanitize the full answer before persistence and final response
        full_answer = sanitize_answer(full_answer, ctx.citations)

        if not citations and ctx.citations:
            logger.warning("No citation markers in response; falling back to all retrieved citations", session_id=body.session_id)
            citations = ctx.citations

        # Persist both sides of the exchange using a fresh connection.
        async with pg_pool.get_connection() as persist_conn:
            repo = SessionRepository(persist_conn)
            await repo.add_message(body.session_id, MessageRole.USER, sanitized_message)
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

        new_title: str | None = None
        if session.title == "New Chat":
            new_title = await _update_title(body.session_id, body.message, llm=stream_override_llm)

        yield f"data: {json.dumps({'type': 'done', 'message_id': msg.id, 'citations': citation_data, 'title': new_title})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
