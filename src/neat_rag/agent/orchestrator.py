"""
Agent orchestrator — the single entry point for all RAG queries.

The Agent and retriever objects are initialised lazily on the first call to
run_query() so that importing this module does not require a live LLM API key
or a database connection. This makes testing and CLI usage straightforward.

Public surface:
    run_query(question, session_id, user_id) → str
    get_agent()                               → Agent[AgentContext, str]

Circular-import note: AgentContext and tool functions live in tools.py so
that orchestrator.py can import them without tools.py importing back.
"""
from __future__ import annotations

from pydantic_ai import Agent

from neat_rag.agent.memory import load_history
from neat_rag.agent.prompts import build_system_prompt
from neat_rag.agent.tools import (
    AgentContext,
    get_document,
    hybrid_search,
    AGENT_TOOLS,
    vector_search,
)
from neat_rag.db.documents import DocumentRepository
from neat_rag.db.pool import pg_pool
from neat_rag.db.sessions import SessionRepository
from neat_rag.logger import get_logger
from neat_rag.models import MessageRole
from neat_rag.providers.embedding import get_embedder
from neat_rag.providers.llm import get_llm
from neat_rag.retrieval.retrievers import HybridRetriever, VectorRetriever

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons — None until the first run_query() call.
# This avoids requiring an LLM API key / DB connection at import time.
# ---------------------------------------------------------------------------
_agent: Agent[AgentContext, str] | None = None
_vector_retriever: VectorRetriever | None = None
_hybrid_retriever: HybridRetriever | None = None


def get_agent() -> Agent[AgentContext, str]:
    """
    Return the singleton neat_agent, creating it on first access.
    Thread-safety note: asyncio is single-threaded so a simple global is fine.
    """
    global _agent, _vector_retriever, _hybrid_retriever

    if _agent is None:
        embedder = get_embedder()
        _vector_retriever = VectorRetriever(embedder, pg_pool)
        _hybrid_retriever = HybridRetriever(embedder, pg_pool)

        _agent = Agent(
            get_llm(),
            deps_type=AgentContext,
            system_prompt=build_system_prompt(),
        )
        
        for tool in AGENT_TOOLS:
            _agent.tool(tool)

        logger.info(
            "neat_agent initialised",
            tools=[t.__name__ for t in AGENT_TOOLS],
        )

    return _agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_query(
    question: str,
    session_id: str,
    user_id: str | None = None,
) -> str:
    """
    Run a RAG query through neat_agent and persist the exchange to the session.

    Args:
        question:   The user's question (natural language).
        session_id: UUID of the active chat session (must already exist in DB).
        user_id:    Optional caller identity for audit / personalisation.

    Returns:
        The agent's final text answer as a plain string.
    """
    agent = get_agent()

    async with pg_pool.get_connection() as conn:
        doc_repo = DocumentRepository(conn)
        session_repo = SessionRepository(conn)

        # Load prior conversation turns so the agent has multi-turn context
        history = await load_history(session_repo, session_id)

        ctx = AgentContext(
            session_id=session_id,
            vector_retriever=_vector_retriever,  # type: ignore[arg-type]
            hybrid_retriever=_hybrid_retriever,  # type: ignore[arg-type]
            doc_repo=doc_repo,
            session_repo=session_repo,
            user_id=user_id,
        )

        logger.info(
            "Running agent query",
            session_id=session_id,
            history_turns=len(history),
            question=question[:80],
        )

        result = await agent.run(
            question,
            deps=ctx,
            message_history=history,
        )

        answer: str = result.output

        # Persist both sides of the exchange in the session
        await session_repo.add_message(session_id, MessageRole.USER, question)
        await session_repo.add_message(session_id, MessageRole.AGENT, answer)

        logger.info(
            "Agent query complete",
            session_id=session_id,
            answer_chars=len(answer),
        )
        return answer
