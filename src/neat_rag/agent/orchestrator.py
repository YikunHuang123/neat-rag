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
from neat_rag.db.vector_store import get_vector_store
from neat_rag.logger import get_logger
from neat_rag.providers.embedding import get_embedder
from neat_rag.providers.llm import get_llm
from neat_rag.providers.reranker import get_reranker, CrossEncoderReranker, CohereReranker
from neat_rag.retrieval.retrievers import HybridRetriever, VectorRetriever
from neat_rag.retrieval.citation import extract_citations
from neat_rag.models import MessageRole, SearchType, Citation

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons — None until the first run_query() call.
# This avoids requiring an LLM API key / DB connection at import time.
# ---------------------------------------------------------------------------
_agent: Agent[AgentContext, str] | None = None
_vector_retriever: VectorRetriever | None = None
_hybrid_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | CohereReranker | None = None


def get_agent() -> Agent[AgentContext, str]:
    """
    Return the singleton neat_agent, creating it on first access.
    Thread-safety note: asyncio is single-threaded so a simple global is fine.
    """
    global _agent, _vector_retriever, _hybrid_retriever, _reranker

    if _agent is None:
        embedder = get_embedder()
        store = get_vector_store()
        _vector_retriever = VectorRetriever(embedder, store)
        _hybrid_retriever = HybridRetriever(embedder, store)
        _reranker = get_reranker()

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

def build_agent_context(
    session_id: str,
    user_id: str | None = None,
    search_type: SearchType = SearchType.HYBRID,
) -> AgentContext:
    """
    Build an AgentContext with the lazy-initialised retrievers.
    Callers must invoke get_agent() first to ensure the retrievers exist.
    """
    return AgentContext(
        session_id=session_id,
        pg_pool=pg_pool,
        vector_store=get_vector_store(),
        vector_retriever=_vector_retriever,  # type: ignore[arg-type]
        hybrid_retriever=_hybrid_retriever,  # type: ignore[arg-type]
        reranker=_reranker,
        user_id=user_id,
        search_type=search_type,
    )


def _inject_language_directive(question: str) -> str:
    """
    Prepend a language directive to reinforce that the model must reply in the
    same language as the question — needed for weaker local models (e.g. llama3.1)
    that ignore system-prompt instructions when RAG context is in another language.
    """
    return f"[IMPORTANT: Respond in the same language as this question, regardless of the language of the retrieved context.]\n{question}"


async def run_query(
    question: str,
    session_id: str,
    user_id: str | None = None,
    search_type: SearchType = SearchType.HYBRID,
    agent_override: "Agent[AgentContext, str] | None" = None,
) -> tuple[str, list[Citation]]:
    """
    Run a RAG query through neat_agent and persist the exchange to the session.

    Args:
        question:       The user's question (natural language).
        session_id:     UUID of the active chat session (must already exist in DB).
        user_id:        Optional caller identity for audit / personalisation.
        search_type:    The search strategy to use (vector or hybrid).
        agent_override: Optional pre-built Agent to use instead of the singleton
                        (e.g. when the UI selects a custom LLM provider).

    Returns:
        (answer_text, citations_referenced)
    """
    _default = get_agent()  # always call to ensure retrievers are initialised
    agent = agent_override if agent_override is not None else _default

    # Load prior conversation turns so the agent has multi-turn context
    async with pg_pool.get_connection() as conn:
        session_repo = SessionRepository(conn)
        history = await load_history(session_repo, session_id)

    # Sanitize question to remove potential Unicode surrogates that cause encoding errors
    question = question.encode("utf-8", "ignore").decode("utf-8")
    original_question = question

    # Prepend explicit language directive so weak local models don't default to a language
    # when retrieved RAG context is in a different language.
    question = _inject_language_directive(question)

    ctx = build_agent_context(session_id, user_id, search_type=search_type)

    logger.info(
        "Running agent query",
        session_id=session_id,
        history_turns=len(history),
        search_type=search_type,
        question=question[:80],
    )

    result = await agent.run(
        question,
        deps=ctx,
        message_history=history,
    )

    answer: str = result.output
    # Filter only the citations that the agent actually mentioned in its response
    citations = extract_citations(answer, ctx.citations)

    # Persist both sides of the exchange in the session
    async with pg_pool.get_connection() as conn:
        session_repo = SessionRepository(conn)
        await session_repo.add_message(session_id, MessageRole.USER, original_question)
        await session_repo.add_message(session_id, MessageRole.AGENT, answer)

    logger.info(
        "Agent query complete",
        session_id=session_id,
        answer_chars=len(answer),
        citations=len(citations),
    )
    return answer, citations
