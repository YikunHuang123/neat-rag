"""
Agent tools and runtime context for neat_agent.

AgentContext is defined here (not in orchestrator.py) so that tool functions
can reference it without creating a circular import:

    tools.py   → no imports from orchestrator.py
    orchestrator.py → imports AgentContext + tool functions from tools.py
"""
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from pydantic_ai import RunContext

from neat_rag.config import settings
from neat_rag.db.documents import DocumentRepository
from neat_rag.db.sessions import SessionRepository
from neat_rag.db.pool import PgPool
from neat_rag.exceptions import ToolExecutionError
from neat_rag.logger import get_logger
from neat_rag.models import SearchType, Citation
from neat_rag.retrieval.retrievers import HybridRetriever, VectorRetriever
from neat_rag.providers.reranker import CrossEncoderReranker, CohereReranker
from neat_rag.retrieval.rerank import rerank_hits
from neat_rag.retrieval.rewrite import hyde_rewrite, multi_query_rewrite, rrf_merge
from neat_rag.retrieval.citation import build_citation_context

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Runtime context — injected by pydantic-ai into every tool call via deps
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Carries per-request dependencies into every tool call."""
    session_id: str
    pg_pool: PgPool
    vector_retriever: VectorRetriever
    hybrid_retriever: HybridRetriever
    reranker: Optional[CrossEncoderReranker | CohereReranker] = None
    user_id: str | None = None
    search_type: SearchType = SearchType.HYBRID
    default_top_k: int = 10
    default_text_weight: float = 0.3
    # Citation tracking — populated by search tools, read by orchestrator
    citations: List[Citation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def hybrid_search(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int = 10,
) -> str:
    """
    Search for relevant information using both semantic and keyword matching.

    This is the recommended tool for most questions. It combines vector
    similarity with PostgreSQL full-text search for the best overall accuracy.

    Args:
        query: The search query — a question or keywords to look up.
        limit: Maximum number of chunks to return (1–50, default 10).
    """
    return await _run_advanced_search(ctx, query, limit, search_mode="hybrid")


async def vector_search(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int = 10,
) -> str:
    """
    Search for semantically similar content in the knowledge base.

    Use this for conceptual or abstract questions where exact keyword
    matching is less important than semantic understanding.

    Args:
        query: The search query.
        limit: Maximum number of chunks to return (1–50, default 10).
    """
    return await _run_advanced_search(ctx, query, limit, search_mode="vector")


async def _run_advanced_search(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int,
    search_mode: str = "hybrid",
) -> str:
    """Internal helper to handle the full retrieval -> rerank -> citation pipeline."""
    try:
        limit = max(1, min(limit, 50))
        
        # 1. Query Rewriting (HyDE / Multi-Query)
        # Note: In a production tool, we might skip this if the query is very short
        queries = [query]
        if settings.ENABLE_HYDE:
            hyde_query = await hyde_rewrite(query)
            queries.append(hyde_query)
        elif settings.ENABLE_MULTI_QUERY:
            queries = await multi_query_rewrite(query, n=settings.MULTI_QUERY_COUNT)

        # 2. Retrieval
        all_results = []
        for q in queries:
            # Use candidate_k if reranking is enabled to give the reranker more to work with
            k = settings.RETRIEVE_CANDIDATE_K if settings.ENABLE_RERANK else limit
            
            if search_mode == "hybrid":
                res = await ctx.deps.hybrid_retriever.search(
                    q, top_k=k, text_weight=ctx.deps.default_text_weight
                )
            else:
                res = await ctx.deps.vector_retriever.search(q, top_k=k)
            all_results.append(res.hits)

        # 3. Merge (RRF if multi-query)
        hits = all_results[0] if len(all_results) == 1 else rrf_merge(all_results)

        # 4. Rerank
        if settings.ENABLE_RERANK and ctx.deps.reranker and hits:
            hits = await rerank_hits(
                query=query, 
                hits=hits, 
                reranker=ctx.deps.reranker, 
                top_k=limit
            )
        else:
            hits = hits[:limit]

        if not hits:
            return "No relevant information found in the knowledge base."

        # 5. Citation Formatting
        context_str, citations = build_citation_context(hits)
        
        # Store citations in the context so the orchestrator can retrieve them later
        # We append to avoid losing citations from previous tool calls in the same run
        ctx.deps.citations.extend(citations)

        return context_str

    except Exception as e:
        logger.error(f"{search_mode}_search tool error", query=query[:60], error=str(e))
        return f"Error during search: {str(e)}"


async def get_document(
    ctx: RunContext[AgentContext],
    document_id: str,
) -> Any:
    """
    Retrieve the complete content and metadata of a specific document.

    Use this when you need the full document rather than individual chunks,
    for example when you already know the document_id from a previous search.
    
    CRITICAL: The `document_id` parameter MUST be a valid UUID string (e.g.,
    "123e4567-e89b-12d3-a456-426614174000"). Do NOT pass the document title.

    Args:
        document_id: The UUID string of the document to retrieve.
    """
    try:
        # Validate UUID format to prevent database errors and guide the Agent
        try:
            uuid.UUID(document_id)
        except ValueError:
            return f"Error: '{document_id}' is not a valid UUID. Please use the technical 'id' field returned by list_documents, not the title."

        async with ctx.deps.pg_pool.get_connection() as conn:
            doc_repo = DocumentRepository(conn)
            doc = await doc_repo.get_document(document_id)
            if doc is None:
                return f"Document with ID {document_id} not found."
            
            chunks = await doc_repo.get_document_chunks(document_id)
            return {
                "id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "chunk_count": doc.chunk_count,
                "content": "\n\n".join(c.content for c in chunks),
                "created_at": doc.created_at.isoformat(),
            }
    except Exception as e:
        logger.error("get_document tool error", document_id=document_id, error=str(e))
        return f"Error retrieving document: {str(e)}"


async def list_documents(
    ctx: RunContext[AgentContext],
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    List all documents available in the knowledge base with their metadata.

    Use this to understand what information sources are available before
    searching, or when the user asks what documents are in the system.

    Args:
        limit:  Maximum number of documents to return (1–100, default 20).
        offset: Number of documents to skip for pagination (default 0).
    """
    try:
        limit = max(1, min(limit, 100))
        async with ctx.deps.pg_pool.get_connection() as conn:
            doc_repo = DocumentRepository(conn)
            docs = await doc_repo.list_documents(limit=limit, offset=offset)
            return [
                {
                    "id": d.id,
                    "title": d.title,
                    "source": d.source,
                    "chunk_count": d.chunk_count,
                    "created_at": d.created_at.isoformat(),
                }
                for d in docs
            ]
    except Exception as e:
        logger.error("list_documents tool error", error=str(e))
        return []

# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------
AGENT_TOOLS = [
    hybrid_search,
    vector_search,
    get_document,
    list_documents,
]

