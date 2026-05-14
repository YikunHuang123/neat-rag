"""
Agent tools and runtime context for neat_agent.

AgentContext is defined here (not in orchestrator.py) so that tool functions
can reference it without creating a circular import:

    tools.py   → no imports from orchestrator.py
    orchestrator.py → imports AgentContext + tool functions from tools.py
"""
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Dict

from pydantic_ai import RunContext

from neat_rag.db.documents import DocumentRepository
from neat_rag.db.sessions import SessionRepository
from neat_rag.db.pool import PgPool
from neat_rag.exceptions import ToolExecutionError
from neat_rag.logger import get_logger
from neat_rag.retrieval.retrievers import HybridRetriever, VectorRetriever

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
    user_id: str | None = None
    default_top_k: int = 10
    default_text_weight: float = 0.3


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def hybrid_search(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search for relevant information using both semantic and keyword matching.

    This is the recommended tool for most questions. It combines vector
    similarity with PostgreSQL full-text search for the best overall accuracy.

    Args:
        query: The search query — a question or keywords to look up.
        limit: Maximum number of chunks to return (1–50, default 10).
    """
    try:
        limit = max(1, min(limit, 50))
        result = await ctx.deps.hybrid_retriever.search(
            query,
            top_k=limit,
            text_weight=ctx.deps.default_text_weight,
        )
        return [
            {
                "content": h.content,
                "score": round(h.score, 4),
                "document_title": h.document_title,
                "document_source": h.document_source,
                "chunk_id": h.chunk_id,
            }
            for h in result.hits
        ]
    except Exception as e:
        logger.error("hybrid_search tool error", query=query[:60], error=str(e))
        return []


async def vector_search(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search for semantically similar content in the knowledge base.

    Use this for conceptual or abstract questions where exact keyword
    matching is less important than semantic understanding.

    Args:
        query: The search query.
        limit: Maximum number of chunks to return (1–50, default 10).
    """
    try:
        limit = max(1, min(limit, 50))
        result = await ctx.deps.vector_retriever.search(query, top_k=limit)
        return [
            {
                "content": h.content,
                "score": round(h.score, 4),
                "document_title": h.document_title,
                "document_source": h.document_source,
                "chunk_id": h.chunk_id,
            }
            for h in result.hits
        ]
    except Exception as e:
        logger.error("vector_search tool error", query=query[:60], error=str(e))
        return []


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

