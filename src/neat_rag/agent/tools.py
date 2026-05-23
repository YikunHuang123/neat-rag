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
from neat_rag.db.vector_store import VectorStoreBase
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
    vector_store: VectorStoreBase
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
    document_ids: list[str] | None = None,
) -> str:
    """
    Search for specific facts, answers, or details within the document content.

    Use this tool when you need to answer "how" or "why" questions, or when
    you need to extract specific information from the text of documents.
    It combines semantic and keyword search for high precision.

    Args:
        query:        The specific question or information to look for.
        limit:        Maximum number of chunks to return (1–50, default 10).
        document_ids: Optional list of document UUIDs to restrict the search.
    """
    return await _run_advanced_search(ctx, query, limit, search_mode="hybrid", document_ids=document_ids)


async def vector_search(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int = 10,
    document_ids: list[str] | None = None,
) -> str:
    """
    Search for conceptual or abstract content using semantic similarity.

    Use this for open-ended or high-level questions where exact keyword
    matching is less important than understanding the overall meaning.

    Args:
        query:        The search query.
        limit:        Maximum number of chunks to return (1–50, default 10).
        document_ids: Optional list of document UUIDs to restrict the search.
    """
    return await _run_advanced_search(ctx, query, limit, search_mode="vector", document_ids=document_ids)


async def _run_advanced_search(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int,
    search_mode: str = "hybrid",
    document_ids: list[str] | None = None,
) -> str:
    """Internal helper to handle the full retrieval -> rerank -> citation pipeline."""
    try:
        limit = max(1, min(limit, 50))

        # 1. Query Rewriting (HyDE / Multi-Query)
        queries = [query]
        if settings.ENABLE_HYDE:
            hyde_query = await hyde_rewrite(query)
            queries.append(hyde_query)
        elif settings.ENABLE_MULTI_QUERY:
            queries = await multi_query_rewrite(query, n=settings.MULTI_QUERY_COUNT)

        # 2. Retrieval
        all_results = []
        for q in queries:
            k = settings.RETRIEVE_CANDIDATE_K if settings.ENABLE_RERANK else limit

            if search_mode == "hybrid":
                res = await ctx.deps.hybrid_retriever.search(
                    q, top_k=k, text_weight=ctx.deps.default_text_weight,
                    user_id=ctx.deps.user_id, document_ids=document_ids,
                )
            else:
                res = await ctx.deps.vector_retriever.search(
                    q, top_k=k, user_id=ctx.deps.user_id, document_ids=document_ids,
                )
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
            return (
                "No relevant information found in the knowledge base. "
                "CRITICAL: Do NOT use any [n] citation markers in your response, "
                "as there is no context to reference."
            )

        # 5. Context Augmentation — prepend document-level structured metadata so
        #    the LLM can reason over document-wide facts (dates, parties, type)
        #    without needing extra tool calls.
        preamble = ""
        if settings.ENABLE_CONTEXT_AUGMENTATION:
            unique_doc_ids = list({h.document_id for h in hits})
            async with ctx.deps.pg_pool.get_connection() as conn:
                doc_repo = DocumentRepository(conn)
                schemas = await doc_repo.get_document_schemas_batch(unique_doc_ids)
            if schemas:
                lines = ["[Document Metadata Context — use to support reasoning, do not cite directly]"]
                for doc_id, schema in schemas.items():
                    title = next(
                        (h.document_title for h in hits if h.document_id == doc_id),
                        doc_id,
                    )
                    lines.append(
                        f'  "{title}": type={schema.get("document_type", "unknown")}'
                        f', key_dates={schema.get("key_dates", [])}'
                        f', entities={schema.get("entities", [])[:5]}'
                        f', summary={schema.get("summary", "")}'
                    )
                preamble = "\n".join(lines) + "\n\n"

        # 6. Citation Formatting — start numbering after already-accumulated citations
        offset = len(ctx.deps.citations)
        context_str, citations = build_citation_context(hits, offset=offset)
        ctx.deps.citations.extend(citations)

        return preamble + context_str

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
            # Filter by user_id for security
            doc = await doc_repo.get_document(document_id, user_id=ctx.deps.user_id)
            if doc is None:
                return f"Document with ID {document_id} not found."

        # Fetch chunks via the active vector store backend
        chunks = await ctx.deps.vector_store.get_chunks_by_document(document_id)
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
            docs = await doc_repo.list_documents(limit=limit, offset=offset, user_id=ctx.deps.user_id)
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

async def search_by_schema_field(
    ctx: RunContext[AgentContext],
    document_type: str | None = None,
    entity: str | None = None,
    topic: str | None = None,
) -> str:
    """
    Find documents matching structured metadata (type, entities, topics).

    Use this tool when the query asks to identify, list, or filter documents
    by high-level attributes rather than searching their internal text.
    It is the primary tool for queries like:
      - "What documents mention [Person/Company]?"
      - "List all [Category] documents."
      - "Show me documents about [Topic]."

    After calling this tool, you can use the returned IDs with hybrid_search
    to dive deeper into the content if needed.

    Args:
        document_type: Category (e.g. 'contract', 'invoice', 'report').
        entity:        Named entity (person, organization, location).
        topic:         High-level topic or keyword.
    """
    if not any([document_type, entity, topic]):
        return "Provide at least one filter: document_type, entity, or topic."

    try:
        # Use JSONB @> containment for document_type
        metadata_filter: dict | None = None
        if document_type:
            metadata_filter = {"structured_schema": {"document_type": document_type}}

        async with ctx.deps.pg_pool.get_connection() as conn:
            doc_repo = DocumentRepository(conn)
            # Increase limit to 100 to make post-filtering more effective
            docs = await doc_repo.list_documents(
                limit=100,
                metadata_filter=metadata_filter,
                user_id=ctx.deps.user_id,
            )

        # Post-filter for entity / topic (case-insensitive substring match)
        if entity:
            entity_lower = entity.lower()
            docs = [
                d for d in docs
                if any(
                    entity_lower in e.lower()
                    for e in d.metadata.get("structured_schema", {}).get("entities", [])
                )
            ]
        if topic:
            topic_lower = topic.lower()
            docs = [
                d for d in docs
                if any(
                    topic_lower in t.lower()
                    for t in d.metadata.get("structured_schema", {}).get("key_topics", [])
                )
            ]

        # Further narrow to documents that actually have a structured_schema
        docs = [d for d in docs if d.metadata.get("structured_schema")]

        if not docs:
            return (
                "No documents found matching the specified schema filters. "
                "Try hybrid_search if you are looking for specific text content."
            )

        # Generate citations for the metadata findings so the model can reference them
        offset = len(ctx.deps.citations)
        lines = [f"Found {len(docs)} document(s) matching the schema filter:"]
        
        for i, d in enumerate(docs, 1):
            schema = d.metadata.get("structured_schema", {})
            doc_type = schema.get("document_type", "unknown")
            summary = schema.get("summary", "No summary available.")
            citation_num = offset + i
            
            lines.append(f"  [{citation_num}] \"{d.title}\" (ID: {d.id}, type={doc_type}): {summary}")
            
            ctx.deps.citations.append(Citation(
                citation_number=citation_num,
                chunk_id=f"doc-metadata-{d.id}",
                document_id=d.id,
                document_title=d.title,
                document_source=d.source,
                content_snippet=f"Document Metadata: type={doc_type}, summary={summary}"
            ))

        lines.append(
            "\nTo search content within these specific documents, call:"
            "\n  hybrid_search(query=<your question>, document_ids=[<id1>, <id2>, ...])"
        )
        return "\n".join(lines)

    except Exception as e:
        logger.error("search_by_schema_field tool error", error=str(e))
        return f"Error filtering documents by schema: {e}"


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------
AGENT_TOOLS = [
    hybrid_search,
    vector_search,
    search_by_schema_field,
    get_document,
    list_documents,
]
