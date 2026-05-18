"""
Query transformation for improved retrieval recall.

HyDE (Hypothetical Document Embeddings)
    Ask the LLM to draft a short hypothetical answer, then embed that text
    instead of the raw query. The hypothesis lives in "answer space" which
    tends to be closer to real answers in the vector index.

Multi-Query
    Generate N paraphrases of the query, retrieve with each independently,
    and merge the ranked lists via Reciprocal Rank Fusion (RRF).
"""
from typing import Optional

from pydantic_ai import Agent

from neat_rag.logger import get_logger
from neat_rag.models import SearchHit

logger = get_logger(__name__)

# Lazily-initialised singleton — shares the provider config from providers/llm.py
# so any LLM_PROVIDER setting is honoured, including Anthropic and local models.
_rewrite_agent: Optional[Agent] = None


def _get_rewrite_agent() -> Agent:
    global _rewrite_agent
    if _rewrite_agent is None:
        from neat_rag.providers.llm import get_llm
        _rewrite_agent = Agent(get_llm())
    return _rewrite_agent

_HYDE_PROMPT = """\
Write a short passage (2–4 sentences) that directly answers the following question.
The passage should read like an excerpt from a reference document, not a conversational reply.
Do NOT use first-person language.

Question: {query}

Passage:"""

_MULTI_QUERY_PROMPT = """\
Generate {n} different phrasings of the following question.
Each phrasing must preserve the original intent but use different vocabulary or structure.
Output ONLY the questions, one per line, with no numbering or extra text.

Question: {query}

Phrasings:"""


async def _llm_complete(prompt: str) -> str:
    """Single LLM completion for query rewriting. Delegates to the configured provider via providers/llm.py."""
    agent = _get_rewrite_agent()
    result = await agent.run(prompt)
    return result.output or ""


async def hyde_rewrite(query: str) -> str:
    """
    Generate a hypothetical passage that would answer `query`.
    Returns the hypothetical text to be embedded instead of the raw query.
    Falls back to the original query on any failure.
    """
    try:
        text = (await _llm_complete(_HYDE_PROMPT.format(query=query))).strip()
        if not text:
            return query
        logger.info("HyDE rewrite done", original_chars=len(query), hypo_chars=len(text))
        return text
    except Exception as e:
        logger.warning("HyDE rewrite failed, using original query", error=str(e))
        return query


async def multi_query_rewrite(query: str, n: int = 3) -> list[str]:
    """
    Generate `n` paraphrases of `query`.
    Returns the original query + up to `n` paraphrases (deduped, original first).
    Falls back to [query] on any failure.
    """
    try:
        text = await _llm_complete(_MULTI_QUERY_PROMPT.format(query=query, n=n))
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        # Dedup while preserving order; keep original query first.
        seen: set[str] = {query}
        queries = [query]
        for line in lines[:n]:
            if line not in seen:
                seen.add(line)
                queries.append(line)
        logger.info("Multi-query rewrite done", variants=len(queries))
        return queries
    except Exception as e:
        logger.warning("Multi-query rewrite failed, using original query", error=str(e))
        return [query]


def rrf_merge(results_list: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """
    Merge multiple independently ranked hit lists using Reciprocal Rank Fusion.

    A higher RRF score means the chunk appeared near the top of several lists.
    The `k` constant (default 60) dampens the impact of rank position — a
    standard value from the original RRF paper.
    """
    rrf_scores: dict[str, float] = {}
    hits_map: dict[str, SearchHit] = {}

    for hits in results_list:
        for rank, hit in enumerate(hits):
            rrf_scores[hit.chunk_id] = (
                rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            )
            hits_map.setdefault(hit.chunk_id, hit)

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    return [hits_map[cid] for cid in sorted_ids]
