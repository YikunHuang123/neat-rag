"""
Reranking step: takes raw retrieval hits and re-scores them with a cross-encoder.

Flow:  retrieve top-CANDIDATE_K  →  cross-encode  →  return top-FINAL_K to agent.
"""
from neat_rag.exceptions import RerankerProviderError
from neat_rag.logger import get_logger
from neat_rag.models import SearchHit

logger = get_logger(__name__)


async def rerank_hits(
    query: str,
    hits: list[SearchHit],
    reranker,
    top_k: int,
) -> list[SearchHit]:
    """
    Rerank a list of SearchHits using a cross-encoder and return the top hits.

    Args:
        query:    The original user query (used as the anchor for scoring).
        hits:     Candidate hits from initial retrieval (e.g. top-20 or top-50).
        reranker: A CrossEncoderReranker or CohereReranker instance.
        top_k:    Number of hits to keep after reranking.

    Returns:
        At most `top_k` SearchHit objects sorted by reranker score (descending).
        Falls back to original order on reranker failure.
    """
    if not hits:
        return hits

    passages = [h.content for h in hits]
    try:
        scores = await reranker.score(query, passages)
    except RerankerProviderError:
        raise
    except Exception as e:
        logger.error("Reranker scoring failed, falling back to original order", error=str(e))
        return hits[:top_k]

    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
    logger.info("Reranked hits", input=len(hits), top_k=top_k)

    return [
        hit.model_copy(update={"score": float(score)})
        for hit, score in ranked[:top_k]
    ]
