"""
Citation injection and extraction for RAG responses.

Flow:
  1. build_citation_context(hits)
       → formats hits as "[1] content\\n    Source: title (source)"
       → returns (formatted_str, [Citation, ...]) aligned 1-to-1 with hits

  2. extract_citations(answer, citations)
       → scans the agent's answer for [n] markers
       → returns only the Citations that were actually referenced, in order
"""
import re

from neat_rag.models import Citation, SearchHit
from neat_rag.logger import get_logger

logger = get_logger(__name__)

def build_citation_context(hits: list[SearchHit]) -> tuple[str, list[Citation]]:
    """
    Format a list of SearchHits into a cited context block for the agent.

    Each hit is rendered as:
        [n] <chunk content>
            Source: <document_title> (<document_source>)

    The citation numbers are sequential starting from 1 and correspond
    directly to the position of each hit in the input list.

    Returns:
        (formatted_context_string, citations)
        where citations[i].citation_number == i + 1.
    """
    if not hits:
        return "", []

    citations: list[Citation] = []
    parts: list[str] = []

    for i, hit in enumerate(hits, start=1):
        citations.append(
            Citation(
                citation_number=i,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_title=hit.document_title,
                document_source=hit.document_source,
                content_snippet=hit.content[:300],
            )
        )
        parts.append(
            f"[{i}] {hit.content}\n"
            f"    Source: {hit.document_title} ({hit.document_source})"
        )

    return "\n\n".join(parts), citations


def extract_citations(answer: str, citations: list[Citation]) -> list[Citation]:
    """
    Parse [n] markers from the agent's answer and return the matched Citations.

    Args:
        answer:    The full text of the agent's response.
        citations: The complete citation list built from citation_hits.

    Returns:
        Citations referenced in `answer`, sorted by citation_number.
    """
    # 1. Identify all [n] style markers in the text
    found_numbers = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}

    # 2. Get the set of valid numbers we actually provided to the agent
    valid_numbers = {c.citation_number for c in citations}

    # 3. Intersection: only keep what's real
    real_refs = found_numbers & valid_numbers

    # 4. Log if the agent hallucinated a citation number we didn't give it
    hallucinated = found_numbers - valid_numbers
    if hallucinated:
        logger.warning(
            "Agent generated hallucinated citation markers",
            numbers=sorted(list(hallucinated)),
            valid_range=f"1-{len(citations)}" if citations else "none"
        )

    return sorted(
        [c for c in citations if c.citation_number in real_refs],
        key=lambda c: c.citation_number,
    )
