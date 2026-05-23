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

def build_citation_context(hits: list[SearchHit], offset: int = 0) -> tuple[str, list[Citation]]:
    """
    Format a list of SearchHits into a cited context block for the agent.

    Each hit is rendered as:
        [n] <chunk content>
            Source: <document_title> (<document_source>)

    Numbers start from offset+1 so that successive calls within the same
    session produce a single monotonically-increasing sequence rather than
    resetting to [1] each time.

    Args:
        hits:   Search hits to format.
        offset: Number of citations already accumulated (pass
                ``len(ctx.deps.citations)`` before extending the list).

    Returns:
        (formatted_context_string, citations)
        where citations[i].citation_number == offset + i + 1.
    """
    if not hits:
        return "", []

    citations: list[Citation] = []
    parts: list[str] = []

    for i, hit in enumerate(hits, start=offset + 1):
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

    header = (
        "CITATION RULE: You MUST place [n] immediately after EVERY fact you use from these sources.\n"
        'Example: "The system uses FastAPI [1] and PostgreSQL [2]."\n\n'
    )
    footer = (
        "\n\nREMINDER: Every factual sentence MUST be immediately followed by [n]. "
        "A response with no [n] markers is incomplete and invalid."
    )
    return header + "\n\n".join(parts) + footer, citations


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


def sanitize_answer(answer: str, valid_citations: list[Citation]) -> str:
    """
    Remove [n] markers from the answer that do not exist in the valid_citations list.

    This handles "benign hallucinations" where the LLM adds citation markers
    to appear professional even when no documents were retrieved or when
    referencing non-existent chunks.
    """
    valid_numbers = {c.citation_number for c in valid_citations}

    def _replace_marker(match: re.Match) -> str:
        num = int(match.group(1))
        return match.group(0) if num in valid_numbers else ""

    # Replace [n] with empty string if n is not in valid_numbers
    sanitized = re.sub(r"\[(\d+)\]", _replace_marker, answer)

    # Clean up double spaces or trailing spaces created by removal
    sanitized = re.sub(r" +", " ", sanitized).strip()
    return sanitized
