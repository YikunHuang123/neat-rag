"""
Schema extraction — uses a Pydantic AI agent to pull structured metadata
from raw document text, enforcing a JSON schema defined by DocumentStructuredData.

The agent's result_type is the Pydantic model itself: pydantic-ai validates the
LLM output against the schema and retries on parse failure, so callers always
receive a fully-typed object or None (on unrecoverable failure).

XML serialisation is provided alongside JSON so the same extracted data can be
stored in two formats, satisfying downstream consumers that prefer XML.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

from pydantic_ai import Agent

from neat_rag.config import settings
from neat_rag.logger import get_logger
from neat_rag.models import DocumentStructuredData
from neat_rag.providers.llm import get_llm

logger = get_logger(__name__)

# Lazy singleton — not created until the first extract_document_schema() call
# so importing this module never requires a live LLM key or network connection.
_agent: Optional[Agent] = None


def _get_agent() -> Agent:
    """Return the singleton schema extraction agent, creating it on first access."""
    global _agent
    if _agent is None:
        types_hint = ", ".join(settings.SCHEMA_DOCUMENT_TYPES)
        _agent = Agent(
            get_llm(),
            output_type=DocumentStructuredData,
            system_prompt=(
                "You are a document metadata extractor. "
                "Given raw document text, extract structured metadata as JSON. "
                f"For document_type, choose exactly one of: {types_hint}. "
                "For entities, list up to 10 key named entities "
                "(people, organisations, locations). "
                "For key_topics, list up to 5 main topics or themes. "
                "For key_dates, list dates in ISO 8601 format (YYYY-MM-DD) when found; "
                "omit if none are present. "
                "The summary must be at most 2 concise sentences."
            ),
        )
        logger.info(
            "Schema extraction agent initialised",
            document_types=settings.SCHEMA_DOCUMENT_TYPES,
        )
    return _agent


def to_xml(data: DocumentStructuredData) -> str:
    """
    Serialise a DocumentStructuredData instance to a compact XML string.

    Produces a <document_schema> root element with child elements for each
    field.  List fields (entities, key_topics, key_dates) are wrapped in a
    parent element containing one child per item.

    Example output (abbreviated):
        <document_schema>
          <title>Annual Report 2023</title>
          <summary>Covers fiscal results …</summary>
          <document_type>report</document_type>
          <entities><entity>Acme Corp</entity></entities>
          <key_topics><topic>revenue</topic></key_topics>
          <key_dates><date>2023-12-31</date></key_dates>
        </document_schema>
    """
    root = ET.Element("document_schema")

    ET.SubElement(root, "title").text = data.title
    ET.SubElement(root, "summary").text = data.summary
    ET.SubElement(root, "document_type").text = data.document_type

    entities_el = ET.SubElement(root, "entities")
    for entity in data.entities:
        ET.SubElement(entities_el, "entity").text = entity

    topics_el = ET.SubElement(root, "key_topics")
    for topic in data.key_topics:
        ET.SubElement(topics_el, "topic").text = topic

    dates_el = ET.SubElement(root, "key_dates")
    for date in data.key_dates:
        ET.SubElement(dates_el, "date").text = date

    return ET.tostring(root, encoding="unicode", xml_declaration=False)


async def extract_document_schema(content: str) -> Optional[DocumentStructuredData]:
    """
    Run the schema extraction agent on the given document content.

    Truncates to SCHEMA_EXTRACTION_MAX_CHARS before calling the LLM so that
    token costs stay bounded for large documents — the beginning of a document
    is usually the most metadata-rich part (title, abstract, introduction).

    Returns None on any failure so the ingestion pipeline can continue cleanly
    without the schema rather than marking the whole ingestion job as failed.
    """
    text = content[: settings.SCHEMA_EXTRACTION_MAX_CHARS]
    try:
        result = await _get_agent().run(text)
        return result.output
    except Exception as e:
        logger.warning(
            "Schema extraction failed — schema will be omitted for this document",
            error=str(e),
        )
        return None
