from __future__ import annotations

from pydantic_ai import Agent

from neat_rag.logger import get_logger
from neat_rag.providers.llm import get_llm

logger = get_logger(__name__)

_title_agent: Agent | None = None

_SYSTEM_PROMPT = (
    "You are a concise chat-session title generator. "
    "Given the user's first message, output a short title that captures the topic. "
    "Rules:\n"
    "- Be extremely concise (e.g., 5-12 words or equivalent length)\n"
    "- No quotes, no trailing punctuation, no explanation — title only\n"
    "- ALWAYS output the title in the SAME language used in the message\n"
    "- If the message has no clear topic or is just a greeting, "
    "output a generic 'New Chat' equivalent in that same language\n"
    "- Never copy the message verbatim; always extract the core topic"
)


def _get_title_agent() -> Agent:
    global _title_agent
    if _title_agent is None:
        _title_agent = Agent(get_llm(), system_prompt=_SYSTEM_PROMPT)
    return _title_agent


async def generate_session_title(user_message: str) -> str:
    """Return a short LLM-generated title for the session, or 'New Chat' on failure."""
    try:
        result = await _get_title_agent().run(user_message)
        title = result.output.strip().strip("\"'").strip()
        return title[:60] if title else "New Chat"
    except Exception as exc:
        logger.warning("Session title generation failed", error=str(exc))
        return "New Chat"
