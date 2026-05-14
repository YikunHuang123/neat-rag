"""
Conversation memory: converts between DB Message rows and pydantic-ai
ModelMessage objects so that multi-turn context is preserved across requests.

Flow:
    DB messages (Message)  →  load_history()  →  list[ModelMessage]
                                                   ↓ passed to agent.run()
    agent output           →  orchestrator saves new messages to DB
"""
from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from neat_rag.db.sessions import SessionRepository
from neat_rag.logger import get_logger
from neat_rag.models import Message, MessageRole

logger = get_logger(__name__)

# Limit how many past messages are sent to the LLM to keep the context window
# manageable. Each pair of (user + agent) messages counts as 2.
_MAX_HISTORY = 20


async def load_history(
    session_repo: SessionRepository,
    session_id: str,
) -> list[ModelMessage]:
    """
    Load the recent message history for a session from the database and
    convert it to the pydantic-ai ModelMessage format expected by agent.run().

    Returns an empty list when no prior messages exist (first turn).
    """
    db_messages = await session_repo.get_session_messages(
        session_id, limit=_MAX_HISTORY
    )
    history = _to_pydantic_messages(db_messages)
    logger.debug("Loaded message history", session_id=session_id, count=len(history))
    return history


def _to_pydantic_messages(messages: list[Message]) -> list[ModelMessage]:
    """
    Convert DB Message records to pydantic-ai ModelMessage objects.

    User messages  → ModelRequest  with a UserPromptPart
    Agent messages → ModelResponse with a TextPart
    """
    result: list[ModelMessage] = []
    for msg in messages:
        if msg.role == MessageRole.USER:
            result.append(
                ModelRequest(parts=[UserPromptPart(content=msg.content)])
            )
        elif msg.role == MessageRole.AGENT:
            result.append(
                ModelResponse(
                    parts=[TextPart(content=msg.content)],
                    model_name="history",
                )
            )
    return result
