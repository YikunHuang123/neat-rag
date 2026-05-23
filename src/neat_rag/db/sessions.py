import json
from typing import List, Optional
from asyncpg import Connection

from neat_rag.models import Session, Message, MessageRole
from neat_rag.exceptions import RecordNotFoundError, DatabaseError
from neat_rag.logger import get_logger

logger = get_logger(__name__)


class SessionRepository:
    """
    Repository for managing Session and Message aggregates in the database.
    Session and its messages are always queried together, so they share one repo.
    """

    def __init__(self, connection: Connection):
        self.conn = connection

    async def create_session(self, user_id: Optional[str] = None, title: str = "New Chat") -> Session:
        """Create a new chat session and return it."""
        try:
            row = await self.conn.fetchrow(
                """
                INSERT INTO sessions (user_id, title)
                VALUES ($1, $2)
                RETURNING id::text, user_id, title, created_at, updated_at
                """,
                user_id,
                title,
            )
            return Session(
                id=row["id"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except Exception as e:
            logger.error("Failed to create session", user_id=user_id, error=str(e))
            raise DatabaseError(f"Failed to create session: {e}")

    async def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        """Fetch a session by its ID, optionally filtered by user_id."""
        try:
            if user_id is not None:
                row = await self.conn.fetchrow(
                    """
                    SELECT id::text, user_id, title, created_at, updated_at
                    FROM sessions
                    WHERE id = $1::uuid AND user_id = $2
                    """,
                    session_id,
                    user_id,
                )
            else:
                row = await self.conn.fetchrow(
                    """
                    SELECT id::text, user_id, title, created_at, updated_at
                    FROM sessions
                    WHERE id = $1::uuid
                    """,
                    session_id,
                )
            if not row:
                return None
            return Session(
                id=row["id"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except Exception as e:
            logger.error("Failed to fetch session", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to fetch session: {e}")

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Session]:
        """List sessions, optionally filtered by user_id."""
        query = "SELECT id::text, user_id, title, created_at, updated_at FROM sessions"
        params: list = []

        if user_id is not None:
            params.append(user_id)
            query += f" WHERE user_id = ${len(params)}"

        query += " ORDER BY created_at DESC"
        params.append(limit)
        query += f" LIMIT ${len(params)}"
        params.append(offset)
        query += f" OFFSET ${len(params)}"

        try:
            rows = await self.conn.fetch(query, *params)
            return [
                Session(
                    id=row["id"],
                    user_id=row["user_id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to list sessions", user_id=user_id, error=str(e))
            raise DatabaseError(f"Failed to list sessions: {e}")

    async def count_sessions(self, user_id: Optional[str] = None) -> int:
        """Return the total number of sessions, optionally filtered by user_id."""
        query = "SELECT COUNT(*) FROM sessions"
        params = []
        if user_id is not None:
            query += " WHERE user_id = $1"
            params.append(user_id)
        
        try:
            val = await self.conn.fetchval(query, *params)
            return int(val or 0)
        except Exception as e:
            logger.error("Failed to count sessions", user_id=user_id, error=str(e))
            raise DatabaseError(f"Failed to count sessions: {e}")

    async def update_session_title(self, session_id: str, title: str, user_id: Optional[str] = None) -> None:
        """Update the title of a session."""
        try:
            if user_id is not None:
                result = await self.conn.execute(
                    """
                    UPDATE sessions
                    SET title = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1::uuid AND user_id = $3
                    """,
                    session_id,
                    title,
                    user_id,
                )
            else:
                result = await self.conn.execute(
                    """
                    UPDATE sessions
                    SET title = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1::uuid
                    """,
                    session_id,
                    title,
                )
            if result == "UPDATE 0":
                raise RecordNotFoundError(f"Session with ID {session_id} not found.")
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update session title", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to update session title: {e}")

    async def delete_session(self, session_id: str, user_id: Optional[str] = None) -> None:
        """
        Delete a session.
        Associated messages are removed automatically by ON DELETE CASCADE.
        """
        try:
            if user_id is not None:
                result = await self.conn.execute(
                    "DELETE FROM sessions WHERE id = $1::uuid AND user_id = $2", 
                    session_id, 
                    user_id
                )
            else:
                result = await self.conn.execute(
                    "DELETE FROM sessions WHERE id = $1::uuid", 
                    session_id
                )
            if result == "DELETE 0":
                raise RecordNotFoundError(f"Session with ID {session_id} not found.")
            logger.info("Deleted session", session_id=session_id)
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to delete session", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to delete session: {e}")

    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Append a message to a session and update the session's updated_at timestamp."""
        try:
            row = await self.conn.fetchrow(
                """
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES ($1::uuid, $2, $3, $4::jsonb)
                RETURNING id::text, session_id::text, role, content, metadata, created_at
                """,
                session_id,
                role.value,
                content,
                json.dumps(metadata or {}),
            )
            # Keep session's updated_at current so list_sessions sorts correctly
            await self.conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = $1::uuid",
                session_id,
            )
            return Message(
                id=row["id"],
                session_id=row["session_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
            )
        except Exception as e:
            logger.error("Failed to add message", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to add message: {e}")

    async def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """
        Fetch messages for a session in chronological order.
        Fixes the original db_utils f-string LIMIT injection by using a bound parameter.
        """
        try:
            if limit is not None:
                rows = await self.conn.fetch(
                    """
                    SELECT * FROM (
                        SELECT id::text, session_id::text, role, content, metadata, created_at
                        FROM messages
                        WHERE session_id = $1::uuid
                        ORDER BY created_at DESC
                        LIMIT $2
                    ) sub
                    ORDER BY created_at ASC
                    """,
                    session_id,
                    limit,
                )
            else:
                rows = await self.conn.fetch(
                    """
                    SELECT id::text, session_id::text, role, content, metadata, created_at
                    FROM messages
                    WHERE session_id = $1::uuid
                    ORDER BY created_at ASC
                    """,
                    session_id,
                )
            return [
                Message(
                    id=row["id"],
                    session_id=row["session_id"],
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to fetch messages", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to fetch messages: {e}")
