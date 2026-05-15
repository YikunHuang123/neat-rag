from typing import List, Optional
from asyncpg import Connection

from neat_rag.models import Feedback, FeedbackRating
from neat_rag.exceptions import DatabaseError
from neat_rag.logger import get_logger

logger = get_logger(__name__)


def _row_to_feedback(row) -> Feedback:
    return Feedback(
        id=row["id"],
        message_id=row["message_id"],
        user_id=row["user_id"],
        rating=FeedbackRating(row["rating"]),
        comment=row["comment"],
        created_at=row["created_at"],
    )


class FeedbackRepository:
    """
    Repository for user feedback on agent messages (feedback table).
    Each message can have at most one feedback entry per user.
    """

    def __init__(self, connection: Connection):
        self.conn = connection

    async def create_feedback(
        self,
        message_id: str,
        rating: FeedbackRating,
        user_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> Feedback:
        """
        Record thumbs-up/down feedback for an agent message.
        Uses UPSERT so re-submitting a rating overwrites the previous one.
        """
        try:
            row = await self.conn.fetchrow(
                """
                INSERT INTO feedback (message_id, user_id, rating, comment)
                VALUES ($1::uuid, $2, $3, $4)
                ON CONFLICT (message_id, user_id) DO UPDATE SET
                    rating  = EXCLUDED.rating,
                    comment = EXCLUDED.comment
                RETURNING id::text, message_id::text, user_id, rating, comment, created_at
                """,
                message_id,
                user_id,
                rating.value,
                comment,
            )
            return _row_to_feedback(row)
        except Exception as e:
            logger.error("Failed to create feedback", message_id=message_id, error=str(e))
            raise DatabaseError(f"Failed to create feedback: {e}")

    async def get_feedback(self, message_id: str, user_id: Optional[str] = None) -> Optional[Feedback]:
        """Fetch feedback for a specific message, optionally scoped to a user."""
        try:
            if user_id is not None:
                row = await self.conn.fetchrow(
                    """
                    SELECT id::text, message_id::text, user_id, rating, comment, created_at
                    FROM feedback
                    WHERE message_id = $1::uuid AND user_id = $2
                    """,
                    message_id,
                    user_id,
                )
            else:
                row = await self.conn.fetchrow(
                    """
                    SELECT id::text, message_id::text, user_id, rating, comment, created_at
                    FROM feedback
                    WHERE message_id = $1::uuid
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    message_id,
                )
            return _row_to_feedback(row) if row else None
        except Exception as e:
            logger.error("Failed to fetch feedback", message_id=message_id, error=str(e))
            raise DatabaseError(f"Failed to fetch feedback: {e}")

    async def list_feedback(
        self,
        rating: Optional[FeedbackRating] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Feedback]:
        """List feedback entries for offline analysis, optionally filtered by rating."""
        query = "SELECT id::text, message_id::text, user_id, rating, comment, created_at FROM feedback"
        params: list = []

        if rating is not None:
            params.append(rating.value)
            query += f" WHERE rating = ${len(params)}"

        query += " ORDER BY created_at DESC"
        params.append(limit)
        query += f" LIMIT ${len(params)}"
        params.append(offset)
        query += f" OFFSET ${len(params)}"

        try:
            rows = await self.conn.fetch(query, *params)
            return [_row_to_feedback(r) for r in rows]
        except Exception as e:
            logger.error("Failed to list feedback", error=str(e))
            raise DatabaseError(f"Failed to list feedback: {e}")
