from typing import List, Optional
from asyncpg import Connection

from neat_rag.models import Job, JobStatus, Feedback, FeedbackRating
from neat_rag.exceptions import RecordNotFoundError, DatabaseError
from neat_rag.logger import get_logger

logger = get_logger(__name__)


class JobRepository:
    """
    Repository for async ingestion jobs (ingest_jobs table).
    The ingestion pipeline creates a job on upload and updates it as chunks are stored.
    """

    def __init__(self, connection: Connection):
        self.conn = connection

    async def create_job(self, filename: str) -> Job:
        """Create a new ingestion job in PENDING state and return it."""
        try:
            row = await self.conn.fetchrow(
                """
                INSERT INTO ingest_jobs (filename, status, progress)
                VALUES ($1, $2, 0.0)
                RETURNING id::text, filename, status, progress, error, created_at, updated_at
                """,
                filename,
                JobStatus.PENDING.value,
            )
            return _row_to_job(row)
        except Exception as e:
            logger.error("Failed to create job", filename=filename, error=str(e))
            raise DatabaseError(f"Failed to create job: {e}")

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Fetch a job by ID. Returns None if not found."""
        try:
            row = await self.conn.fetchrow(
                """
                SELECT id::text, filename, status, progress, error, created_at, updated_at
                FROM ingest_jobs
                WHERE id = $1::uuid
                """,
                job_id,
            )
            return _row_to_job(row) if row else None
        except Exception as e:
            logger.error("Failed to fetch job", job_id=job_id, error=str(e))
            raise DatabaseError(f"Failed to fetch job: {e}")

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> List[Job]:
        """List all ingestion jobs ordered by creation time descending."""
        try:
            rows = await self.conn.fetch(
                """
                SELECT id::text, filename, status, progress, error, created_at, updated_at
                FROM ingest_jobs
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            return [_row_to_job(r) for r in rows]
        except Exception as e:
            logger.error("Failed to list jobs", error=str(e))
            raise DatabaseError(f"Failed to list jobs: {e}")

    async def update_progress(self, job_id: str, progress: float, status: JobStatus) -> None:
        """
        Update a job's progress (0.0–1.0) and status.
        Called repeatedly by the ingestion pipeline as chunks are processed.
        """
        try:
            result = await self.conn.execute(
                """
                UPDATE ingest_jobs
                SET progress = $2, status = $3, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1::uuid
                """,
                job_id,
                progress,
                status.value,
            )
            if result == "UPDATE 0":
                raise RecordNotFoundError(f"Job with ID {job_id} not found.")
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update job progress", job_id=job_id, error=str(e))
            raise DatabaseError(f"Failed to update job progress: {e}")

    async def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a job as FAILED and record the error message."""
        try:
            result = await self.conn.execute(
                """
                UPDATE ingest_jobs
                SET status = $2, error = $3, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1::uuid
                """,
                job_id,
                JobStatus.FAILED.value,
                error,
            )
            if result == "UPDATE 0":
                raise RecordNotFoundError(f"Job with ID {job_id} not found.")
            logger.warning("Ingestion job failed", job_id=job_id, error=error)
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to mark job as failed", job_id=job_id, error=str(e))
            raise DatabaseError(f"Failed to mark job as failed: {e}")

    async def mark_completed(self, job_id: str) -> None:
        """Mark a job as COMPLETED with progress=1.0."""
        try:
            result = await self.conn.execute(
                """
                UPDATE ingest_jobs
                SET status = $2, progress = 1.0, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1::uuid
                """,
                job_id,
                JobStatus.COMPLETED.value,
            )
            if result == "UPDATE 0":
                raise RecordNotFoundError(f"Job with ID {job_id} not found.")
            logger.info("Ingestion job completed", job_id=job_id)
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to mark job as completed", job_id=job_id, error=str(e))
            raise DatabaseError(f"Failed to mark job as completed: {e}")


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


# ---------------------------------------------------------------------------
# Private helpers — keep row → model mapping out of every method body
# ---------------------------------------------------------------------------

def _row_to_job(row) -> Job:
    return Job(
        id=row["id"],
        filename=row["filename"],
        status=JobStatus(row["status"]),
        progress=row["progress"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_feedback(row) -> Feedback:
    return Feedback(
        id=row["id"],
        message_id=row["message_id"],
        user_id=row["user_id"],
        rating=FeedbackRating(row["rating"]),
        comment=row["comment"],
        created_at=row["created_at"],
    )


# todo
# ---------------------------------------------------------------------------
# api_keys — Phase 4 (authentication middleware)
# Planned table: api_keys(id, hashed_key, owner, scopes[], created_at, last_used_at)
# Will be added to ApiKeyRepository here once the auth middleware (api/middleware.py)
# and the corresponding Alembic migration are in place.
# ---------------------------------------------------------------------------
