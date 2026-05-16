from datetime import datetime, timezone
from typing import Optional

from asyncpg import Connection

from neat_rag.models import InviteToken
from neat_rag.exceptions import RecordNotFoundError, DatabaseError
from neat_rag.logger import get_logger

logger = get_logger(__name__)


def _row_to_invite(row) -> InviteToken:
    return InviteToken(
        id=str(row["id"]),
        token=row["token"],
        owner=row["owner"],
        used=row["used"],
        used_at=row["used_at"],
        expires_at=row["expires_at"],
        created_at=row["created_at"],
    )


class InviteTokenRepository:
    def __init__(self, connection: Connection):
        self.conn = connection

    async def create(self, token: str, owner: str, expires_at: datetime) -> InviteToken:
        try:
            row = await self.conn.fetchrow(
                """
                INSERT INTO invite_tokens (token, owner, expires_at)
                VALUES ($1, $2, $3)
                RETURNING id::text, token, owner, used, used_at, expires_at, created_at
                """,
                token, owner, expires_at,
            )
            return _row_to_invite(row)
        except Exception as e:
            logger.error("Failed to create invite token", owner=owner, error=str(e))
            raise DatabaseError(f"Failed to create invite token: {e}")

    async def get_by_token(self, token: str) -> Optional[InviteToken]:
        try:
            row = await self.conn.fetchrow(
                """
                SELECT id::text, token, owner, used, used_at, expires_at, created_at
                FROM invite_tokens WHERE token = $1
                """,
                token,
            )
            return _row_to_invite(row) if row else None
        except Exception as e:
            logger.error("Failed to look up invite token", error=str(e))
            raise DatabaseError(f"Failed to look up invite token: {e}")

    async def mark_used(self, token: str) -> None:
        try:
            await self.conn.execute(
                """
                UPDATE invite_tokens
                SET used = TRUE, used_at = CURRENT_TIMESTAMP
                WHERE token = $1
                """,
                token,
            )
        except Exception as e:
            logger.error("Failed to mark invite token as used", error=str(e))
            raise DatabaseError(f"Failed to mark invite token as used: {e}")

    async def delete(self, token_id: str) -> None:
        try:
            result = await self.conn.execute(
                "DELETE FROM invite_tokens WHERE id = $1::uuid", token_id
            )
            if result == "DELETE 0":
                raise RecordNotFoundError(f"Invite token {token_id} not found.")
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to delete invite token", token_id=token_id, error=str(e))
            raise DatabaseError(f"Failed to delete invite token: {e}")

    async def list_all(self) -> list[InviteToken]:
        try:
            rows = await self.conn.fetch(
                """
                SELECT id::text, token, owner, used, used_at, expires_at, created_at
                FROM invite_tokens ORDER BY created_at DESC
                """
            )
            return [_row_to_invite(r) for r in rows]
        except Exception as e:
            logger.error("Failed to list invite tokens", error=str(e))
            raise DatabaseError(f"Failed to list invite tokens: {e}")
