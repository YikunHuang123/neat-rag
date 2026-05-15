from typing import Optional
from asyncpg import Connection

from neat_rag.models import ApiKey
from neat_rag.exceptions import RecordNotFoundError, DatabaseError
from neat_rag.logger import get_logger

logger = get_logger(__name__)


def _row_to_api_key(row) -> ApiKey:
    return ApiKey(
        id=row["id"],
        hashed_key=row["hashed_key"],
        owner=row["owner"],
        scopes=list(row["scopes"]) if row["scopes"] else [],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
    )


class ApiKeyRepository:
    """
    Repository for API key management (api_keys table).
    Keys are stored as SHA-256 hashes — the raw key is never persisted.
    """

    def __init__(self, connection: Connection):
        self.conn = connection

    async def create_key(self, hashed_key: str, owner: str, scopes: list[str] | None = None) -> ApiKey:
        """Persist a new API key record. The caller must hash the raw key first."""
        try:
            row = await self.conn.fetchrow(
                """
                INSERT INTO api_keys (hashed_key, owner, scopes)
                VALUES ($1, $2, $3)
                RETURNING id::text, hashed_key, owner, scopes, created_at, last_used_at
                """,
                hashed_key,
                owner,
                scopes or [],
            )
            return _row_to_api_key(row)
        except Exception as e:
            logger.error("Failed to create API key", owner=owner, error=str(e))
            raise DatabaseError(f"Failed to create API key: {e}")

    async def get_key_by_hash(self, hashed_key: str) -> Optional[ApiKey]:
        """Look up an API key by its hash. Returns None if not found."""
        try:
            row = await self.conn.fetchrow(
                """
                SELECT id::text, hashed_key, owner, scopes, created_at, last_used_at
                FROM api_keys
                WHERE hashed_key = $1
                """,
                hashed_key,
            )
            return _row_to_api_key(row) if row else None
        except Exception as e:
            logger.error("Failed to look up API key", error=str(e))
            raise DatabaseError(f"Failed to look up API key: {e}")

    async def touch_last_used(self, hashed_key: str) -> None:
        """Update last_used_at to now. Best-effort — callers should not raise on failure."""
        try:
            await self.conn.execute(
                "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE hashed_key = $1",
                hashed_key,
            )
        except Exception as e:
            logger.warning("Failed to update last_used_at", error=str(e))

    async def delete_key(self, key_id: str) -> None:
        """Delete an API key by its ID."""
        try:
            result = await self.conn.execute(
                "DELETE FROM api_keys WHERE id = $1::uuid", key_id
            )
            if result == "DELETE 0":
                raise RecordNotFoundError(f"API key with ID {key_id} not found.")
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to delete API key", key_id=key_id, error=str(e))
            raise DatabaseError(f"Failed to delete API key: {e}")

    async def list_keys(self, owner: Optional[str] = None) -> list[ApiKey]:
        """List all API keys, optionally filtered by owner."""
        query = "SELECT id::text, hashed_key, owner, scopes, created_at, last_used_at FROM api_keys"
        params: list = []
        if owner is not None:
            params.append(owner)
            query += f" WHERE owner = ${len(params)}"
        query += " ORDER BY created_at DESC"
        try:
            rows = await self.conn.fetch(query, *params)
            return [_row_to_api_key(r) for r in rows]
        except Exception as e:
            logger.error("Failed to list API keys", error=str(e))
            raise DatabaseError(f"Failed to list API keys: {e}")
