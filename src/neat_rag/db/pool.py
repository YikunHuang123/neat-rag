import asyncpg
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager
from neat_rag.config import settings
from neat_rag.logger import get_logger

logger = get_logger(__name__)

class PgPool:
    """Manages the PostgreSQL asyncpg connection pool."""
    
    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize the connection pool."""
        if self._pool is None:
            logger.info("Initializing asyncpg connection pool", db_host=settings.DB_HOST, db_name=settings.DB_NAME)
            self._pool = await asyncpg.create_pool(
                dsn=settings.DATABASE_URL,
                min_size=2,
                max_size=20,
                command_timeout=60,
            )
            logger.info("Database pool initialized successfully.")

    async def disconnect(self):
        """Close the connection pool."""
        if self._pool is not None:
            logger.info("Closing database pool.")
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        """Get the raw asyncpg pool object."""
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized. Call connect() first.")
        return self._pool

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[asyncpg.Connection]:
        """Yields a database connection from the pool."""
        async with self.pool.acquire() as connection:
            yield connection

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """Yields a database connection within a transaction context."""
        async with self.get_connection() as connection:
            async with connection.transaction():
                yield connection

# Global instance to be initialized during app startup (e.g., FastAPI lifespan)
pg_pool = PgPool()
