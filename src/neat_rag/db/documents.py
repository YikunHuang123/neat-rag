import json
from typing import List, Optional, Dict, Any
from asyncpg import Connection

from neat_rag.models import Document, Chunk
from neat_rag.exceptions import RecordNotFoundError, DatabaseError
from neat_rag.logger import get_logger

logger = get_logger(__name__)

class DocumentRepository:
    """
    Repository for managing Document and Chunk aggregates in the database.
    Responsible for all SQL operations related to ingestion metadata and document retrieval.
    """

    def __init__(self, connection: Connection):
        self.conn = connection

    async def get_document(self, document_id: str, user_id: Optional[str] = None) -> Optional[Document]:
        """Fetch a document by its ID, optionally filtered by user_id."""
        try:
            if user_id is not None:
                row = await self.conn.fetchrow(
                    """
                    SELECT id::text, title, source, mime_type, metadata, created_at, updated_at, user_id
                    FROM documents
                    WHERE id = $1::uuid AND user_id = $2
                    """,
                    document_id,
                    user_id,
                )
            else:
                row = await self.conn.fetchrow(
                    """
                    SELECT id::text, title, source, mime_type, metadata, created_at, updated_at, user_id
                    FROM documents
                    WHERE id = $1::uuid
                    """,
                    document_id,
                )
            if not row:
                return None

            count_row = await self.conn.fetchval(
                "SELECT COUNT(id) FROM chunks WHERE document_id = $1::uuid", document_id
            )

            return Document(
                id=row["id"],
                title=row["title"],
                source=row["source"],
                mime_type=row.get("mime_type", "application/octet-stream"),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                chunk_count=count_row or 0,
                user_id=row["user_id"],
            )
        except Exception as e:
            logger.error("Error fetching document", document_id=document_id, error=str(e))
            raise DatabaseError(f"Error fetching document: {e}")

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        metadata_filter: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """List documents, optionally filtered by user_id and/or metadata."""
        query = """
            SELECT
                d.id::text, d.title, d.source, d.mime_type, d.metadata,
                d.created_at, d.updated_at, d.user_id,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
        """

        params: list = []
        conditions: list = []

        if user_id is not None:
            params.append(user_id)
            conditions.append(f"d.user_id = ${len(params)}")

        if metadata_filter:
            params.append(json.dumps(metadata_filter))
            conditions.append(f"d.metadata @> ${len(params)}::jsonb")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY d.id ORDER BY d.created_at DESC"

        params.append(limit)
        limit_idx = len(params)
        params.append(offset)
        offset_idx = len(params)
        query += f" LIMIT ${limit_idx} OFFSET ${offset_idx}"

        try:
            rows = await self.conn.fetch(query, *params)
            return [
                Document(
                    id=row["id"],
                    title=row["title"],
                    source=row["source"],
                    mime_type=row.get("mime_type", "application/octet-stream"),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    chunk_count=row["chunk_count"],
                    user_id=row["user_id"],
                )
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to list documents", error=str(e))
            raise DatabaseError(f"Failed to list documents: {e}")

    async def save_document_and_chunks(self, document: Document, chunks: List[Chunk]):
        """
        Saves a document and its associated chunks in a single operation.
        Note: The caller MUST wrap this method call inside `async with pg_pool.transaction():`
        """
        try:
            # 1. Insert Document (Upsert logic to allow overwriting)
            await self.conn.execute(
                """
                INSERT INTO documents (id, title, source, mime_type, metadata, created_at, updated_at, user_id)
                VALUES ($1::uuid, $2, $3, $4, $5::jsonb, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    updated_at = EXCLUDED.updated_at
                """,
                document.id,
                document.title,
                document.source,
                document.mime_type,
                json.dumps(document.metadata),
                document.created_at,
                document.updated_at,
                document.user_id,
            )

            # 2. Insert Chunks
            if chunks:
                chunk_records = [
                    (
                        c.id,
                        c.document_id,
                        c.content,
                        f"[{','.join(map(str, c.embedding))}]" if c.embedding else None,
                        c.start_char,
                        c.end_char,
                        json.dumps(c.metadata),
                        c.created_at
                    )
                    for c in chunks
                ]
                # Executemany is highly optimized in asyncpg for bulk inserts
                await self.conn.executemany(
                    """
                    INSERT INTO chunks (id, document_id, content, embedding, start_char, end_char, metadata, created_at)
                    VALUES ($1::uuid, $2::uuid, $3, $4::vector, $5, $6, $7::jsonb, $8)
                    """,
                    chunk_records
                )
                logger.info("Saved document and chunks", document_id=document.id, chunk_count=len(chunks))
        except Exception as e:
            logger.error("Failed to save document and chunks", document_id=document.id, error=str(e))
            raise DatabaseError(f"Failed to save document and chunks: {e}")

    async def get_document_chunks(self, document_id: str) -> List[Chunk]:
        """Fetch all chunks associated with a specific document."""
        try:
            rows = await self.conn.fetch(
                """
                SELECT id::text, document_id::text, content, start_char, end_char, metadata, created_at
                FROM chunks
                WHERE document_id = $1::uuid
                ORDER BY start_char ASC NULLS LAST, created_at ASC
                """,
                document_id
            )
            return [
                Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    start_char=row.get("start_char"),
                    end_char=row.get("end_char"),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"]
                )
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to fetch chunks", document_id=document_id, error=str(e))
            raise DatabaseError(f"Failed to fetch chunks: {e}")

    async def delete_document(self, document_id: str, user_id: Optional[str] = None):
        """
        Deletes a document. When user_id is provided, only deletes if the document
        belongs to that user (returns RecordNotFoundError otherwise).
        Associated chunks are removed automatically by ON DELETE CASCADE.
        """
        try:
            if user_id is not None:
                result = await self.conn.execute(
                    "DELETE FROM documents WHERE id = $1::uuid AND user_id = $2",
                    document_id,
                    user_id,
                )
            else:
                result = await self.conn.execute(
                    "DELETE FROM documents WHERE id = $1::uuid", document_id
                )
            if result == "DELETE 0":
                raise RecordNotFoundError(f"Document with ID {document_id} not found.")
            logger.info("Deleted document", document_id=document_id)
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to delete document", document_id=document_id, error=str(e))
            raise DatabaseError(f"Failed to delete document: {e}")

    async def update_metadata(self, document_id: str, new_metadata: Dict[str, Any]):
        """Updates (merges) the metadata of a document."""
        try:
            result = await self.conn.execute(
                """
                UPDATE documents 
                SET metadata = metadata || $2::jsonb, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1::uuid
                """,
                document_id,
                json.dumps(new_metadata)
            )
            if result == "UPDATE 0":
                raise RecordNotFoundError(f"Document with ID {document_id} not found.")
        except RecordNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update document metadata", document_id=document_id, error=str(e))
            raise DatabaseError(f"Failed to update document metadata: {e}")
