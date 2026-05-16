import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Any

from neat_rag.config import settings
from neat_rag.db.pool import PgPool
from neat_rag.db.vector_store import VectorStoreBase, get_vector_store
from neat_rag.db.documents import DocumentRepository
from neat_rag.db.jobs import JobRepository
from neat_rag.models import Document, Chunk, JobStatus
from neat_rag.providers.embedding import OpenAIEmbedder, get_embedder
from neat_rag.ingestion.extractors import dispatch_by_ext
from neat_rag.ingestion.chunkers import get_chunker, RawChunk
from neat_rag.exceptions import IngestionError
from neat_rag.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates: extract → chunk → embed → persist for a single file.

    Document metadata is always saved to PostgreSQL.
    Chunk vectors are saved via the active VectorStoreBase backend
    (pgvector or Qdrant, depending on VECTOR_STORE_BACKEND in config).

    Job progress updates use a dedicated connection so they are committed
    immediately and survive even if the document insert transaction rolls back.
    """

    def __init__(
        self,
        pg_pool: PgPool,
        embedder: Optional[OpenAIEmbedder] = None,
        vector_store: Optional[VectorStoreBase] = None,
        chunking_strategy: str = "recursive",
        chunker_kwargs: Optional[dict] = None,
    ):
        self.pg_pool = pg_pool
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or get_vector_store()

        kwargs = chunker_kwargs or {}
        if chunking_strategy == "recursive" and not kwargs:
            kwargs = {"chunk_size": 1000, "chunk_overlap": 200}

        self.chunker = get_chunker(strategy=chunking_strategy, **kwargs)

    async def run(
        self,
        file_path: Path,
        job_id: Optional[str] = None,
        original_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Document:
        """
        Ingest a single file end-to-end. If job_id is provided, progress is written
        to the ingest_jobs table in real time (separate from the document transaction).
        Returns the persisted Document model.
        """
        logger.info("Ingestion started", file=file_path.name, job_id=job_id)

        # Use a dedicated connection for job status — commits immediately per call,
        # so progress is visible even if the document insert later fails.
        async with self.pg_pool.get_connection() as status_conn:
            job_repo = JobRepository(status_conn) if job_id else None

            try:
                # ── Step 1: Extract ──────────────────────────────────────
                # Run in a thread pool — extraction (especially Docling PDF) is
                # synchronous and CPU-intensive; blocking here would freeze the
                # entire async event loop for all concurrent requests.
                await _update_job(job_repo, job_id, 0.1, JobStatus.PROCESSING)
                extractor = dispatch_by_ext(file_path)
                content, metadata = await asyncio.to_thread(extractor.extract, file_path)
                if not content.strip():
                    raise IngestionError(f"Extraction produced empty content for '{file_path.name}'")
                logger.info("Extraction done", file=file_path.name, content_len=len(content))

                # ── Step 2: Chunk ────────────────────────────────────────
                await _update_job(job_repo, job_id, 0.3, JobStatus.PROCESSING)
                raw_chunks: List[RawChunk] = await asyncio.to_thread(
                    self.chunker.chunk, content, metadata
                )
                if not raw_chunks:
                    raise IngestionError(f"Chunking produced no chunks for '{file_path.name}'")
                logger.info("Chunking done", file=file_path.name, chunks=len(raw_chunks))

                # ── Step 3: Embed ────────────────────────────────────────
                await _update_job(job_repo, job_id, 0.5, JobStatus.PROCESSING)
                vectors = await self.embedder.embed([c.content for c in raw_chunks])
                logger.info("Embedding done", file=file_path.name, vectors=len(vectors))

                # ── Step 4: Assemble domain models ───────────────────────
                await _update_job(job_repo, job_id, 0.8, JobStatus.PROCESSING)
                now = datetime.now(timezone.utc)
                display_name = original_name or file_path.name
                document = Document(
                    id=str(uuid.uuid4()),
                    title=display_name,
                    source=display_name,
                    mime_type=metadata.get("mime_type", "application/octet-stream"),
                    metadata={k: v for k, v in metadata.items() if k != "title"},
                    created_at=now,
                    updated_at=now,
                    chunk_count=len(raw_chunks),
                    user_id=user_id,
                )
                chunks = [
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=document.id,
                        content=raw.content,
                        embedding=vectors[i],
                        start_char=raw.start_char,
                        end_char=raw.end_char,
                        metadata=raw.metadata,
                        created_at=now,
                    )
                    for i, raw in enumerate(raw_chunks)
                ]

                # ── Step 5: Persist ──────────────────────────────────────
                # 5a. Save chunk vectors via the active backend FIRST.
                # If this fails, the document won't be visible in the UI (PostgreSQL).
                await self.vector_store.upsert_chunks(document, chunks)

                # 5b. Save document metadata to PostgreSQL (commits the document).
                async with self.pg_pool.transaction() as doc_conn:
                    doc_repo = DocumentRepository(doc_conn)
                    await doc_repo.save_document(document)

                if job_repo and job_id:
                    await job_repo.mark_completed(job_id)

                logger.info(
                    "Ingestion complete",
                    file=file_path.name,
                    document_id=document.id,
                    chunks=len(chunks),
                )
                return document

            except IngestionError:
                if job_repo and job_id:
                    await job_repo.mark_failed(job_id, "IngestionError — see server logs")
                raise
            except Exception as e:
                if job_repo and job_id:
                    await job_repo.mark_failed(job_id, str(e))
                logger.error("Ingestion failed", file=file_path.name, error=str(e))
                raise IngestionError(f"Ingestion failed for '{file_path.name}': {e}") from e


async def _update_job(
    job_repo: Optional[JobRepository],
    job_id: Optional[str],
    progress: float,
    status: JobStatus,
) -> None:
    if job_repo and job_id:
        await job_repo.update_progress(job_id, progress, status)
