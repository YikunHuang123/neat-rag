import asyncio
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Any

from arq.connections import RedisSettings

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

        # Generate document ID early — needed to name the persisted image file
        # before the temp file is cleaned up by the caller.
        doc_id = str(uuid.uuid4())

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

                if metadata.get("is_image"):
                    # Images skip the generic chunker — we build exactly two
                    # targeted chunks: one for the VLM description (captures
                    # visual semantics) and one for the OCR text (captures
                    # embedded characters).  Both chunks carry image_path so
                    # the orchestrator can attach the raw image to multimodal
                    # LLM requests.
                    image_path_str = await asyncio.to_thread(
                        _persist_image, file_path, doc_id, settings.IMAGE_STORAGE_PATH
                    )
                    metadata["image_path"] = image_path_str
                    raw_chunks = _build_image_chunks(metadata)
                else:
                    raw_chunks = await asyncio.to_thread(
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
                    id=doc_id,
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


def _persist_image(file_path: Path, doc_id: str, storage_path_str: str) -> str:
    """Copy the uploaded image to persistent storage and return the saved path.

    Called in a thread pool (same as extraction) to avoid blocking the event loop.
    The caller's temp file is deleted after pipeline.run() returns, so we copy here.
    """
    storage = Path(storage_path_str)
    storage.mkdir(parents=True, exist_ok=True)
    ext = file_path.suffix.lower()
    dest = storage / f"{doc_id}{ext}"
    shutil.copy2(file_path, dest)
    logger.info("Image persisted", dest=str(dest))
    return str(dest)


def _build_image_chunks(metadata: dict) -> List[RawChunk]:
    """Build description and OCR chunks for an image document.

    Creates at most two chunks:
      - image_description: VLM-generated natural language description
      - image_ocr:         pytesseract-extracted text

    Either may be absent (e.g. scenic photos have no OCR text; pure-text
    screenshots may produce an empty description).  At least one will exist
    because ImageExtractor.extract() already guards against both being empty.
    """
    image_path = metadata["image_path"]
    chunks: List[RawChunk] = []

    description = metadata.get("description", "").strip()
    ocr_text = metadata.get("ocr_text", "").strip()

    if description:
        chunks.append(RawChunk(
            content=description,
            start_char=0,
            end_char=len(description),
            metadata={"chunk_type": "image_description", "image_path": image_path},
        ))
    if ocr_text:
        chunks.append(RawChunk(
            content=ocr_text,
            start_char=0,
            end_char=len(ocr_text),
            metadata={"chunk_type": "image_ocr", "image_path": image_path},
        ))
    return chunks


async def _update_job(
    job_repo: Optional[JobRepository],
    job_id: Optional[str],
    progress: float,
    status: JobStatus,
) -> None:
    if job_repo and job_id:
        await job_repo.update_progress(job_id, progress, status)


# ── arq worker task ───────────────────────────────────────────────────────────

async def ingest_document(
    ctx: dict,
    file_path_str: str,
    job_id: str,
    original_name: str,
    user_id: Optional[str] = None,
) -> None:
    pipeline: IngestionPipeline = ctx["pipeline"]
    path = Path(file_path_str)
    try:
        await pipeline.run(path, job_id=job_id, original_name=original_name, user_id=user_id)
    finally:
        path.unlink(missing_ok=True)


async def _worker_startup(ctx: dict) -> None:
    pool = PgPool()
    await pool.connect()
    ctx["pg_pool"] = pool
    ctx["pipeline"] = IngestionPipeline(pg_pool=pool)
    logger.info("arq worker started")


async def _worker_shutdown(ctx: dict) -> None:
    pool: PgPool = ctx.get("pg_pool")
    if pool:
        await pool.disconnect()
    logger.info("arq worker stopped")


class WorkerSettings:
    functions = [ingest_document]
    on_startup = _worker_startup
    on_shutdown = _worker_shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
