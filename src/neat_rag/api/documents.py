import tempfile
from pathlib import Path
from typing import Optional

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile

from neat_rag.api.deps import get_connection, get_pipeline, get_store
from neat_rag.api.middleware import verify_api_key
from neat_rag.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    JobListResponse,
    JobResponse,
    UpdateDocumentRequest,
    UploadResponse,
)
from neat_rag.db.documents import DocumentRepository
from neat_rag.db.jobs import JobRepository
from neat_rag.db.vector_store import VectorStoreBase
from neat_rag.exceptions import RecordNotFoundError
from neat_rag.ingestion.pipeline import IngestionPipeline
from neat_rag.logger import get_logger
from neat_rag.models import Document, Job

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".html"}
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

router = APIRouter(tags=["documents"])
jobs_router = APIRouter(tags=["jobs"])


# ── Documents ──────────────────────────────────────────────────────────────────

@router.post("/documents/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conn: asyncpg.Connection = Depends(get_connection),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    owner: Optional[str] = Depends(verify_api_key),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 50 MB limit.")

    job_repo = JobRepository(conn)
    job = await job_repo.create_job(file.filename or "unknown", user_id=owner)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()

    original_name = file.filename or "unknown"
    background_tasks.add_task(_run_ingestion, pipeline, Path(tmp.name), job.id, original_name, owner)

    logger.info("Document upload accepted", filename=original_name, job_id=job.id, owner=owner)
    return UploadResponse(job_id=job.id, filename=original_name)


async def _run_ingestion(
    pipeline: IngestionPipeline,
    path: Path,
    job_id: str,
    original_name: str,
    user_id: Optional[str] = None,
) -> None:
    try:
        await pipeline.run(path, job_id=job_id, original_name=original_name, user_id=user_id)
    except Exception as e:
        logger.error("Background ingestion task failed", job_id=job_id, error=str(e))
    finally:
        path.unlink(missing_ok=True)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    doc_repo = DocumentRepository(conn)
    docs = await doc_repo.list_documents(limit=limit, offset=offset, user_id=owner)
    return DocumentListResponse(
        items=[_to_doc_response(d) for d in docs],
        total=len(docs),
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    doc_repo = DocumentRepository(conn)
    doc = await doc_repo.get_document(document_id, user_id=owner)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return _to_doc_response(doc)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    conn: asyncpg.Connection = Depends(get_connection),
    store: VectorStoreBase = Depends(get_store),
    owner: Optional[str] = Depends(verify_api_key),
):
    doc_repo = DocumentRepository(conn)
    try:
        await doc_repo.delete_document(document_id, user_id=owner)
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    # For pgvector: chunks already removed by ON DELETE CASCADE.
    # For Qdrant: explicitly remove vectors from the collection.
    await store.delete_by_document(document_id)


@router.patch("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    body: UpdateDocumentRequest,
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    doc_repo = DocumentRepository(conn)
    doc = await doc_repo.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    if owner is not None and doc.user_id != owner:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    try:
        await doc_repo.update_metadata(document_id, body.metadata)
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    doc = await doc_repo.get_document(document_id, user_id=owner)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return _to_doc_response(doc)
  # type: ignore[arg-type]


# ── Jobs ──────────────────────────────────────────────────────────────────────

@jobs_router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    job_repo = JobRepository(conn)
    jobs = await job_repo.list_jobs(limit=limit, offset=offset, user_id=owner)
    return JobListResponse(items=[_to_job_response(j) for j in jobs])


@jobs_router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    conn: asyncpg.Connection = Depends(get_connection),
    owner: Optional[str] = Depends(verify_api_key),
):
    job_repo = JobRepository(conn)
    job = await job_repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if owner is not None and job.user_id != owner:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _to_job_response(job)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_doc_response(d: Document) -> DocumentResponse:
    return DocumentResponse(
        id=d.id,
        title=d.title,
        source=d.source,
        mime_type=d.mime_type,
        metadata=d.metadata,
        created_at=d.created_at,
        updated_at=d.updated_at,
        chunk_count=d.chunk_count,
    )


def _to_job_response(j: Job) -> JobResponse:
    return JobResponse(
        id=j.id,
        filename=j.filename,
        status=j.status,
        progress=j.progress,
        error=j.error,
        created_at=j.created_at,
        updated_at=j.updated_at,
    )
