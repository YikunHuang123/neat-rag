import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neat_rag.api.schemas import ErrorResponse
from neat_rag.config import settings
from neat_rag.db.pool import pg_pool
from neat_rag.exceptions import NeatRagError, RecordNotFoundError
from neat_rag.logger import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await pg_pool.connect()
    logger.info("Neat-RAG API started", env=settings.APP_ENV)
    yield
    await pg_pool.disconnect()
    logger.info("Neat-RAG API stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Neat-RAG API",
        version="0.1.0",
        description="A modern Agentic RAG system with Pydantic AI and PostgreSQL",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request-ID middleware ─────────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(RecordNotFoundError)
    async def not_found_handler(request: Request, exc: RecordNotFoundError):
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(error="Not Found", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(NeatRagError)
    async def neat_rag_error_handler(request: Request, exc: NeatRagError):
        logger.error("Application error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="Internal Error", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="Internal Server Error").model_dump(),
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    from neat_rag.api.chat import router as chat_router
    from neat_rag.api.documents import jobs_router, router as documents_router
    from neat_rag.api.feedback import router as feedback_router
    from neat_rag.api.health import router as health_router
    from neat_rag.api.sessions import router as sessions_router

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(chat_router)
    app.include_router(sessions_router)
    app.include_router(feedback_router)

    return app


# Module-level app instance for uvicorn: `uvicorn neat_rag.api:app`
app = create_app()
