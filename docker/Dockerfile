# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — install uv, create venv, install dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy only the dependency manifest first for better layer caching.
COPY pyproject.toml .

# Create a virtual environment and install all project dependencies.
RUN uv venv /venv && \
    uv pip install --python /venv/bin/python -r pyproject.toml

# Copy source code after dependencies so code edits don't invalidate the
# dependency layer.
COPY src/ src/

# Install the package itself (no-deps because they were already installed).
RUN uv pip install --python /venv/bin/python --no-deps -e .


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime base — lean image with just the venv and source
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# curl is needed for the Docker healthcheck command.
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy the fully-built virtualenv from the builder stage.
COPY --from=builder /venv /venv
COPY --from=builder /build/src /app/src

# Also copy migration files so `alembic upgrade head` works in the api container.
COPY alembic.ini /app/
COPY migrations/ /app/migrations/

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3a: api — FastAPI server
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime AS api

EXPOSE 8058

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8058/health/live || exit 1

CMD ["uvicorn", "neat_rag.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8058", \
     "--workers", "1"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b: worker — arq background task worker
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime AS worker

CMD ["arq", "neat_rag.ingestion.pipeline.WorkerSettings"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3c: ui — Streamlit frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime AS ui

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/neat_rag/ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
