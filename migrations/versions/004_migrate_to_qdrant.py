"""Add chunk_count column to documents.

Revision ID: 004
Down revision: 003

Why: The chunk_count is now maintained by the ingestion pipeline on write,
so we can avoid an expensive COUNT JOIN on every document read.  This column
is used by both the pgvector and Qdrant backends — neither backend is removed
here; the factory pattern in db/vector_store.py lets operators switch via
VECTOR_STORE_BACKEND without a schema change.
"""
from __future__ import annotations

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS chunk_count")
