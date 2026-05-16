"""Add invite_tokens table for self-service API key acquisition.

Revision ID: 003
Down-revision: 002
"""
from __future__ import annotations

from textwrap import dedent

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(dedent("""
        CREATE TABLE IF NOT EXISTS invite_tokens (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            token       TEXT NOT NULL UNIQUE,
            owner       TEXT NOT NULL,
            used        BOOLEAN NOT NULL DEFAULT FALSE,
            used_at     TIMESTAMPTZ,
            expires_at  TIMESTAMPTZ NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """))

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_invite_tokens_token ON invite_tokens (token)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS invite_tokens CASCADE")
