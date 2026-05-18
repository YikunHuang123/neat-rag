"""Add per-user permission flags to api_keys.

Revision ID: 005
Down revision: 004

Adds four boolean columns to control what a regular user key can do:
  is_active   — master on/off switch (soft-disable without revoking the key)
  can_upload  — allow document uploads
  can_delete  — allow document deletion
  can_chat    — allow chat/query endpoints

All columns default to TRUE so existing keys are unaffected after upgrade.
"""
from __future__ import annotations

from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE api_keys
        ADD COLUMN IF NOT EXISTS is_active  BOOLEAN NOT NULL DEFAULT TRUE,
        ADD COLUMN IF NOT EXISTS can_upload BOOLEAN NOT NULL DEFAULT TRUE,
        ADD COLUMN IF NOT EXISTS can_delete BOOLEAN NOT NULL DEFAULT TRUE,
        ADD COLUMN IF NOT EXISTS can_chat   BOOLEAN NOT NULL DEFAULT TRUE
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE api_keys
        DROP COLUMN IF EXISTS is_active,
        DROP COLUMN IF EXISTS can_upload,
        DROP COLUMN IF EXISTS can_delete,
        DROP COLUMN IF EXISTS can_chat
    """)
