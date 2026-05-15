"""Alembic environment — bridges Alembic's migration engine to the project's Settings."""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the src/ tree importable when running `alembic` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neat_rag.config import settings  # noqa: E402 — must come after sys.path patch

# ---------------------------------------------------------------------------
# Alembic Config object (provides access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Override the blank sqlalchemy.url in alembic.ini with the live settings value.
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL_SYNC)

# Apply logging config from alembic.ini if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# target_metadata enables --autogenerate.  We use raw SQL migrations so we
# leave it as None — autogenerate is disabled.
target_metadata = None


# ---------------------------------------------------------------------------
# Offline mode (no live DB required — emits SQL to stdout)
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode (connects to the real database)
# ---------------------------------------------------------------------------
def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
