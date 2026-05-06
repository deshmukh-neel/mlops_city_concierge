"""Alembic environment.

Reads the database URL via `app.db_url.resolve_alembic_database_url()`, which
reuses `app.config.resolve_database_url` so the env-var resolution rules
(DATABASE_URL vs POSTGRES_* vs Cloud SQL socket) live in one place. The helper
is in its own module so tests can exercise it without booting an Alembic
context.
"""

from __future__ import annotations

from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context
from app.db_url import resolve_alembic_database_url

# Load .env so `alembic <cmd>` works the same way `python scripts/*.py` does.
# Every script in this repo calls load_dotenv(); Alembic must too, or
# `make migrate` fails for users who put DATABASE_URL only in .env.
load_dotenv()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", resolve_alembic_database_url())

# Migrations are hand-written SQL via op.execute() — autogenerate is off.
target_metadata = None


def run_migrations_offline() -> None:
    """Emit migrations as SQL without connecting (used for `alembic upgrade --sql`)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
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
