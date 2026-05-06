"""Smoke tests for the Alembic database-URL resolver.

Real migrations run in the `migrations` CI job against a live Postgres. These
tests just guard the contract `alembic/env.py` depends on — if `app.db_url`
breaks, this fails before CI does.
"""

from __future__ import annotations

import pytest

from app.db_url import resolve_alembic_database_url


def test_raises_when_url_is_unset() -> None:
    """Must fail loudly, not silently default to a wrong DB."""
    with pytest.raises(RuntimeError, match="could not resolve a database URL"):
        resolve_alembic_database_url(env={})


def test_resolves_explicit_database_url() -> None:
    url = resolve_alembic_database_url(env={"DATABASE_URL": "postgresql://u:p@host:5432/db"})
    assert url == "postgresql://u:p@host:5432/db"


def test_resolves_postgres_components() -> None:
    """Falls back to POSTGRES_* vars when DATABASE_URL is unset."""
    url = resolve_alembic_database_url(
        env={
            "POSTGRES_USER": "u",
            "POSTGRES_PASSWORD": "p",
            "POSTGRES_DB": "db",
            "POSTGRES_HOST": "host",
            "POSTGRES_PORT": "5432",
        }
    )
    assert url == "postgresql://u:p@host:5432/db"


def test_raises_when_postgres_components_partial() -> None:
    """Missing one of user/password/db must raise — never silently fall back."""
    with pytest.raises(RuntimeError, match="could not resolve a database URL"):
        resolve_alembic_database_url(
            env={
                "POSTGRES_USER": "u",
                "POSTGRES_DB": "db",
                # POSTGRES_PASSWORD intentionally omitted
            }
        )
