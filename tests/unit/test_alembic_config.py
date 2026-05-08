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


def test_iam_token_with_percent_breaks_configparser_without_escape() -> None:
    """Regression guard: alembic.Config.set_main_option goes through
    configparser, which treats '%' as interpolation. env.py:30 escapes
    %→%% to work around this; this test proves the escape is load-bearing.
    """
    from alembic.config import Config

    url_with_iam_token = "postgresql://user%40svc:abc%def%ghi@host:5432/db"
    config = Config()

    # Without the escape, configparser rejects the value at set-time.
    with pytest.raises(ValueError, match="invalid interpolation syntax"):
        config.set_main_option("sqlalchemy.url", url_with_iam_token)

    # With the escape (the env.py:30 workaround), the original URL round-trips.
    config.set_main_option("sqlalchemy.url", url_with_iam_token.replace("%", "%%"))
    assert config.get_main_option("sqlalchemy.url") == url_with_iam_token
