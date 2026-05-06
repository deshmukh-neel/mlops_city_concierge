"""Database URL resolution for Alembic.

Lives outside `alembic/env.py` so it's importable from tests without booting
an Alembic context. The actual resolution rules (DATABASE_URL precedence over
POSTGRES_*, Cloud SQL socket vs TCP) are owned by `app.config.resolve_database_url` —
this module is a thin wrapper that adds "fail loudly if unresolved."
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from app.config import resolve_database_url


def resolve_alembic_database_url(env: Mapping[str, str | None] | None = None) -> str:
    """Return the URL Alembic should connect to. Raise if it can't be resolved.

    Alembic must not silently fall back to a default — running migrations
    against the wrong database (or no database) is one of the worst things
    we can do.

    Reads from `os.environ` by default. Tests pass an explicit mapping to
    avoid bleeding through whatever `.env` happens to be on disk.
    """
    url = resolve_database_url(env if env is not None else os.environ)
    if not url:
        raise RuntimeError(
            "Alembic could not resolve a database URL. Set DATABASE_URL or the "
            "POSTGRES_* env vars (see app/config.py:resolve_database_url)."
        )
    return url
