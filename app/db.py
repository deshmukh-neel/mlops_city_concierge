from collections.abc import Generator
from contextlib import contextmanager

import psycopg2
from psycopg2.extensions import connection

from .config import get_settings


def _resolve_url() -> str:
    url = get_settings().resolved_database_url
    if not url:
        raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")
    return url


def get_db() -> Generator[connection, None, None]:
    conn = psycopg2.connect(_resolve_url())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_conn() -> Generator[connection, None, None]:
    """Context-manager Postgres connection for scripts and agent tools.

    Opens a fresh psycopg2 connection per call. PR #56 will swap this body
    to borrow from the shared pool with no caller change.
    """
    conn = psycopg2.connect(_resolve_url())
    try:
        yield conn
    finally:
        conn.close()
