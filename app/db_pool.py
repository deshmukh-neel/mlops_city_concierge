from __future__ import annotations

from threading import Lock

from psycopg2.extensions import connection
from psycopg2.pool import ThreadedConnectionPool

from .config import get_settings

_pool: ThreadedConnectionPool | None = None
_pool_config: tuple[str, int, int] | None = None
_pool_lock = Lock()


def init_db_pool(
    database_url: str,
    min_connections: int,
    max_connections: int,
) -> ThreadedConnectionPool:
    """Create the process-local DB pool if it does not already exist.

    Raises if called a second time with a different (url, min, max) — silently
    keeping the original pool would mean later callers think they're connected
    to one database while actually borrowing from another.
    """
    if min_connections < 0:
        raise ValueError("db_pool_min_connections must be greater than or equal to 0.")
    if max_connections < 1:
        raise ValueError("db_pool_max_connections must be greater than or equal to 1.")
    if min_connections > max_connections:
        raise ValueError("db_pool_min_connections cannot exceed db_pool_max_connections.")

    global _pool, _pool_config
    with _pool_lock:
        requested = (database_url, min_connections, max_connections)
        if _pool is not None:
            if _pool_config != requested:
                raise RuntimeError(
                    "init_db_pool was called with different parameters than the "
                    "existing pool. Call close_db_pool() before re-initialising."
                )
            return _pool
        _pool = ThreadedConnectionPool(
            min_connections,
            max_connections,
            dsn=database_url,
        )
        _pool_config = requested
        return _pool


def get_connection() -> connection:
    """Borrow a connection from the shared pool, creating the pool lazily if needed."""
    return _ensure_db_pool().getconn()


def return_connection(conn: connection, *, close: bool = False) -> None:
    """Return a borrowed connection to the shared pool."""
    pool = _pool
    if pool is None:
        conn.close()
        return

    pool.putconn(conn, close=close)


def close_db_pool() -> None:
    """Close every connection owned by the process-local pool."""
    global _pool, _pool_config
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None
            _pool_config = None


def _ensure_db_pool() -> ThreadedConnectionPool:
    pool = _pool
    if pool is not None:
        return pool

    # init_db_pool re-checks under _pool_lock, so concurrent lazy callers converge safely.
    settings = get_settings()
    database_url = settings.resolved_database_url
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")

    return init_db_pool(
        database_url,
        settings.db_pool_min_connections,
        settings.db_pool_max_connections,
    )
