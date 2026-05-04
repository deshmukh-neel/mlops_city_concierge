from __future__ import annotations

from threading import Lock

from psycopg2.extensions import connection
from psycopg2.pool import ThreadedConnectionPool

from .config import get_settings

_pool: ThreadedConnectionPool | None = None
_pool_lock = Lock()


def init_db_pool(
    database_url: str,
    min_connections: int,
    max_connections: int,
) -> ThreadedConnectionPool:
    """Create the process-local DB pool if it does not already exist."""
    if min_connections < 0:
        raise ValueError("db_pool_min_connections must be greater than or equal to 0.")
    if max_connections < 1:
        raise ValueError("db_pool_max_connections must be greater than or equal to 1.")
    if min_connections > max_connections:
        raise ValueError("db_pool_min_connections cannot exceed db_pool_max_connections.")

    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadedConnectionPool(
                min_connections,
                max_connections,
                dsn=database_url,
            )
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
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


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
