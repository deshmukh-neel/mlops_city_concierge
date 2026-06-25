from __future__ import annotations

import logging
from threading import Lock

from psycopg2.extensions import connection
from psycopg2.pool import ThreadedConnectionPool

from .config import get_settings

logger = logging.getLogger(__name__)

connection_pool: ThreadedConnectionPool | None = None
connection_pool_config: tuple[str, int, int] | None = None
connection_pool_lock = Lock()


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

    global connection_pool, connection_pool_config
    with connection_pool_lock:
        requested = (database_url, min_connections, max_connections)
        if connection_pool is not None:
            if connection_pool_config != requested:
                raise RuntimeError(
                    "init_db_pool was called with different parameters than the "
                    "existing pool. Call close_db_pool() before re-initialising."
                )
            return connection_pool
        connection_pool = ThreadedConnectionPool(
            min_connections,
            max_connections,
            dsn=database_url,
        )
        connection_pool_config = requested
        return connection_pool


def get_connection() -> connection:
    """Borrow a connection from the shared pool, creating the pool lazily if needed."""
    return ensure_db_pool().getconn()


def return_connection(conn: connection, *, close: bool = False) -> None:
    """Return a borrowed connection to the shared pool.

    If the pool has already been closed (e.g. lifespan shutdown ran while a
    request was in flight), close the connection directly. The bare close()
    can still raise if psycopg2 already disposed of the underlying socket; we
    swallow that — the connection is gone either way and the caller doesn't
    benefit from a noisy traceback in shutdown logs.
    """
    pool = connection_pool
    if pool is None:
        try:
            conn.close()
        except Exception:
            logger.debug(
                "return_connection: pool already closed and conn.close() raised; ignoring.",
                exc_info=True,
            )
        return

    pool.putconn(conn, close=close)


def close_db_pool() -> None:
    """Close every connection owned by the process-local pool."""
    global connection_pool, connection_pool_config
    with connection_pool_lock:
        if connection_pool is not None:
            connection_pool.closeall()
            connection_pool = None
            connection_pool_config = None


def ensure_db_pool() -> ThreadedConnectionPool:
    pool = connection_pool
    if pool is not None:
        return pool

    settings = get_settings()
    database_url = settings.resolved_database_url
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")

    return init_db_pool(
        database_url,
        settings.db_pool_min_connections,
        settings.db_pool_max_connections,
    )
