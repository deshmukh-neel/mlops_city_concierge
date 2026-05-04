import logging
from collections.abc import Generator
from contextlib import contextmanager

from psycopg2.extensions import connection

from .db_pool import get_connection, return_connection

logger = logging.getLogger(__name__)


def _return_connection_safely(conn: connection) -> None:
    close_connection = False
    try:
        if conn.closed:
            close_connection = True
        else:
            conn.rollback()
    except Exception:
        logger.warning(
            "Failed to reset DB connection before returning it to the pool.",
            exc_info=True,
        )
        close_connection = True
    return_connection(conn, close=close_connection)


def get_db() -> Generator[connection, None, None]:
    conn = get_connection()
    try:
        yield conn
    finally:
        _return_connection_safely(conn)


@contextmanager
def get_conn() -> Generator[connection, None, None]:
    """Context-manager Postgres connection for scripts and agent tools.

    Borrows from the shared pool with the same lifecycle guarantees as get_db().
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        _return_connection_safely(conn)
