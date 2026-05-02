import logging
from collections.abc import Generator, Iterator
from contextlib import contextmanager

from psycopg2.extensions import connection
from psycopg2.pool import ThreadedConnectionPool

from .config import require_database_url

logger = logging.getLogger(__name__)

_pool: ThreadedConnectionPool | None = None
_POOL_MIN = 1
_POOL_MAX = 10


def get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=_POOL_MIN, maxconn=_POOL_MAX, dsn=require_database_url()
        )
        logger.info("postgres pool created (min=%d, max=%d)", _POOL_MIN, _POOL_MAX)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("postgres pool closed")


@contextmanager
def borrow_connection() -> Iterator[connection]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def get_db() -> Generator[connection, None, None]:
    with borrow_connection() as conn:
        yield conn
