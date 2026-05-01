from collections.abc import Generator, Iterator
from contextlib import contextmanager

from psycopg2.extensions import connection
from psycopg2.pool import ThreadedConnectionPool

from .config import require_database_url

_pool: ThreadedConnectionPool | None = None


def get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=require_database_url())
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


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
