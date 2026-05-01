from collections.abc import Generator

import psycopg2
from psycopg2.extensions import connection

from .config import require_database_url


def get_db() -> Generator[connection, None, None]:
    conn = psycopg2.connect(require_database_url())
    try:
        yield conn
    finally:
        conn.close()
