from collections.abc import Generator

import psycopg2
from psycopg2.extensions import connection

from .config import settings


def get_db() -> Generator[connection, None, None]:
    database_url = settings.resolved_database_url
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")
    conn = psycopg2.connect(database_url)
    try:
        yield conn
    finally:
        conn.close()
