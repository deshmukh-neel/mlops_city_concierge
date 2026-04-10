from collections.abc import Generator

import psycopg2
from psycopg2.extensions import connection

from .config import get_settings


def get_db() -> Generator[connection, None, None]:
    conn = psycopg2.connect(get_settings().database_url)
    try:
        yield conn
    finally:
        conn.close()
