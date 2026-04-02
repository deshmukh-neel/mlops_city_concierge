from fastapi import Depends, FastAPI
from psycopg2.extensions import connection

from .config import settings
from .db import get_db

app = FastAPI(title=settings.api_title, version=settings.api_version)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/db")
def health_db(conn: connection = Depends(get_db)) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        _ = cur.fetchone()
    return {"status": "ok"}
