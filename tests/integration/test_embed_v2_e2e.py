"""Integration tests for the v2 embeddings pipeline.

Skipped unless APP_ENV=integration. Assumes both place_embeddings and
place_embeddings_v2 have been populated from the same places_raw snapshot
(run `make embed-places` and `make embed-v2` first).

Verifies the v2 outputs satisfy the role taxonomy from
implementation_plan/james/w0a_embeddings_v2.md.
"""

from __future__ import annotations

import os
from statistics import median

import psycopg2
import pytest

from app.config import resolve_database_url

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)


@pytest.fixture(scope="module")
def conn():
    database_url = resolve_database_url(os.environ)
    if not database_url:
        pytest.skip("No DATABASE_URL configured.")
    connection = psycopg2.connect(database_url)
    yield connection
    connection.close()


def _fetch_one(conn, sql: str) -> object:
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def test_v2_table_populated(conn) -> None:
    count = _fetch_one(conn, "SELECT COUNT(*) FROM place_embeddings_v2")
    assert count > 0, "place_embeddings_v2 is empty — run `make embed-v2` first."


def test_v2_chunks_are_meaningfully_smaller_than_v1(conn) -> None:
    """Median v2 length should be < 70% of median v1 length on the same place_ids."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT LENGTH(v1.embedding_text), LENGTH(v2.embedding_text)
            FROM place_embeddings    v1
            JOIN place_embeddings_v2 v2 USING (place_id)
            """
        )
        rows = cur.fetchall()

    assert rows, "No overlapping rows between place_embeddings and place_embeddings_v2."
    v1_median = median(r[0] for r in rows)
    v2_median = median(r[1] for r in rows)
    assert v2_median < 0.70 * v1_median, (
        f"Expected v2 median chunk length < 70% of v1 median. "
        f"Got v1={v1_median}, v2={v2_median}."
    )


def test_v2_chunks_contain_no_urls(conn) -> None:
    forbidden_count = _fetch_one(
        conn,
        """
        SELECT COUNT(*) FROM place_embeddings_v2
        WHERE embedding_text ILIKE '%http%'
           OR embedding_text ILIKE '%g_mp=%'
           OR embedding_text ILIKE '%googleMapsLinks%'
        """,
    )
    assert forbidden_count == 0, f"{forbidden_count} v2 rows still contain URL/link payloads."


def test_v2_chunks_contain_no_disclosure_boilerplate(conn) -> None:
    boilerplate_count = _fetch_one(
        conn,
        """
        SELECT COUNT(*) FROM place_embeddings_v2
        WHERE embedding_text ILIKE '%Summarized with Gemini%'
           OR embedding_text ILIKE '%languageCode%'
        """,
    )
    assert (
        boilerplate_count == 0
    ), f"{boilerplate_count} v2 rows still contain disclosure / language-code boilerplate."
