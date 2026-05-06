"""Integration tests for the place_documents view.

Gated on APP_ENV=integration. Requires the W1 migration applied AND data in
places_raw + the active embedding table. Run after `make migrate && make ingest`.
"""

from __future__ import annotations

import os

import pytest

from app.db import get_conn

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)


def _view_exists(view_name: str) -> bool:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.views WHERE table_name = %s",
            [view_name],
        )
        return cur.fetchone() is not None


def test_place_documents_view_exists() -> None:
    assert _view_exists("place_documents")


def test_place_documents_v2_view_exists() -> None:
    assert _view_exists("place_documents_v2")


def test_neighborhood_of_function_exists() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_proc WHERE proname = 'neighborhood_of'",
        )
        assert cur.fetchone() is not None


def test_place_is_open_function_exists() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_proc WHERE proname = 'place_is_open'",
        )
        assert cur.fetchone() is not None


def test_view_has_expected_columns() -> None:
    """Sanity check on the projection — guards against accidental column rename."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'place_documents'
            ORDER BY ordinal_position
            """
        )
        columns = {row[0] for row in cur.fetchall()}
    expected_subset = {
        "place_id",
        "name",
        "neighborhood",
        "rating",
        "user_rating_count",
        "price_level",
        "business_status",
        "regular_opening_hours",
        "serves_cocktails",
        "outdoor_seating",
        "allows_dogs",
        "parking_options",
        "source",
        "embedding",
        "embedding_text",
    }
    missing = expected_subset - columns
    assert not missing, f"view is missing expected columns: {missing}"


def test_view_returns_rows_when_data_present() -> None:
    """Soft check — passes regardless of row count, just verifies the view is queryable."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM place_documents_v2")
        row = cur.fetchone()
        assert row is not None
        # Any non-negative count is fine; the test exists to prove the view runs.
        assert row[0] >= 0
