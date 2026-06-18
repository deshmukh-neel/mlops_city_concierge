"""Integration tests for app.query_log.log_user_query.

These tests require a running Postgres instance with the user_query_log table
created by the 17-01 migration (revision d1be72aea7d4).

Run locally with:
    make db-up && make migrate
    APP_ENV=integration DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge \
        poetry run pytest tests/integration/test_query_log.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

# Skip the entire module if not in an integration environment.
pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)

from app.db import get_conn  # noqa: E402
from app.query_log import log_user_query  # noqa: E402


class TestQueryLogIntegration:
    """Real-DB INSERT round-trip tests for log_user_query."""

    @pytest.fixture(autouse=True)
    def _table_writable_or_skip(self) -> None:
        """Skip when user_query_log is absent or the DB role can't write it.

        Mirrors the guard in ``test_build_place_relations.py`` /
        ``test_coverage_agent.py``: CI integration runs against the shared Cloud
        SQL instance (which may lag the 17-01 migration) authenticating as an
        IAM role that may have read-only access. Existence isn't enough — without
        INSERT/DELETE grants the round-trip hard-errors instead of skipping.
        """
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.user_query_log')")
            if cur.fetchone()[0] is None:
                pytest.skip("user_query_log not migrated on this DB (run `make migrate`)")
            cur.execute(
                "SELECT has_table_privilege(current_user, 'user_query_log', 'INSERT, DELETE')"
            )
            if not cur.fetchone()[0]:
                pytest.skip("current DB role lacks INSERT/DELETE on user_query_log")

    def test_populated_round_trip(self) -> None:
        """log_user_query inserts a real row that SELECTs back verbatim."""
        session_marker = f"it-marker-{uuid.uuid4()}"
        try:
            log_user_query(
                message="integration test query",
                requested_primary_types=["restaurant"],
                num_stops=2,
                rag_label="test-label",
                session_id=session_marker,
            )

            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT message, requested_primary_types, num_stops, rag_label, session_id
                    FROM user_query_log
                    WHERE session_id = %s
                    """,
                    [session_marker],
                )
                rows = cur.fetchall()

            assert len(rows) == 1, f"Expected exactly 1 row, got {len(rows)}"
            row = rows[0]
            assert row[0] == "integration test query"
            assert row[1] == ["restaurant"]
            assert row[2] == 2
            assert row[3] == "test-label"
            assert row[4] == session_marker
        finally:
            # Clean up so repeated runs stay idempotent.
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_query_log WHERE session_id = %s",
                    [session_marker],
                )
                conn.commit()

    def test_empty_array_round_trip(self) -> None:
        """The dominant free-text shape: requested_primary_types=[] reads back as [] (not None).

        Proves psycopg2/Postgres text[] adapts an empty Python list as an
        empty Postgres array, not NULL — the common free-text /chat case.
        """
        session_marker = f"it-empty-{uuid.uuid4()}"
        try:
            log_user_query(
                message="real-db empty-array case",
                requested_primary_types=[],
                num_stops=None,
                rag_label="test-label",
                session_id=session_marker,
            )

            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT requested_primary_types, num_stops
                    FROM user_query_log
                    WHERE session_id = %s
                    """,
                    [session_marker],
                )
                rows = cur.fetchall()

            assert len(rows) == 1
            row = rows[0]
            # Must read back as empty list, NOT None.
            assert row[0] == [], f"Expected [] but got {row[0]!r}"
            assert row[1] is None
        finally:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_query_log WHERE session_id = %s",
                    [session_marker],
                )
                conn.commit()
