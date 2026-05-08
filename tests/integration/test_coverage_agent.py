"""Integration tests for the W5 coverage agent.

Gated on APP_ENV=integration. Requires the migration to be applied
(`make migrate`). Each test seeds + cleans its own fixture data so the
suite is order-independent and won't pollute a real DB beyond its window.
"""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import MagicMock

import pytest

from app.db import get_conn
from scripts import coverage_agent

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)


_SF_CENTER_LAT = 37.7749
_SF_CENTER_LNG = -122.4194


@pytest.fixture
def _seeded_sparse_bucket():
    """Seed one place in places_raw under a rare cuisine + a hit row, yield the
    place_id, and clean up afterward."""
    place_id = f"w5-int-{uuid.uuid4().hex[:8]}"
    rare_cuisine = "burmese"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO places_raw
                (place_id, name, primary_type, formatted_address,
                 latitude, longitude, types, source_city, source_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                place_id,
                "Test Burmese Spot",
                "restaurant",
                "1 Test St, Mission, San Francisco, CA",
                _SF_CENTER_LAT,
                _SF_CENTER_LNG,
                [rare_cuisine, "restaurant"],
                "San Francisco",
                json.dumps({"id": place_id}),
            ],
        )
        cur.execute(
            """
            INSERT INTO place_query_hits
                (place_id, query_text, field_mode, page_number, rank_in_page)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [place_id, f"{rare_cuisine}-hit-{uuid.uuid4().hex[:6]}", "all", 1, 1],
        )
        conn.commit()

    try:
        yield place_id
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM places_raw WHERE place_id = %s", [place_id])
            conn.commit()


def _purge_test_proposals(prefix: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM places_ingest_query_proposals WHERE query_text LIKE %s",
            [f"{prefix}%"],
        )
        conn.commit()


def test_proposals_table_exists() -> None:
    """Migration was applied and the proposals table is present."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'places_ingest_query_proposals'"
        )
        assert cur.fetchone() is not None


def test_apply_inserts_proposal_row(_seeded_sparse_bucket, monkeypatch) -> None:
    """End-to-end: gather → propose (mocked LLM) → insert lands a real row.

    Also asserts that an LLM proposal that collides with the static seed list
    (`vietnamese restaurants in San Francisco`) is filtered out before insert,
    not silently accepted.
    """
    marker = f"w5-int-{uuid.uuid4().hex[:8]}"
    proposal_text = f"{marker} burmese restaurants in San Francisco"
    seed_collision_text = "vietnamese restaurants in San Francisco"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = json.dumps(
        [
            {
                "query_text": proposal_text,
                "field_mode": "enriched",
                "rationale": "burmese coverage too thin",
            },
            {
                "query_text": seed_collision_text,
                "field_mode": "enriched",
                "rationale": "should be filtered — already in seed list",
            },
        ]
    )
    monkeypatch.setattr(coverage_agent.vibe, "make_judge", lambda: fake_llm)

    monkeypatch.setattr(coverage_agent.mlflow, "set_experiment", MagicMock())
    start_run = MagicMock()
    start_run.return_value.__enter__ = MagicMock(return_value=None)
    start_run.return_value.__exit__ = MagicMock(return_value=None)
    monkeypatch.setattr(coverage_agent.mlflow, "start_run", start_run)
    monkeypatch.setattr(coverage_agent.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_dict", MagicMock())

    try:
        rc = coverage_agent.main(["--days", "30"])
        assert rc == 0

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status, rationale FROM places_ingest_query_proposals WHERE query_text = %s",
                [proposal_text],
            )
            row = cur.fetchone()
            cur.execute(
                "SELECT 1 FROM places_ingest_query_proposals WHERE query_text = %s",
                [seed_collision_text],
            )
            collision_row = cur.fetchone()
        assert row is not None, "marker proposal row should have been inserted"
        assert row[0] == "pending"
        assert "burmese" in row[1]
        assert collision_row is None, "seed-list collision should have been filtered before insert"
    finally:
        _purge_test_proposals(marker)


def test_dry_run_inserts_nothing(_seeded_sparse_bucket, monkeypatch) -> None:
    """Dry-run path emits MLflow artifacts but never writes proposal rows."""
    marker = f"w5-dry-{uuid.uuid4().hex[:8]}"
    proposal_text = f"{marker} thai restaurants in San Francisco"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = json.dumps(
        [
            {
                "query_text": proposal_text,
                "field_mode": "enriched",
                "rationale": "thai gap",
            }
        ]
    )
    monkeypatch.setattr(coverage_agent.vibe, "make_judge", lambda: fake_llm)
    monkeypatch.setattr(coverage_agent.mlflow, "set_experiment", MagicMock())
    start_run = MagicMock()
    start_run.return_value.__enter__ = MagicMock(return_value=None)
    start_run.return_value.__exit__ = MagicMock(return_value=None)
    monkeypatch.setattr(coverage_agent.mlflow, "start_run", start_run)
    monkeypatch.setattr(coverage_agent.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_dict", MagicMock())

    try:
        rc = coverage_agent.main(["--dry-run", "--days", "30"])
        assert rc == 0

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM places_ingest_query_proposals WHERE query_text = %s",
                [proposal_text],
            )
            assert cur.fetchone() is None
    finally:
        _purge_test_proposals(marker)
