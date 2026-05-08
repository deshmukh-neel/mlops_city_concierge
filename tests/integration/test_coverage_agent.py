"""Integration tests for the W5 coverage agent.

Gated on APP_ENV=integration. The CI integration job applies migrations
before running this suite, so the proposals table is guaranteed to exist.

We don't write to places_raw because the CI service account lacks INSERT
on it. Instead, gather_stats is stubbed to return a synthetic gap; the
real-DB coverage we care about is the INSERT into places_ingest_query_proposals
and the dedup against the live seed/checkpoint state.
"""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import MagicMock

import pytest

from app.db import get_conn
from scripts import coverage_agent
from scripts.coverage_agent import CoverageStat

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)


def _purge_test_proposals(prefix: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM places_ingest_query_proposals WHERE query_text LIKE %s",
            [f"{prefix}%"],
        )
        conn.commit()


def _stub_synthetic_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace gather_stats with a fixed sparse-cuisine gap so tests don't
    need INSERT on places_raw."""
    monkeypatch.setattr(
        coverage_agent,
        "gather_stats",
        lambda days: [CoverageStat("cuisine:burmese", 0, 0, None)],
    )


def _stub_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coverage_agent.mlflow, "set_experiment", MagicMock())
    start_run = MagicMock()
    start_run.return_value.__enter__ = MagicMock(return_value=None)
    start_run.return_value.__exit__ = MagicMock(return_value=None)
    monkeypatch.setattr(coverage_agent.mlflow, "start_run", start_run)
    monkeypatch.setattr(coverage_agent.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_dict", MagicMock())


def test_proposals_table_exists() -> None:
    """Migration was applied and the proposals table is present."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'places_ingest_query_proposals'"
        )
        assert cur.fetchone() is not None


def test_apply_inserts_proposal_row(monkeypatch) -> None:
    """End-to-end: stubbed gap → propose (mocked LLM) → real INSERT lands a row.

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
    _stub_synthetic_gap(monkeypatch)
    _stub_mlflow(monkeypatch)

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


def test_dry_run_inserts_nothing(monkeypatch) -> None:
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
    _stub_synthetic_gap(monkeypatch)
    _stub_mlflow(monkeypatch)

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
