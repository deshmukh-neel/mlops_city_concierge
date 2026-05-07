"""Integration tests for the deterministic critique checks against a real DB.

Gated on APP_ENV=integration. Assumes the standard dev setup (migrations
applied, places_raw seeded) — same prerequisites as the rest of the
integration suite. Each test that needs a real place_id pulls one from the
DB and skips if the table is empty.

Run locally with:
    make db-up
    make migrate
    python scripts/seed.py
    APP_ENV=integration make test-integration
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from app.agent.critique.checks import (
    constraints_satisfied,
    no_hallucinated_place_ids,
    temporal_coherence,
)
from app.agent.state import ItineraryState, Stop, UserConstraints
from app.db import get_conn

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)


def _any_seeded_place_id() -> str:
    """Return a place_id that exists in the DB. Skip the test if none."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT place_id FROM places_raw LIMIT 1")
        row = cur.fetchone()
    if row is None:
        pytest.skip("no rows in places_raw; seed the DB first")
    return row[0]


def _stop(place_id: str, **kwargs) -> Stop:
    return Stop(
        place_id=place_id,
        name=kwargs.pop("name", "X"),
        source="google_places",
        rationale=kwargs.pop("rationale", ""),
        **kwargs,
    )


def test_no_hallucinated_passes_for_real_place_id() -> None:
    pid = _any_seeded_place_id()
    state = ItineraryState(stops=[_stop(pid)])
    assert no_hallucinated_place_ids(state) == 1.0


def test_no_hallucinated_fails_for_fake_place_id() -> None:
    fake = f"fake_{uuid.uuid4().hex}"
    state = ItineraryState(stops=[_stop(fake)])
    assert no_hallucinated_place_ids(state) == 0.0


def test_no_hallucinated_fails_when_one_of_many_is_fake() -> None:
    real = _any_seeded_place_id()
    fake = f"fake_{uuid.uuid4().hex}"
    state = ItineraryState(stops=[_stop(real), _stop(fake)])
    assert no_hallucinated_place_ids(state) == 0.0


def test_temporal_coherence_runs_against_real_hours() -> None:
    """Smoke test that the SQL helper integrates without erroring. The exact
    open/closed result depends on the seeded place's hours — we just assert
    we get a 0.0 or 1.0 back, not that it's a particular value."""
    pid = _any_seeded_place_id()
    arrival = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    state = ItineraryState(stops=[_stop(pid, arrival_time=arrival)])
    score = temporal_coherence(state)
    assert score in (0.0, 1.0)


def test_constraints_satisfied_runs_against_real_row() -> None:
    """Sanity-check the SQL: ask for very lenient constraints (no minimums)
    and expect a pass for a real seeded place."""
    pid = _any_seeded_place_id()
    state = ItineraryState(
        constraints=UserConstraints(min_user_rating_count=0, min_rating=0.0),
        stops=[_stop(pid)],
    )
    score = constraints_satisfied(state)
    assert 0.0 <= score <= 1.0
    # Lenient floors should pass for any real place.
    assert score == 1.0
