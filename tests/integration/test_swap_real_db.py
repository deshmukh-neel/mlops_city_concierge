"""Integration tests for the closure-aware swap node — gated on APP_ENV=integration.

These exercise the real `place_is_open()` PL/pgSQL helper, the real
`_PRIMARY_TYPE_FAMILIES` SQL clauses, and the real `nearby()` projection.
They're the only layer that can catch hours-data drift (Google Places hours
changes) that mocked tests can't.

Run with:
    APP_ENV=integration make test-integration
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)


from app.agent.state import ClosureContext, ItineraryState, Stop  # noqa: E402
from app.agent.swap import (  # noqa: E402
    _per_stop_closure_status,
    _try_walking_distance_swap,
    swap_closed_stops,
)

SF = ZoneInfo("America/Los_Angeles")


def _make_stop(
    place_id: str,
    name: str,
    primary_type: str,
    arrival: datetime,
) -> Stop:
    """Build a real-DB-shaped Stop; coords are filled by enrich at runtime."""
    return Stop(
        place_id=place_id,
        name=name,
        rationale="integration",
        source="google_places",
        arrival_time=arrival,
        primary_type=primary_type,
        planned_duration_min=30,
        latitude=37.78,
        longitude=-122.41,
    )


def test_per_stop_closure_status_call_against_live_db_does_not_raise() -> None:
    """For an arbitrary place_id and a wee-hours arrival, the helper must
    return a list-of-bools without raising — exercises the live SQL helper.

    The exact True/False outcome depends on the place's hours and on whether
    the place_id exists in places_raw at test time (a missing row defaults
    to "open" per the spec). Both outcomes are acceptable; we only assert
    the contract.
    """
    stops = [
        _make_stop(
            place_id="ChIJ_a_real_place_id_in_the_index",
            name="Test Place",
            primary_type="Bar",
            arrival=datetime(2026, 5, 19, 3, 0, tzinfo=SF),
        ),
    ]
    statuses = _per_stop_closure_status(stops)
    assert isinstance(statuses, list)
    assert len(statuses) == 1
    assert isinstance(statuses[0], bool)


def test_swap_node_runs_against_live_db_without_raising() -> None:
    """Smoke: the node must execute against the real DB without exceptions
    even when no stops are closed (or the place_id isn't in the index).
    """
    stops = [
        _make_stop(
            place_id="ChIJ_a_real_place_id_in_the_index",
            name="Anchor",
            primary_type="Bar",
            arrival=datetime(2026, 5, 19, 19, 0, tzinfo=SF),
        ),
    ]
    state = ItineraryState(stops=stops)
    update = asyncio.run(swap_closed_stops(state))
    assert isinstance(update, dict)


def test_walking_distance_search_uses_real_family_mapping() -> None:
    """A walking-distance search for the dessert family must return only
    candidates whose primary_type belongs to that family list (per the live
    `_PRIMARY_TYPE_FAMILIES` SQL clauses)."""
    from app.tools.filters import _PRIMARY_TYPE_FAMILIES

    dessert_primaries = set(_PRIMARY_TYPE_FAMILIES["dessert"]["primary_types"])
    closed = _make_stop(
        place_id="ChIJ_a_dessert_place_id",
        name="Closed Dessert",
        primary_type="Dessert Shop",
        arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
    )
    state = ItineraryState(stops=[closed])
    ctx = ClosureContext(
        place_id=closed.place_id,
        place_name=closed.name,
        family="dessert",
        attempted_arrival=closed.arrival_time or datetime.now(SF),
        outcome="pending_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=0,
    )
    match = _try_walking_distance_swap(state, ctx, anchor_place_id=closed.place_id)
    if match is None:
        pytest.skip("No walking-distance match in the live DB for this anchor.")
    assert match.stop.primary_type is None or match.stop.primary_type in dessert_primaries
