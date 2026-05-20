"""Unit tests for app/agent/swap.py.

All DB access (place_is_open SQL function) is mocked via _execute_closure_query
or the helper's direct cursor calls. Live-DB behavior is covered separately
in tests/integration/test_swap_real_db.py.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.agent.state import Stop

SF = ZoneInfo("America/Los_Angeles")


def _stop(
    place_id: str = "p",
    name: str = "X",
    arrival_iso: str = "2026-05-19T19:00:00-07:00",
    lat: float = 37.78,
    lng: float = -122.41,
    primary_type: str = "Bar",
) -> Stop:
    return Stop(
        place_id=place_id,
        name=name,
        rationale="r",
        source="google_places",
        arrival_time=datetime.fromisoformat(arrival_iso),
        latitude=lat,
        longitude=lng,
        primary_type=primary_type,
        planned_duration_min=60,
    )


def test_per_stop_closure_status_all_open(mocker) -> None:
    """Every stop returns is_open=True → list of all False (False = not closed)."""
    from app.agent.swap import _per_stop_closure_status

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"p1": True, "p2": True},
    )
    stops = [_stop(place_id="p1"), _stop(place_id="p2")]
    statuses = _per_stop_closure_status(stops)
    # _per_stop_closure_status returns True for "closed", False for "open".
    assert statuses == [False, False]


def test_per_stop_closure_status_one_closed(mocker) -> None:
    from app.agent.swap import _per_stop_closure_status

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"p1": True, "p2": False},
    )
    stops = [_stop(place_id="p1"), _stop(place_id="p2")]
    statuses = _per_stop_closure_status(stops)
    assert statuses == [False, True]


def test_per_stop_closure_status_skips_stops_without_arrival_time(mocker) -> None:
    """A stop with arrival_time=None can't be checked; treat as not-closed."""
    from app.agent.swap import _per_stop_closure_status

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"p1": True},
    )
    stops = [_stop(place_id="p1")]
    stops.append(
        Stop(
            place_id="p2",
            name="no time",
            rationale="r",
            source="google_places",
            arrival_time=None,
            latitude=37.78,
            longitude=-122.41,
            primary_type="Bar",
            planned_duration_min=60,
        )
    )
    statuses = _per_stop_closure_status(stops)
    assert statuses == [False, False]


def test_per_stop_closure_status_db_failure_fails_open(mocker) -> None:
    """A DB blip must NOT block /chat — the helper returns [False] * n
    (no closure detected) so the plan ships unchanged. Matches checks.py
    fail-open precedent at lines 200-205."""
    from app.agent.swap import _per_stop_closure_status

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        side_effect=Exception("db down"),
    )
    statuses = _per_stop_closure_status([_stop(place_id="p1"), _stop(place_id="p2")])
    assert statuses == [False, False]
