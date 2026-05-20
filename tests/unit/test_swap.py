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


# ─── _resolve_insert_position + _score_candidate (Task 8) ───────────────


def test_resolve_insert_position_uses_insert_after_when_anchor_present() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _resolve_insert_position

    stops = [_stop(place_id="a"), _stop(place_id="b"), _stop(place_id="c")]
    ctx = ClosureContext(
        place_id="closed",
        place_name="X",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="a",
        insert_before_place_id=None,
        stop_index_hint=99,
    )
    # insert_after a (index 0) -> position 1
    assert _resolve_insert_position(ctx, stops) == 1


def test_resolve_insert_position_falls_back_to_insert_before() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _resolve_insert_position

    stops = [_stop(place_id="a"), _stop(place_id="b")]
    ctx = ClosureContext(
        place_id="closed",
        place_name="X",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="missing",
        insert_before_place_id="b",
        stop_index_hint=99,
    )
    # insert_after missing; insert_before b (index 1) -> position 1
    assert _resolve_insert_position(ctx, stops) == 1


def test_resolve_insert_position_falls_back_to_index_hint_clamped() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _resolve_insert_position

    stops = [_stop(place_id="a"), _stop(place_id="b")]
    ctx = ClosureContext(
        place_id="closed",
        place_name="X",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="missing",
        insert_before_place_id="also_missing",
        stop_index_hint=99,
    )
    # Both anchors absent; clamp hint to len(stops)
    assert _resolve_insert_position(ctx, stops) == 2


def test_score_candidate_prefers_lower_route_impact() -> None:
    """Two candidates with identical family-match: the one with smaller
    combined prev+next distance scores higher."""
    from app.agent.swap import _score_candidate

    closed = _stop(place_id="closed", lat=37.78, lng=-122.41)
    prev_ = _stop(place_id="prev", lat=37.78, lng=-122.41)
    next_ = _stop(place_id="next", lat=37.785, lng=-122.41)

    close_candidate = _stop(place_id="c1", lat=37.78, lng=-122.41)
    far_candidate = _stop(place_id="c2", lat=37.90, lng=-122.41)

    s_close = _score_candidate(close_candidate, closed, prev_, next_, family_match=True)
    s_far = _score_candidate(far_candidate, closed, prev_, next_, family_match=True)
    assert s_close > s_far


def test_score_candidate_prefers_family_match() -> None:
    """All else equal, a family-matching candidate beats one that doesn't."""
    from app.agent.swap import _score_candidate

    closed = _stop(place_id="closed")
    prev_ = _stop(place_id="prev")
    next_ = _stop(place_id="next")
    candidate = _stop(place_id="c", lat=37.78, lng=-122.41)

    s_match = _score_candidate(candidate, closed, prev_, next_, family_match=True)
    s_nomatch = _score_candidate(candidate, closed, prev_, next_, family_match=False)
    assert s_match > s_nomatch
