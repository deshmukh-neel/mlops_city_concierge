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


# ─── walking-distance + citywide candidate search (Task 9) ──────────────


def test_try_walking_distance_swap_uses_family_and_exclusion(mocker) -> None:
    """The swap helper must:
    1. Pass the family to nearby() via SearchFilters.primary_type_family.
    2. Pass closure_context + current-stops exclusions via excluded_place_ids.
    3. Pass open_at = attempted_arrival.
    4. Return the highest-scoring candidate, if any.
    """
    from app.agent.state import ClosureContext, ItineraryState
    from app.agent.swap import _try_walking_distance_swap
    from app.tools.retrieval import PlaceHit

    captured_calls: list = []

    def _fake_nearby(place_id, radius_m, filters, k):
        captured_calls.append((place_id, radius_m, filters, k))
        return [
            PlaceHit(
                place_id="alt1",
                name="Alt 1",
                primary_type="Dessert Shop",
                latitude=37.78,
                longitude=-122.41,
                rating=4.5,
                source="google_places",
                similarity=0.0,
                dist_m=300.0,
            )
        ]

    mocker.patch("app.agent.swap._nearby_search", side_effect=_fake_nearby)
    closed = _stop(place_id="closed", primary_type="Dessert Shop")
    prev_ = _stop(place_id="prev", lat=37.78, lng=-122.41)
    state = ItineraryState(stops=[prev_, closed], closure_context=[])
    ctx = ClosureContext(
        place_id="closed",
        place_name="Closed",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="prev",
        insert_before_place_id=None,
        stop_index_hint=1,
    )

    match = _try_walking_distance_swap(state, ctx, anchor_place_id="prev")

    assert match is not None
    assert match.stop.place_id == "alt1"
    # captured: (place_id, radius_m, filters, k)
    _, radius_m, filters, _ = captured_calls[0]
    assert radius_m <= 500  # walking budget
    assert filters.primary_type_family == "dessert"
    assert "closed" in (filters.excluded_place_ids or [])
    assert filters.open_at == datetime(2026, 5, 19, 20, 0, tzinfo=SF)


def test_try_walking_distance_swap_returns_none_when_no_candidates(mocker) -> None:
    from app.agent.state import ClosureContext, ItineraryState
    from app.agent.swap import _try_walking_distance_swap

    mocker.patch("app.agent.swap._nearby_search", return_value=[])
    closed = _stop(place_id="closed", primary_type="Dessert Shop")
    state = ItineraryState(stops=[_stop(place_id="prev"), closed])
    ctx = ClosureContext(
        place_id="closed",
        place_name="Closed",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="prev",
        insert_before_place_id=None,
        stop_index_hint=1,
    )
    assert _try_walking_distance_swap(state, ctx, anchor_place_id="prev") is None


def test_try_any_distance_search_uses_citywide_radius(mocker) -> None:
    """Fallback search uses _CITYWIDE_RADIUS_M (30 km) so the question can
    propose a drive-distance alternative when nothing is within walking
    distance."""
    from app.agent.state import ClosureContext, ItineraryState
    from app.agent.swap import _try_any_distance_search
    from app.tools.retrieval import PlaceHit

    captured = []

    def _fake_nearby(place_id, radius_m, filters, k):
        captured.append(radius_m)
        return [
            PlaceHit(
                place_id="alt2",
                name="Alt 2 (far)",
                primary_type="Dessert Shop",
                latitude=37.80,
                longitude=-122.45,
                source="google_places",
                similarity=0.0,
                dist_m=4800.0,
            )
        ]

    mocker.patch("app.agent.swap._nearby_search", side_effect=_fake_nearby)
    closed = _stop(place_id="closed", primary_type="Dessert Shop")
    state = ItineraryState(stops=[_stop(place_id="prev"), closed])
    ctx = ClosureContext(
        place_id="closed",
        place_name="Closed",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id="prev",
        insert_before_place_id=None,
        stop_index_hint=1,
    )

    match = _try_any_distance_search(state, ctx, anchor_place_id="prev")
    assert match is not None
    assert match.distance_m == 4800.0
    assert captured[0] == 30_000
