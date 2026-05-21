"""Smoke tests for the swap_closed_stops graph node.

Uses real ItineraryState + Stop models, with the SQL helpers and Routes API
mocked. Verifies the orchestration: no-op when nothing's closed, auto-swap
batches, escalation to pending, queueing additional pending entries, etc.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from app.agent.state import ClosureContext, ItineraryState, Stop
from app.tools.directions import DirectionsLeg, DirectionsResult
from app.tools.retrieval import PlaceHit

SF = ZoneInfo("America/Los_Angeles")


def _stop(
    place_id: str,
    *,
    name: str = "X",
    arrival_iso: str = "2026-05-19T19:00:00-07:00",
    primary_type: str = "Bar",
    lat: float = 37.78,
    lng: float = -122.41,
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


def _fake_route_factory():
    """Returns an async fake for route_legs that returns one leg per pair."""

    async def _r(stops, mode="walk"):
        legs = [DirectionsLeg(duration_s=600, distance_m=400.0)] * max(len(stops) - 1, 1)
        return DirectionsResult(
            legs=legs,
            total_duration_s=600 * len(legs),
            mode=mode,
            source="haversine_fallback",
        )

    return _r


def test_swap_node_noop_when_nothing_closed(mocker) -> None:
    from app.agent.swap import swap_closed_stops

    mocker.patch("app.agent.swap._execute_closure_query", return_value={"a": True, "b": True})
    state = ItineraryState(stops=[_stop("a"), _stop("b")])
    update = asyncio.run(swap_closed_stops(state))
    # No-op -> empty update (the graph state stays as-is)
    assert update == {}


def test_swap_node_auto_swap_silent_when_candidate_found(mocker) -> None:
    """Closure detected at stop 1; walking-distance candidate exists ->
    silent swap, closure_context records auto_swapped, summary reply (no
    "Caveats:" text)."""
    from app.agent.swap import swap_closed_stops

    # Stop b is closed initially; after the swap, all are open
    mocker.patch(
        "app.agent.swap._execute_closure_query",
        side_effect=[
            {"a": True, "b": False, "c": True},  # initial closure check
            {"a": True, "b_alt": True, "c": True},  # post-swap re-check
        ],
    )
    candidate = PlaceHit(
        place_id="b_alt",
        name="B Alt",
        primary_type="Bar",
        latitude=37.78,
        longitude=-122.41,
        source="google_places",
        similarity=0.0,
        dist_m=200.0,
    )
    mocker.patch("app.agent.swap._nearby_search", return_value=[candidate])
    mocker.patch("app.agent.swap.route_legs", side_effect=_fake_route_factory())
    mocker.patch("app.agent.swap.enrich_stops_with_booking", return_value=None)

    state = ItineraryState(
        stops=[
            _stop("a", primary_type="Bar"),
            _stop("b", primary_type="Bar"),
            _stop("c", primary_type="Bar"),
        ],
    )
    update = asyncio.run(swap_closed_stops(state))
    new_stops = update.get("stops")
    assert new_stops is not None
    assert [s.place_id for s in new_stops] == ["a", "b_alt", "c"]
    new_ctx = update["closure_context"]
    assert any(c.outcome == "auto_swapped" and c.place_id == "b" for c in new_ctx)
    # Silent swap -> reply is the regenerated summary
    reply = update.get("final_reply", "")
    assert "Caveats" not in reply
    assert "B Alt" in reply


def test_swap_node_escalates_to_pending_when_no_walking_match(mocker) -> None:
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"a": True, "b": False},
    )
    # Walking search returns empty; citywide search returns one far candidate.
    mocker.patch(
        "app.agent.swap._nearby_search",
        side_effect=[
            [],  # walking-distance result
            [
                PlaceHit(
                    place_id="b_far",
                    name="B Far",
                    primary_type="Bar",
                    latitude=37.80,
                    longitude=-122.45,
                    source="google_places",
                    similarity=0.0,
                    dist_m=4800.0,
                )
            ],  # citywide fallback
        ],
    )
    state = ItineraryState(
        stops=[_stop("a", primary_type="Bar"), _stop("b", primary_type="Bar")],
    )
    update = asyncio.run(swap_closed_stops(state))
    new_ctx = update["closure_context"]
    pending = [c for c in new_ctx if c.outcome == "pending_user_decision"]
    assert len(pending) == 1
    assert pending[0].place_id == "b"
    assert pending[0].proposed_alternative is not None
    assert pending[0].proposed_alternative.place_id == "b_far"
    # Reply is the question text, not a summary
    assert update["final_reply"]
    assert "B" in update["final_reply"]


def test_swap_node_queues_additional_pending_closures(mocker) -> None:
    """Two stops closed, neither walking-fixable -> first becomes pending,
    second becomes queued."""
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"a": False, "b": False},
    )
    mocker.patch("app.agent.swap._nearby_search", return_value=[])
    state = ItineraryState(
        stops=[_stop("a", primary_type="Bar"), _stop("b", primary_type="Bar")],
    )
    update = asyncio.run(swap_closed_stops(state))
    outcomes = [c.outcome for c in update["closure_context"]]
    assert outcomes.count("pending_user_decision") == 1
    assert outcomes.count("queued_user_decision") == 1


def test_swap_node_skips_when_family_unresolved(mocker) -> None:
    """A stop whose primary_type has no family (e.g. 'Spaceship') escalates
    to a pending entry with no proposal — we can't search."""
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        return_value={"a": True, "b": False},
    )
    mocker.patch("app.agent.swap._nearby_search", return_value=[])
    state = ItineraryState(
        stops=[
            _stop("a", primary_type="Bar"),
            _stop("b", primary_type="Spaceship"),  # unknown family
        ],
    )
    update = asyncio.run(swap_closed_stops(state))
    new_ctx = update["closure_context"]
    pending = [c for c in new_ctx if c.outcome == "pending_user_decision"]
    assert len(pending) == 1
    assert pending[0].family == ""
    assert pending[0].proposed_alternative is None


def test_swap_node_caps_closure_context_at_max(mocker) -> None:
    """When closure_context grows past MAX_CLOSURE_CONTEXT_ENTRIES, oldest
    entries are dropped."""
    from app.agent.state import MAX_CLOSURE_CONTEXT_ENTRIES
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        side_effect=[{"x": False}, {"y_alt": True}],
    )
    candidate = PlaceHit(
        place_id="y_alt",
        name="Y Alt",
        primary_type="Bar",
        latitude=37.78,
        longitude=-122.41,
        source="google_places",
        similarity=0.0,
        dist_m=200.0,
    )
    mocker.patch("app.agent.swap._nearby_search", return_value=[candidate])
    mocker.patch("app.agent.swap.route_legs", side_effect=_fake_route_factory())
    mocker.patch("app.agent.swap.enrich_stops_with_booking", return_value=None)

    existing = [
        ClosureContext(
            place_id=f"old_{i}",
            place_name=f"Old {i}",
            family="bar",
            attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
            outcome="auto_swapped",
            insert_after_place_id=None,
            insert_before_place_id=None,
            stop_index_hint=0,
        )
        for i in range(MAX_CLOSURE_CONTEXT_ENTRIES)
    ]
    state = ItineraryState(
        stops=[_stop("x", primary_type="Bar")],
        closure_context=existing,
    )
    update = asyncio.run(swap_closed_stops(state))
    new_ctx = update["closure_context"]
    assert len(new_ctx) == MAX_CLOSURE_CONTEXT_ENTRIES
    # The oldest "old_0" should be dropped; the new entry should be present.
    place_ids = {c.place_id for c in new_ctx}
    assert "old_0" not in place_ids
    assert "x" in place_ids


def test_swap_node_fail_open_on_initial_db_error(mocker) -> None:
    """If the initial closure query fails, the node is a no-op and ships
    the plan as-is. Matches checks.py:200-205 precedent."""
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap._execute_closure_query",
        side_effect=Exception("db down"),
    )
    state = ItineraryState(
        stops=[_stop("a", primary_type="Bar"), _stop("b", primary_type="Bar")],
    )
    update = asyncio.run(swap_closed_stops(state))
    # No-op (closure-status helper returns [False, False] on DB error)
    assert update == {}
