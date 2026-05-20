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


# ─── apply_swap + bounded_retime + promote_pending + question (Task 10) ─


def test_apply_swap_replaces_stop_at_position(mocker) -> None:
    from app.agent.state import ItineraryState
    from app.agent.swap import _apply_swap

    s1 = _stop(place_id="s1", name="S1")
    s2_closed = _stop(place_id="s2_closed", name="S2 closed")
    s3 = _stop(place_id="s3", name="S3")
    state = ItineraryState(stops=[s1, s2_closed, s3])
    replacement = _stop(place_id="s2_new", name="S2 new")
    leg_durations_min = [10.0, 5.0]

    # Avoid touching the real DB during enrich
    mocker.patch("app.agent.swap.enrich_stops_with_booking", return_value=None)

    new_stops = _apply_swap(
        state,
        stop_index=1,
        replacement=replacement,
        leg_durations_min=leg_durations_min,
    )

    assert [s.place_id for s in new_stops] == ["s1", "s2_new", "s3"]


def test_bounded_retime_after_swap_calls_route_legs_once(mocker) -> None:
    """The bounded retime helper makes at most ONE extra route_legs call per
    swap-node invocation. Mock route_legs and confirm call_count == 1."""
    import asyncio

    from app.agent.swap import _bounded_retime_after_swap
    from app.tools.directions import DirectionsLeg, DirectionsResult

    call_count = {"n": 0}

    async def _fake_route(stops, mode="walk"):
        call_count["n"] += 1
        legs = [DirectionsLeg(duration_s=600, distance_m=400.0)] * max(len(stops) - 1, 1)
        return DirectionsResult(
            legs=legs,
            total_duration_s=600 * len(legs),
            mode=mode,
            source="haversine_fallback",
        )

    mocker.patch("app.agent.swap.route_legs", side_effect=_fake_route)
    stops = [_stop(place_id="a"), _stop(place_id="b"), _stop(place_id="c")]
    retimed = asyncio.run(_bounded_retime_after_swap(stops))
    assert call_count["n"] == 1
    assert len(retimed) == 3


def test_promote_pending_flips_first_queued_to_pending() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _promote_pending

    queued1 = ClosureContext(
        place_id="q1",
        place_name="Q1",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="queued_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=0,
    )
    queued2 = queued1.model_copy(update={"place_id": "q2"})
    auto = queued1.model_copy(update={"place_id": "a", "outcome": "auto_swapped"})

    promoted = _promote_pending([auto, queued1, queued2])
    outcomes = [c.outcome for c in promoted]
    assert outcomes == ["auto_swapped", "pending_user_decision", "queued_user_decision"]


def test_promote_pending_is_noop_when_no_queued() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _promote_pending

    auto = ClosureContext(
        place_id="a",
        place_name="A",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="auto_swapped",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=0,
    )
    assert [c.outcome for c in _promote_pending([auto])] == ["auto_swapped"]


def test_promote_pending_is_noop_when_pending_already_present() -> None:
    """If pending already exists, don't promote a queued one too."""
    from app.agent.state import ClosureContext
    from app.agent.swap import _promote_pending

    pending = ClosureContext(
        place_id="p",
        place_name="P",
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=0,
    )
    queued = pending.model_copy(update={"place_id": "q", "outcome": "queued_user_decision"})
    promoted = _promote_pending([pending, queued])
    outcomes = [c.outcome for c in promoted]
    assert outcomes == ["pending_user_decision", "queued_user_decision"]


def test_formulate_closure_question_with_proposal() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _formulate_closure_question

    proposal = _stop(place_id="alt", name="Sophie's Crepes")
    ctx = ClosureContext(
        place_id="closed",
        place_name="Mochill Mochidonut",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=2,
        proposed_alternative=proposal,
        proposed_distance_m=4800.0,
    )
    q = _formulate_closure_question(ctx)
    assert "Sophie's Crepes" in q
    assert "Mochill Mochidonut" in q
    # ~3 mi rounding from 4800m is expected
    assert "3" in q


def test_formulate_closure_question_without_proposal() -> None:
    from app.agent.state import ClosureContext
    from app.agent.swap import _formulate_closure_question

    ctx = ClosureContext(
        place_id="closed",
        place_name="Mochill Mochidonut",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome="pending_user_decision",
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=2,
        proposed_alternative=None,
        proposed_distance_m=None,
    )
    q = _formulate_closure_question(ctx)
    assert "Mochill Mochidonut" in q
    # No proposal -> message should ask user to pick / change category
    assert "pick" in q.lower() or "different" in q.lower() or "skip" in q.lower()


# ─── _inject_closure_exclusions (Task 12) ────────────────────────────────


def _closure_entry(place_id: str = "closed1", outcome: str = "auto_swapped"):
    from app.agent.state import ClosureContext

    return ClosureContext(
        place_id=place_id,
        place_name=place_id,
        family="bar",
        attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=SF),
        outcome=outcome,  # type: ignore[arg-type]
        insert_after_place_id=None,
        insert_before_place_id=None,
        stop_index_hint=0,
    )


def test_inject_closure_exclusions_merges_into_semantic_search_filters() -> None:
    """Filters arrive as either a Pydantic SearchFilters (rare — direct
    test usage) or a dict (the actual LangChain wire shape). Either way,
    the returned `filters` MUST be a plain dict so the result is
    json-serializable when langchain re-sends the AIMessage.tool_calls."""
    from app.agent.swap import _inject_closure_exclusions
    from app.tools.filters import SearchFilters

    ctx = [
        _closure_entry("closed1", "auto_swapped"),
        _closure_entry("closed2", "user_accepted_drive"),
    ]
    args = {
        "query": "ramen",
        "filters": SearchFilters(min_rating=4.0, excluded_place_ids=["llm_excluded"]),
    }
    out = _inject_closure_exclusions("semantic_search", args, ctx)
    assert isinstance(out["filters"], dict)
    excluded = out["filters"]["excluded_place_ids"]
    assert set(excluded) == {"llm_excluded", "closed1", "closed2"}
    # Returns a new args dict (not in-place mutation)
    assert out is not args


def test_inject_closure_exclusions_creates_filters_when_absent() -> None:
    from app.agent.swap import _inject_closure_exclusions

    ctx = [_closure_entry("closed1", "auto_swapped")]
    args = {"query": "ramen"}
    out = _inject_closure_exclusions("semantic_search", args, ctx)
    assert "filters" in out
    assert isinstance(out["filters"], dict)
    assert out["filters"]["excluded_place_ids"] == ["closed1"]


def test_inject_closure_exclusions_kg_traverse_is_top_level() -> None:
    """kg_traverse takes excluded_place_ids as a top-level arg, not via filters."""
    from app.agent.swap import _inject_closure_exclusions

    ctx = [_closure_entry("closed1", "auto_swapped")]
    args = {"place_id": "anchor", "relation_type": "SIMILAR_VECTOR"}
    out = _inject_closure_exclusions("kg_traverse", args, ctx)
    assert out["excluded_place_ids"] == ["closed1"]


def test_inject_closure_exclusions_empty_context_is_noop() -> None:
    from app.agent.swap import _inject_closure_exclusions

    args = {"query": "ramen"}
    out = _inject_closure_exclusions("semantic_search", args, [])
    assert out == args
    assert "filters" not in out


def test_inject_closure_exclusions_unknown_tool_is_noop() -> None:
    from app.agent.swap import _inject_closure_exclusions

    ctx = [_closure_entry("closed1", "auto_swapped")]
    args = {"foo": "bar"}
    out = _inject_closure_exclusions("get_details", args, ctx)
    assert out == args


def test_inject_closure_exclusions_accepts_dict_filters_from_llm() -> None:
    """LangChain delivers `filters` as a dict in tool_call args (the
    StructuredTool args_schema is for validation, not for the on-the-wire
    shape). The helper must accept dicts AND return dicts — anything else
    breaks JSON serialization when the AIMessage is later sent back to the
    LLM with the tool_call still attached.
    """
    from app.agent.swap import _inject_closure_exclusions

    ctx = [_closure_entry("closed1", "auto_swapped")]
    args = {"query": "ramen", "filters": {"min_rating": 4.0, "excluded_place_ids": ["llm_excl"]}}
    out = _inject_closure_exclusions("semantic_search", args, ctx)
    assert isinstance(out["filters"], dict), "filters must round-trip as a dict"
    assert "closed1" in out["filters"]["excluded_place_ids"]
    assert "llm_excl" in out["filters"]["excluded_place_ids"]


def test_inject_closure_exclusions_output_is_json_serializable() -> None:
    """Regression test for the SearchFilters-in-AIMessage.tool_calls crash:
    `args["filters"]` was stored as a Pydantic instance and json.dumps on
    the re-sent message blew up. The helper's output must be a plain dict
    tree so langchain can serialize it as the tool_call args.
    """
    import json as _json

    from app.agent.swap import _inject_closure_exclusions
    from app.tools.filters import SearchFilters

    ctx = [_closure_entry("closed1", "auto_swapped")]
    # All three shapes LangChain might deliver: dict, SearchFilters, absent
    for filters_in in (
        {"min_rating": 4.0},
        SearchFilters(min_rating=4.0),
        None,
    ):
        args = (
            {"query": "ramen"} if filters_in is None else {"query": "ramen", "filters": filters_in}
        )
        out = _inject_closure_exclusions("semantic_search", args, ctx)
        # If json.dumps doesn't raise, langchain's _lc_tool_call_to_openai_tool_call
        # won't crash on the round trip.
        _json.dumps(out)
