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


def stop(
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


def fake_route_factory():
    """Returns an async fake for route_legs that returns one leg per pair."""

    async def r(stops, mode="walk"):
        legs = [DirectionsLeg(duration_s=600, distance_m=400.0)] * max(len(stops) - 1, 1)
        return DirectionsResult(
            legs=legs,
            total_duration_s=600 * len(legs),
            mode=mode,
            source="haversine_fallback",
        )

    return r


def test_swap_node_noop_when_nothing_closed(mocker) -> None:
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap.execute_closure_query",
        return_value={"ChIJtest_a_aaaaaaaaa": True, "ChIJtest_b_aaaaaaaaa": True},
    )
    state = ItineraryState(stops=[stop("ChIJtest_a_aaaaaaaaa"), stop("ChIJtest_b_aaaaaaaaa")])
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
        "app.agent.swap.execute_closure_query",
        side_effect=[
            {
                "ChIJtest_a_aaaaaaaaa": True,
                "ChIJtest_b_aaaaaaaaa": False,
                "ChIJtest_c_aaaaaaaaa": True,
            },  # initial closure check
            {
                "ChIJtest_a_aaaaaaaaa": True,
                "ChIJtest_b_alt_aaaaa": True,
                "ChIJtest_c_aaaaaaaaa": True,
            },  # post-swap re-check
        ],
    )
    candidate = PlaceHit(
        place_id="ChIJtest_b_alt_aaaaa",
        name="B Alt",
        primary_type="Bar",
        latitude=37.78,
        longitude=-122.41,
        source="google_places",
        similarity=0.0,
        dist_m=200.0,
    )
    mocker.patch("app.agent.swap.nearby_search", return_value=[candidate])
    mocker.patch("app.agent.swap.route_legs", side_effect=fake_route_factory())
    mocker.patch("app.agent.swap.enrich_stops_with_booking", return_value=None)

    state = ItineraryState(
        stops=[
            stop("ChIJtest_a_aaaaaaaaa", primary_type="Bar"),
            stop("ChIJtest_b_aaaaaaaaa", primary_type="Bar"),
            stop("ChIJtest_c_aaaaaaaaa", primary_type="Bar"),
        ],
    )
    update = asyncio.run(swap_closed_stops(state))
    new_stops = update.get("stops")
    assert new_stops is not None
    assert [s.place_id for s in new_stops] == [
        "ChIJtest_a_aaaaaaaaa",
        "ChIJtest_b_alt_aaaaa",
        "ChIJtest_c_aaaaaaaaa",
    ]
    new_ctx = update["closure_context"]
    assert any(
        c.outcome == "auto_swapped" and c.place_id == "ChIJtest_b_aaaaaaaaa" for c in new_ctx
    )
    # Silent swap -> reply is the regenerated summary
    reply = update.get("final_reply", "")
    assert "Caveats" not in reply
    assert "B Alt" in reply


def test_swap_node_escalates_to_pending_when_no_walking_match(mocker) -> None:
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap.execute_closure_query",
        return_value={"ChIJtest_a_aaaaaaaaa": True, "ChIJtest_b_aaaaaaaaa": False},
    )
    # Walking search returns empty; citywide search returns one far candidate.
    mocker.patch(
        "app.agent.swap.nearby_search",
        side_effect=[
            [],  # walking-distance result
            [
                PlaceHit(
                    place_id="ChIJtest_b_far_aaaaa",
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
        stops=[
            stop("ChIJtest_a_aaaaaaaaa", primary_type="Bar"),
            stop("ChIJtest_b_aaaaaaaaa", primary_type="Bar"),
        ],
    )
    update = asyncio.run(swap_closed_stops(state))
    new_ctx = update["closure_context"]
    pending = [c for c in new_ctx if c.outcome == "pending_user_decision"]
    assert len(pending) == 1
    assert pending[0].place_id == "ChIJtest_b_aaaaaaaaa"
    assert pending[0].proposed_alternative is not None
    assert pending[0].proposed_alternative.place_id == "ChIJtest_b_far_aaaaa"
    # Reply is the question text, not a summary
    assert update["final_reply"]
    assert "B" in update["final_reply"]


def test_swap_node_queues_additional_pending_closures(mocker) -> None:
    """Two stops closed, neither walking-fixable -> first becomes pending,
    second becomes queued."""
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap.execute_closure_query",
        return_value={"ChIJtest_a_aaaaaaaaa": False, "ChIJtest_b_aaaaaaaaa": False},
    )
    mocker.patch("app.agent.swap.nearby_search", return_value=[])
    state = ItineraryState(
        stops=[
            stop("ChIJtest_a_aaaaaaaaa", primary_type="Bar"),
            stop("ChIJtest_b_aaaaaaaaa", primary_type="Bar"),
        ],
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
        "app.agent.swap.execute_closure_query",
        return_value={"ChIJtest_a_aaaaaaaaa": True, "ChIJtest_b_aaaaaaaaa": False},
    )
    mocker.patch("app.agent.swap.nearby_search", return_value=[])
    state = ItineraryState(
        stops=[
            stop("ChIJtest_a_aaaaaaaaa", primary_type="Bar"),
            stop("ChIJtest_b_aaaaaaaaa", primary_type="Spaceship"),  # unknown family
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
        "app.agent.swap.execute_closure_query",
        side_effect=[{"ChIJtest_x_aaaaaaaaa": False}, {"ChIJtest_y_alt_aaaaa": True}],
    )
    candidate = PlaceHit(
        place_id="ChIJtest_y_alt_aaaaa",
        name="Y Alt",
        primary_type="Bar",
        latitude=37.78,
        longitude=-122.41,
        source="google_places",
        similarity=0.0,
        dist_m=200.0,
    )
    mocker.patch("app.agent.swap.nearby_search", return_value=[candidate])
    mocker.patch("app.agent.swap.route_legs", side_effect=fake_route_factory())
    mocker.patch("app.agent.swap.enrich_stops_with_booking", return_value=None)

    # Padded place_ids satisfy the 06-01 Task 3 Google-Place-ID-format validator.
    old_pid = lambda i: f"ChIJtest_old_{i}_aaaaa"  # noqa: E731 — 20+ chars for i in 0..9
    existing = [
        ClosureContext(
            place_id=old_pid(i),
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
        stops=[stop("ChIJtest_x_aaaaaaaaa", primary_type="Bar")],
        closure_context=existing,
    )
    update = asyncio.run(swap_closed_stops(state))
    new_ctx = update["closure_context"]
    assert len(new_ctx) == MAX_CLOSURE_CONTEXT_ENTRIES
    # The oldest entry (index 0) should be dropped; the new entry should be present.
    place_ids = {c.place_id for c in new_ctx}
    assert old_pid(0) not in place_ids
    assert "ChIJtest_x_aaaaaaaaa" in place_ids


def test_swap_node_fail_open_on_initial_db_error(mocker) -> None:
    """If the initial closure query fails, the node is a no-op and ships
    the plan as-is. Matches checks.py:200-205 precedent."""
    from app.agent.swap import swap_closed_stops

    mocker.patch(
        "app.agent.swap.execute_closure_query",
        side_effect=Exception("db down"),
    )
    state = ItineraryState(
        stops=[
            stop("ChIJtest_a_aaaaaaaaa", primary_type="Bar"),
            stop("ChIJtest_b_aaaaaaaaa", primary_type="Bar"),
        ],
    )
    update = asyncio.run(swap_closed_stops(state))
    # No-op (closure-status helper returns [False, False] on DB error)
    assert update == {}


def test_swap_node_post_swap_rationale_satisfies_alignment_scorer(mocker) -> None:
    """Functional regression for RAT-02 / D-05-03 (Phase 5 plan 02).

    Drives the full `swap_closed_stops` graph node end-to-end (not just the
    inner `candidates_to_matches` helper that plan 01 unit-tested) on a
    state with one closed stop + one walking-distance candidate, then asserts
    BOTH halves of D-05-03:

    1. The user-visible `final_reply` does NOT contain the substring
       "Walking-distance alternative for" (the placeholder that triggered
       the bug).
    2. `rationale_stop_alignment(post_state) == 1.0` where post_state is
       built from `update["stops"]` — the existing Phase 3 scorer assents
       to every stop in the post-swap state.

    Mock strategy notes:
    - Unlike the analog `test_swap_node_auto_swap_silent_when_candidate_found`
      above, this test does NOT mock `enrich_stops_with_booking` to a no-op.
      Instead it patches `app.agent.commit.get_details_many` to return `{}`,
      which makes `enrich_stops_with_booking` run for real but skip each
      per-stop branch via its `details is None` guard at commit.py:114. This
      preserves the function call boundary so a future planner extending
      the fix into the enrichment loop (path (b)-extend variant) is not
      silently bypassed by this test.
    - Also patches `app.agent.revision.itinerary_violations` to `[]` per
      `project_full_suite_db_pool_contamination.md` — the full suite leaks a
      live DB pool through this function otherwise.
    """
    from app.agent.critique.checks import rationale_stop_alignment
    from app.agent.swap import swap_closed_stops

    # Stop b is closed initially; after the swap, all are open.
    mocker.patch(
        "app.agent.swap.execute_closure_query",
        side_effect=[
            {
                "ChIJtest_a_aaaaaaaaa": True,
                "ChIJtest_b_aaaaaaaaa": False,
                "ChIJtest_c_aaaaaaaaa": True,
            },  # initial closure check
            {
                "ChIJtest_a_aaaaaaaaa": True,
                "ChIJtest_b_alt_aaaaa": True,
                "ChIJtest_c_aaaaaaaaa": True,
            },  # post-swap re-check
        ],
    )
    candidate = PlaceHit(
        place_id="ChIJtest_b_alt_aaaaa",
        name="B Alt",
        primary_type="Bar",
        latitude=37.78,
        longitude=-122.41,
        source="google_places",
        similarity=0.0,
        dist_m=200.0,
    )
    mocker.patch("app.agent.swap.nearby_search", return_value=[candidate])
    mocker.patch("app.agent.swap.route_legs", side_effect=fake_route_factory())
    # Structurally-real no-op for enrich_stops_with_booking: the function runs
    # but each per-stop branch hits the `details is None` guard at commit.py:114.
    mocker.patch("app.agent.commit.get_details_many", return_value={})
    # Full-suite DB-pool contamination defense per project memory.
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

    # Stops a and c carry rationales that already satisfy `is_rationale_aligned`
    # (each contains its own stop's name) so the scorer's assertion below
    # measures the post-swap stop b_alt's rationale in isolation. The default
    # `stop` factory rationale "r" would dilute the scorer to ~0.33 even with
    # a perfect swap and obscure what this test is actually pinning.
    state = ItineraryState(
        stops=[
            stop("ChIJtest_a_aaaaaaaaa", name="Alpha", primary_type="Bar").model_copy(
                update={"rationale": "Alpha is a lively cocktail bar."}
            ),
            stop("ChIJtest_b_aaaaaaaaa", name="Beta", primary_type="Bar"),
            stop("ChIJtest_c_aaaaaaaaa", name="Gamma", primary_type="Bar").model_copy(
                update={"rationale": "Gamma rounds out the night with craft beer."}
            ),
        ],
    )
    update = asyncio.run(swap_closed_stops(state))

    # Half (1) of D-05-03: no placeholder substring in user-visible reply.
    final_reply = update.get("final_reply", "")
    assert final_reply, "swap node returned empty final_reply"
    assert "Walking-distance alternative for" not in final_reply

    # Half (2) of D-05-03: post-swap state passes the Phase 3 alignment scorer.
    new_stops = update.get("stops")
    assert new_stops is not None
    assert [s.place_id for s in new_stops] == [
        "ChIJtest_a_aaaaaaaaa",
        "ChIJtest_b_alt_aaaaa",
        "ChIJtest_c_aaaaaaaaa",
    ]
    post_state = ItineraryState(stops=new_stops)
    assert rationale_stop_alignment(post_state) == 1.0

    # Sanity: the swap path was actually exercised (not a no-op short-circuit).
    assert any(
        c.outcome == "auto_swapped" and c.place_id == "ChIJtest_b_aaaaaaaaa"
        for c in update["closure_context"]
    )
    # Sanity: the candidate's name made it into the rendered summary.
    assert "B Alt" in final_reply
