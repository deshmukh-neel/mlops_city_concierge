"""Unit + mock tests for app/agent/critique/checks.py.

Pure functions get pure tests. SQL paths get mocked via the same FakeConnection
pattern used in tests/unit/test_tools_retrieval.py — we don't bring up a DB.

Real DB integration lives in tests/integration/test_critique_checks_db.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.agent.critique.checks import (
    CRITIQUE_THRESHOLDS,
    category_compliance,
    category_compliance_strict,
    constraints_satisfied,
    geographic_coherence,
    is_rationale_aligned,
    itinerary_violations,
    no_hallucinated_place_ids,
    rationale_stop_alignment,
    stop_count_satisfied,
    temporal_coherence,
    walking_budget_respected,
)
from app.agent.state import ItineraryState, Stop, UserConstraints

# --- DB fakes ---------------------------------------------------------------


class _FakeCursor:
    def __init__(self, queue: list[list[dict] | list[tuple]]) -> None:
        self.queue = list(queue)
        self.executed: list[tuple[str, list]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: list) -> None:
        self.executed.append((sql, list(params)))
        self._next = self.queue.pop(0) if self.queue else []

    def fetchall(self) -> list:
        return self._next

    def fetchone(self) -> Any:
        return self._next[0] if self._next else None


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self, **_kw: Any) -> _FakeCursor:
        return self._cursor


@pytest.fixture
def patch_db(mocker):
    """Patch app.agent.critique.checks.get_conn to yield a controllable cursor."""

    def _patch(rows_per_execute: list[list]) -> _FakeCursor:
        cursor = _FakeCursor(rows_per_execute)
        conn = _FakeConn(cursor)

        @contextmanager
        def _get_conn():
            yield conn

        mocker.patch("app.agent.critique.checks.get_conn", _get_conn)
        return cursor

    return _patch


def _stop(
    place_id: str = "p1",
    *,
    arrival: datetime | None = None,
    lat: float | None = None,
    lng: float | None = None,
    name: str | None = None,
    rationale: str = "",
    primary_type: str | None = None,
) -> Stop:
    return Stop(
        place_id=place_id,
        name=name if name is not None else place_id.upper(),
        source="google_places",
        rationale=rationale,
        arrival_time=arrival,
        latitude=lat,
        longitude=lng,
        primary_type=primary_type,
    )


# --- no_hallucinated_place_ids ----------------------------------------------


def test_no_hallucinated_passes_when_all_resolve(patch_db) -> None:
    patch_db([[("p1",), ("p2",)]])
    state = ItineraryState(stops=[_stop("p1"), _stop("p2")])
    assert no_hallucinated_place_ids(state) == 1.0


def test_no_hallucinated_fails_when_any_missing(patch_db) -> None:
    """One missing place_id is critical — score drops to 0."""
    patch_db([[("p1",)]])
    state = ItineraryState(stops=[_stop("p1"), _stop("p2")])
    assert no_hallucinated_place_ids(state) == 0.0


def test_no_hallucinated_returns_one_for_empty_stops() -> None:
    """Vacuous truth — empty itinerary has no hallucinations."""
    assert no_hallucinated_place_ids(ItineraryState()) == 1.0


# --- temporal_coherence -----------------------------------------------------


def test_temporal_returns_one_when_no_arrival_times() -> None:
    state = ItineraryState(stops=[_stop("p1"), _stop("p2")])
    # No DB call needed — function short-circuits.
    assert temporal_coherence(state) == 1.0


def test_temporal_fractional_when_some_closed(patch_db) -> None:
    """One coalesced query now returns one row per matched stop."""
    arrival = datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc)
    patch_db(
        [
            [
                {"place_id": "p1", "is_open": True},
                {"place_id": "p2", "is_open": False},
            ]
        ]
    )
    state = ItineraryState(
        stops=[
            _stop("p1", arrival=arrival),
            _stop("p2", arrival=arrival + timedelta(hours=2)),
        ]
    )
    assert temporal_coherence(state) == 0.5


def test_temporal_treats_missing_row_as_open(patch_db) -> None:
    """A stop whose place_id has no places_raw row is absent from the JOIN
    result; the function defaults that stop to open."""
    arrival = datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc)
    patch_db([[]])  # JOIN returned zero rows
    state = ItineraryState(stops=[_stop("p1", arrival=arrival)])
    assert temporal_coherence(state) == 1.0


# --- geographic_coherence ---------------------------------------------------


def test_geographic_perfect_when_close() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            _stop("p1", lat=37.78, lng=-122.41),
            _stop("p2", lat=37.781, lng=-122.411),
        ],
    )
    assert geographic_coherence(state) == 1.0


def test_geographic_fractional_when_one_leg_too_long() -> None:
    """Per-leg budget = 2400/2 = 1200m. Leg 1 is short, leg 2 is ~1.5km."""
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            _stop("p1", lat=37.78, lng=-122.41),
            _stop("p2", lat=37.781, lng=-122.411),  # tiny hop
            _stop("p3", lat=37.794, lng=-122.41),  # ~1.5km from p2
        ],
    )
    assert geographic_coherence(state) == 0.5


def test_geographic_skips_pairs_missing_coords() -> None:
    """Pairs without coordinates aren't measurable — score on what we can."""
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            _stop("p1", lat=37.78, lng=-122.41),
            _stop("p2"),  # no coords
            _stop("p3", lat=37.781, lng=-122.411),
        ],
    )
    # Neither leg measurable — both endpoints incomplete.
    assert geographic_coherence(state) == 1.0


def test_geographic_returns_one_for_single_stop() -> None:
    state = ItineraryState(stops=[_stop("p1", lat=37.78, lng=-122.41)])
    assert geographic_coherence(state) == 1.0


# --- walking_budget_respected -----------------------------------------------


def test_walking_budget_respected_when_under() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            _stop("p1", lat=37.78, lng=-122.41),
            _stop("p2", lat=37.781, lng=-122.411),
        ],
    )
    assert walking_budget_respected(state) == 1.0


def test_walking_budget_violated_when_over() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=100),
        stops=[
            _stop("p1", lat=37.78, lng=-122.41),
            _stop("p2", lat=37.80, lng=-122.41),  # ~2.2km
        ],
    )
    assert walking_budget_respected(state) == 0.0


# --- constraints_satisfied --------------------------------------------------


def test_constraints_satisfied_full_pass(patch_db) -> None:
    patch_db(
        [
            [
                {
                    "place_id": "p1",
                    "price_rank": 2,
                    "rating": 4.5,
                    "user_rating_count": 200,
                    "neighborhood": "Mission",
                    "formatted_address": "x",
                }
            ]
        ]
    )
    state = ItineraryState(
        constraints=UserConstraints(
            price_level_max=3,
            min_rating=4.0,
            min_user_rating_count=50,
            neighborhood="Mission",
        ),
        stops=[_stop("p1")],
    )
    assert constraints_satisfied(state) == 1.0


def test_constraints_satisfied_partial(patch_db) -> None:
    """Place is OK on rating + neighborhood but blows the price cap."""
    patch_db(
        [
            [
                {
                    "place_id": "p1",
                    "price_rank": 4,  # over cap of 2
                    "rating": 4.5,
                    "user_rating_count": 200,
                    "neighborhood": "Mission",
                    "formatted_address": "x",
                }
            ]
        ]
    )
    state = ItineraryState(
        constraints=UserConstraints(
            price_level_max=2,
            min_rating=4.0,
            min_user_rating_count=50,
            neighborhood="Mission",
        ),
        stops=[_stop("p1")],
    )
    # 3 of 4 expressed constraints satisfied = 0.75
    assert constraints_satisfied(state) == pytest.approx(0.75)


def test_constraints_satisfied_neighborhood_falls_back_to_address(patch_db) -> None:
    """When the structured neighborhood column is empty, address ILIKE wins."""
    patch_db(
        [
            [
                {
                    "place_id": "p1",
                    "price_rank": None,
                    "rating": None,
                    "user_rating_count": None,
                    "neighborhood": None,
                    "formatted_address": "1 Main St, North Beach, SF",
                }
            ]
        ]
    )
    state = ItineraryState(
        constraints=UserConstraints(neighborhood="North Beach"),
        stops=[_stop("p1")],
    )
    assert constraints_satisfied(state) == 1.0


def test_constraints_satisfied_skips_hallucinated_pids(patch_db) -> None:
    """Stops whose place_id isn't in the DB row dump aren't scored — that's
    no_hallucinated_place_ids' job."""
    patch_db([[]])  # no rows
    state = ItineraryState(
        constraints=UserConstraints(price_level_max=2),
        stops=[_stop("p1"), _stop("p2")],
    )
    # No rows = no scoring possible — return 1.0 vacuously.
    assert constraints_satisfied(state) == 1.0


# --- category_compliance (EVAL-01) ------------------------------------------
# Pure-function scorer: no DB access. D-03 abstain contract = return 1.0 when
# the user didn't name category slots (requested_primary_types == []).


def test_category_compliance_abstains_when_no_requested_types() -> None:
    """D-03 abstain contract: empty requested_primary_types -> 1.0."""
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=[]),
        stops=[_stop("p1", primary_type="Sushi Restaurant")],
    )
    assert category_compliance(state) == 1.0


def test_category_compliance_returns_one_for_empty_stops() -> None:
    """Fail-open: no committed stops -> 1.0 (nothing to score)."""
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[],
    )
    assert category_compliance(state) == 1.0


def test_category_compliance_single_exact_family_match() -> None:
    """Same exact primary_type -> family matches -> 1.0."""
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[_stop("p1", primary_type="Sushi Restaurant")],
    )
    assert category_compliance(state) == 1.0


def test_category_compliance_multi_slot_all_match() -> None:
    """Per-index family match across multiple slots -> 1.0."""
    state = ItineraryState(
        constraints=UserConstraints(
            requested_primary_types=["Sushi Restaurant", "Cocktail Bar"],
        ),
        stops=[
            _stop("p1", primary_type="Sushi Restaurant"),
            _stop("p2", primary_type="Cocktail Bar"),
        ],
    )
    assert category_compliance(state) == 1.0


def test_category_compliance_family_mismatch_single_stop() -> None:
    """Family mismatch (restaurant vs bar) -> 0.0."""
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[_stop("p1", primary_type="Cocktail Bar")],
    )
    assert category_compliance(state) == 0.0


def test_category_compliance_partial_match_two_slots() -> None:
    """One of two slots matches: requested=[restaurant, bar], stops=[restaurant, cafe]."""
    state = ItineraryState(
        constraints=UserConstraints(
            requested_primary_types=["Sushi Restaurant", "Cocktail Bar"],
        ),
        stops=[
            _stop("p1", primary_type="Sushi Restaurant"),
            _stop("p2", primary_type="Coffee Shop"),  # cafe family, not bar
        ],
    )
    assert category_compliance(state) == 0.5


def test_category_compliance_length_mismatch_more_stops_than_requested() -> None:
    """More committed stops than named slots: extras are mismatches (denom = max).

    requested=[restaurant], stops=[restaurant, bar] -> 1 match / 2 = 0.5.
    Documented in scorer docstring: scoring the overlap and penalizing the gap
    prevents an agent from gaming the scorer by committing extra stops.
    """
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[
            _stop("p1", primary_type="Sushi Restaurant"),
            _stop("p2", primary_type="Cocktail Bar"),
        ],
    )
    assert category_compliance(state) == 0.5


def test_category_compliance_length_mismatch_fewer_stops_than_requested() -> None:
    """Fewer committed stops than named slots: missing slots count as mismatches.

    requested=[restaurant, bar], stops=[restaurant] -> 1 match / 2 = 0.5.
    """
    state = ItineraryState(
        constraints=UserConstraints(
            requested_primary_types=["Sushi Restaurant", "Cocktail Bar"],
        ),
        stops=[_stop("p1", primary_type="Sushi Restaurant")],
    )
    assert category_compliance(state) == 0.5


def test_category_compliance_none_primary_type_is_mismatch() -> None:
    """primary_type=None can't be scored as a match — score as mismatch (strict)."""
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[_stop("p1", primary_type=None)],
    )
    assert category_compliance(state) == 0.0


def test_category_compliance_registered_in_thresholds() -> None:
    """CRITIQUE_THRESHOLDS must include the new key so itinerary_violations
    knows the cutoff."""
    assert "category_compliance" in CRITIQUE_THRESHOLDS
    assert CRITIQUE_THRESHOLDS["category_compliance"] == 1.0


def test_category_compliance_pure_function_no_db_access(mocker) -> None:
    """Smoke test that the scorer does NOT touch the DB — if get_conn is
    called the test fails. Pure-state scorers must not regress into DB calls."""
    sentinel = mocker.patch("app.agent.critique.checks.get_conn")
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Cocktail Bar"]),
        stops=[_stop("p1", primary_type="Cocktail Bar")],
    )
    assert category_compliance(state) == 1.0
    sentinel.assert_not_called()


# --- category_compliance_strict (D-04-10) -----------------------------------
# Strict scorer catches within-family drift that family-level category_compliance
# deliberately allows. Pure-function scorer: no DB access.


def test_category_compliance_strict_abstains_when_no_requested_types() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=[]),
        stops=[_stop("p1", primary_type="Sushi Restaurant")],
    )
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_returns_one_for_empty_stops() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[],
    )
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_exact_keyword_match() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[_stop("p1", primary_type="Sushi Restaurant")],
    )
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_within_family_drift_returns_zero() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[_stop("p1", primary_type="Pizza Restaurant")],
    )
    family_state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[_stop("p1", primary_type="Pizza Restaurant")],
    )
    assert category_compliance(family_state) == 1.0
    assert category_compliance_strict(state) == 0.0


def test_category_compliance_strict_multi_slot_all_match() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase", "cocktails"]),
        stops=[
            _stop("p1", primary_type="Sushi Restaurant"),
            _stop("p2", primary_type="Cocktail Bar"),
        ],
    )
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_partial_match() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase", "cocktails"]),
        stops=[
            _stop("p1", primary_type="Sushi Restaurant"),
            _stop("p2", primary_type="Coffee Shop"),
        ],
    )
    assert category_compliance_strict(state) == 0.5


def test_category_compliance_strict_falls_back_to_family_on_unmapped_keyword() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Italian Restaurant"]),
        stops=[_stop("p1", primary_type="Restaurant")],
    )
    assert category_compliance(state) == 1.0
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_length_mismatch_dilutes() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[
            _stop("p1", primary_type="Sushi Restaurant"),
            _stop("p2", primary_type="Cocktail Bar"),
        ],
    )
    assert category_compliance_strict(state) == 0.5


def test_category_compliance_strict_none_primary_type_is_mismatch() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[_stop("p1", primary_type=None)],
    )
    assert category_compliance_strict(state) == 0.0


def test_category_compliance_strict_keyword_lookup_is_case_insensitive() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["OMAKASE"]),
        stops=[_stop("p1", primary_type="Sushi Restaurant")],
    )
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_primary_type_match_is_case_preserving() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[_stop("p1", primary_type="sushi restaurant")],
    )
    assert category_compliance_strict(state) == 0.0


def test_category_compliance_strict_registered_in_thresholds() -> None:
    assert "category_compliance_strict" in CRITIQUE_THRESHOLDS
    assert CRITIQUE_THRESHOLDS["category_compliance_strict"] == 1.0


def test_category_compliance_strict_itinerary_violations_registration(mocker) -> None:
    mocker.patch("app.agent.critique.checks.no_hallucinated_place_ids", return_value=1.0)
    mocker.patch("app.agent.critique.checks.stop_count_satisfied", return_value=1.0)
    mocker.patch("app.agent.critique.checks.category_compliance", return_value=1.0)
    mocker.patch("app.agent.critique.checks.category_compliance_strict", return_value=0.0)
    mocker.patch("app.agent.critique.checks.temporal_coherence", return_value=1.0)
    mocker.patch("app.agent.critique.checks.geographic_coherence", return_value=1.0)
    mocker.patch("app.agent.critique.checks.walking_budget_respected", return_value=1.0)
    mocker.patch("app.agent.critique.checks.constraints_satisfied", return_value=1.0)
    mocker.patch("app.agent.critique.checks.rationale_stop_alignment", return_value=1.0)

    assert itinerary_violations(ItineraryState(stops=[_stop("p1")])) == [
        "category_compliance_strict"
    ]


def test_category_compliance_strict_is_pure_function_no_db() -> None:
    import inspect

    source = inspect.getsource(category_compliance_strict)
    blocked = ("get_conn", "engine", "session", "execute", "connection")
    assert not any(token in source for token in blocked), [
        token for token in blocked if token in source
    ]


# --- rationale_stop_alignment (EVAL-02) -------------------------------------
# Catches rationale drift: refinement-turn bleed AND closure-swap placeholder
# bleed. Pure function. The closure-swap placeholder lives in
# app/agent/swap.py:238 (f"Walking-distance alternative for {closed_stop.name}");
# the regression test below pins the live text so renames are caught.


def test_rationale_stop_alignment_returns_one_for_empty_stops() -> None:
    """Fail-open: no committed stops -> 1.0."""
    assert rationale_stop_alignment(ItineraryState()) == 1.0


def test_rationale_stop_alignment_name_substring_match() -> None:
    """Stop name appearing in the rationale is enough — case-insensitive."""
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="Kaiseki Yuzu",
                rationale="Kaiseki Yuzu offers a tasting menu",
                primary_type="Sushi Restaurant",
            ),
        ],
    )
    assert rationale_stop_alignment(state) == 1.0


def test_rationale_stop_alignment_family_keyword_match() -> None:
    """No name match but family-keyword 'restaurant' present -> match."""
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="Lazy Bear",
                rationale="An intimate restaurant experience",
                primary_type="American Restaurant",
            ),
        ],
    )
    assert rationale_stop_alignment(state) == 1.0


def test_rationale_stop_alignment_catches_closure_swap_placeholder_bleed() -> None:
    """REGRESSION: closure-swap placeholder must score < 1.0.

    The exact placeholder string lives in app/agent/swap.py at line 238 as
    f"Walking-distance alternative for {closed_stop.name}". For a stop with
    name='Lazy Bear' / primary_type='American Restaurant', the placeholder
    rationale 'Walking-distance alternative for Kaiseki Yuzu' contains
    neither 'lazy bear' (the current stop's name) nor any restaurant-family
    keyword — so the scorer must return 0.0.

    If you renamed the placeholder template in swap.py and this test broke,
    update the literal below to match — the regression target is "rationale
    text that does not describe the current stop must fail the scorer".
    """
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="Lazy Bear",
                rationale="Walking-distance alternative for Kaiseki Yuzu",
                primary_type="American Restaurant",
            ),
        ],
    )
    assert rationale_stop_alignment(state) == 0.0


def test_rationale_stop_alignment_bar_family_keyword() -> None:
    """No name match but 'cocktail' is a bar-family keyword -> 1.0."""
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="Stookey's",
                rationale="Excellent cocktails in a vintage setting",
                primary_type="Cocktail Bar",
            ),
        ],
    )
    assert rationale_stop_alignment(state) == 1.0


def test_rationale_stop_alignment_multi_stop_fractional() -> None:
    """Score is matches / len(stops) across multiple stops."""
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="Kaiseki Yuzu",
                rationale="Kaiseki Yuzu offers a tasting menu",
                primary_type="Sushi Restaurant",
            ),
            _stop(
                "p2",
                name="Lazy Bear",
                rationale="Walking-distance alternative for Kaiseki Yuzu",
                primary_type="American Restaurant",
            ),
        ],
    )
    # 1 match / 2 stops = 0.5
    assert rationale_stop_alignment(state) == 0.5


def test_rationale_stop_alignment_none_primary_type_no_name_match() -> None:
    """primary_type=None and no name substring -> 0.0 for that stop.

    Without a family, we can't derive keywords; the only way to save the
    stop is the name substring path, which here is absent.
    """
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="Mystery Spot",
                rationale="A pleasant place with great vibes",
                primary_type=None,
            ),
        ],
    )
    assert rationale_stop_alignment(state) == 0.0


def test_rationale_stop_alignment_case_insensitive_name_match() -> None:
    """Name substring match is case-insensitive (lowercased on both sides)."""
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="KAISEKI YUZU",
                # uppercase name, lowercase rationale
                rationale="we recommend kaiseki yuzu for the tasting menu",
                primary_type="Sushi Restaurant",
            ),
        ],
    )
    assert rationale_stop_alignment(state) == 1.0


def test_rationale_stop_alignment_registered_in_thresholds() -> None:
    """CRITIQUE_THRESHOLDS must include the new key."""
    assert "rationale_stop_alignment" in CRITIQUE_THRESHOLDS
    assert CRITIQUE_THRESHOLDS["rationale_stop_alignment"] == 1.0


def test_rationale_stop_alignment_pure_function_no_db_access(mocker) -> None:
    """Smoke test that the scorer does NOT touch the DB."""
    sentinel = mocker.patch("app.agent.critique.checks.get_conn")
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="Kaiseki Yuzu",
                rationale="Kaiseki Yuzu omakase",
                primary_type="Sushi Restaurant",
            ),
        ],
    )
    assert rationale_stop_alignment(state) == 1.0
    sentinel.assert_not_called()


# --- is_rationale_aligned (public helper, plan 04-05) -----------------------
# The per-stop boolean rule used by rationale_stop_alignment was extracted into
# a public helper so the revision dispatcher (_first_misaligned_stop_index in
# app/agent/revision.py) can call it without duplicating the rule. Tests below
# pin (a) helper behavior on each branch and (b) byte-identical scorer behavior
# pre/post extraction so the DRY refactor can't regress the existing scorer.


def test_is_rationale_aligned_name_substring_match() -> None:
    """Name appearing in the rationale (case-insensitive) -> True."""
    stop = _stop(
        "p1",
        name="Lazy Bear",
        rationale="Lazy Bear is excellent",
        primary_type="American Restaurant",
    )
    assert is_rationale_aligned(stop) is True


def test_is_rationale_aligned_family_keyword_match() -> None:
    """No name match but a family keyword in rationale -> True."""
    stop = _stop(
        "p1",
        name="Lazy Bear",
        rationale="An intimate restaurant experience downtown",
        primary_type="American Restaurant",
    )
    assert is_rationale_aligned(stop) is True


def test_is_rationale_aligned_neither_match_returns_false() -> None:
    """Closure-swap placeholder bleed: no name AND no family keyword -> False."""
    stop = _stop(
        "p1",
        name="Lazy Bear",
        rationale="Walking-distance alternative for Kaiseki Yuzu",
        primary_type="American Restaurant",
    )
    assert is_rationale_aligned(stop) is False


def test_is_rationale_aligned_none_primary_type_no_name_match() -> None:
    """primary_type=None means we can't derive family keywords; name substring
    is the only path. No name match -> False."""
    stop = _stop(
        "p1",
        name="Mystery Spot",
        rationale="A pleasant place with great vibes",
        primary_type=None,
    )
    assert is_rationale_aligned(stop) is False


def test_is_rationale_aligned_none_or_empty_rationale_returns_false() -> None:
    """Defensive: empty rationale -> False (would crash on .lower() pre-extraction
    if name were also None; the helper coerces None defensively)."""
    # rationale="" branch
    stop_empty = _stop(
        "p1",
        name="Lazy Bear",
        rationale="",
        primary_type="American Restaurant",
    )
    assert is_rationale_aligned(stop_empty) is False


def test_rationale_stop_alignment_behavior_unchanged_after_extraction() -> None:
    """REGRESSION (ADVISORY 3): pin rationale_stop_alignment's output on a fixed
    fixture so the is_rationale_aligned extraction is byte-identical.

    Fixture covers every branch of the per-stop rule:
      - stop[0]: aligned by name substring (Sushi family)
      - stop[1]: aligned by family-keyword 'cocktail' (Bar family)
      - stop[2]: misaligned — closure-swap placeholder bleed (no name, no family kw)
      - stop[3]: no primary_type and no name match — misaligned

    Expected: 2 of 4 stops align -> 0.5. This is hand-computed against the
    pre-extraction scorer body so any drift in is_rationale_aligned will trip.
    """
    state = ItineraryState(
        stops=[
            _stop(
                "p1",
                name="Kaiseki Yuzu",
                rationale="Kaiseki Yuzu offers a tasting menu",
                primary_type="Sushi Restaurant",
            ),
            _stop(
                "p2",
                name="Stookey's",
                rationale="Excellent cocktails in a vintage setting",
                primary_type="Cocktail Bar",
            ),
            _stop(
                "p3",
                name="Lazy Bear",
                rationale="Walking-distance alternative for Kaiseki Yuzu",
                primary_type="American Restaurant",
            ),
            _stop(
                "p4",
                name="Mystery Spot",
                rationale="A pleasant place with great vibes",
                primary_type=None,
            ),
        ],
    )
    # Pin the exact pre-extraction value. matches=2 (p1 by name, p2 by 'cocktail'
    # in bar family); total=4 -> 0.5.
    assert rationale_stop_alignment(state) == 0.5


# --- itinerary_violations aggregation ---------------------------------------


def test_stop_count_satisfied_checks_explicit_request() -> None:
    assert stop_count_satisfied(ItineraryState(stops=[_stop("p1")])) == 1.0
    assert (
        stop_count_satisfied(
            ItineraryState(
                stops=[_stop("p1"), _stop("p2")],
                constraints=UserConstraints(num_stops=3),
            )
        )
        == 0.0
    )
    assert (
        stop_count_satisfied(
            ItineraryState(
                stops=[_stop("p1"), _stop("p2"), _stop("p3")],
                constraints=UserConstraints(num_stops=3),
            )
        )
        == 1.0
    )


def test_itinerary_violations_reports_failed_checks_in_order(mocker) -> None:
    """Order matters: hallucination first, then stop count, category compliance,
    strict category compliance, temporal, geo, walking, constraints, rationale.
    Mocks each check independently so we can assert the order."""
    mocker.patch("app.agent.critique.checks.no_hallucinated_place_ids", return_value=0.0)
    mocker.patch("app.agent.critique.checks.stop_count_satisfied", return_value=0.0)
    mocker.patch("app.agent.critique.checks.category_compliance", return_value=0.0)
    mocker.patch("app.agent.critique.checks.category_compliance_strict", return_value=0.0)
    mocker.patch("app.agent.critique.checks.temporal_coherence", return_value=0.0)
    mocker.patch("app.agent.critique.checks.geographic_coherence", return_value=0.5)
    mocker.patch("app.agent.critique.checks.walking_budget_respected", return_value=0.0)
    mocker.patch("app.agent.critique.checks.constraints_satisfied", return_value=0.5)
    mocker.patch("app.agent.critique.checks.rationale_stop_alignment", return_value=0.0)
    state = ItineraryState(stops=[_stop("p1")])
    assert itinerary_violations(state) == [
        "no_hallucinated_place_ids",
        "stop_count_satisfied",
        "category_compliance",
        "category_compliance_strict",
        "temporal_coherence",
        "geographic_coherence",
        "walking_budget_respected",
        "constraints_satisfied",
        "rationale_stop_alignment",
    ]


def test_itinerary_violations_empty_when_all_pass(mocker) -> None:
    for fn in (
        "no_hallucinated_place_ids",
        "stop_count_satisfied",
        "category_compliance",
        "category_compliance_strict",
        "temporal_coherence",
        "geographic_coherence",
        "walking_budget_respected",
        "constraints_satisfied",
        "rationale_stop_alignment",
    ):
        mocker.patch(f"app.agent.critique.checks.{fn}", return_value=1.0)
    assert itinerary_violations(ItineraryState(stops=[_stop("p1")])) == []


def test_itinerary_violations_fails_open_on_db_error(mocker) -> None:
    """If a check raises (e.g. DB unreachable), itinerary_violations skips it
    rather than treating it as a violation. The user gets a plan rather than
    a 500."""
    mocker.patch(
        "app.agent.critique.checks.no_hallucinated_place_ids",
        side_effect=RuntimeError("db down"),
    )
    mocker.patch("app.agent.critique.checks.stop_count_satisfied", return_value=1.0)
    mocker.patch("app.agent.critique.checks.category_compliance", return_value=1.0)
    mocker.patch("app.agent.critique.checks.category_compliance_strict", return_value=1.0)
    mocker.patch("app.agent.critique.checks.temporal_coherence", return_value=1.0)
    mocker.patch("app.agent.critique.checks.geographic_coherence", return_value=1.0)
    mocker.patch("app.agent.critique.checks.walking_budget_respected", return_value=1.0)
    mocker.patch("app.agent.critique.checks.constraints_satisfied", return_value=1.0)
    mocker.patch("app.agent.critique.checks.rationale_stop_alignment", return_value=1.0)
    assert itinerary_violations(ItineraryState(stops=[_stop("p1")])) == []


def test_thresholds_are_strict_enough() -> None:
    """A failing geographic check must trip the threshold even at 0.99 — the
    plan calls for zero tolerance on coherence."""
    assert CRITIQUE_THRESHOLDS["geographic_coherence"] == 1.0
    assert CRITIQUE_THRESHOLDS["temporal_coherence"] == 1.0
    assert CRITIQUE_THRESHOLDS["no_hallucinated_place_ids"] == 1.0
    assert CRITIQUE_THRESHOLDS["stop_count_satisfied"] == 1.0
    assert CRITIQUE_THRESHOLDS["walking_budget_respected"] == 1.0
    assert CRITIQUE_THRESHOLDS["category_compliance"] == 1.0
    assert CRITIQUE_THRESHOLDS["category_compliance_strict"] == 1.0
    assert CRITIQUE_THRESHOLDS["rationale_stop_alignment"] == 1.0
    # Constraint satisfaction has wiggle room — not every constraint is hard.
    assert CRITIQUE_THRESHOLDS["constraints_satisfied"] == 0.8
