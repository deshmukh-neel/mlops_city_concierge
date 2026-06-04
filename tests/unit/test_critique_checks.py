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
    refinement_minimal_edit,
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
    place_id: str = "ChIJtest_p1_aaaaaaaa",
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
    patch_db([[("ChIJtest_p1_aaaaaaaa",), ("ChIJtest_p2_aaaaaaaa",)]])
    state = ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa"), _stop("ChIJtest_p2_aaaaaaaa")])
    assert no_hallucinated_place_ids(state) == 1.0


def test_no_hallucinated_fails_when_any_missing(patch_db) -> None:
    """One missing place_id is critical — score drops to 0."""
    patch_db([[("ChIJtest_p1_aaaaaaaa",)]])
    state = ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa"), _stop("ChIJtest_p2_aaaaaaaa")])
    assert no_hallucinated_place_ids(state) == 0.0


def test_no_hallucinated_returns_one_for_empty_stops() -> None:
    """Vacuous truth — empty itinerary has no hallucinations."""
    assert no_hallucinated_place_ids(ItineraryState()) == 1.0


# --- temporal_coherence -----------------------------------------------------


def test_temporal_returns_one_when_no_arrival_times() -> None:
    state = ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa"), _stop("ChIJtest_p2_aaaaaaaa")])
    # No DB call needed — function short-circuits.
    assert temporal_coherence(state) == 1.0


def test_temporal_fractional_when_some_closed(patch_db) -> None:
    """One coalesced query now returns one row per matched stop."""
    arrival = datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc)
    patch_db(
        [
            [
                {"place_id": "ChIJtest_p1_aaaaaaaa", "is_open": True},
                {"place_id": "ChIJtest_p2_aaaaaaaa", "is_open": False},
            ]
        ]
    )
    state = ItineraryState(
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", arrival=arrival),
            _stop("ChIJtest_p2_aaaaaaaa", arrival=arrival + timedelta(hours=2)),
        ]
    )
    assert temporal_coherence(state) == 0.5


def test_temporal_treats_missing_row_as_open(patch_db) -> None:
    """A stop whose place_id has no places_raw row is absent from the JOIN
    result; the function defaults that stop to open."""
    arrival = datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc)
    patch_db([[]])  # JOIN returned zero rows
    state = ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa", arrival=arrival)])
    assert temporal_coherence(state) == 1.0


# --- geographic_coherence ---------------------------------------------------


def test_geographic_perfect_when_close() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", lat=37.78, lng=-122.41),
            _stop("ChIJtest_p2_aaaaaaaa", lat=37.781, lng=-122.411),
        ],
    )
    assert geographic_coherence(state) == 1.0


def test_geographic_fractional_when_one_leg_too_long() -> None:
    """Per-leg budget = 2400/2 = 1200m. Leg 1 is short, leg 2 is ~1.5km."""
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", lat=37.78, lng=-122.41),
            _stop("ChIJtest_p2_aaaaaaaa", lat=37.781, lng=-122.411),  # tiny hop
            _stop("ChIJtest_p3_aaaaaaaa", lat=37.794, lng=-122.41),  # ~1.5km from p2
        ],
    )
    assert geographic_coherence(state) == 0.5


def test_geographic_skips_pairs_missing_coords() -> None:
    """Pairs without coordinates aren't measurable — score on what we can."""
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", lat=37.78, lng=-122.41),
            _stop("ChIJtest_p2_aaaaaaaa"),  # no coords
            _stop("ChIJtest_p3_aaaaaaaa", lat=37.781, lng=-122.411),
        ],
    )
    # Neither leg measurable — both endpoints incomplete.
    assert geographic_coherence(state) == 1.0


def test_geographic_returns_one_for_single_stop() -> None:
    state = ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa", lat=37.78, lng=-122.41)])
    assert geographic_coherence(state) == 1.0


# --- walking_budget_respected -----------------------------------------------


def test_walking_budget_respected_when_under() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", lat=37.78, lng=-122.41),
            _stop("ChIJtest_p2_aaaaaaaa", lat=37.781, lng=-122.411),
        ],
    )
    assert walking_budget_respected(state) == 1.0


def test_walking_budget_violated_when_over() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=100),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", lat=37.78, lng=-122.41),
            _stop("ChIJtest_p2_aaaaaaaa", lat=37.80, lng=-122.41),  # ~2.2km
        ],
    )
    assert walking_budget_respected(state) == 0.0


# --- constraints_satisfied --------------------------------------------------


def test_constraints_satisfied_full_pass(patch_db) -> None:
    patch_db(
        [
            [
                {
                    "place_id": "ChIJtest_p1_aaaaaaaa",
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
        stops=[_stop("ChIJtest_p1_aaaaaaaa")],
    )
    assert constraints_satisfied(state) == 1.0


def test_constraints_satisfied_partial(patch_db) -> None:
    """Place is OK on rating + neighborhood but blows the price cap."""
    patch_db(
        [
            [
                {
                    "place_id": "ChIJtest_p1_aaaaaaaa",
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
        stops=[_stop("ChIJtest_p1_aaaaaaaa")],
    )
    # 3 of 4 expressed constraints satisfied = 0.75
    assert constraints_satisfied(state) == pytest.approx(0.75)


def test_constraints_satisfied_neighborhood_falls_back_to_address(patch_db) -> None:
    """When the structured neighborhood column is empty, address ILIKE wins."""
    patch_db(
        [
            [
                {
                    "place_id": "ChIJtest_p1_aaaaaaaa",
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
        stops=[_stop("ChIJtest_p1_aaaaaaaa")],
    )
    assert constraints_satisfied(state) == 1.0


def test_constraints_satisfied_skips_hallucinated_pids(patch_db) -> None:
    """Stops whose place_id isn't in the DB row dump aren't scored — that's
    no_hallucinated_place_ids' job."""
    patch_db([[]])  # no rows
    state = ItineraryState(
        constraints=UserConstraints(price_level_max=2),
        stops=[_stop("ChIJtest_p1_aaaaaaaa"), _stop("ChIJtest_p2_aaaaaaaa")],
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
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant")],
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
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant")],
    )
    assert category_compliance(state) == 1.0


def test_category_compliance_multi_slot_all_match() -> None:
    """Per-index family match across multiple slots -> 1.0."""
    state = ItineraryState(
        constraints=UserConstraints(
            requested_primary_types=["Sushi Restaurant", "Cocktail Bar"],
        ),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant"),
            _stop("ChIJtest_p2_aaaaaaaa", primary_type="Cocktail Bar"),
        ],
    )
    assert category_compliance(state) == 1.0


def test_category_compliance_family_mismatch_single_stop() -> None:
    """Family mismatch (restaurant vs bar) -> 0.0."""
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Cocktail Bar")],
    )
    assert category_compliance(state) == 0.0


def test_category_compliance_partial_match_two_slots() -> None:
    """One of two slots matches: requested=[restaurant, bar], stops=[restaurant, cafe]."""
    state = ItineraryState(
        constraints=UserConstraints(
            requested_primary_types=["Sushi Restaurant", "Cocktail Bar"],
        ),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant"),
            _stop("ChIJtest_p2_aaaaaaaa", primary_type="Coffee Shop"),  # cafe family, not bar
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
            _stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant"),
            _stop("ChIJtest_p2_aaaaaaaa", primary_type="Cocktail Bar"),
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
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant")],
    )
    assert category_compliance(state) == 0.5


def test_category_compliance_none_primary_type_is_mismatch() -> None:
    """primary_type=None can't be scored as a match — score as mismatch (strict)."""
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type=None)],
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
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Cocktail Bar")],
    )
    assert category_compliance(state) == 1.0
    sentinel.assert_not_called()


# --- category_compliance_strict (D-04-10) -----------------------------------
# Strict scorer catches within-family drift that family-level category_compliance
# deliberately allows. Pure-function scorer: no DB access.


def test_category_compliance_strict_abstains_when_no_requested_types() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=[]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant")],
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
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant")],
    )
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_within_family_drift_returns_zero() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Pizza Restaurant")],
    )
    family_state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Pizza Restaurant")],
    )
    assert category_compliance(family_state) == 1.0
    assert category_compliance_strict(state) == 0.0


def test_category_compliance_strict_multi_slot_all_match() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase", "cocktails"]),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant"),
            _stop("ChIJtest_p2_aaaaaaaa", primary_type="Cocktail Bar"),
        ],
    )
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_partial_match() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase", "cocktails"]),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant"),
            _stop("ChIJtest_p2_aaaaaaaa", primary_type="Coffee Shop"),
        ],
    )
    assert category_compliance_strict(state) == 0.5


def test_category_compliance_strict_falls_back_to_family_on_unmapped_keyword() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["Italian Restaurant"]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Restaurant")],
    )
    assert category_compliance(state) == 1.0
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_length_mismatch_dilutes() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[
            _stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant"),
            _stop("ChIJtest_p2_aaaaaaaa", primary_type="Cocktail Bar"),
        ],
    )
    assert category_compliance_strict(state) == 0.5


def test_category_compliance_strict_none_primary_type_is_mismatch() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type=None)],
    )
    assert category_compliance_strict(state) == 0.0


def test_category_compliance_strict_keyword_lookup_is_case_insensitive() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["OMAKASE"]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="Sushi Restaurant")],
    )
    assert category_compliance_strict(state) == 1.0


def test_category_compliance_strict_primary_type_match_is_case_preserving() -> None:
    state = ItineraryState(
        constraints=UserConstraints(requested_primary_types=["omakase"]),
        stops=[_stop("ChIJtest_p1_aaaaaaaa", primary_type="sushi restaurant")],
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

    assert itinerary_violations(ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa")])) == [
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
                "ChIJtest_p1_aaaaaaaa",
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
                "ChIJtest_p1_aaaaaaaa",
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
                "ChIJtest_p1_aaaaaaaa",
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
                "ChIJtest_p1_aaaaaaaa",
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
                "ChIJtest_p1_aaaaaaaa",
                name="Kaiseki Yuzu",
                rationale="Kaiseki Yuzu offers a tasting menu",
                primary_type="Sushi Restaurant",
            ),
            _stop(
                "ChIJtest_p2_aaaaaaaa",
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
                "ChIJtest_p1_aaaaaaaa",
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
                "ChIJtest_p1_aaaaaaaa",
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
                "ChIJtest_p1_aaaaaaaa",
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
        "ChIJtest_p1_aaaaaaaa",
        name="Lazy Bear",
        rationale="Lazy Bear is excellent",
        primary_type="American Restaurant",
    )
    assert is_rationale_aligned(stop) is True


def test_is_rationale_aligned_family_keyword_match() -> None:
    """No name match but a family keyword in rationale -> True."""
    stop = _stop(
        "ChIJtest_p1_aaaaaaaa",
        name="Lazy Bear",
        rationale="An intimate restaurant experience downtown",
        primary_type="American Restaurant",
    )
    assert is_rationale_aligned(stop) is True


def test_is_rationale_aligned_neither_match_returns_false() -> None:
    """Closure-swap placeholder bleed: no name AND no family keyword -> False."""
    stop = _stop(
        "ChIJtest_p1_aaaaaaaa",
        name="Lazy Bear",
        rationale="Walking-distance alternative for Kaiseki Yuzu",
        primary_type="American Restaurant",
    )
    assert is_rationale_aligned(stop) is False


def test_is_rationale_aligned_none_primary_type_no_name_match() -> None:
    """primary_type=None means we can't derive family keywords; name substring
    is the only path. No name match -> False."""
    stop = _stop(
        "ChIJtest_p1_aaaaaaaa",
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
        "ChIJtest_p1_aaaaaaaa",
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
                "ChIJtest_p1_aaaaaaaa",
                name="Kaiseki Yuzu",
                rationale="Kaiseki Yuzu offers a tasting menu",
                primary_type="Sushi Restaurant",
            ),
            _stop(
                "ChIJtest_p2_aaaaaaaa",
                name="Stookey's",
                rationale="Excellent cocktails in a vintage setting",
                primary_type="Cocktail Bar",
            ),
            _stop(
                "ChIJtest_p3_aaaaaaaa",
                name="Lazy Bear",
                rationale="Walking-distance alternative for Kaiseki Yuzu",
                primary_type="American Restaurant",
            ),
            _stop(
                "ChIJtest_p4_aaaaaaaa",
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
    assert stop_count_satisfied(ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa")])) == 1.0
    assert (
        stop_count_satisfied(
            ItineraryState(
                stops=[_stop("ChIJtest_p1_aaaaaaaa"), _stop("ChIJtest_p2_aaaaaaaa")],
                constraints=UserConstraints(num_stops=3),
            )
        )
        == 0.0
    )
    assert (
        stop_count_satisfied(
            ItineraryState(
                stops=[
                    _stop("ChIJtest_p1_aaaaaaaa"),
                    _stop("ChIJtest_p2_aaaaaaaa"),
                    _stop("ChIJtest_p3_aaaaaaaa"),
                ],
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
    state = ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa")])
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
    assert itinerary_violations(ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa")])) == []


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
    assert itinerary_violations(ItineraryState(stops=[_stop("ChIJtest_p1_aaaaaaaa")])) == []


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


# --- PROMPT-02 grep gate (Phase 7 / D-07-04) --------------------------------
# Source-file guard: the six canonical behavioral phrases that Phase 7 moved
# OUT of the prompt body and INTO the `refinement_minimal_edit` scorer must
# never reappear in `app/agent/prompts.py` or `app/agent/io.py`. This is a
# SOURCE FILE substring check (not a runtime import + module-state check),
# because some of the forbidden text lives in docstrings/comments adjacent to
# the relevant code constants — a runtime import would only see the post-load
# string constant, missing the commentary that the prompt rewrite was meant
# to remove. PATTERNS.md confirms: "asserts canonical behavioral phrases are
# absent from prompts.py + io.py".


def test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts() -> None:
    """PROMPT-02 / D-07-04: the six canonical behavioral phrases that moved
    from the prompt body into `refinement_minimal_edit` must NOT reappear in
    `app/agent/prompts.py` or `app/agent/io.py`.

    Fails CI if a future PR resurrects any of the phrases in the prompt body.
    The scorer is the single source of truth for these behavioral rules; the
    prompt body describes the refinement TASK only.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    prompts_path = repo_root / "app/agent/prompts.py"
    io_path = repo_root / "app/agent/io.py"

    combined_lower = (prompts_path.read_text() + "\n" + io_path.read_text()).lower()

    # The six D-07-04 canonical behavioral phrases (case-insensitive). Each
    # entry is matched as a literal substring against the lowercased combined
    # source. The `SAME primary_type` variant is covered by `.lower()`
    # normalization of the prior entry.
    forbidden = [
        "keep same stop count",
        "do not ask clarifying questions",
        "preserve `place_id` byte-for-byte",
        "byte-for-byte",
        "same primary_type".lower(),
        "SAME primary_type".lower(),
    ]
    found = [phrase for phrase in forbidden if phrase in combined_lower]
    assert not found, (
        f"PROMPT-02 violation: behavioral phrases found in prompt body: {found}. "
        "These rules must live in refinement_minimal_edit, not in the prompt."
    )


def test_phase7_known_d_07_10_preamble_exception() -> None:
    """D-07-10 KNOWN EXCEPTION to D-07-03 (preamble is task-only).

    Plan 07-07's live PROMPT-04 measurement showed `openai/gpt-4o-mini`
    regressing from `refinement_minimal_edit.median == 1.0` (pre-Phase-7)
    to `0.0` under the pure task-only preamble. After the user-approved
    D-07-10 iteration, one behavioral-prescription sentence was added back
    to `_REFINEMENT_PREAMBLE`. It was deliberately phrased to avoid the
    six D-07-04 forbidden literal substrings ("byte-for-byte", etc.) but
    IS semantically equivalent — it tells the model to reuse `place_id` /
    `slot` on non-target stops, which is the byte-equality rule the scorer
    now enforces.

    The grep-gate test above uses literal-substring matching and therefore
    does NOT catch the paraphrase. This test PINS the exact sentence so:

      1. Any drift in the prescription wording (rewording, deletion,
         re-introduction of forbidden phrases) fails CI loudly with a
         pointer to 07-07-SUMMARY.md's accept-with-notes resolution.
      2. Future contributors reading `io.py` + the green grep gate cannot
         mistake the prescription for an oversight.

    Resolved by Phase 6 D-06-09 part 2 precedent (accept-with-notes when
    scorer tightening + prompt rewrite jointly move a baseline under a
    deliberate contract change). See 07-07-SUMMARY.md for the full
    measurement + decision record.
    """
    from app.agent.io import _REFINEMENT_PREAMBLE

    expected_d_07_10_exception_sentence = (
        "Reuse the `place_id` and `slot` index of every stop you are not changing "
        "exactly as listed; only the slot named by the user gets a new `place_id`."
    )
    assert expected_d_07_10_exception_sentence in _REFINEMENT_PREAMBLE, (
        "Phase 7 / D-07-10 KNOWN-EXCEPTION sentence drifted. "
        "If you intentionally removed or rewrote the prescription, update this "
        "test AND 07-07-SUMMARY.md's PROMPT-04 outcome record together. "
        "If you re-introduced a D-07-04 forbidden phrase, the grep gate above "
        "will also fail."
    )


# --- refinement_minimal_edit smoke (Task 1 driver; full class in Task 2) -----


def test_refinement_minimal_edit_smoke_threshold_registered() -> None:
    """Task 1 RED: CRITIQUE_THRESHOLDS must include the new strict scorer key."""
    assert "refinement_minimal_edit" in CRITIQUE_THRESHOLDS
    assert CRITIQUE_THRESHOLDS["refinement_minimal_edit"] == 1.0


def test_refinement_minimal_edit_smoke_callable_returns_float() -> None:
    """Task 1 RED: the scorer must be importable, callable on an empty state,
    and return a float in [0.0, 1.0]. Empty state has no refinement_context →
    Branch 1 abstain → 1.0."""
    score = refinement_minimal_edit(ItineraryState())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == 1.0


# --- refinement_minimal_edit full coverage (Task 2) -------------------------
# Per 06-REVIEWS.md § Pass 2 N-3, every branch in the five-branch precedence
# has its own dedicated test method so a regression localizes to one branch
# in CI output. Test fixtures use >= 20-char place_ids per the HIGH-4
# residual fix from plan 06-01 Task 3 (defensive against future regex
# validation on Stop.place_id).


_PID_LEN20_A = "ChIJtest_fixture_a_aaaaa"  # 24 chars, satisfies ^[A-Za-z0-9_-]{20,255}$
_PID_LEN20_B = "ChIJtest_fixture_b_bbbbb"
_PID_LEN20_C = "ChIJtest_fixture_c_ccccc"
_PID_LEN20_D = "ChIJtest_fixture_d_ddddd"
_PID_LEN20_E = "ChIJtest_fixture_e_eeeee"
_PID_LEN20_NEW2 = "ChIJtest_new_slot2_xxxxx"
_PID_LEN20_NEW4 = "ChIJtest_new_slot4_yyyyy"
_PID_LEN20_LONE = "ChIJlone_valid_id_aaaa"


def _refinement_stop(place_id: str, primary_type: str | None = None) -> Stop:
    """Build a Stop fixture suitable for the refinement_minimal_edit scorer.

    Only ``place_id`` matters for the scorer's byte-equal comparison. Required
    fields (``name``, ``source``, ``rationale``) get minimal values; the scorer
    never touches them.

    Phase 7 / D-07-05 / D-07-07: ``primary_type`` is now also relevant to the
    scorer's Branch-5 same-category sub-check on the TARGET slot. The kwarg is
    optional with a default of ``None`` so every pre-Phase-7 call site stays
    backward-compatible (a Stop with ``primary_type=None`` triggers the
    D-07-07 fail-loud / abstain branches in Branch 5 depending on the prior).
    """
    return Stop(
        place_id=place_id,
        name=place_id[-5:].upper(),
        source="google_places",
        rationale="",
        primary_type=primary_type,
    )


class TestRefinementMinimalEdit:
    """Coverage for refinement_minimal_edit (D-06-08, D-06-09).

    Every branch of the N-3 five-branch precedence has its own test method.
    HIGH-2 regression guards live alongside the branch-5 happy-path test
    (drop and insert variants). HIGH-1 cross-table registration check lives
    in tests/unit/test_eval_agent.py:TestDeterministicChecksRegistration.

    DB-pool-contamination guard (per project_full_suite_db_pool_contamination.md):
    every test patches `app.agent.critique.checks.itinerary_violations`'s
    DB-touching scorers so the full suite (`make test`) cannot leak a live
    DB pool from this class.
    """

    # --- Branch 1: abstain (refinement_context absent or False) -------------

    def test_branch_1_no_refinement_context_returns_1_0_abstain(self) -> None:
        """Empty scratch → no refinement_context key → Branch 1 abstain → 1.0."""
        state = ItineraryState()
        assert refinement_minimal_edit(state) == 1.0

    def test_branch_1_refinement_context_explicit_false_returns_1_0_abstain(
        self,
    ) -> None:
        """Explicit False is equivalent to absent for Branch 1 abstain."""
        state = ItineraryState()
        state.scratch = {
            "refinement_context": False,
            "prior_committed_stops": [{"slot": 1, "place_id": _PID_LEN20_A}],
            "refinement_target_slot": 2,
        }
        assert refinement_minimal_edit(state) == 1.0

    # --- Branch 2: refinement context but prior missing/empty → fail-loud ---

    def test_branch_2_refinement_context_true_empty_prior_returns_0_0_fail_loud(
        self,
    ) -> None:
        """N-2 fix regression guard: turn 0 was supposed to commit but didn't.

        The runner sets refinement_context=True regardless of turn-0 commit
        outcome, so the scorer must signal the failure as 0.0 rather than
        silently abstain at 1.0.
        """
        state = ItineraryState()
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [],
        }
        assert refinement_minimal_edit(state) == 0.0

    def test_branch_2_refinement_context_true_missing_target_slot_returns_0_0(
        self,
    ) -> None:
        """target_slot is required when refinement_context is True."""
        state = ItineraryState()
        state.scratch = {
            "refinement_context": True,
            "prior_committed_stops": [{"slot": 1, "place_id": _PID_LEN20_A}],
        }
        assert refinement_minimal_edit(state) == 0.0

    def test_branch_2_refinement_context_true_none_prior_returns_0_0(self) -> None:
        """prior_committed_stops explicitly None → Branch 2 fail-loud."""
        state = ItineraryState()
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": None,
        }
        assert refinement_minimal_edit(state) == 0.0

    # --- Branch 3: refinement context + malformed prior → fail-loud ---------

    def test_branch_3_refinement_context_true_all_malformed_prior_returns_0_0_fail_loud(
        self,
    ) -> None:
        """Every entry malformed (missing slot/place_id or empty place_id)
        collapses prior_by_slot to {} → Branch 3 fail-loud → 0.0.

        Eval-runner contract violation: surface it, don't hide it.
        """
        state = ItineraryState()
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1},  # missing place_id
                {"place_id": _PID_LEN20_A},  # missing slot
                {},  # empty
                {"slot": 2, "place_id": ""},  # empty place_id
            ],
        }
        assert refinement_minimal_edit(state) == 0.0

    # --- Branch 4: legitimate zero-denom (lone-stop-target) -----------------

    def test_branch_4_single_stop_target_lone_stop_returns_1_0(self) -> None:
        """When the lone prior stop IS the target, prior_non_target_slots is
        empty → Branch 4 legitimate zero-denom → 1.0 (no ZeroDivisionError).

        Nothing to preserve — the user asked to change the only stop.
        """
        state = ItineraryState(
            stops=[_refinement_stop(_PID_LEN20_NEW2)],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 1,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_LONE},
            ],
        }
        assert refinement_minimal_edit(state) == 1.0

    # --- Branch 5: normal — matches / len(prior_non_target_slots) ----------

    def test_branch_5_all_non_target_stops_preserved_returns_1_0(self) -> None:
        """Happy path: prior 3 slots, target=2, slots 1+3 byte-equal in
        current → 2/2 → 1.0."""
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),  # slot 1 — preserved
                _refinement_stop(_PID_LEN20_NEW2),  # slot 2 — changed (target)
                _refinement_stop(_PID_LEN20_C),  # slot 3 — preserved
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 1.0

    def test_branch_5_dropped_non_target_slot_scores_below_1_0(self) -> None:
        """HIGH-2 regression guard: denominator iterates PRIOR non-target
        slots, NOT current non-target slots.

        Prior 3 stops (slots 1, 2, 3), target_slot=2. Current has only 2
        stops: slot 1 preserved byte-equal, slot 2 changed; slot 3 is
        DROPPED ENTIRELY (not present in current at all).

        Pre-HIGH-2 bug: denom=current_non_target=1 → 1/1=1.0 (silent pass).
        Post-fix: denom=prior_non_target=2 → 1/2=0.5 (correctly fails).
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),  # slot 1 preserved
                _refinement_stop(_PID_LEN20_NEW2),  # slot 2 changed (target)
                # slot 3 entirely dropped from current
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 0.5

    def test_branch_5_inserted_non_target_slot_does_not_help_score(self) -> None:
        """HIGH-2 regression guard: insertions are neutral.

        Prior 3 slots (1, 2, 3), target=2. Current has 4 stops: slots 1+3
        preserved byte-equal, slot 2 changed, NEW slot 4 added. Denominator
        still iterates PRIOR slots only → 2/2 → 1.0. An inserted slot
        cannot pad the denominator to hide a drop.
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),  # slot 1 preserved
                _refinement_stop(_PID_LEN20_NEW2),  # slot 2 changed (target)
                _refinement_stop(_PID_LEN20_C),  # slot 3 preserved
                _refinement_stop(_PID_LEN20_NEW4),  # slot 4 NEW (inserted)
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 1.0

    def test_branch_5_one_of_two_non_target_stops_changed_returns_0_5(self) -> None:
        """Branch 5 partial: prior 3 slots, target=2, slot 1 CHANGED (not
        preserved), slot 3 preserved → 1/2 → 0.5."""
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_D),  # slot 1 changed (not preserved)
                _refinement_stop(_PID_LEN20_NEW2),  # slot 2 changed (target)
                _refinement_stop(_PID_LEN20_C),  # slot 3 preserved
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 0.5

    def test_branch_5_all_non_target_stops_changed_returns_0_0(self) -> None:
        """Branch 5 worst case: prior 3 slots, target=2, both slots 1 and 3
        CHANGED → 0/2 → 0.0 (every non-target stop lost)."""
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_D),  # slot 1 changed
                _refinement_stop(_PID_LEN20_NEW2),  # slot 2 changed (target)
                _refinement_stop(_PID_LEN20_E),  # slot 3 changed
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 0.0

    # --- Branch 5: D-07-07 target-slot primary_type sub-check (Phase 7) -----
    # Four-cell matrix (prior x current):
    #   | prior pt        | current pt   | scorer returns        |
    #   |-----------------|--------------|-----------------------|
    #   | None / missing  | (any)        | byte_fraction unchanged (abstain) |
    #   | present         | None         | 0.0 (fail-loud)       |
    #   | present, "X"    | present, "Y" | 0.0 (mismatch)        |
    #   | present, "X"    | present, "X" | byte_fraction unchanged (match) |
    # Plus two edge guards:
    #   - partial byte_fraction (0.5) with category match → 0.5 (multiplication
    #     not override).
    #   - Branch 4 lone-stop short-circuit → 1.0 regardless of category.

    def test_branch_5_target_primary_type_matches_returns_byte_fraction(self) -> None:
        """D-07-07 match cell: prior+current target both 'Cocktail Bar';
        non-target byte-fraction = 1.0; expect 1.0 (category check returns
        byte_fraction unchanged).
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),  # slot 1 preserved
                _refinement_stop(_PID_LEN20_NEW2, primary_type="Cocktail Bar"),
                _refinement_stop(_PID_LEN20_C),  # slot 3 preserved
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B, "primary_type": "Cocktail Bar"},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 1.0

    def test_branch_5_target_primary_type_mismatch_returns_0_0(self) -> None:
        """D-07-07 mismatch cell: prior target 'Cocktail Bar', current target
        'Wine Bar'; byte-fraction would be 1.0 but category mismatch zeros
        the score per D-07-05 binary merge-gate semantic.
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),
                _refinement_stop(_PID_LEN20_NEW2, primary_type="Wine Bar"),
                _refinement_stop(_PID_LEN20_C),
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B, "primary_type": "Cocktail Bar"},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 0.0

    def test_branch_5_prior_primary_type_missing_abstains_on_category(self) -> None:
        """D-07-07 abstain cell (missing-key variant): prior entry dict OMITS
        the `primary_type` key entirely; current target has primary_type='Cafe';
        scorer abstains on the category check and returns byte_fraction (1.0).

        This is the migration path for legacy 06-06 scratch payloads.
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),
                _refinement_stop(_PID_LEN20_NEW2, primary_type="Cafe"),
                _refinement_stop(_PID_LEN20_C),
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                # Target slot 2 entry omits primary_type entirely (legacy payload).
                {"slot": 2, "place_id": _PID_LEN20_B},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 1.0

    def test_branch_5_prior_primary_type_none_abstains_on_category(self) -> None:
        """D-07-07 abstain cell (explicit-None variant): prior entry has
        ``"primary_type": None`` explicitly; current target has any value;
        scorer abstains on the category check and returns byte_fraction (1.0).

        Pins BOTH the missing-key path AND the explicit-None path to the
        D-07-07 abstain branch.
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),
                _refinement_stop(_PID_LEN20_NEW2, primary_type="Cafe"),
                _refinement_stop(_PID_LEN20_C),
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B, "primary_type": None},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 1.0

    def test_branch_5_current_primary_type_none_fails_loud(self) -> None:
        """D-07-07 fail-loud cell: prior carries a real value ('Cocktail Bar')
        but current target's primary_type is None — the commit dropped a real
        field, so the scorer returns 0.0.
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),
                _refinement_stop(_PID_LEN20_NEW2, primary_type=None),
                _refinement_stop(_PID_LEN20_C),
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B, "primary_type": "Cocktail Bar"},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 0.0

    def test_branch_5_target_primary_type_match_with_partial_byte_fraction(
        self,
    ) -> None:
        """D-07-07 match cell with partial byte_fraction (0.5): prior 3 stops,
        target_slot=2, current dropped slot 3 → byte_fraction = 0.5, prior+
        current target both 'Bar' → expect 0.5.

        Proves the category check MULTIPLIES the byte_fraction (rather than
        overriding to 1.0) on a match — the scorer's byte-equality signal must
        survive even when the category sub-check passes.
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),  # slot 1 preserved
                _refinement_stop(_PID_LEN20_NEW2, primary_type="Bar"),  # target
                # slot 3 dropped entirely from current
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B, "primary_type": "Bar"},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 0.5

    def test_branch_4_lone_stop_target_skips_category_check(self) -> None:
        """Branch 4 short-circuit: single-stop prior, that stop IS the target,
        ``prior_non_target_slots`` is empty → Branch 4 returns 1.0 BEFORE the
        D-07-07 category sub-check fires.

        Pins the PATTERNS.md "Preserve abstain semantics on Branch 4" rule:
        the lone-stop case is a degenerate refinement shape and treating it
        as a category abstain keeps the scorer's no-data semantics consistent.
        """
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_NEW2, primary_type="Restaurant"),
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 1,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_LONE, "primary_type": "Cocktail Bar"},
            ],
        }
        assert refinement_minimal_edit(state) == 1.0

    # --- Registration + isolation guards ------------------------------------

    def test_registered_in_thresholds(self) -> None:
        """CRITIQUE_THRESHOLDS must carry the strict 1.0 threshold (D-06-09)."""
        assert "refinement_minimal_edit" in CRITIQUE_THRESHOLDS
        assert CRITIQUE_THRESHOLDS["refinement_minimal_edit"] == 1.0

    def test_registered_in_itinerary_violations_when_refinement_context_present(
        self, mocker
    ) -> None:
        """Mid-refinement: a scorer that returns < 1.0 (Branch 5 partial)
        must surface in `itinerary_violations(state)`.

        Mocks the DB-touching scorers so the full suite never sees a live
        connection from this class.
        """
        # DB-pool contamination guard: stub every scorer that touches the DB.
        mocker.patch("app.agent.critique.checks.no_hallucinated_place_ids", return_value=1.0)
        mocker.patch("app.agent.critique.checks.temporal_coherence", return_value=1.0)
        mocker.patch("app.agent.critique.checks.constraints_satisfied", return_value=1.0)

        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_D),  # slot 1 changed (drops score)
                _refinement_stop(_PID_LEN20_NEW2),  # slot 2 target
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B},
            ],
        }
        violations = itinerary_violations(state)
        assert "refinement_minimal_edit" in violations

    def test_not_in_itinerary_violations_when_no_refinement_context(self, mocker) -> None:
        """Ad-hoc revision-loop invocation: no refinement_context in scratch.

        Branch 1 abstain must return 1.0 so itinerary_violations does NOT
        produce a spurious refinement violation on every revision turn.
        Without this guard, the new scorer would break the revision loop's
        existing fail-open behavior.
        """
        # DB-pool contamination guard.
        mocker.patch("app.agent.critique.checks.no_hallucinated_place_ids", return_value=1.0)
        mocker.patch("app.agent.critique.checks.temporal_coherence", return_value=1.0)
        mocker.patch("app.agent.critique.checks.constraints_satisfied", return_value=1.0)

        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),
                _refinement_stop(_PID_LEN20_B),
            ],
        )
        # No refinement_context in scratch — this is the ad-hoc case.
        violations = itinerary_violations(state)
        assert "refinement_minimal_edit" not in violations

    def test_pure_function_no_db_access(self, mocker) -> None:
        """Pure-state scorer guard: `get_conn` must NEVER be called.

        Mirrors test_category_compliance_pure_function_no_db_access. If the
        scorer ever regresses into a DB call, the merge gate could be broken
        by a DB outage (which would silently fail-open via _try in
        itinerary_violations). This test ensures that path stays closed.
        """
        sentinel = mocker.patch("app.agent.critique.checks.get_conn")
        # Use a real refinement-context state (Branch 5 normal path).
        state = ItineraryState(
            stops=[
                _refinement_stop(_PID_LEN20_A),
                _refinement_stop(_PID_LEN20_NEW2),
                _refinement_stop(_PID_LEN20_C),
            ],
        )
        state.scratch = {
            "refinement_context": True,
            "refinement_target_slot": 2,
            "prior_committed_stops": [
                {"slot": 1, "place_id": _PID_LEN20_A},
                {"slot": 2, "place_id": _PID_LEN20_B},
                {"slot": 3, "place_id": _PID_LEN20_C},
            ],
        }
        assert refinement_minimal_edit(state) == 1.0
        sentinel.assert_not_called()
