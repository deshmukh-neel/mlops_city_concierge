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
    constraints_satisfied,
    geographic_coherence,
    itinerary_violations,
    no_hallucinated_place_ids,
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
) -> Stop:
    return Stop(
        place_id=place_id,
        name=place_id.upper(),
        source="google_places",
        rationale="",
        arrival_time=arrival,
        latitude=lat,
        longitude=lng,
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


# --- itinerary_violations aggregation ---------------------------------------


def test_itinerary_violations_reports_failed_checks_in_order(mocker) -> None:
    """Order matters: hallucination first, then temporal, geo, walking, then
    constraints. Mocks each check independently so we can assert the order."""
    mocker.patch("app.agent.critique.checks.no_hallucinated_place_ids", return_value=0.0)
    mocker.patch("app.agent.critique.checks.temporal_coherence", return_value=0.0)
    mocker.patch("app.agent.critique.checks.geographic_coherence", return_value=0.5)
    mocker.patch("app.agent.critique.checks.walking_budget_respected", return_value=0.0)
    mocker.patch("app.agent.critique.checks.constraints_satisfied", return_value=0.5)
    state = ItineraryState(stops=[_stop("p1")])
    assert itinerary_violations(state) == [
        "no_hallucinated_place_ids",
        "temporal_coherence",
        "geographic_coherence",
        "walking_budget_respected",
        "constraints_satisfied",
    ]


def test_itinerary_violations_empty_when_all_pass(mocker) -> None:
    for fn in (
        "no_hallucinated_place_ids",
        "temporal_coherence",
        "geographic_coherence",
        "walking_budget_respected",
        "constraints_satisfied",
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
    mocker.patch("app.agent.critique.checks.temporal_coherence", return_value=1.0)
    mocker.patch("app.agent.critique.checks.geographic_coherence", return_value=1.0)
    mocker.patch("app.agent.critique.checks.walking_budget_respected", return_value=1.0)
    mocker.patch("app.agent.critique.checks.constraints_satisfied", return_value=1.0)
    assert itinerary_violations(ItineraryState(stops=[_stop("p1")])) == []


def test_thresholds_are_strict_enough() -> None:
    """A failing geographic check must trip the threshold even at 0.99 — the
    plan calls for zero tolerance on coherence."""
    assert CRITIQUE_THRESHOLDS["geographic_coherence"] == 1.0
    assert CRITIQUE_THRESHOLDS["temporal_coherence"] == 1.0
    assert CRITIQUE_THRESHOLDS["no_hallucinated_place_ids"] == 1.0
    assert CRITIQUE_THRESHOLDS["walking_budget_respected"] == 1.0
    # Constraint satisfaction has wiggle room — not every constraint is hard.
    assert CRITIQUE_THRESHOLDS["constraints_satisfied"] == 0.8
