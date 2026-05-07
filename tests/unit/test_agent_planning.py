from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.planning import (
    haversine_m,
    next_arrival_time,
    parse_stops_count,
    remaining_walking_budget_m,
    suggested_radius_m,
    walking_time_min,
)
from app.agent.state import ItineraryState, Stop, UserConstraints


def test_haversine_zero_distance() -> None:
    assert haversine_m((37.7, -122.4), (37.7, -122.4)) == pytest.approx(0.0, abs=1e-6)


def test_haversine_small_distance_within_sf() -> None:
    # Two known SF coordinates ~1 km apart (Mission Dolores to 16th St BART)
    d = haversine_m((37.7596, -122.4269), (37.7651, -122.4194))
    assert 700 < d < 1300


def test_walking_time_min_matches_speed_constant() -> None:
    # 800m at 80 m/min = 10 min.
    minutes = walking_time_min(37.7596, -122.4269, 37.7651, -122.4194)
    assert 7 < minutes < 16


def test_next_arrival_time_chains_correctly() -> None:
    prev = Stop(
        place_id="p1",
        name="X",
        rationale="r",
        source="google_places",
        latitude=37.7596,
        longitude=-122.4269,
        planned_duration_min=90,
        arrival_time=datetime(2026, 5, 6, 19, 0, tzinfo=timezone.utc),
    )
    arrival = next_arrival_time(prev, 37.7651, -122.4194)
    delta_min = (arrival - prev.arrival_time).total_seconds() / 60.0
    # 90 minutes of dinner + ~10 min of walking
    assert 95 < delta_min < 110


def test_next_arrival_time_requires_arrival_time() -> None:
    prev = Stop(
        place_id="p1",
        name="X",
        rationale="r",
        source="google_places",
        latitude=37.7,
        longitude=-122.4,
    )
    with pytest.raises(ValueError, match="arrival_time"):
        next_arrival_time(prev, 37.71, -122.41)


def test_next_arrival_time_requires_coordinates() -> None:
    prev = Stop(
        place_id="p1",
        name="X",
        rationale="r",
        source="google_places",
        arrival_time=datetime(2026, 5, 6, 19, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="coordinates"):
        next_arrival_time(prev, 37.71, -122.41)


def test_remaining_walking_budget_clamps_at_zero() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=1000), walked_meters_so_far=1500
    )
    assert remaining_walking_budget_m(state) == 0.0


def test_remaining_walking_budget_normal() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400), walked_meters_so_far=400
    )
    assert remaining_walking_budget_m(state) == 2000.0


def test_suggested_radius_zero_remaining_stops() -> None:
    state = ItineraryState()
    assert suggested_radius_m(state, remaining_stops=0) == 0


def test_suggested_radius_clamps_minimum() -> None:
    state = ItineraryState(walked_meters_so_far=2400)
    # Budget exhausted -> still returns at least the floor (300m).
    assert suggested_radius_m(state, remaining_stops=2) == 300


def test_suggested_radius_clamps_maximum() -> None:
    state = ItineraryState(constraints=UserConstraints(walking_budget_m=10_000))
    assert suggested_radius_m(state, remaining_stops=1) == 1500


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("just dinner", 1),
        ("only dinner please", 1),
        ("dinner and drinks", 2),
        ("dinner + drinks", 2),
        ("dinner then drinks", 2),
        ("3 spots", 3),
        ("plan 4 places", 4),
        ("you decide", 3),  # default
        ("a wonderful evening", 3),  # default
        ("47 stops", 3),  # bogus -> default
    ],
)
def test_parse_stops_count(text: str, expected: int) -> None:
    assert parse_stops_count(text) == expected


def test_parse_stops_count_custom_default() -> None:
    assert parse_stops_count("you decide", default=5) == 5
