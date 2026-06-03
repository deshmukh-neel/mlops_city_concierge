from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.agent.planning import (
    chain_arrival_times,
    haversine_m,
    next_arrival_time,
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
        place_id="ChIJtest_p1_aaaaaaaa",
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
        place_id="ChIJtest_p1_aaaaaaaa",
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
        place_id="ChIJtest_p1_aaaaaaaa",
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


def _stop_at(pid: str, *, arrival=None, duration=60):
    return Stop(
        place_id=pid,
        name=pid.upper(),
        source="google_places",
        rationale="",
        arrival_time=arrival,
        planned_duration_min=duration,
    )


def test_chain_arrival_times_empty_is_noop() -> None:
    assert chain_arrival_times([], []) == []


def test_chain_arrival_times_single_stop_unchanged() -> None:
    start = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = [_stop_at("ChIJtest_p1_aaaaaaaa", arrival=start)]
    out = chain_arrival_times(stops, [])
    assert out[0].arrival_time == start


def test_chain_arrival_times_chains_with_real_legs() -> None:
    start = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = [
        _stop_at("ChIJtest_p1_aaaaaaaa", arrival=start, duration=90),
        _stop_at("ChIJtest_p2_aaaaaaaa", duration=60),
        _stop_at("ChIJtest_p3_aaaaaaaa", duration=60),
    ]
    # Two legs: 10 min then 25 min of travel.
    out = chain_arrival_times(stops, [10.0, 25.0])
    assert out[0].arrival_time == start  # start preserved
    assert out[1].arrival_time == start + timedelta(minutes=90 + 10)
    assert out[2].arrival_time == start + timedelta(minutes=90 + 10 + 60 + 25)


def test_chain_arrival_times_requires_start_arrival() -> None:
    stops = [_stop_at("ChIJtest_p1_aaaaaaaa", arrival=None), _stop_at("ChIJtest_p2_aaaaaaaa")]
    with pytest.raises(ValueError, match="arrival_time"):
        chain_arrival_times(stops, [10.0])


def test_chain_arrival_times_does_not_mutate_input() -> None:
    start = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = [
        _stop_at("ChIJtest_p1_aaaaaaaa", arrival=start, duration=30),
        _stop_at("ChIJtest_p2_aaaaaaaa"),
    ]
    chain_arrival_times(stops, [5.0])
    assert stops[1].arrival_time is None  # original untouched
