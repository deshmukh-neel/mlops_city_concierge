from __future__ import annotations

import math

import pytest

from app.agent.planning import WALKING_SPEED_M_PER_MIN, haversine_m
from app.tools.directions import (
    DEFAULT_MODE,
    DirectionsLeg,
    DirectionsResult,
    route_legs,
)


def test_models_construct() -> None:
    leg = DirectionsLeg(duration_s=300, distance_m=420.0)
    result = DirectionsResult(
        legs=[leg],
        total_duration_s=300,
        mode="walk",
        source="google",
    )
    assert result.legs[0].duration_s == 300
    assert result.total_duration_s == 300
    assert result.source == "google"
    assert DEFAULT_MODE == "walk"


# Two known SF coords ~1 km apart (Mission Dolores → 16th St BART)
_A = (37.7596, -122.4269)
_B = (37.7651, -122.4194)


async def test_unknown_mode_raises_before_network() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        await route_legs([_A, _B], mode="teleport")


async def test_missing_key_uses_fallback() -> None:
    # conftest does not set GOOGLE_DIRECTIONS_API_KEY -> '' -> fallback,
    # and no HTTP is attempted.
    result = await route_legs([_A, _B], mode="walk")
    assert result.source == "haversine_fallback"
    assert result.mode == "walk"
    assert len(result.legs) == 1


async def test_fallback_durations_match_haversine() -> None:
    result = await route_legs([_A, _B], mode="walk")
    expected_s = round(haversine_m(_A, _B) / WALKING_SPEED_M_PER_MIN * 60)
    assert result.legs[0].duration_s == expected_s
    assert result.total_duration_s == expected_s
    assert math.isclose(result.legs[0].distance_m, haversine_m(_A, _B), rel_tol=1e-9)


async def test_fallback_handles_three_stops() -> None:
    result = await route_legs([_A, _B, _A], mode="walk")
    assert result.source == "haversine_fallback"
    assert len(result.legs) == 2  # N-1 legs for N stops


async def test_single_stop_returns_empty_legs() -> None:
    """The <2-stops early return is a distinct path: empty legs, zero total,
    fallback source — not a haversine computation."""
    result = await route_legs([_A], mode="walk")
    assert result.legs == []
    assert result.total_duration_s == 0
    assert result.source == "haversine_fallback"
    assert result.mode == "walk"
