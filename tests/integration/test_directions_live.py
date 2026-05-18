"""Live Google Routes API v2 check for the W8c Directions tool.

Gated: skipped unless APP_ENV=integration AND GOOGLE_DIRECTIONS_API_KEY is
set, mirroring tests/integration/test_db.py:19-21. Run with:

    APP_ENV=integration GOOGLE_DIRECTIONS_API_KEY=... make test-integration
"""

from __future__ import annotations

import os

import pytest

from app.tools.directions import route_legs

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration" or not os.getenv("GOOGLE_DIRECTIONS_API_KEY"),
    reason=("Set APP_ENV=integration and GOOGLE_DIRECTIONS_API_KEY to run live Directions tests."),
)

# Mission Dolores -> 16th St BART, ~1 km, definitely walkable.
_A = (37.7596, -122.4269)
_B = (37.7651, -122.4194)


async def test_route_legs_live_walk() -> None:
    result = await route_legs([_A, _B], mode="walk")
    assert result.source == "google"
    assert result.mode == "walk"
    assert len(result.legs) == 1
    assert 0 < result.total_duration_s < 7200  # <2h for a 1km walk
