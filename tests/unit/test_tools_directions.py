from __future__ import annotations

from app.tools.directions import (
    DEFAULT_MODE,
    DirectionsLeg,
    DirectionsResult,
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
