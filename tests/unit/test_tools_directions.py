from __future__ import annotations

import math

import httpx
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
A = (37.7596, -122.4269)
B = (37.7651, -122.4194)


async def test_unknown_mode_raises_before_network() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        await route_legs([A, B], mode="teleport")


async def test_missing_key_uses_fallback() -> None:
    # conftest does not set GOOGLE_DIRECTIONS_API_KEY -> '' -> fallback,
    # and no HTTP is attempted.
    result = await route_legs([A, B], mode="walk")
    assert result.source == "haversine_fallback"
    assert result.mode == "walk"
    assert len(result.legs) == 1


async def test_fallback_durations_match_haversine() -> None:
    result = await route_legs([A, B], mode="walk")
    expected_s = round(haversine_m(A, B) / WALKING_SPEED_M_PER_MIN * 60)
    assert result.legs[0].duration_s == expected_s
    assert result.total_duration_s == expected_s
    assert math.isclose(result.legs[0].distance_m, haversine_m(A, B), rel_tol=1e-9)


async def test_fallback_handles_three_stops() -> None:
    result = await route_legs([A, B, A], mode="walk")
    assert result.source == "haversine_fallback"
    assert len(result.legs) == 2  # N-1 legs for N stops


async def test_single_stop_returns_empty_legs() -> None:
    """The <2-stops early return is a distinct path: empty legs, zero total,
    fallback source — not a haversine computation."""
    result = await route_legs([A], mode="walk")
    assert result.legs == []
    assert result.total_duration_s == 0
    assert result.source == "haversine_fallback"
    assert result.mode == "walk"


def key(monkeypatch) -> None:
    """Set a non-empty key so route_legs takes the network path."""
    monkeypatch.setenv("GOOGLE_DIRECTIONS_API_KEY", "test-directions-key")
    from app.config import get_settings

    get_settings.cache_clear()


def ok_response(legs: list[dict]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"routes": [{"legs": legs}]},
        request=httpx.Request("POST", "https://x"),
    )


async def test_route_legs_walk_parses_legs(monkeypatch, mocker) -> None:
    key(monkeypatch)
    post = mocker.AsyncMock(
        return_value=ok_response([{"duration": "780s", "distanceMeters": 1040}])
    )
    mocker.patch("httpx.AsyncClient.post", post)
    result = await route_legs([A, B], mode="walk")
    assert result.source == "google"
    assert result.mode == "walk"
    assert result.legs == [DirectionsLeg(duration_s=780, distance_m=1040.0)]
    assert result.total_duration_s == 780
    headers = post.call_args.kwargs["headers"]
    assert headers["X-Goog-FieldMask"] == "routes.legs.duration,routes.legs.distanceMeters"
    assert headers["X-Goog-Api-Key"] == "test-directions-key"


async def test_route_legs_transit_maps_travelmode(monkeypatch, mocker) -> None:
    key(monkeypatch)
    post = mocker.AsyncMock(
        return_value=ok_response([{"duration": "600s", "distanceMeters": 900}])
    )
    mocker.patch("httpx.AsyncClient.post", post)
    await route_legs([A, B], mode="transit")
    # 2 stops => single point-to-point request even for transit.
    body = post.call_args.kwargs["json"]
    assert body["travelMode"] == "TRANSIT"


async def test_route_legs_drive_maps_travelmode(monkeypatch, mocker) -> None:
    key(monkeypatch)
    post = mocker.AsyncMock(
        return_value=ok_response([{"duration": "300s", "distanceMeters": 2000}])
    )
    mocker.patch("httpx.AsyncClient.post", post)
    await route_legs([A, B], mode="drive")
    assert post.call_args.kwargs["json"]["travelMode"] == "DRIVE"


async def test_walk_three_stops_single_request_with_intermediates(monkeypatch, mocker) -> None:
    key(monkeypatch)
    post = mocker.AsyncMock(
        return_value=ok_response(
            [
                {"duration": "400s", "distanceMeters": 500},
                {"duration": "500s", "distanceMeters": 600},
            ]
        )
    )
    mocker.patch("httpx.AsyncClient.post", post)
    result = await route_legs([A, B, A], mode="walk")
    assert post.call_count == 1  # one multi-leg request
    assert "intermediates" in post.call_args.kwargs["json"]
    assert len(result.legs) == 2
    assert result.total_duration_s == 900


async def test_transit_with_intermediates_routes_per_leg(monkeypatch, mocker) -> None:
    """Routes API TRANSIT + intermediates is unreliable (W8b prod bug). For
    transit/drive with intermediates, route per-leg point-to-point and stitch.
    Not on the canonical /chat path (backend re-times WALK only) but kept
    honest for the integration test / future use."""
    key(monkeypatch)
    post = mocker.AsyncMock(
        side_effect=[
            ok_response([{"duration": "400s", "distanceMeters": 500}]),
            ok_response([{"duration": "500s", "distanceMeters": 600}]),
        ]
    )
    mocker.patch("httpx.AsyncClient.post", post)
    result = await route_legs([A, B, A], mode="transit")
    assert post.call_count == 2  # N-1 point-to-point requests
    assert len(result.legs) == 2
    assert result.total_duration_s == 900


async def test_timeout_falls_back(monkeypatch, mocker) -> None:
    key(monkeypatch)
    mocker.patch(
        "httpx.AsyncClient.post",
        mocker.AsyncMock(side_effect=httpx.TimeoutException("slow")),
    )
    result = await route_legs([A, B], mode="walk")
    assert result.source == "haversine_fallback"


async def test_non_200_falls_back(monkeypatch, mocker) -> None:
    key(monkeypatch)
    mocker.patch(
        "httpx.AsyncClient.post",
        mocker.AsyncMock(
            return_value=httpx.Response(500, text="err", request=httpx.Request("POST", "https://x"))
        ),
    )
    result = await route_legs([A, B], mode="walk")
    assert result.source == "haversine_fallback"


async def test_malformed_body_falls_back(monkeypatch, mocker) -> None:
    key(monkeypatch)
    mocker.patch(
        "httpx.AsyncClient.post",
        mocker.AsyncMock(
            return_value=httpx.Response(
                200,
                json={"no_routes": True},
                request=httpx.Request("POST", "https://x"),
            )
        ),
    )
    result = await route_legs([A, B], mode="walk")
    assert result.source == "haversine_fallback"


async def test_empty_routes_list_falls_back(monkeypatch, mocker) -> None:
    """Routes API returns {"routes": []} (HTTP 200) for an unroutable pair —
    must degrade to haversine, NOT raise IndexError out of route_legs."""
    key(monkeypatch)
    mocker.patch(
        "httpx.AsyncClient.post",
        mocker.AsyncMock(
            return_value=httpx.Response(
                200,
                json={"routes": []},
                request=httpx.Request("POST", "https://x"),
            )
        ),
    )
    result = await route_legs([A, B], mode="walk")
    assert result.source == "haversine_fallback"
