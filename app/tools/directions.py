"""Google Routes API v2 client for re-timing committed itineraries.

Mirrors app/tools/retrieval.py *structure* (typed Pydantic models, key via
get_settings(), module-level functions, __all__) — NOT its sync DB pattern.
This tool is natively async (httpx.AsyncClient): the W8c retime graph node
awaits route_legs() directly, never via asyncio.to_thread (contrast graph.py
act(), which wraps sync DB tools in a worker thread).

Never raises to the caller: every failure (missing key, timeout, non-200,
malformed body) degrades to a haversine-based estimate so /chat never blocks.
"""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

from app.agent.planning import WALKING_SPEED_M_PER_MIN, haversine_m
from app.config import get_settings

_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_TIMEOUT_S = 3.0

DEFAULT_MODE = "walk"
_MODE_TO_ROUTES_TRAVELMODE: dict[str, str] = {
    "walk": "WALK",
    "transit": "TRANSIT",
    "drive": "DRIVE",
}


class DirectionsLeg(BaseModel):
    duration_s: int
    distance_m: float


class DirectionsResult(BaseModel):
    legs: list[DirectionsLeg]
    total_duration_s: int
    mode: str
    source: Literal["google", "haversine_fallback"]


def _haversine_fallback(stops: list[tuple[float, float]], mode: str) -> DirectionsResult:
    legs: list[DirectionsLeg] = []
    for a, b in zip(stops[:-1], stops[1:], strict=False):
        dist_m = haversine_m(a, b)
        dur_s = round(dist_m / WALKING_SPEED_M_PER_MIN * 60)
        legs.append(DirectionsLeg(duration_s=dur_s, distance_m=dist_m))
    return DirectionsResult(
        legs=legs,
        total_duration_s=sum(leg.duration_s for leg in legs),
        mode=mode,
        source="haversine_fallback",
    )


def _parse_duration_s(raw: str) -> int:
    # Routes API returns protobuf Duration strings like "780s".
    return int(raw.rstrip("s"))


def _result_from_legs(legs: list[DirectionsLeg], mode: str) -> DirectionsResult:
    return DirectionsResult(
        legs=legs,
        total_duration_s=sum(leg.duration_s for leg in legs),
        mode=mode,
        source="google",
    )


async def _request_legs(
    client: httpx.AsyncClient,
    stops: list[tuple[float, float]],
    travel_mode: str,
    api_key: str,
) -> list[DirectionsLeg]:
    """One computeRoutes POST. Raises on non-2xx or malformed body so the
    caller's except-clause can fall back."""

    def _waypoint(p: tuple[float, float]) -> dict:
        return {"location": {"latLng": {"latitude": p[0], "longitude": p[1]}}}

    body: dict = {
        "origin": _waypoint(stops[0]),
        "destination": _waypoint(stops[-1]),
        "travelMode": travel_mode,
    }
    if len(stops) > 2:
        body["intermediates"] = [_waypoint(p) for p in stops[1:-1]]

    resp = await client.post(
        _ROUTES_URL,
        json=body,
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.legs.duration,routes.legs.distanceMeters",
        },
    )
    resp.raise_for_status()
    routes = resp.json()["routes"]
    raw_legs = routes[0]["legs"]
    return [
        DirectionsLeg(
            duration_s=_parse_duration_s(leg["duration"]),
            distance_m=float(leg["distanceMeters"]),
        )
        for leg in raw_legs
    ]


async def route_legs(
    stops: list[tuple[float, float]], mode: str = DEFAULT_MODE
) -> DirectionsResult:
    """Per-leg travel time for an ordered list of (lat, lng) stops.

    Never raises: missing key / timeout / non-200 / malformed body all
    degrade to a haversine-based estimate. Unknown mode raises ValueError
    BEFORE any network call (programmer error, not a runtime degradation).
    """
    if mode not in _MODE_TO_ROUTES_TRAVELMODE:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {sorted(_MODE_TO_ROUTES_TRAVELMODE)}"
        )
    if len(stops) < 2:
        return DirectionsResult(legs=[], total_duration_s=0, mode=mode, source="haversine_fallback")

    api_key = get_settings().google_directions_api_key
    if not api_key:
        return _haversine_fallback(stops, mode)

    travel_mode = _MODE_TO_ROUTES_TRAVELMODE[mode]
    try:
        if travel_mode != "WALK" and len(stops) > 2:
            # Routes API TRANSIT/DRIVE + intermediates is unreliable
            # (mirrors the W8b RouteOverlay per-leg fix). Route each leg
            # point-to-point and stitch.
            legs: list[DirectionsLeg] = []
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                for a, b in zip(stops[:-1], stops[1:], strict=False):
                    leg = await _request_legs(client, [a, b], travel_mode, api_key)
                    legs.extend(leg)
            return _result_from_legs(legs, mode)

        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            legs = await _request_legs(client, stops, travel_mode, api_key)
        return _result_from_legs(legs, mode)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return _haversine_fallback(stops, mode)


__all__ = [
    "DirectionsLeg",
    "DirectionsResult",
    "DEFAULT_MODE",
    "route_legs",
]
