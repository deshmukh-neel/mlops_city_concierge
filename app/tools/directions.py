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

from pydantic import BaseModel

from app.agent.planning import WALKING_SPEED_M_PER_MIN, haversine_m  # noqa: F401
from app.config import get_settings  # noqa: F401

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


__all__ = [  # noqa: F822
    "DirectionsLeg",
    "DirectionsResult",
    "DEFAULT_MODE",
    "route_legs",
]
