"""Pure helpers for itinerary planning math. No LLM, no DB."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt

from app.agent.state import Stop

WALKING_SPEED_M_PER_MIN = 80.0  # ~3 mph, casual pace


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371000 * 2 * asin(sqrt(h))


def walking_time_min(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    return haversine_m((a_lat, a_lng), (b_lat, b_lng)) / WALKING_SPEED_M_PER_MIN


def next_arrival_time(prev_stop: Stop, next_lat: float, next_lng: float) -> datetime:
    if prev_stop.arrival_time is None:
        raise ValueError("prev_stop.arrival_time must be set before chaining")
    if prev_stop.latitude is None or prev_stop.longitude is None:
        raise ValueError("prev_stop coordinates must be set before chaining")
    walk = walking_time_min(prev_stop.latitude, prev_stop.longitude, next_lat, next_lng)
    return prev_stop.arrival_time + timedelta(minutes=prev_stop.planned_duration_min + walk)


def remaining_walking_budget_m(state) -> float:
    """How many meters of walking are left in the user's budget."""
    return max(0.0, state.constraints.walking_budget_m - state.walked_meters_so_far)


def suggested_radius_m(state, remaining_stops: int) -> int:
    """Per-stop radius suggestion given how much walking budget is left."""
    if remaining_stops <= 0:
        return 0
    budget = remaining_walking_budget_m(state)
    return int(min(1500, max(300, budget / max(remaining_stops, 1))))


def parse_stops_count(user_text: str, default: int = 3) -> int:
    """Parse '2', 'just dinner', 'dinner + drinks', '3 spots' -> integer."""
    text = user_text.lower().strip()
    if "just dinner" in text or "only dinner" in text:
        return 1
    if "dinner and drinks" in text or "dinner + drinks" in text or "dinner then drinks" in text:
        return 2
    for token in text.split():
        if token.isdigit():
            n = int(token)
            if 1 <= n <= 5:
                return n
    return default
