"""
Shared pytest configuration and fixtures.

Fixtures defined here are available to all tests without explicit import.
Plain helper functions (`make_stop`, `make_hit`) are exposed for `from
tests.conftest import ...` so multiple test modules don't redefine them.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from app.agent.state import Stop
from app.config import get_settings
from app.tools.retrieval import PlaceHit

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide safe defaults for required env vars so unit tests never hit
    real services. setdefault semantics — caller-supplied env wins so
    APP_ENV=integration + a real DATABASE_URL flow through unchanged.
    """
    defaults = {
        "OPENAI_API_KEY": "test-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "DATABASE_URL": "postgresql://postgres:test@localhost:5432/city_concierge_test",
        "APP_ENV": "test",
    }
    for key, value in defaults.items():
        if os.environ.get(key) is None:
            monkeypatch.setenv(key, value)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def make_stop(place_id: str = "p1", **kwargs: Any) -> Stop:
    """Build a Stop with sensible defaults; override any field via kwargs."""
    return Stop(
        place_id=place_id,
        name=kwargs.pop("name", place_id.upper()),
        source=kwargs.pop("source", "google_places"),
        rationale=kwargs.pop("rationale", ""),
        **kwargs,
    )


def make_hit(
    place_id: str = "p1",
    *,
    similarity: float = 0.9,
    business_status: str = "OPERATIONAL",
    **kwargs: Any,
) -> PlaceHit:
    """Build a PlaceHit with sensible defaults for fixture data."""
    return PlaceHit(
        place_id=place_id,
        name=kwargs.pop("name", place_id.upper()),
        source=kwargs.pop("source", "google_places"),
        similarity=similarity,
        latitude=kwargs.pop("latitude", 37.78),
        longitude=kwargs.pop("longitude", -122.41),
        rating=kwargs.pop("rating", 4.5),
        price_level=kwargs.pop("price_level", "PRICE_LEVEL_MODERATE"),
        business_status=business_status,
        primary_type=kwargs.pop("primary_type", "restaurant"),
        formatted_address=kwargs.pop("formatted_address", "123 Main"),
        snippet=kwargs.pop("snippet", None),
        **kwargs,
    )
