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
def neutralize_query_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress app.main.log_user_query globally in all unit/functional tests.

    The new BackgroundTask scheduled by chat() runs synchronously under
    TestClient before client.post() returns.  Without this fixture, the
    no-op spy it installs prevents the BackgroundTask from ever calling the
    real get_conn() / activating a real ThreadedConnectionPool — closing the
    known full-suite DB-pool-contamination class (see project memory
    project_full_suite_db_pool_contamination).

    Contract (document for future contributors): this fixture suppresses
    app.main.log_user_query so the BackgroundTask never opens a real DB pool
    in unit/functional tests.  Any future test that needs the REAL write to
    run (e.g. "real /chat writes a DB row" integration test) MUST override
    this suppression — either re-patch app.main.log_user_query locally via
    mocker.patch(...) or use the APP_ENV=integration integration test in
    tests/integration/test_query_log.py.  The word "override" is intentional:
    grep -q 'override' tests/conftest.py is a contract check.
    """
    monkeypatch.setattr("app.main.log_user_query", lambda **kwargs: None)


@pytest.fixture(autouse=True)
def patch_env(monkeypatch: pytest.MonkeyPatch) -> None:
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


def normalize_test_place_id(value: str) -> str:
    """Pad short descriptive place_ids to satisfy the 06-01 Task 3
    Google-Place-ID-format validator (>= 20 chars, [A-Za-z0-9_-]).
    Inputs already matching the format pass through unchanged.
    """
    import re as re_mod

    if re_mod.fullmatch(r"^[A-Za-z0-9_-]{20,255}$", value):
        return value
    descriptor = re_mod.sub(r"[^A-Za-z0-9_-]", "_", value) or "x"
    base = f"ChIJtest_{descriptor}_"
    while len(base) < 20:
        base += "a"
    return base[:255]


def make_stop(place_id: str = "ChIJtest_p1_aaaaaaaa", **kwargs: Any) -> Stop:
    """Build a Stop with sensible defaults; override any field via kwargs.

    `place_id` is normalized via `normalize_test_place_id` so short
    descriptive ids like "p1" or `f"p{i}"` still work after Plan 06-01
    added the Google-Place-ID-format validator on Stop.
    """
    return Stop(
        place_id=normalize_test_place_id(place_id),
        name=kwargs.pop("name", place_id.upper()),
        source=kwargs.pop("source", "google_places"),
        rationale=kwargs.pop("rationale", ""),
        **kwargs,
    )


def make_hit(
    place_id: str = "ChIJtest_p1_aaaaaaaa",
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
