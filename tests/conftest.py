"""
Shared pytest configuration and fixtures.

Fixtures defined here are available to all tests without explicit import.
"""

from __future__ import annotations

import os

import pytest

from app.config import get_settings

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
