"""
Shared pytest configuration and fixtures.

Fixtures defined here are available to all tests without explicit import.
"""

from __future__ import annotations

import pytest

from app.config import get_settings

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never accidentally read from a real .env file by providing
    safe default values for required environment variables."""
    defaults = {
        "OPENAI_API_KEY": "test-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "DATABASE_URL": "postgresql://postgres:test@localhost:5432/city_concierge_test",
        "APP_ENV": "test",
    }
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
