"""
Integration tests — require a running Postgres + pgvector instance.

These tests are skipped automatically when the DATABASE_URL points to
a host that is not reachable (e.g., in CI without a real DB service).

Run locally with:
    make db-up          # start Postgres in Docker
    make test-integration
"""

from __future__ import annotations

import os

import pytest


# Skip the entire module if we're not in an environment with a real DB.
pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)


class TestDatabaseConnection:
    """Smoke tests that verify DB connectivity and the pgvector extension."""

    def test_placeholder(self) -> None:
        """Placeholder — replace with real connection test once app/ exists."""
        assert True
