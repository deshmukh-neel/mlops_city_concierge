"""Integration tests for the place_is_open() PL/pgSQL helper.

Gated on APP_ENV=integration. Requires the W1 migration to be applied:
    make migrate
    APP_ENV=integration make test-integration
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.db import get_conn

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)


SF = ZoneInfo("America/Los_Angeles")


def _period(
    open_dow: int,
    open_h: int,
    open_m: int,
    close_dow: int,
    close_h: int,
    close_m: int,
) -> dict:
    """Build a single-period regular_opening_hours JSONB matching Google Places v1."""
    return {
        "periods": [
            {
                "open": {"day": open_dow, "hour": open_h, "minute": open_m},
                "close": {"day": close_dow, "hour": close_h, "minute": close_m},
            }
        ]
    }


def _is_open(hours: dict, at: datetime) -> bool:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT place_is_open(%s::jsonb, %s)", [json.dumps(hours), at])
        row = cur.fetchone()
        assert row is not None
        return bool(row[0])


def test_unknown_hours_returns_true() -> None:
    """If we don't know the hours, don't exclude — assume open."""
    assert _is_open({}, datetime(2026, 4, 28, 12, 30, tzinfo=SF))


def test_same_day_window_open() -> None:
    # Tuesday (DOW=2) 12:30, place open Tue 11:00–22:00.
    assert _is_open(_period(2, 11, 0, 2, 22, 0), datetime(2026, 4, 28, 12, 30, tzinfo=SF))


def test_same_day_window_closed_after_hours() -> None:
    assert not _is_open(_period(2, 11, 0, 2, 22, 0), datetime(2026, 4, 28, 23, 0, tzinfo=SF))


def test_overnight_window_after_open_same_day() -> None:
    # Bar open Fri (5) 18:00 → Sat (6) 02:00, query Fri 22:00 → open.
    assert _is_open(_period(5, 18, 0, 6, 2, 0), datetime(2026, 5, 1, 22, 0, tzinfo=SF))


def test_overnight_window_before_close_next_day() -> None:
    # Same period, query Sat 01:30 → still open.
    assert _is_open(_period(5, 18, 0, 6, 2, 0), datetime(2026, 5, 2, 1, 30, tzinfo=SF))


def test_overnight_window_closed_in_morning() -> None:
    # Same period, query Sat 03:00 → closed.
    assert not _is_open(_period(5, 18, 0, 6, 2, 0), datetime(2026, 5, 2, 3, 0, tzinfo=SF))
