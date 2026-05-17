"""Unit tests for app.agent.state helpers."""

from __future__ import annotations

import pytest

from app.agent.state import Stop, price_level_to_rank


@pytest.mark.parametrize(
    ("enum", "expected"),
    [
        ("PRICE_LEVEL_FREE", 0),
        ("PRICE_LEVEL_INEXPENSIVE", 1),
        ("PRICE_LEVEL_MODERATE", 2),
        ("PRICE_LEVEL_EXPENSIVE", 3),
        ("PRICE_LEVEL_VERY_EXPENSIVE", 4),
        ("PRICE_LEVEL_UNSPECIFIED", None),
        ("garbage", None),
        ("", None),
        (None, None),
    ],
)
def test_price_level_to_rank(enum: str | None, expected: int | None) -> None:
    assert price_level_to_rank(enum) == expected


def test_stop_defaults_new_fields_to_none() -> None:
    s = Stop(place_id="p1", name="A", rationale="r", source="google_places")
    assert s.address is None
    assert s.rating is None
    assert s.price_level is None
