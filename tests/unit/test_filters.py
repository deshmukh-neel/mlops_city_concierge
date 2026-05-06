from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.tools.filters import SearchFilters, compile_filters


def test_empty_filters_no_clauses() -> None:
    where, params = compile_filters(SearchFilters(business_status=None, min_user_rating_count=None))
    assert where == ""
    assert params == []


def test_default_excludes_closed_permanently() -> None:
    where, params = compile_filters(SearchFilters())
    assert "business_status = %s" in where
    assert "OPERATIONAL" in params


def test_price_and_rating() -> None:
    where, params = compile_filters(
        SearchFilters(
            price_level_max=2,
            min_rating=4.3,
            business_status=None,
            min_user_rating_count=0,
        )
    )
    assert "price_level_rank(price_level) <= %s" in where
    assert "rating >= %s" in where
    assert 2 in params
    assert 4.3 in params


def test_neighborhood_uses_structured_column_with_fallback() -> None:
    where, params = compile_filters(
        SearchFilters(
            neighborhood="Mission Bay",
            business_status=None,
            min_user_rating_count=0,
        )
    )
    assert "LOWER(neighborhood) = LOWER(%s)" in where
    assert "formatted_address ILIKE %s" in where
    assert "Mission Bay" in params
    assert "%Mission Bay%" in params


def test_types_uses_array_overlap() -> None:
    where, params = compile_filters(
        SearchFilters(
            types_any=["bar", "wine_bar"],
            business_status=None,
            min_user_rating_count=0,
        )
    )
    assert "types && %s" in where
    assert ["bar", "wine_bar"] in params


def test_open_at_calls_helper() -> None:
    ts = datetime(2026, 4, 26, 19, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
    where, params = compile_filters(
        SearchFilters(open_at=ts, business_status=None, min_user_rating_count=0)
    )
    assert "place_is_open(regular_opening_hours, %s)" in where
    assert ts in params


def test_default_user_rating_count_floor_present() -> None:
    where, params = compile_filters(SearchFilters())
    assert "user_rating_count >= %s" in where
    assert 50 in params


def test_user_rating_count_floor_can_be_disabled() -> None:
    where, params = compile_filters(SearchFilters(min_user_rating_count=0))
    # 0 still emits a clause so the agent's intent is auditable in the SQL log.
    assert "user_rating_count >= %s" in where
    assert 0 in params


def test_boolean_amenity_filters_emit_clauses_when_set() -> None:
    where, params = compile_filters(
        SearchFilters(
            serves_cocktails=True,
            outdoor_seating=True,
            allows_dogs=False,
            business_status=None,
            min_user_rating_count=None,
        )
    )
    assert "serves_cocktails = %s" in where
    assert "outdoor_seating = %s" in where
    assert "allows_dogs = %s" in where
    # True/True/False — order matches SearchFilters field order.
    assert params == [True, True, False]


def test_unset_boolean_filters_emit_no_clause() -> None:
    where, _ = compile_filters(SearchFilters(business_status=None, min_user_rating_count=0))
    for col in (
        "serves_cocktails",
        "outdoor_seating",
        "reservable",
        "allows_dogs",
        "live_music",
        "good_for_groups",
    ):
        assert f"{col} =" not in where


def test_open_at_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SearchFilters(open_at=datetime(2026, 4, 26, 19, 30))


def test_open_at_accepts_tz_aware_datetime() -> None:
    sf = ZoneInfo("America/Los_Angeles")
    f = SearchFilters(open_at=datetime(2026, 4, 26, 19, 30, tzinfo=sf))
    assert f.open_at is not None
    assert f.open_at.tzinfo is sf

    f_utc = SearchFilters(open_at=datetime(2026, 4, 26, 19, 30, tzinfo=timezone.utc))
    assert f_utc.open_at is not None
    assert f_utc.open_at.tzinfo is timezone.utc
