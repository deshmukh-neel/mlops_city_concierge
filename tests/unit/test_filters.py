from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from app.tools.filters import (
    _PRIMARY_TYPE_FAMILIES,
    SearchFilters,
    compile_filters,
    family_of,
    family_of_types,
)


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
    assert "(price_level_rank(price_level) IS NULL OR price_level_rank(price_level) <= %s)" in where
    assert "rating >= %s" in where
    assert 2 in params
    assert 4.3 in params


def test_price_filter_keeps_unknown_price_levels() -> None:
    where, params = compile_filters(
        SearchFilters(price_level_max=3, business_status=None, min_user_rating_count=0)
    )
    assert "price_level_rank(price_level) IS NULL OR" in where
    assert 3 in params


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


def test_serves_dessert_maps_to_dessert_place_types() -> None:
    where, params = compile_filters(
        SearchFilters(serves_dessert=True, business_status=None, min_user_rating_count=0)
    )
    assert "types && %s" in where
    assert "primary_type = ANY(%s)" in where
    assert "dessert_shop" in params[1]
    assert "Ice Cream Shop" in params[2]


def test_unknown_filter_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchFilters.model_validate({"serves_desserts": True})


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


def test_open_at_coerces_naive_datetime_to_sf_tz() -> None:
    """A naive open_at must NOT hard-reject (it derailed gpt-4o-mini on its
    first tool call — models frequently omit the offset). The app is SF-only,
    so a naive time unambiguously means SF local; coerce it to
    America/Los_Angeles. Still satisfies the original correctness goal (no
    ambiguous Postgres-session-tz interpretation across DST)."""
    f = SearchFilters(open_at=datetime(2026, 4, 26, 19, 30))
    assert f.open_at is not None
    assert f.open_at.tzinfo == ZoneInfo("America/Los_Angeles")
    # wall-clock preserved (interpreted AS SF local, not converted)
    assert (f.open_at.hour, f.open_at.minute) == (19, 30)


def test_open_at_accepts_tz_aware_datetime() -> None:
    sf = ZoneInfo("America/Los_Angeles")
    f = SearchFilters(open_at=datetime(2026, 4, 26, 19, 30, tzinfo=sf))
    assert f.open_at is not None
    assert f.open_at.tzinfo is sf

    f_utc = SearchFilters(open_at=datetime(2026, 4, 26, 19, 30, tzinfo=timezone.utc))
    assert f_utc.open_at is not None
    assert f_utc.open_at.tzinfo is timezone.utc


# ─── _PRIMARY_TYPE_FAMILIES + family_of (closure-aware swap) ─────────────


def test_primary_type_families_all_have_types_and_primary_types() -> None:
    for family, members in _PRIMARY_TYPE_FAMILIES.items():
        assert set(members.keys()) == {"types", "primary_types"}, family
        assert members["types"], f"{family} types must not be empty"
        assert members["primary_types"], f"{family} primary_types must not be empty"


def test_dessert_family_preserves_existing_dessert_members() -> None:
    dessert = _PRIMARY_TYPE_FAMILIES["dessert"]
    assert "dessert_shop" in dessert["types"]
    assert "Dessert Shop" in dessert["primary_types"]
    assert "Ice Cream Shop" in dessert["primary_types"]


def test_family_of_resolves_primary_type() -> None:
    assert family_of("Dessert Shop") == "dessert"
    assert family_of("Bar") == "bar"
    assert family_of("Cafe") == "cafe"
    assert family_of("Italian Restaurant") == "restaurant"


def test_family_of_returns_none_for_unknown_primary_type() -> None:
    assert family_of("Spaceship") is None
    assert family_of(None) is None
    assert family_of("") is None


def test_family_of_types_resolves_via_types_array() -> None:
    assert family_of_types(["italian_restaurant", "restaurant"]) == "restaurant"
    assert family_of_types(["dessert_shop"]) == "dessert"
    assert family_of_types([]) is None
