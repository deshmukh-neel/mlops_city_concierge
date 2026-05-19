"""Structured filters compiled to parameterized SQL fragments.

Every filter here corresponds to a real column on the place_documents view
(see alembic/versions/*_create_place_documents_view.py). A filter that maps
to a JSONB lookup (e.g. open_at) is implemented as a function the LLM does
not need to know about — it just sets `open_at`.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The concierge is San Francisco-only; a naive open_at unambiguously means
# SF local time.
_CITY_TZ = ZoneInfo("America/Los_Angeles")


class SearchFilters(BaseModel):
    """Structured constraints the agent passes to retrieval tools.

    All fields are optional but the quality-floor defaults
    (`min_user_rating_count = 50`, `business_status = 'OPERATIONAL'`) apply
    unless explicitly overridden. Empty SearchFilters() does NOT match
    everything — it matches operational places with at least 50 raters. The
    agent must opt out of the floors deliberately.
    """

    model_config = ConfigDict(extra="forbid")

    price_level_max: int | None = Field(
        default=None,
        ge=0,
        le=4,
        description=(
            "Max Google price_level on the documented 0..4 scale "
            "(0=FREE, 1=INEXPENSIVE, 2=MODERATE, 3=EXPENSIVE, 4=VERY_EXPENSIVE). "
            "places_raw stores the enum string; the filter compares via "
            "price_level_rank() so the comparison is an integer comparison."
        ),
    )
    min_rating: float | None = Field(default=None, ge=0.0, le=5.0)
    min_user_rating_count: int | None = Field(
        default=50,
        ge=0,
        description=(
            "Quality floor. Default 50 to keep single-rater places out. "
            "Set to 0 to include any number of raters."
        ),
    )
    open_at: datetime | None = Field(
        default=None,
        description=(
            "If set, restrict to places open at this local time. Must be "
            "timezone-aware (the agent should attach the city's local tz). "
            "Used per-stop with planned arrival time, NOT the user's prompt time."
        ),
    )
    neighborhood: str | None = Field(
        default=None,
        description=(
            "Exact match against the structured neighborhood column "
            "(case-insensitive). Falls back to formatted_address ILIKE only "
            "when no row in the neighborhood column matches."
        ),
    )
    types_any: list[str] | None = Field(
        default=None,
        description="Match if any of these strings appears in types[].",
    )
    business_status: str | None = Field(
        default="OPERATIONAL",
        description="Default OPERATIONAL. Set None to include closed/permanently_closed.",
    )
    source: str | None = Field(
        default=None,
        description="One of 'google_places', 'editorial'. None = any.",
    )

    serves_cocktails: bool | None = None
    serves_beer: bool | None = None
    serves_wine: bool | None = None
    serves_coffee: bool | None = None
    serves_breakfast: bool | None = None
    serves_brunch: bool | None = None
    serves_lunch: bool | None = None
    serves_dinner: bool | None = None
    serves_vegetarian: bool | None = None
    serves_dessert: bool | None = Field(
        default=None,
        description=(
            "Dessert category helper. True matches dessert-oriented Google "
            "types such as dessert_shop, bakery, ice_cream_shop, cafe, "
            "tea_house, and confectionery."
        ),
    )
    outdoor_seating: bool | None = None
    reservable: bool | None = None
    allows_dogs: bool | None = None
    live_music: bool | None = None
    good_for_groups: bool | None = None
    good_for_children: bool | None = None
    good_for_sports: bool | None = None

    @field_validator("open_at")
    @classmethod
    def _ensure_tz_aware(cls, v: datetime | None) -> datetime | None:
        # A naive datetime would be interpreted in Postgres's session timezone,
        # silently producing wrong DOW/hour answers across DST boundaries.
        # Hard-rejecting it derailed models that omit the offset (a very common
        # LLM mistake — it broke gpt-4o-mini on its first tool call). The app
        # is SF-only, so a naive time unambiguously means SF local: attach the
        # city tz instead of rejecting. Same correctness, no derailment.
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=_CITY_TZ)
        return v


_BOOL_COLUMNS: tuple[str, ...] = (
    "serves_cocktails",
    "serves_beer",
    "serves_wine",
    "serves_coffee",
    "serves_breakfast",
    "serves_brunch",
    "serves_lunch",
    "serves_dinner",
    "serves_vegetarian",
    "outdoor_seating",
    "reservable",
    "allows_dogs",
    "live_music",
    "good_for_groups",
    "good_for_children",
    "good_for_sports",
)

_DESSERT_TYPES: tuple[str, ...] = (
    "dessert_shop",
    "bakery",
    "ice_cream_shop",
    "candy_store",
    "chocolate_shop",
    "coffee_shop",
    "cafe",
    "confectionery",
    "donut_shop",
    "tea_house",
)

_DESSERT_PRIMARY_TYPES: tuple[str, ...] = (
    "Dessert Shop",
    "Bakery",
    "Ice Cream Shop",
    "Candy Store",
    "Chocolate Shop",
    "Coffee Shop",
    "Cafe",
    "Confectionery store",
    "Donut Shop",
    "Tea House",
)


def compile_filters(f: SearchFilters) -> tuple[str, list]:
    """Return (sql_where_fragment, params_list).

    The fragment begins with ' AND ' (caller prepends a WHERE clause). Params
    are positional (psycopg2 style).
    """
    clauses: list[str] = []
    params: list = []

    if f.price_level_max is not None:
        # places_raw.price_level is a Google v1 enum string ('PRICE_LEVEL_*');
        # price_level_rank() (alembic c428add573d7) maps it to 0..4 so we can
        # compare against the integer the agent passes. Unknown price passes
        # rather than disappearing; constraints_satisfied() uses the same
        # "missing data should not fail" semantics.
        clauses.append(
            "(price_level_rank(price_level) IS NULL OR price_level_rank(price_level) <= %s)"
        )
        params.append(f.price_level_max)

    if f.min_rating is not None:
        clauses.append("rating >= %s")
        params.append(f.min_rating)

    if f.min_user_rating_count is not None:
        clauses.append("user_rating_count >= %s")
        params.append(f.min_user_rating_count)

    if f.business_status is not None:
        clauses.append("business_status = %s")
        params.append(f.business_status)

    if f.neighborhood:
        clauses.append("(LOWER(neighborhood) = LOWER(%s) OR formatted_address ILIKE %s)")
        params.append(f.neighborhood)
        params.append(f"%{f.neighborhood}%")

    for column in _BOOL_COLUMNS:
        value = getattr(f, column)
        if value is not None:
            clauses.append(f"{column} = %s")
            params.append(value)

    if f.serves_dessert is not None:
        dessert_clause = "(types && %s OR primary_type = ANY(%s))"
        if f.serves_dessert:
            clauses.append(dessert_clause)
        else:
            clauses.append(f"NOT {dessert_clause}")
        params.append(list(_DESSERT_TYPES))
        params.append(list(_DESSERT_PRIMARY_TYPES))

    if f.types_any:
        clauses.append("types && %s")
        params.append(f.types_any)

    if f.source:
        clauses.append("source = %s")
        params.append(f.source)

    if f.open_at is not None:
        clauses.append("place_is_open(regular_opening_hours, %s)")
        params.append(f.open_at)

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params
