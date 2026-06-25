"""Structured filters compiled to parameterized SQL fragments.

Every filter here corresponds to a real column on the place_documents view
(see alembic/versions/*_create_place_documents_view.py). A filter that maps
to a JSONB lookup (e.g. open_at) is implemented as a function the LLM does
not need to know about — it just sets `open_at`.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The concierge is San Francisco-only; a naive open_at unambiguously means
# SF local time.
CITY_TZ = ZoneInfo("America/Los_Angeles")


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
    primary_type_family: Literal["dessert", "bar", "restaurant", "cafe"] | None = Field(
        default=None,
        description=(
            "Restrict candidates to a category family (dessert/bar/restaurant/"
            "cafe). Expands to a `(types && %s OR primary_type = ANY(%s))` "
            "clause against both column conventions. Distinct from `types_any`, "
            "which only matches the types[] array."
        ),
    )
    excluded_place_ids: list[str] | None = Field(
        default=None,
        description=(
            "place_ids to exclude from results. Used by the closure-aware "
            "swap path to prevent re-suggesting a place that was previously "
            "found closed in the same conversation. Empty list (or None) is a "
            "no-op."
        ),
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
    def ensure_tz_aware(cls, v: datetime | None) -> datetime | None:
        # A naive datetime would be interpreted in Postgres's session timezone,
        # silently producing wrong DOW/hour answers across DST boundaries.
        # Hard-rejecting it derailed models that omit the offset (a very common
        # LLM mistake — it broke gpt-4o-mini on its first tool call). The app
        # is SF-only, so a naive time unambiguously means SF local: attach the
        # city tz instead of rejecting. Same correctness, no derailment.
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=CITY_TZ)
        return v


BOOL_COLUMNS: tuple[str, ...] = (
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

# Maps a family name to the column conventions place_documents/places_raw
# uses for category matching:
#   - "types":         snake_case strings used in the `types[]` array column
#   - "primary_types": Title Case strings used in the `primary_type` scalar
# Both lists are required for each family because Postgres rows can carry
# either or both. The `serves_dessert` helper and the `primary_type_family`
# filter both compile to `(types && %s OR primary_type = ANY(%s))` against
# these two lists so the row matches on either column.
PRIMARY_TYPE_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "dessert": {
        "types": (
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
        ),
        "primary_types": (
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
        ),
    },
    "bar": {
        "types": (
            "bar",
            "cocktail_bar",
            "wine_bar",
            "pub",
            "sports_bar",
            "night_club",
        ),
        "primary_types": (
            "Bar",
            "Cocktail Bar",
            "Wine Bar",
            "Pub",
            "Sports Bar",
            "Night Club",
        ),
    },
    "restaurant": {
        "types": (
            "restaurant",
            "fine_dining_restaurant",
            "italian_restaurant",
            "japanese_restaurant",
            "chinese_restaurant",
            "mexican_restaurant",
            "thai_restaurant",
            "indian_restaurant",
            "french_restaurant",
            "vietnamese_restaurant",
            "korean_restaurant",
            "mediterranean_restaurant",
            "seafood_restaurant",
            "steak_house",
            "sushi_restaurant",
            "ramen_restaurant",
            "pizza_restaurant",
            "american_restaurant",
        ),
        "primary_types": (
            "Restaurant",
            "Fine Dining Restaurant",
            "Italian Restaurant",
            "Japanese Restaurant",
            "Chinese Restaurant",
            "Mexican Restaurant",
            "Thai Restaurant",
            "Indian Restaurant",
            "French Restaurant",
            "Vietnamese Restaurant",
            "Korean Restaurant",
            "Mediterranean Restaurant",
            "Seafood Restaurant",
            "Steak House",
            "Sushi Restaurant",
            "Ramen Restaurant",
            "Pizza Restaurant",
            "American Restaurant",
        ),
    },
    "cafe": {
        "types": ("cafe", "coffee_shop", "tea_house"),
        "primary_types": ("Cafe", "Coffee Shop", "Tea House"),
    },
}


# Priority order for the reverse-lookup helpers below. The dessert family
# intentionally overlaps with the cafe family (a cafe serves desserts), but
# when *identifying* a closed stop's category for the closure-aware swap,
# we want the more specific family — a closed "Cafe" should swap to another
# cafe, not arbitrarily into the dessert family. Order: most specific first.
FAMILY_LOOKUP_PRIORITY: tuple[str, ...] = ("bar", "restaurant", "cafe", "dessert")


# Free-text keywords that signal each family in an agent-issued query string.
# Used by `family_from_query` to bind a per-slot family filter when the model
# omits `slot_index`. Lowercase, matched on whole words. Kept deliberately
# tight (high-precision) so an ambiguous query infers nothing rather than the
# wrong family — the fail-open default (no filter) is preferable to a wrong
# filter. Walked in `FAMILY_LOOKUP_PRIORITY` order by the caller.
FAMILY_QUERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "bar": ("drinks", "drink", "cocktail", "cocktails", "bar", "wine", "pub", "nightcap"),
    "restaurant": ("dinner", "lunch", "restaurant", "eat", "meal", "dining", "brunch", "supper"),
    "cafe": ("coffee", "cafe", "espresso", "latte"),
    "dessert": ("dessert", "desserts", "sweets", "bakery", "pastry", "pastries", "gelato"),
}


def family_of(primary_type: str | None) -> str | None:
    """Reverse lookup: scalar primary_type column value -> family name.

    Case-preserving comparison — the DB column preserves Title Case verbatim,
    and `PRIMARY_TYPE_FAMILIES` stores both casings exactly as the columns do.
    Returns the first match in `FAMILY_LOOKUP_PRIORITY` order so overlapping
    categories (Cafe is in both "cafe" and "dessert") resolve deterministically.
    Returns None for unknown / empty / None inputs.
    """
    if not primary_type:
        return None
    for family in FAMILY_LOOKUP_PRIORITY:
        if primary_type in PRIMARY_TYPE_FAMILIES[family]["primary_types"]:
            return family
    return None


def family_of_types(types: list[str] | None) -> str | None:
    """Reverse lookup: types[] array values -> family name.

    Returns the first family that overlaps the input list, walking families
    in `FAMILY_LOOKUP_PRIORITY` order. Lets the swap node fall back when
    `primary_type` isn't in the index but `types[]` is populated. Returns
    None for an empty list.
    """
    if not types:
        return None
    for family in FAMILY_LOOKUP_PRIORITY:
        if any(t in PRIMARY_TYPE_FAMILIES[family]["types"] for t in types):
            return family
    return None


def family_from_query(query: str | None, requested_primary_types: list[str] | None) -> str | None:
    """Infer a slot family from free-text query when the model omits `slot_index`.

    `inject_primary_type_family` only binds a per-slot family filter when the
    model voluntarily emits `slot_index`. Live traces (2026-06-15) show the
    models (incl. gpt-4o-mini and the reasoning models) routinely DON'T — they
    emit thin, unfiltered queries like "drinks in Hayes Valley", so the
    typed-slot viability gate never sees per-slot candidates and the agent
    fails to commit on multi-type requests (the refinement_cheaper failure).

    This is the slot-index-free fallback: derive the family from the query
    text, but ONLY return a family the user actually requested. The query's
    words are matched (lowercased, whole-word) against `FAMILY_QUERY_KEYWORDS`
    for each requested type's family. It never invents a family outside
    `requested_primary_types`, so a "coffee shop" query cannot smuggle in the
    `cafe` family when the user asked only for Restaurant + Bar.

    Returns None when there is no query, no requested types, or no keyword from
    a requested family appears — leaving the existing fail-open (no filter)
    behavior untouched in the ambiguous case.
    """
    if not query or not requested_primary_types:
        return None

    requested_families: list[str] = []
    for pt in requested_primary_types:
        fam = family_of(pt)
        if fam is not None and fam not in requested_families:
            requested_families.append(fam)
    if not requested_families:
        return None

    q_words = set(re.findall(r"[a-z]+", query.lower()))
    # Collect EVERY requested family whose keywords appear. If more than one
    # matches, the query is ambiguous (e.g. "dinner and drinks" → restaurant AND
    # bar) — return None rather than silently filtering out a requested category
    # Only an unambiguous single match injects a filter.
    matched_families = [
        family
        for family in FAMILY_LOOKUP_PRIORITY
        if family in requested_families
        and any(kw in q_words for kw in FAMILY_QUERY_KEYWORDS.get(family, ()))
    ]
    return matched_families[0] if len(matched_families) == 1 else None


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

    for column in BOOL_COLUMNS:
        value = getattr(f, column)
        if value is not None:
            clauses.append(f"{column} = %s")
            params.append(value)

    if f.serves_dessert is not None:
        dessert = PRIMARY_TYPE_FAMILIES["dessert"]
        dessert_clause = "(types && %s OR primary_type = ANY(%s))"
        if f.serves_dessert:
            clauses.append(dessert_clause)
        else:
            clauses.append(f"NOT {dessert_clause}")
        params.append(list(dessert["types"]))
        params.append(list(dessert["primary_types"]))

    if f.types_any:
        clauses.append("types && %s")
        params.append(f.types_any)

    if f.primary_type_family is not None:
        family = PRIMARY_TYPE_FAMILIES[f.primary_type_family]
        clauses.append("(types && %s OR primary_type = ANY(%s))")
        params.append(list(family["types"]))
        params.append(list(family["primary_types"]))

    if f.excluded_place_ids:
        # Empty list is intentionally a no-op (caller signals "no exclusions"
        # without having to pass None).
        clauses.append("place_id != ALL(%s)")
        params.append(f.excluded_place_ids)

    if f.source:
        clauses.append("source = %s")
        params.append(f.source)

    if f.open_at is not None:
        clauses.append("place_is_open(regular_opening_hours, %s)")
        params.append(f.open_at)

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params
