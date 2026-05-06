#!/usr/bin/env python3
"""End-to-end smoke for W1 retrieval tools.

Exercises semantic_search / nearby / get_details with several filter
combinations against whichever DATABASE_URL + EMBEDDING_TABLE the env
provides. Prints results; raises on any unexpected failure so this script
can also run in CI as a post-deploy gate.

Usage:
    EMBEDDING_TABLE=place_embeddings_v2 \\
    DATABASE_URL='postgresql://postgres:cityconcierge@127.0.0.1:5433/mlops-city-concierge' \\
    poetry run python scripts/smoke_w1.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from app.tools.filters import SearchFilters
from app.tools.retrieval import get_details, nearby, semantic_search

load_dotenv()

SF = ZoneInfo("America/Los_Angeles")


def _print_hits(label: str, hits: list) -> None:
    print(f"\n=== {label} ({len(hits)} hits) ===")
    for h in hits:
        rating = f"{h.rating:>3.1f}" if h.rating is not None else "  ?"
        price = (h.price_level or "").replace("PRICE_LEVEL_", "").lower() or "?"
        print(f"  {rating}  {price:<15s}  {h.name:<45s}  {h.formatted_address}")


def smoke_basic_neighborhood() -> list:
    """Original plan smoke — neighborhood + min_rating."""
    hits = semantic_search(
        "romantic italian",
        SearchFilters(min_rating=4.3, neighborhood="North Beach"),
        k=5,
    )
    _print_hits("1. romantic italian in North Beach (rating>=4.3)", hits)
    assert all(h.rating is None or h.rating >= 4.3 for h in hits), "min_rating violated"
    return hits


def smoke_price_filter() -> list:
    """Tests price_level_max — was the bug we just fixed."""
    hits = semantic_search(
        "casual lunch spot",
        SearchFilters(price_level_max=2, min_user_rating_count=100),
        k=5,
    )
    _print_hits("2. casual lunch (price_level <= MODERATE, raters >= 100)", hits)
    allowed = {None, "PRICE_LEVEL_FREE", "PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_MODERATE"}
    assert all(h.price_level in allowed for h in hits), (
        f"price_level filter violated: {[h.price_level for h in hits]}"
    )
    return hits


def smoke_amenities() -> list:
    """Tests boolean amenity columns from W0a's exposed-as-columns work."""
    hits = semantic_search(
        "wine bar",
        SearchFilters(serves_wine=True, outdoor_seating=True),
        k=5,
    )
    _print_hits("3. wine bar (serves_wine + outdoor_seating)", hits)
    return hits


def smoke_open_at_tz_aware() -> list:
    """Tests place_is_open + the tz-aware Pydantic validator."""
    sf_friday_8pm = datetime(2026, 5, 1, 20, 0, tzinfo=SF)
    hits = semantic_search(
        "cocktail bar",
        SearchFilters(serves_cocktails=True, open_at=sf_friday_8pm),
        k=5,
    )
    _print_hits(f"4. cocktail bar open at {sf_friday_8pm.isoformat()}", hits)
    return hits


def smoke_get_details_and_nearby(seed_hits: list) -> None:
    """Tests get_details on a known place_id, then nearby() around it."""
    if not seed_hits:
        print("\n=== 5. get_details + nearby SKIPPED (no seed hit) ===")
        return
    seed = seed_hits[0]
    details = get_details(seed.place_id)
    assert details is not None, "get_details returned None for a known place_id"
    print(f"\n=== 5a. get_details({seed.place_id}) ===")
    print(f"  name: {details.name}")
    print(f"  types: {details.types}")
    print(f"  user_rating_count: {details.user_rating_count}")
    print(f"  has hours: {bool(details.regular_opening_hours)}")

    neighbors = nearby(seed.place_id, radius_m=500, k=5)
    _print_hits(f"5b. nearby {seed.name} (500m)", neighbors)
    assert all(h.place_id != seed.place_id for h in neighbors), "anchor leaked into nearby"


def main() -> int:
    print("W1 smoke — exercising semantic_search / nearby / get_details\n")

    failures: list[str] = []

    try:
        seed = smoke_basic_neighborhood()
    except Exception as exc:
        failures.append(f"smoke_basic_neighborhood: {exc!r}")
        seed = []

    for fn in (smoke_price_filter, smoke_amenities, smoke_open_at_tz_aware):
        try:
            fn()
        except Exception as exc:
            failures.append(f"{fn.__name__}: {exc!r}")

    try:
        smoke_get_details_and_nearby(seed)
    except Exception as exc:
        failures.append(f"smoke_get_details_and_nearby: {exc!r}")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} smoke(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — all smokes passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
