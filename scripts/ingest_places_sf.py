#!/usr/bin/env python3
"""
Ingest Google Places data scoped to San Francisco into Postgres.

This script pulls Places data (with paging), deduplicates by place_id,
and upserts rows into places_raw so you can query locally without repeated
Google API calls.

Usage:
    python scripts/ingest_places_sf.py

Optional env vars:
    GOOGLE_PLACES_API_KEY      Google Places API key
    DATABASE_URL               Postgres connection URL
    PLACES_MAX_PAGES_PER_QUERY Max pages per query (default: 1)
    PLACES_QUERY_LIMIT         Optional cap on number of generated queries
    PLACES_MAX_API_CALLS       Optional hard cap on API calls per run (0 = no cap)
    PLACES_MIN_REQUEST_INTERVAL_SECONDS Minimum delay between calls (default: 0.25)
    PLACES_API_MAX_RETRIES     Retry attempts for transient API errors (default: 4)
    PLACES_API_BACKOFF_BASE_SECONDS Base exponential backoff sleep (default: 1.0)
    PLACES_API_BACKOFF_MAX_SECONDS Max backoff sleep per retry (default: 20.0)
    PLACES_SKIP_COMPLETED_QUERIES Skip previously completed seed queries on reruns (default: true)
    PLACES_CHECKPOINT_RESET_CONTAINS Comma-separated keywords to clear matching checkpoints before run
    PLACES_FIELD_MODE         Field mask mode: lean (default) or enriched
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from urllib.parse import quote_plus

import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import Json

load_dotenv()

# Global variables dictating Google Places API limits and interactions (ie minimizing cost)

BASE_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE-PLACES-API-KEY")


def resolve_database_url() -> str | None:
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    dbname = os.getenv("POSTGRES_DB")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    if not (user and password and dbname):
        return None

    return f"postgresql://{user}:{quote_plus(password)}@{host}:{port}/{dbname}"


DATABASE_URL = resolve_database_url()
MAX_PAGES_PER_QUERY = int(os.getenv("PLACES_MAX_PAGES_PER_QUERY", "1"))
QUERY_LIMIT = int(os.getenv("PLACES_QUERY_LIMIT", "0"))
MAX_API_CALLS = int(os.getenv("PLACES_MAX_API_CALLS", "1800"))
MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("PLACES_MIN_REQUEST_INTERVAL_SECONDS", "0.25"))
API_MAX_RETRIES = int(os.getenv("PLACES_API_MAX_RETRIES", "4"))
API_BACKOFF_BASE_SECONDS = float(os.getenv("PLACES_API_BACKOFF_BASE_SECONDS", "1.0"))
API_BACKOFF_MAX_SECONDS = float(os.getenv("PLACES_API_BACKOFF_MAX_SECONDS", "20.0"))
FIELD_MODE = os.getenv("PLACES_FIELD_MODE", "lean").strip().lower()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


SKIP_COMPLETED_QUERIES = env_bool("PLACES_SKIP_COMPLETED_QUERIES", True)
CHECKPOINT_RESET_CONTAINS = [
    token.strip().lower()
    for token in os.getenv("PLACES_CHECKPOINT_RESET_CONTAINS", "").split(",")
    if token.strip()
]

LEAN_FIELDS = [
    # "Lean" is optimized for lower cost and semantic usefulness.
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.types",
    "places.location",
    "places.editorialSummary",
]

ENRICHED_EXTRA_FIELDS = [
    "places.primaryTypeDisplayName",
    "places.websiteUri",
    "places.businessStatus",
    "places.googleMapsUri",
    "places.regularOpeningHours",
]


def build_field_mask() -> str:
    if FIELD_MODE == "enriched":
        return ",".join(LEAN_FIELDS + ENRICHED_EXTRA_FIELDS)
    if FIELD_MODE != "lean":
        print(f"Unknown PLACES_FIELD_MODE={FIELD_MODE}; falling back to lean mode.")
    return ",".join(LEAN_FIELDS)


FIELDS = build_field_mask()

SF_CENTER = {"latitude": 37.7749, "longitude": -122.4194}
SF_RADIUS_METERS = 10_000.0

NEIGHBORHOODS = [
    "Mission District",
    "SOMA",
    "North Beach",
    "Chinatown",
    "Financial District",
    "Nob Hill",
    "Russian Hill",
    "Marina District",
    "Cow Hollow",
    "Pacific Heights",
    "Hayes Valley",
    "Castro",
    "Noe Valley",
    "Bernal Heights",
    "Potrero Hill",
    "Dogpatch",
    "Inner Sunset",
    "Outer Sunset",
    "Inner Richmond",
    "Outer Richmond",
    "Presidio Heights",
    "Haight-Ashbury",
    "Western Addition",
    "Japantown",
    "Tenderloin",
    "Union Square",
    "Civic Center",
    "Bayview",
    "Visitacion Valley",
    "Excelsior",
]

CUISINES = [
    "italian",
    "french",
    "spanish",
    "portuguese",
    "greek",
    "turkish",
    "lebanese",
    "persian",
    "moroccan",
    "ethiopian",
    "indian",
    "pakistani",
    "nepalese",
    "thai",
    "vietnamese",
    "chinese",
    "japanese",
    "korean",
    "filipino",
    "indonesian",
    "malaysian",
    "taiwanese",
    "mexican",
    "salvadoran",
    "guatemalan",
    "peruvian",
    "brazilian",
    "argentinian",
    "american",
    "californian",
    "seafood",
    "steakhouse",
    "vegan",
    "vegetarian",
    "gluten free",
    "halal",
    "kosher",
    "ramen",
    "sushi",
    "pizza",
    "burger",
    "bbq",
    "dimsum",
    "taco",
    "mediterranean",
]

EATERY_TYPES = [
    "restaurants",
    "fine dining restaurants",
    "casual restaurants",
    "family restaurants",
    "cafes",
    "coffee shops",
    "espresso bars",
    "tea houses",
    "bakeries",
    "pastry shops",
    "dessert shops",
    "ice cream shops",
    "donut shops",
    "breakfast restaurants",
    "brunch spots",
    "lunch spots",
    "dinner restaurants",
    "late night food",
    "food trucks",
    "sandwich shops",
    "delis",
    "pizzerias",
    "noodle shops",
    "ramen shops",
    "sushi bars",
    "taquerias",
    "poke shops",
    "salad shops",
    "juice bars",
    "vegetarian restaurants",
    "vegan restaurants",
]

BAR_TYPES = [
    "bars",
    "wine bars",
    "cocktail bars",
    "speakeasy bars",
    "sports bars",
    "dive bars",
    "rooftop bars",
    "hotel bars",
    "craft beer bars",
    "breweries",
    "taprooms",
    "gastropubs",
    "pubs",
    "lounges",
    "live music bars",
    "karaoke bars",
]

ATTRACTION_TYPES = [
    "tourist attractions",
    "things to do",
    "museums",
    "art museums",
    "science museums",
    "history museums",
    "art galleries",
    "public parks",
    "botanical gardens",
    "viewpoints",
    "landmarks",
    "historic sites",
    "neighborhood walking tours",
    "shopping districts",
    "waterfront attractions",
    "live music venues",
    "theaters",
    "comedy clubs",
    "bookstores",
    "movie theaters",
]


def build_seed_queries() -> list[str]:
    queries: list[str] = []

    # City-wide category coverage.
    queries.extend([f"{kind} in San Francisco" for kind in EATERY_TYPES])
    queries.extend([f"{kind} in San Francisco" for kind in BAR_TYPES])
    queries.extend([f"{kind} in San Francisco" for kind in ATTRACTION_TYPES])
    queries.extend([f"{cuisine} restaurants in San Francisco" for cuisine in CUISINES])

    # Neighborhood-level coverage for food, bar, and attraction intent.
    for neighborhood in NEIGHBORHOODS:
        queries.extend([f"{kind} in {neighborhood} San Francisco" for kind in EATERY_TYPES])
        queries.extend([f"{kind} in {neighborhood} San Francisco" for kind in BAR_TYPES])
        queries.extend([f"{kind} in {neighborhood} San Francisco" for kind in ATTRACTION_TYPES])
        queries.extend(
            [f"{cuisine} restaurants in {neighborhood} San Francisco" for cuisine in CUISINES]
        )

    # Preserve order while removing accidental duplicates.
    deduped = list(dict.fromkeys(queries))
    if QUERY_LIMIT > 0:
        return deduped[:QUERY_LIMIT]
    return deduped


def checkpoint_key(query_text: str) -> str:
    # Scope checkpoints by field mode so lean/enriched runs can progress independently.
    return f"{FIELD_MODE}::{query_text}"


@dataclass
class PullStats:
    rows_seen: int = 0
    rows_upserted: int = 0
    api_calls: int = 0
    queries_processed: int = 0
    queries_completed: int = 0
    queries_skipped: int = 0


def _sleep_for_rate_limit(last_request_at: float | None) -> None:
    if last_request_at is None:
        return

    elapsed = time.time() - last_request_at
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def search_places(query: str, page_token: str | None = None) -> tuple[list[dict], str | None]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_KEY,
        "X-Goog-FieldMask": FIELDS,
    }

    body: dict = {
        "textQuery": query,
        "maxResultCount": 20,
        "locationBias": {
            "circle": {
                "center": SF_CENTER,
                "radius": SF_RADIUS_METERS,
            }
        },
    }

    if page_token:
        body["pageToken"] = page_token

    for attempt in range(1, API_MAX_RETRIES + 1):
        response = requests.post(BASE_URL, json=body, headers=headers, timeout=30)
        if response.ok:
            payload = response.json()
            return payload.get("places", []), payload.get("nextPageToken")

        if not _is_retryable_status(response.status_code) or attempt == API_MAX_RETRIES:
            response.raise_for_status()

        sleep_seconds = min(API_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), API_BACKOFF_MAX_SECONDS)
        print(
            f"Retrying query after HTTP {response.status_code} "
            f"(attempt {attempt}/{API_MAX_RETRIES}, sleep {sleep_seconds:.1f}s)"
        )
        time.sleep(sleep_seconds)

    raise RuntimeError("Unexpected API retry flow reached.")


def _place_row(place: dict, source_query: str) -> tuple:
    location = place.get("location") or {}
    display_name = place.get("displayName") or {}
    primary_type_display = place.get("primaryTypeDisplayName") or {}
    editorial_summary = place.get("editorialSummary") or {}

    return (
        place.get("id"),
        display_name.get("text"),
        primary_type_display.get("text"),
        place.get("formattedAddress"),
        location.get("latitude"),
        location.get("longitude"),
        place.get("rating"),
        place.get("userRatingCount"),
        place.get("priceLevel"),
        place.get("websiteUri"),
        place.get("businessStatus"),
        place.get("googleMapsUri"),
        place.get("types") or [],
        editorial_summary.get("text"),
        Json(place.get("regularOpeningHours") or {}),
        source_query,
        Json(place),
    )


def ensure_query_checkpoint_table(conn: psycopg2.extensions.connection) -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS places_ingest_query_checkpoints (
        query_text              TEXT PRIMARY KEY,
        status                  TEXT NOT NULL,
        pages_processed         INTEGER NOT NULL DEFAULT 0,
        api_calls               INTEGER NOT NULL DEFAULT 0,
        rows_seen               INTEGER NOT NULL DEFAULT 0,
        rows_changed            INTEGER NOT NULL DEFAULT 0,
        last_error              TEXT,
        last_run_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        completed_at            TIMESTAMPTZ
    )
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def ensure_query_hits_table(conn: psycopg2.extensions.connection) -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS place_query_hits (
        id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        place_id        TEXT NOT NULL REFERENCES places_raw(place_id) ON DELETE CASCADE,
        query_text      TEXT NOT NULL,
        field_mode      TEXT NOT NULL,
        page_number     INTEGER NOT NULL,
        rank_in_page    INTEGER NOT NULL,
        seen_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (place_id, query_text, field_mode, page_number, rank_in_page)
    )
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def get_completed_queries(conn: psycopg2.extensions.connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT query_text
            FROM places_ingest_query_checkpoints
            WHERE status = 'completed'
            """
        )
        rows = cur.fetchall()
    return {row[0] for row in rows}


def mark_query_progress(
    conn: psycopg2.extensions.connection,
    *,
    query_text: str,
    status: str,
    pages_processed: int,
    api_calls: int,
    rows_seen: int,
    rows_changed: int,
    last_error: str | None = None,
) -> None:
    sql = """
    INSERT INTO places_ingest_query_checkpoints (
        query_text,
        status,
        pages_processed,
        api_calls,
        rows_seen,
        rows_changed,
        last_error,
        last_run_at,
        completed_at
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), CASE WHEN %s = 'completed' THEN NOW() ELSE NULL END)
    ON CONFLICT (query_text) DO UPDATE SET
        status = EXCLUDED.status,
        pages_processed = EXCLUDED.pages_processed,
        api_calls = EXCLUDED.api_calls,
        rows_seen = EXCLUDED.rows_seen,
        rows_changed = EXCLUDED.rows_changed,
        last_error = EXCLUDED.last_error,
        last_run_at = NOW(),
        completed_at = CASE WHEN EXCLUDED.status = 'completed' THEN NOW() ELSE NULL END
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (query_text, status, pages_processed, api_calls, rows_seen, rows_changed, last_error, status),
        )
    conn.commit()


def insert_query_hits(
    conn: psycopg2.extensions.connection,
    *,
    query_text: str,
    page_number: int,
    places: list[dict],
) -> int:
    sql = """
    INSERT INTO place_query_hits (
        place_id,
        query_text,
        field_mode,
        page_number,
        rank_in_page,
        seen_at
    )
    VALUES (%s, %s, %s, %s, %s, NOW())
    ON CONFLICT (place_id, query_text, field_mode, page_number, rank_in_page) DO NOTHING
    """

    inserted = 0
    with conn.cursor() as cur:
        for rank, place in enumerate(places, start=1):
            place_id = place.get("id")
            if not place_id:
                continue
            cur.execute(sql, (place_id, query_text, FIELD_MODE, page_number, rank))
            inserted += cur.rowcount
    conn.commit()
    return inserted


def reset_checkpoints_by_keywords(
    conn: psycopg2.extensions.connection,
    keywords: list[str],
) -> int:
    if not keywords:
        return 0

    where_clause = " OR ".join(["LOWER(query_text) LIKE %s" for _ in keywords])
    params = [f"%{keyword}%" for keyword in keywords]

    sql = f"DELETE FROM places_ingest_query_checkpoints WHERE {where_clause}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        deleted = cur.rowcount
    conn.commit()
    return deleted


def upsert_places(conn: psycopg2.extensions.connection, places: list[dict], source_query: str) -> int:
    if not places:
        return 0

    sql = """
    INSERT INTO places_raw (
        place_id,
        name,
        primary_type,
        formatted_address,
        latitude,
        longitude,
        rating,
        user_rating_count,
        price_level,
        website_uri,
        business_status,
        maps_uri,
        types,
        editorial_summary,
        regular_opening_hours,
        source_query,
        source_json,
        source_updated_at,
        updated_at
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s,
        NOW(), NOW()
    )
    ON CONFLICT (place_id) DO UPDATE SET
        name = EXCLUDED.name,
        primary_type = EXCLUDED.primary_type,
        formatted_address = EXCLUDED.formatted_address,
        latitude = EXCLUDED.latitude,
        longitude = EXCLUDED.longitude,
        rating = EXCLUDED.rating,
        user_rating_count = EXCLUDED.user_rating_count,
        price_level = EXCLUDED.price_level,
        website_uri = EXCLUDED.website_uri,
        business_status = EXCLUDED.business_status,
        maps_uri = EXCLUDED.maps_uri,
        types = EXCLUDED.types,
        editorial_summary = EXCLUDED.editorial_summary,
        regular_opening_hours = EXCLUDED.regular_opening_hours,
        source_json = EXCLUDED.source_json,
        source_updated_at = NOW(),
        updated_at = NOW()
    WHERE
        places_raw.name IS DISTINCT FROM EXCLUDED.name OR
        places_raw.primary_type IS DISTINCT FROM EXCLUDED.primary_type OR
        places_raw.formatted_address IS DISTINCT FROM EXCLUDED.formatted_address OR
        places_raw.latitude IS DISTINCT FROM EXCLUDED.latitude OR
        places_raw.longitude IS DISTINCT FROM EXCLUDED.longitude OR
        places_raw.rating IS DISTINCT FROM EXCLUDED.rating OR
        places_raw.user_rating_count IS DISTINCT FROM EXCLUDED.user_rating_count OR
        places_raw.price_level IS DISTINCT FROM EXCLUDED.price_level OR
        places_raw.website_uri IS DISTINCT FROM EXCLUDED.website_uri OR
        places_raw.business_status IS DISTINCT FROM EXCLUDED.business_status OR
        places_raw.maps_uri IS DISTINCT FROM EXCLUDED.maps_uri OR
        places_raw.types IS DISTINCT FROM EXCLUDED.types OR
        places_raw.editorial_summary IS DISTINCT FROM EXCLUDED.editorial_summary OR
        places_raw.regular_opening_hours IS DISTINCT FROM EXCLUDED.regular_opening_hours OR
        places_raw.source_json IS DISTINCT FROM EXCLUDED.source_json
    """

    rows_changed = 0
    with conn.cursor() as cur:
        for place in places:
            cur.execute(sql, _place_row(place, source_query))
            rows_changed += cur.rowcount
    conn.commit()
    return rows_changed


def run() -> None:
    if not GOOGLE_KEY:
        raise RuntimeError("Missing GOOGLE_PLACES_API_KEY in environment.")
    if not DATABASE_URL:
        raise RuntimeError("Missing DATABASE_URL in environment.")

    stats = PullStats()
    all_seed_queries = build_seed_queries()
    print(f"Generated {len(all_seed_queries)} seed queries.")
    print(f"Field mode: {FIELD_MODE}; fields requested: {len(FIELDS.split(','))}")

    last_request_at: float | None = None

    with psycopg2.connect(DATABASE_URL) as conn:
        ensure_query_checkpoint_table(conn)
        ensure_query_hits_table(conn)

        if CHECKPOINT_RESET_CONTAINS:
            deleted = reset_checkpoints_by_keywords(conn, CHECKPOINT_RESET_CONTAINS)
            print(
                "Reset checkpoints by keywords "
                f"{CHECKPOINT_RESET_CONTAINS}; deleted {deleted} checkpoint rows."
            )

        if SKIP_COMPLETED_QUERIES:
            completed = get_completed_queries(conn)
            seed_queries = [
                query
                for query in all_seed_queries
                if checkpoint_key(query) not in completed and query not in completed
            ]
            stats.queries_skipped = len(all_seed_queries) - len(seed_queries)
            print(f"Skipping {stats.queries_skipped} completed queries from prior runs.")
        else:
            seed_queries = all_seed_queries

        for query in seed_queries:
            if MAX_API_CALLS > 0 and stats.api_calls >= MAX_API_CALLS:
                print("Reached PLACES_MAX_API_CALLS budget; stopping run early.")
                break

            stats.queries_processed += 1
            query_api_calls = 0
            query_rows_seen = 0
            query_rows_changed = 0
            query_pages_processed = 0
            query_fully_processed = True

            page_token: str | None = None
            for page in range(1, MAX_PAGES_PER_QUERY + 1):
                if MAX_API_CALLS > 0 and stats.api_calls >= MAX_API_CALLS:
                    print("Reached PLACES_MAX_API_CALLS budget; stopping run early.")
                    query_fully_processed = False
                    break

                _sleep_for_rate_limit(last_request_at)
                places, next_token = search_places(query=query, page_token=page_token)
                last_request_at = time.time()
                stats.api_calls += 1
                query_api_calls += 1
                query_pages_processed += 1
                if not places:
                    break

                stats.rows_seen += len(places)
                query_rows_seen += len(places)

                changed_rows = upsert_places(conn, places, source_query=query)
                stats.rows_upserted += changed_rows
                query_rows_changed += changed_rows

                insert_query_hits(
                    conn,
                    query_text=query,
                    page_number=page,
                    places=places,
                )

                print(f"[{query}] page {page}: received {len(places)} places")

                if not next_token:
                    break

                # Google page tokens may take a short delay before becoming valid.
                time.sleep(2)
                page_token = next_token

            if query_fully_processed:
                mark_query_progress(
                    conn,
                    query_text=checkpoint_key(query),
                    status="completed",
                    pages_processed=query_pages_processed,
                    api_calls=query_api_calls,
                    rows_seen=query_rows_seen,
                    rows_changed=query_rows_changed,
                )
                stats.queries_completed += 1
            else:
                mark_query_progress(
                    conn,
                    query_text=checkpoint_key(query),
                    status="incomplete",
                    pages_processed=query_pages_processed,
                    api_calls=query_api_calls,
                    rows_seen=query_rows_seen,
                    rows_changed=query_rows_changed,
                    last_error="Run stopped before query completed (budget or early termination).",
                )

            if MAX_API_CALLS > 0 and stats.api_calls >= MAX_API_CALLS:
                break

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM places_raw")
            unique_total = cur.fetchone()[0]

    print("\\nIngestion complete")
    print(f"Queries processed: {stats.queries_processed}")
    print(f"Queries completed this run: {stats.queries_completed}")
    print(f"Queries skipped from checkpoints: {stats.queries_skipped}")
    print(f"API calls made: {stats.api_calls}")
    print(f"Rows seen this run: {stats.rows_seen}")
    print(f"Rows inserted/updated this run: {stats.rows_upserted}")
    print(f"Unique places currently stored: {unique_total}")


if __name__ == "__main__":
    run()
