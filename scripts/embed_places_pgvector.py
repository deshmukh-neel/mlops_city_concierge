#!/usr/bin/env python3
"""
Generate embeddings for places_raw rows and upsert into place_embeddings.

This script only embeds rows that are new or whose source_updated_at changed,
so it is safe to run repeatedly.

Usage:
    python scripts/embed_places_pgvector.py
    make-embed

Required env vars:
    OPENAI_API_KEY

Optional env vars:
    DATABASE_URL              Postgres/Cloud SQL connection URL
    CLOUD_SQL_INSTANCE_CONNECTION_NAME Cloud SQL instance connection name for socket auth
    CLOUD_SQL_SOCKET_DIR      Cloud SQL Unix socket directory (default: /cloudsql)
    POSTGRES_SSLMODE          Optional sslmode for env-built direct DB connections
    PLACES_EMBED_MODEL       (default: text-embedding-3-small)
    PLACES_EMBED_BATCH_SIZE  (default: 50)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

from app.config import resolve_database_url

load_dotenv()

DATABASE_URL = resolve_database_url(os.environ)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("PLACES_EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = 1000
MAX_EMBED_INPUT_CHARS_PER_REQUEST = 600_000

SERVICE_FEATURES = {
    "curbsidePickup": "curbside pickup",
    "delivery": "delivery",
    "dineIn": "dine-in",
    "reservable": "reservations",
    "takeout": "takeout",
}

DINING_FEATURES = {
    "allowsDogs": "dogs allowed",
    "goodForChildren": "good for children",
    "goodForGroups": "good for groups",
    "goodForWatchingSports": "good for watching sports",
    "liveMusic": "live music",
    "menuForChildren": "children's menu",
    "outdoorSeating": "outdoor seating",
    "restroom": "restroom",
}

FOOD_DRINK_FEATURES = {
    "servesBeer": "beer",
    "servesBreakfast": "breakfast",
    "servesBrunch": "brunch",
    "servesCocktails": "cocktails",
    "servesCoffee": "coffee",
    "servesDessert": "dessert",
    "servesDinner": "dinner",
    "servesLunch": "lunch",
    "servesVegetarianFood": "vegetarian food",
    "servesWine": "wine",
}

JSON_FLAG_GROUPS = {
    "Accessibility": "accessibilityOptions",
    "Parking": "parkingOptions",
    "Payment": "paymentOptions",
}


@dataclass
class PlaceRow:
    place_id: str
    source_updated_at: str
    text: str


def _join_nonempty(values: list[str]) -> str:
    return ", ".join(value for value in values if value)


def _localized_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
    return ""


def _humanize_key(key: str) -> str:
    words: list[str] = []
    current = ""
    for char in key:
        if char.isupper() and current:
            words.append(current)
            current = char.lower()
        else:
            current += char.lower()
    if current:
        words.append(current)
    return " ".join(words)


def _enabled_features(source_json: dict, fields: dict[str, str]) -> str:
    return _join_nonempty([label for key, label in fields.items() if source_json.get(key) is True])


def _json_flags(source_json: dict, key: str) -> str:
    value = source_json.get(key) or {}
    if not isinstance(value, dict):
        return ""
    return _join_nonempty(
        [_humanize_key(field) for field, enabled in value.items() if enabled is True]
    )


def _opening_hours_text(source_json: dict, key: str) -> str:
    value = source_json.get(key) or {}
    if not isinstance(value, dict):
        return ""
    weekday_descriptions = value.get("weekdayDescriptions") or []
    if isinstance(weekday_descriptions, list):
        return " | ".join(str(day) for day in weekday_descriptions if day)
    return ""


def _price_range_text(source_json: dict) -> str:
    price_range = source_json.get("priceRange") or {}
    if not isinstance(price_range, dict):
        return ""

    prices = []
    for key in ("startPrice", "endPrice"):
        price = price_range.get(key) or {}
        if not isinstance(price, dict):
            continue
        units = price.get("units")
        nanos = price.get("nanos")
        currency = price.get("currencyCode")
        if units is None and nanos is None:
            continue
        amount = float(units or 0) + float(nanos or 0) / 1_000_000_000
        prices.append(f"{amount:g} {currency}".strip())

    return " to ".join(prices)


def _collect_text_values(value: object, *, limit: int = 8) -> list[str]:
    texts: list[str] = []

    def visit(item: object) -> None:
        if len(texts) >= limit:
            return
        if isinstance(item, str):
            if item:
                texts.append(item)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            text = _localized_text(item)
            if text:
                texts.append(text)
            for child in item.values():
                visit(child)

    visit(value)
    return list(dict.fromkeys(texts))


def _summary_text(source_json: dict, key: str) -> str:
    value = source_json.get(key)
    return " | ".join(_collect_text_values(value))


def compose_embedding_text(record: dict) -> str:
    """
    Creates one string with all fields to generate embeddings. Only updates with new information
    """
    source_json = record.get("source_json") or {}
    opening_hours = record.get("regular_opening_hours") or {}
    weekday_descriptions = opening_hours.get("weekdayDescriptions") or []

    parts = [
        f"Name: {record.get('name') or ''}",
        f"Primary Type: {record.get('primary_type') or ''}",
        f"Address: {record.get('formatted_address') or ''}",
        f"Types: {', '.join(record.get('types') or [])}",
        f"Editorial Summary: {record.get('editorial_summary') or ''}",
        f"Rating: {record.get('rating') if record.get('rating') is not None else ''}",
        f"User Ratings: {record.get('user_rating_count') if record.get('user_rating_count') is not None else ''}",
        f"Price Level: {record.get('price_level') or ''}",
        f"Website: {record.get('website_uri') or ''}",
        f"Business Status: {record.get('business_status') or ''}",
        f"Opening Hours: {' | '.join(weekday_descriptions)}",
        f"Latitude: {record.get('latitude') if record.get('latitude') is not None else ''}",
        f"Longitude: {record.get('longitude') if record.get('longitude') is not None else ''}",
    ]

    if isinstance(source_json, dict):
        richer_parts = [
            f"Short Address: {source_json.get('shortFormattedAddress') or ''}",
            f"Primary Type ID: {source_json.get('primaryType') or ''}",
            f"Price Range: {_price_range_text(source_json)}",
            f"Current Opening Hours: {_opening_hours_text(source_json, 'currentOpeningHours')}",
            f"Regular Secondary Hours: {_opening_hours_text(source_json, 'regularSecondaryOpeningHours')}",
            f"Current Secondary Hours: {_opening_hours_text(source_json, 'currentSecondaryOpeningHours')}",
            f"Service Options: {_enabled_features(source_json, SERVICE_FEATURES)}",
            f"Dining Features: {_enabled_features(source_json, DINING_FEATURES)}",
            f"Food and Drink: {_enabled_features(source_json, FOOD_DRINK_FEATURES)}",
            f"National Phone: {source_json.get('nationalPhoneNumber') or ''}",
            f"International Phone: {source_json.get('internationalPhoneNumber') or ''}",
            f"Google Maps Links: {_summary_text(source_json, 'googleMapsLinks')}",
            f"Generative Summary: {_summary_text(source_json, 'generativeSummary')}",
            f"Review Summary: {_summary_text(source_json, 'reviewSummary')}",
            f"Neighborhood Summary: {_summary_text(source_json, 'neighborhoodSummary')}",
            f"Reviews: {_summary_text(source_json, 'reviews')}",
            f"EV Charging: {_summary_text(source_json, 'evChargeAmenitySummary')}",
            f"Fuel Options: {_summary_text(source_json, 'fuelOptions')}",
        ]
        for label, key in JSON_FLAG_GROUPS.items():
            richer_parts.append(f"{label}: {_json_flags(source_json, key)}")
        parts.extend(richer_parts)

    return "\n".join(part for part in parts if not part.endswith(": "))


def fetch_rows_to_embed(conn: psycopg2.extensions.connection, limit: int) -> list[PlaceRow]:
    sql = """
    SELECT
        p.place_id,
        p.source_updated_at,
        p.name,
        p.primary_type,
        p.formatted_address,
        p.types,
        p.editorial_summary,
        p.regular_opening_hours,
        p.rating,
        p.user_rating_count,
        p.price_level,
        p.website_uri,
        p.business_status,
        p.latitude,
        p.longitude,
        p.source_json
    FROM places_raw p
    LEFT JOIN place_embeddings e ON e.place_id = p.place_id
    WHERE e.place_id IS NULL
       OR e.source_updated_at < p.source_updated_at
    ORDER BY p.source_updated_at ASC
    LIMIT %s
    """

    rows: list[PlaceRow] = []
    with conn.cursor() as cur:
        cur.execute(sql, (limit,))
        for row in cur.fetchall():
            record = {
                "place_id": row[0],
                "source_updated_at": row[1],
                "name": row[2],
                "primary_type": row[3],
                "formatted_address": row[4],
                "types": row[5],
                "editorial_summary": row[6],
                "regular_opening_hours": row[7],
                "rating": row[8],
                "user_rating_count": row[9],
                "price_level": row[10],
                "website_uri": row[11],
                "business_status": row[12],
                "latitude": row[13],
                "longitude": row[14],
                "source_json": row[15],
            }
            rows.append(
                PlaceRow(
                    place_id=record["place_id"],
                    source_updated_at=record["source_updated_at"],
                    text=compose_embedding_text(record),
                )
            )
    return rows


def vector_to_pg(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"


def upsert_embedding(
    conn: psycopg2.extensions.connection,
    place_id: str,
    source_updated_at: str,
    text: str,
    embedding: list[float],
) -> None:
    sql = """
    INSERT INTO place_embeddings (
        place_id,
        embedding,
        embedding_model,
        embedding_text,
        embedded_at,
        source_updated_at
    )
    VALUES (%s, %s::vector, %s, %s, NOW(), %s)
    ON CONFLICT (place_id) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        embedding_model = EXCLUDED.embedding_model,
        embedding_text = EXCLUDED.embedding_text,
        embedded_at = NOW(),
        source_updated_at = EXCLUDED.source_updated_at
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (place_id, vector_to_pg(embedding), EMBED_MODEL, text, source_updated_at),
        )
    conn.commit()


def iter_embedding_batches(rows: list[PlaceRow]) -> list[list[PlaceRow]]:
    batches: list[list[PlaceRow]] = []
    current_batch: list[PlaceRow] = []
    current_chars = 0

    for row in rows:
        row_chars = len(row.text)
        if current_batch and current_chars + row_chars > MAX_EMBED_INPUT_CHARS_PER_REQUEST:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append(row)
        current_chars += row_chars

    if current_batch:
        batches.append(current_batch)

    return batches


def run() -> None:
    if not DATABASE_URL:
        raise RuntimeError("Missing DATABASE_URL in environment.")
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    with psycopg2.connect(DATABASE_URL) as conn:
        rows = fetch_rows_to_embed(conn, limit=BATCH_SIZE)
        if not rows:
            print("No new or updated places to embed.")
            return

        embedded_count = 0
        batches = iter_embedding_batches(rows)
        for index, batch in enumerate(batches, start=1):
            inputs = [row.text for row in batch]
            input_chars = sum(len(text) for text in inputs)
            print(
                f"Embedding request {index}/{len(batches)}: "
                f"{len(batch)} places, {input_chars} chars"
            )
            response = client.embeddings.create(model=EMBED_MODEL, input=inputs)

            for row, item in zip(batch, response.data, strict=True):
                upsert_embedding(
                    conn,
                    place_id=row.place_id,
                    source_updated_at=row.source_updated_at,
                    text=row.text,
                    embedding=item.embedding,
                )
                embedded_count += 1

    print(f"Embedded and upserted {embedded_count} places with model {EMBED_MODEL}.")


if __name__ == "__main__":
    run()
