#!/usr/bin/env python3
"""
Generate cleaned embeddings for places_raw rows and upsert into place_embeddings_v2.

This is a fork of scripts/embed_places_pgvector.py with a rewritten
compose_embedding_text. The two scripts intentionally coexist so we can
re-run either one and compare retrieval quality. See implementation_plan/
james/w0a_embeddings_v2.md for the rationale.

Usage:
    python scripts/embed_places_pgvector_v2.py
    make embed-v2

Required env vars:
    OPENAI_API_KEY

Optional env vars:
    DATABASE_URL              Postgres/Cloud SQL connection URL
    CLOUD_SQL_INSTANCE_CONNECTION_NAME Cloud SQL instance connection name for socket auth
    CLOUD_SQL_SOCKET_DIR      Cloud SQL Unix socket directory (default: /cloudsql)
    POSTGRES_SSLMODE          Optional sslmode for env-built direct DB connections
    PLACES_EMBED_MODEL       (default: text-embedding-3-small)
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
TARGET_TABLE = "place_embeddings_v2"

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


# ---- text composition ------------------------------------------------------


def _summary_object_text(value: object) -> str:
    """Extract the human-readable text from Places v1 summary objects.

    Shape: {"text": {"text": "...", "languageCode": "en-US"},
            "disclosureText": {...}, "flagContentUri": "...", "reviewsUri": "..."}
    We want ONLY the inner text; everything else is noise (URLs, language tags,
    "Summarized with Gemini" boilerplate).
    """
    if not isinstance(value, dict):
        return ""
    inner = value.get("text") or value.get("overview")
    if isinstance(inner, dict):
        text = inner.get("text")
        if isinstance(text, str):
            return text
    if isinstance(inner, str):
        return inner
    return ""


def _neighborhood_from_address_components(source_json: dict) -> str:
    """Extract the structured neighborhood from addressComponents.

    Shape: addressComponents: [{"types": ["neighborhood", "political"],
                                "longText": "Mission Bay", ...}, ...]
    """
    components = source_json.get("addressComponents") or []
    if not isinstance(components, list):
        return ""
    for component in components:
        if not isinstance(component, dict):
            continue
        types = component.get("types") or []
        if "neighborhood" in types:
            text = component.get("longText") or component.get("shortText")
            if isinstance(text, str):
                return text
    return ""


def _containing_area_names(source_json: dict) -> str:
    """Names of containing areas from addressDescriptor.areas[].displayName.

    Shape: addressDescriptor.areas: [{"displayName": {"text": "Mission Bay", ...},
                                       "containment": "WITHIN"}, ...]
    """
    descriptor = source_json.get("addressDescriptor") or {}
    if not isinstance(descriptor, dict):
        return ""
    areas = descriptor.get("areas") or []
    if not isinstance(areas, list):
        return ""
    names: list[str] = []
    for area in areas:
        if not isinstance(area, dict):
            continue
        text = _localized_text(area.get("displayName"))
        if text and text not in names:
            names.append(text)
    return ", ".join(names)


def _nearby_landmark_names(source_json: dict, max_landmarks: int = 5) -> str:
    """Names of nearby landmarks (no distances, no place IDs).

    Distance + place IDs are consumed by W7 (knowledge graph) directly from
    source_json — they are KG inputs, not embedding inputs.
    """
    descriptor = source_json.get("addressDescriptor") or {}
    if not isinstance(descriptor, dict):
        return ""
    landmarks = descriptor.get("landmarks") or []
    if not isinstance(landmarks, list):
        return ""
    names: list[str] = []
    for landmark in landmarks[:max_landmarks]:
        if not isinstance(landmark, dict):
            continue
        text = _localized_text(landmark.get("displayName"))
        if text:
            names.append(text)
    return ", ".join(names)


def compose_embedding_text_v2(record: dict) -> str:
    """Cleaned embedding text. See w0a_embeddings_v2.md for the role taxonomy.

    KEEP (Role 1 — semantic signal):
      name, primary_type, types, editorial_summary, generative summary text,
      review summary text, service / dining / food-drink boolean labels,
      neighborhood, containing areas, nearby landmark names, accessibility /
      parking / payment labels.

    DROP (Role 2 / Role 3 — facts and action payloads, do NOT embed):
      rating, user_rating_count, price_level, price_range, opening hours text,
      lat/lng, business_status, phone numbers, all URLs (websiteUri,
      googleMapsUri, googleMapsLinks, flagContentUri, reviewsUri inside
      summaries), language codes, "Summarized with Gemini" disclosure text.
    """
    source_json = record.get("source_json") or {}
    if not isinstance(source_json, dict):
        source_json = {}

    parts: list[str] = []

    # --- core identification ------------------------------------------------
    parts.append(f"Name: {record.get('name') or ''}")
    parts.append(f"Primary Type: {record.get('primary_type') or ''}")
    types = record.get("types") or []
    if types:
        parts.append(f"Types: {', '.join(types)}")

    # --- geographic context (names only, never numbers) ---------------------
    neighborhood = _neighborhood_from_address_components(source_json)
    if neighborhood:
        parts.append(f"Neighborhood: {neighborhood}")
    containing = _containing_area_names(source_json)
    if containing:
        parts.append(f"Containing Areas: {containing}")
    landmarks = _nearby_landmark_names(source_json)
    if landmarks:
        parts.append(f"Nearby Landmarks: {landmarks}")

    # --- editorial / generative / review prose ------------------------------
    editorial = record.get("editorial_summary") or _summary_object_text(
        source_json.get("editorialSummary")
    )
    if editorial:
        parts.append(f"Editorial Summary: {editorial}")
    generative = _summary_object_text(source_json.get("generativeSummary"))
    if generative:
        parts.append(f"Generative Summary: {generative}")
    review_summary = _summary_object_text(source_json.get("reviewSummary"))
    if review_summary:
        parts.append(f"Review Summary: {review_summary}")

    # --- amenities (boolean labels only — no values, no JSON) ---------------
    service = _enabled_features(source_json, SERVICE_FEATURES)
    if service:
        parts.append(f"Service Options: {service}")
    dining = _enabled_features(source_json, DINING_FEATURES)
    if dining:
        parts.append(f"Dining Features: {dining}")
    food_drink = _enabled_features(source_json, FOOD_DRINK_FEATURES)
    if food_drink:
        parts.append(f"Food and Drink: {food_drink}")
    for label, key in JSON_FLAG_GROUPS.items():
        flags = _json_flags(source_json, key)
        if flags:
            parts.append(f"{label}: {flags}")

    # --- raw reviews (only if ingest started capturing places.reviews) ------
    reviews = source_json.get("reviews") or []
    if isinstance(reviews, list) and reviews:
        review_texts: list[str] = []
        for review in reviews[:5]:
            if not isinstance(review, dict):
                continue
            text = _localized_text(review.get("text"))
            if text:
                review_texts.append(text)
        if review_texts:
            parts.append("Reviews: " + " | ".join(review_texts))

    return "\n".join(part for part in parts if not part.endswith(": "))


def fetch_rows_to_embed(conn: psycopg2.extensions.connection, limit: int) -> list[PlaceRow]:
    # TARGET_TABLE is a module-level constant, not user input.
    sql = f"""
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
    LEFT JOIN {TARGET_TABLE} e ON e.place_id = p.place_id
    WHERE e.place_id IS NULL
       OR e.source_updated_at < p.source_updated_at
    ORDER BY p.source_updated_at ASC
    LIMIT %s
    """  # noqa: S608

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
                    text=compose_embedding_text_v2(record),
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
    # TARGET_TABLE is a module-level constant, not user input.
    sql = f"""
    INSERT INTO {TARGET_TABLE} (
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
    """  # noqa: S608

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
            print(f"No new or updated places to embed into {TARGET_TABLE}.")
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

    print(
        f"Embedded and upserted {embedded_count} places into {TARGET_TABLE} "
        f"with model {EMBED_MODEL}."
    )


if __name__ == "__main__":
    run()
