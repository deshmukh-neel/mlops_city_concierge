"""Retrieval tools the agent calls.

Talks to the place_documents view. Reuses build_embedding() from app.retriever
so the OpenAI embedding code stays in one place.
"""

from __future__ import annotations

from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from app.config import ALLOWED_EMBEDDING_TABLES, get_settings
from app.db import get_conn
from app.retriever import build_embedding, vector_to_pg
from app.tools.filters import SearchFilters, compile_filters

# Maps each entry in ALLOWED_EMBEDDING_TABLES (app/config.py) to the view that
# joins that embedding table. The contract test in tests/unit/test_tools_retrieval.py
# enforces that this dict's keys match the allowlist exactly.
_VIEW_FOR_TABLE: dict[str, str] = {
    "place_embeddings": "place_documents",
    "place_embeddings_v2": "place_documents_v2",
}


# When W6 evals show recall regressing on tightly-filtered queries, bump this
# so semantic_search retrieves k * _OVERFETCH_FACTOR rows from HNSW and lets the
# WHERE clauses filter inside the over-fetched set. Default 1 = no over-fetch.
_OVERFETCH_FACTOR: int = 1


class PlaceHit(BaseModel):
    place_id: str
    name: str
    primary_type: str | None = None
    formatted_address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    rating: float | None = None
    price_level: int | None = None
    business_status: str | None = None
    source: str
    similarity: float
    snippet: str | None = None


class PlaceDetails(PlaceHit):
    types: list[str] = []
    user_rating_count: int | None = None
    website_uri: str | None = None
    maps_uri: str | None = None
    editorial_summary: str | None = None
    regular_opening_hours: dict = {}


def _view_name() -> str:
    settings = get_settings()
    # The embedding_table validator (app/config.py) guarantees membership in
    # ALLOWED_EMBEDDING_TABLES, so this lookup cannot raise in normal use.
    return _VIEW_FOR_TABLE[settings.embedding_table]


def _execute(sql: str, params: list) -> list[dict]:
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def semantic_search(
    query: str,
    filters: SearchFilters | None = None,
    k: int = 10,
) -> list[PlaceHit]:
    """Vector similarity over place_documents with optional structured filters."""
    settings = get_settings()
    filters = filters or SearchFilters()
    where_fragment, filter_params = compile_filters(filters)
    embedding = build_embedding(query, settings.openai_embedding_model)
    vector_literal = vector_to_pg(embedding)
    view = _view_name()  # validated by ALLOWED_EMBEDDING_TABLES — safe to f-string

    sql = f"""
        SELECT
            place_id, name, primary_type, formatted_address,
            latitude, longitude, rating, price_level, business_status,
            source,
            1 - (embedding <=> %s::vector) AS similarity,
            LEFT(embedding_text, 400) AS snippet
        FROM {view}
        WHERE embedding_model = %s
        {where_fragment}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """  # noqa: S608

    params = [
        vector_literal,
        settings.openai_embedding_model,
        *filter_params,
        vector_literal,
        k * _OVERFETCH_FACTOR,
    ]
    rows = _execute(sql, params)
    return [PlaceHit(**row) for row in rows[:k]]


def nearby(
    place_id: str,
    radius_m: int = 800,
    filters: SearchFilters | None = None,
    k: int = 10,
) -> list[PlaceHit]:
    """Geographic 'within radius_m' search around an anchor place.

    Anchor coords come from places_raw directly to skip the embedding JOIN.
    The neighbor SELECT does not project the embedding column — we don't need
    1536-dim vectors in the result payload.
    """
    filters = filters or SearchFilters()
    where_fragment, filter_params = compile_filters(filters)
    view = _view_name()  # validated allowlist member — safe to f-string

    sql = f"""
        WITH anchor AS (
            SELECT latitude, longitude
            FROM places_raw
            WHERE place_id = %s
            LIMIT 1
        ),
        candidates AS (
            SELECT
                pd.place_id, pd.name, pd.primary_type, pd.formatted_address,
                pd.latitude, pd.longitude, pd.rating, pd.price_level,
                pd.business_status, pd.source,
                LEFT(pd.embedding_text, 400) AS snippet,
                6371000 * 2 * ASIN(SQRT(
                    POWER(SIN(RADIANS(pd.latitude  - a.latitude)  / 2), 2) +
                    COS(RADIANS(a.latitude)) * COS(RADIANS(pd.latitude)) *
                    POWER(SIN(RADIANS(pd.longitude - a.longitude) / 2), 2)
                )) AS dist_m
            FROM {view} pd, anchor a
            WHERE pd.place_id <> %s
            {where_fragment}
        )
        SELECT
            place_id, name, primary_type, formatted_address,
            latitude, longitude, rating, price_level, business_status,
            source,
            0.0 AS similarity,
            snippet
        FROM candidates
        WHERE dist_m <= %s
        ORDER BY dist_m ASC
        LIMIT %s
    """  # noqa: S608

    params = [place_id, place_id, *filter_params, radius_m, k]
    rows = _execute(sql, params)
    return [PlaceHit(**row) for row in rows]


def get_details(place_id: str) -> PlaceDetails | None:
    view = _view_name()  # validated allowlist member — safe to f-string
    sql = f"""
        SELECT
            place_id, name, primary_type, types, formatted_address,
            latitude, longitude, rating, user_rating_count, price_level,
            business_status, website_uri, maps_uri, editorial_summary,
            regular_opening_hours, source,
            LEFT(embedding_text, 800) AS snippet,
            0.0 AS similarity
        FROM {view}
        WHERE place_id = %s
        LIMIT 1
    """  # noqa: S608
    rows = _execute(sql, [place_id])
    if not rows:
        return None
    return PlaceDetails(**rows[0])


# Re-export for the test that asserts the mapping covers the allowlist.
__all__ = [
    "PlaceHit",
    "PlaceDetails",
    "semantic_search",
    "nearby",
    "get_details",
    "_VIEW_FOR_TABLE",
    "ALLOWED_EMBEDDING_TABLES",
]
