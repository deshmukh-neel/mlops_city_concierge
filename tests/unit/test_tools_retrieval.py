from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from app.config import ALLOWED_EMBEDDING_TABLES
from app.tools.filters import SearchFilters
from app.tools.retrieval import (
    _VIEW_FOR_TABLE,
    PlaceDetails,
    PlaceHit,
    get_details,
    nearby,
    semantic_search,
)

# --- contract test for the view mapping ------------------------------------


def test_view_for_table_covers_allowlist() -> None:
    """Every entry in ALLOWED_EMBEDDING_TABLES must have a view in the map."""
    assert set(_VIEW_FOR_TABLE.keys()) == set(ALLOWED_EMBEDDING_TABLES)


# --- smoke tests -----------------------------------------------------------


def test_pydantic_models_construct() -> None:
    hit = PlaceHit(place_id="abc", name="X", source="google_places", similarity=0.5)
    assert hit.place_id == "abc"
    details = PlaceDetails(
        place_id="abc",
        name="X",
        source="google_places",
        similarity=0.0,
        types=["bar"],
    )
    assert details.types == ["bar"]


# --- fake cursor / connection plumbing -------------------------------------


class FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.executed_sql: str = ""
        self.executed_params: list = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: list) -> None:
        self.executed_sql = sql
        self.executed_params = list(params)

    def fetchall(self) -> list[dict]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self, **_kwargs: Any) -> FakeCursor:  # ignore cursor_factory
        return self._cursor


@pytest.fixture
def patch_get_conn(mocker):
    """Returns a helper that patches get_conn() to yield a FakeConnection
    over the given rows, and exposes the FakeCursor for assertions."""

    def _patch(rows: list[dict]) -> FakeCursor:
        cursor = FakeCursor(rows)
        connection = FakeConnection(cursor)

        @contextmanager
        def fake_get_conn():
            yield connection

        mocker.patch("app.tools.retrieval.get_conn", fake_get_conn)
        return cursor

    return _patch


# --- semantic_search -------------------------------------------------------


def test_semantic_search_builds_expected_sql_with_no_filters(patch_get_conn, mocker) -> None:
    cursor = patch_get_conn(
        [
            {
                "place_id": "abc",
                "name": "Tartine",
                "primary_type": "bakery",
                "formatted_address": "600 Guerrero St",
                "latitude": 37.76,
                "longitude": -122.42,
                "rating": 4.6,
                "price_level": 2,
                "business_status": "OPERATIONAL",
                "source": "google_places",
                "similarity": 0.95,
                "snippet": "Name: Tartine\n...",
            }
        ]
    )
    mocker.patch("app.tools.retrieval.build_embedding", return_value=[0.1, 0.2, 0.3])

    hits = semantic_search("croissants", k=5)

    # business_status default OPERATIONAL + min_user_rating_count default 50
    # are both present unless explicitly disabled.
    assert "FROM place_documents" in cursor.executed_sql
    assert "ORDER BY embedding <=> %s::vector" in cursor.executed_sql
    assert "embedding_model = %s" in cursor.executed_sql
    assert "business_status = %s" in cursor.executed_sql
    assert "user_rating_count >= %s" in cursor.executed_sql

    # Last two params should be the vector literal and the LIMIT.
    assert cursor.executed_params[-1] == 5  # k * _OVERFETCH_FACTOR == 5*1
    assert cursor.executed_params[-2].startswith("[")  # vector literal

    assert len(hits) == 1
    assert hits[0].name == "Tartine"
    assert hits[0].similarity == 0.95


def test_semantic_search_injects_filter_fragments(patch_get_conn, mocker) -> None:
    cursor = patch_get_conn([])
    mocker.patch("app.tools.retrieval.build_embedding", return_value=[0.1])

    semantic_search(
        "wine bar",
        SearchFilters(
            min_rating=4.5,
            outdoor_seating=True,
            min_user_rating_count=100,
        ),
        k=3,
    )

    sql = cursor.executed_sql
    assert "rating >= %s" in sql
    assert "outdoor_seating = %s" in sql
    assert "user_rating_count >= %s" in sql
    # 4.5, 100, True should all be in params.
    assert 4.5 in cursor.executed_params
    assert 100 in cursor.executed_params
    assert True in cursor.executed_params


# --- nearby ----------------------------------------------------------------


def test_nearby_excludes_anchor_and_filters_distance(patch_get_conn) -> None:
    cursor = patch_get_conn(
        [
            {
                "place_id": "neighbor1",
                "name": "Bar Next Door",
                "primary_type": "bar",
                "formatted_address": "601 Guerrero St",
                "latitude": 37.76,
                "longitude": -122.42,
                "rating": 4.4,
                "price_level": 2,
                "business_status": "OPERATIONAL",
                "source": "google_places",
                "similarity": 0.0,
                "snippet": "...",
            }
        ]
    )

    hits = nearby("anchor_id", radius_m=500, k=10)

    sql = cursor.executed_sql
    # Anchor reads from places_raw (P15).
    assert "FROM places_raw" in sql
    # Candidate query reads from the view.
    assert "FROM place_documents pd" in sql
    # Distance filtering moved to outer WHERE per CTE rewrite (Issue #8).
    assert "WHERE dist_m <= %s" in sql
    assert "ORDER BY dist_m ASC" in sql
    # Anchor self-exclusion: anchor place_id passed twice (anchor lookup + neighbor exclusion).
    assert cursor.executed_params[0] == "anchor_id"
    assert cursor.executed_params[1] == "anchor_id"
    # radius_m and k are the last two params.
    assert cursor.executed_params[-2] == 500
    assert cursor.executed_params[-1] == 10
    assert len(hits) == 1


# --- get_details -----------------------------------------------------------


def test_get_details_returns_none_when_missing(patch_get_conn) -> None:
    patch_get_conn([])
    assert get_details("nonexistent_id") is None


def test_get_details_returns_place_details_when_found(patch_get_conn) -> None:
    patch_get_conn(
        [
            {
                "place_id": "abc",
                "name": "Tartine",
                "primary_type": "bakery",
                "types": ["bakery", "cafe"],
                "formatted_address": "600 Guerrero St",
                "latitude": 37.76,
                "longitude": -122.42,
                "rating": 4.6,
                "user_rating_count": 1234,
                "price_level": 2,
                "business_status": "OPERATIONAL",
                "website_uri": "https://tartinebakery.com",
                "maps_uri": "https://maps.google.com/?cid=...",
                "editorial_summary": "Iconic SF bakery.",
                "regular_opening_hours": {"periods": []},
                "source": "google_places",
                "snippet": "Name: Tartine\n...",
                "similarity": 0.0,
            }
        ]
    )
    details = get_details("abc")
    assert details is not None
    assert details.types == ["bakery", "cafe"]
    assert details.user_rating_count == 1234
