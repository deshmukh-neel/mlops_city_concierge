from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from app.tools.graph import kg_traverse


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

    def cursor(self, **_kwargs: Any) -> FakeCursor:
        return self._cursor


@pytest.fixture
def patch_get_conn(mocker):
    def _patch(rows: list[dict]) -> FakeCursor:
        cursor = FakeCursor(rows)
        connection = FakeConnection(cursor)

        @contextmanager
        def fake_get_conn():
            yield connection

        mocker.patch("app.tools.retrieval.get_conn", fake_get_conn)
        return cursor

    return _patch


def _row(weight: float, relation_type: str, **overrides: Any) -> dict:
    base = {
        "place_id": f"p-{weight}",
        "name": "Test",
        "primary_type": None,
        "formatted_address": None,
        "latitude": None,
        "longitude": None,
        "rating": None,
        "price_level": None,
        "business_status": None,
        "source": "test",
        "similarity": 0.0,
        "snippet": None,
        "relation_type": relation_type,
        "weight": weight,
        "relation_metadata": {},
    }
    base.update(overrides)
    return base


def test_near_returns_places_ordered_by_weight_asc(patch_get_conn) -> None:
    rows = [_row(100, "NEAR"), _row(50, "NEAR"), _row(200, "NEAR")]
    cursor = patch_get_conn(rows)
    out = kg_traverse("anchor", "NEAR", k=3)
    # SQL carries the asc-for-NEAR ordering (DB does the sort, not Python).
    assert "WHEN 'NEAR'           THEN  r.weight" in cursor.executed_sql
    # Rows pass through unchanged in the order the cursor provided.
    assert [r.weight for r in out] == [100, 50, 200]
    assert len(out) == 3


def test_similar_vector_returns_places_ordered_desc_in_sql(patch_get_conn) -> None:
    rows = [_row(0.9, "SIMILAR_VECTOR"), _row(0.7, "SIMILAR_VECTOR")]
    cursor = patch_get_conn(rows)
    out = kg_traverse("anchor", "SIMILAR_VECTOR", k=2)
    assert "WHEN 'SIMILAR_VECTOR' THEN -r.weight" in cursor.executed_sql
    assert [r.relation_type for r in out] == ["SIMILAR_VECTOR", "SIMILAR_VECTOR"]


def test_metadata_preserved_through_pipeline(patch_get_conn) -> None:
    meta = {"displayName": "Ferry Building", "types": ["landmark"]}
    patch_get_conn([_row(1.0, "NEAR_LANDMARK", relation_metadata=meta)])
    out = kg_traverse("anchor", "NEAR_LANDMARK", k=1)
    assert out[0].relation_metadata == meta
