from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from app.tools.graph import RelatedPlace, kg_traverse


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
    """Patch get_conn() (resolved inside app.tools.retrieval._execute) to yield
    a FakeConnection over the given rows; returns the FakeCursor for asserts."""

    def _patch(rows: list[dict]) -> FakeCursor:
        cursor = FakeCursor(rows)
        connection = FakeConnection(cursor)

        @contextmanager
        def fake_get_conn():
            yield connection

        mocker.patch("app.tools.retrieval.get_conn", fake_get_conn)
        return cursor

    return _patch


def _row(**overrides: Any) -> dict:
    base = {
        "place_id": "ChIJtest_p2_aaaaaaaa",
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
        "relation_type": "NEAR",
        "weight": 42.0,
        "relation_metadata": {},
    }
    base.update(overrides)
    return base


def test_invalid_relation_raises() -> None:
    with pytest.raises(ValueError, match="Unknown relation_type"):
        kg_traverse("ChIJtest_p1_aaaaaaaa", "BOGUS")


def test_related_place_shape(patch_get_conn) -> None:
    patch_get_conn([_row(relation_type="NEAR", weight=42.0, relation_metadata={"k": "v"})])
    out = kg_traverse("ChIJtest_p1_aaaaaaaa", "NEAR", k=5)
    assert len(out) == 1
    rp = out[0]
    assert isinstance(rp, RelatedPlace)
    assert rp.relation_type == "NEAR"
    assert rp.weight == 42.0
    assert rp.relation_metadata == {"k": "v"}


def test_ordering_clause_near_ascending(patch_get_conn) -> None:
    cursor = patch_get_conn([])
    kg_traverse("ChIJtest_p1_aaaaaaaa", "NEAR", k=5)
    assert "CASE r.relation_type" in cursor.executed_sql
    assert "WHEN 'NEAR'           THEN  r.weight" in cursor.executed_sql


def test_ordering_clause_similar_vector_descending(patch_get_conn) -> None:
    cursor = patch_get_conn([])
    kg_traverse("ChIJtest_p1_aaaaaaaa", "SIMILAR_VECTOR", k=5)
    assert "WHEN 'SIMILAR_VECTOR' THEN -r.weight" in cursor.executed_sql


def test_join_drops_missing_dst(patch_get_conn) -> None:
    cursor = patch_get_conn([])
    out = kg_traverse("ChIJtest_p1_aaaaaaaa", "NEAR", k=5)
    assert out == []
    # Inner JOIN (not LEFT JOIN) so dst missing from the view is dropped.
    assert "JOIN " in cursor.executed_sql
    assert "LEFT JOIN" not in cursor.executed_sql


def test_view_name_resolved(patch_get_conn, monkeypatch) -> None:
    monkeypatch.setattr("app.tools.graph._view_name", lambda: "place_documents_v2")
    cursor = patch_get_conn([])
    kg_traverse("ChIJtest_p1_aaaaaaaa", "NEAR", k=5)
    assert "place_documents_v2" in cursor.executed_sql
    assert "JOIN place_documents pd" not in cursor.executed_sql


# --- excluded_place_ids (closure-aware swap) -----------------------------


def test_kg_traverse_excluded_place_ids_filters_at_sql_layer(patch_get_conn) -> None:
    """Closure-swap exclusion: kg_traverse must drop excluded place_ids in
    the WHERE clause at the SQL layer, not in Python."""
    cursor = patch_get_conn([])
    kg_traverse("ChIJtest_anchor_aaaa", excluded_place_ids=["ChIJ_a", "ChIJ_b"])

    # The SQL must include the exclusion guarded by a NULL check so the
    # no-exclusion call site stays backward-compatible.
    assert "pd.place_id != ALL(%s::text[])" in cursor.executed_sql
    assert ["ChIJ_a", "ChIJ_b"] in cursor.executed_params


def test_kg_traverse_no_exclusions_is_no_op(patch_get_conn) -> None:
    """Without excluded_place_ids the SQL must still emit the NULL-guarded
    clause; the parameter is None so the guard short-circuits and rows
    aren't filtered."""
    cursor = patch_get_conn([])
    kg_traverse("ChIJtest_anchor_aaaa")
    assert "IS NULL OR pd.place_id != ALL" in cursor.executed_sql
    # Excluded slot appears twice in params (NULL-guard + comparison); both None.
    assert cursor.executed_params.count(None) == 2
