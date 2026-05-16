from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool

from app.agent.tools import (
    _args_schema_for,
    all_tools,
    semantic_search,
)


def test_all_tools_returns_expected_names() -> None:
    tools = all_tools()
    assert {t.name for t in tools} == {
        "semantic_search",
        "nearby",
        "get_details",
        "kg_traverse",
        "commit_itinerary",
    }
    for tool in tools:
        assert isinstance(tool, BaseTool)
        assert tool.description  # tool docstrings flow through to LLM


def test_all_tools_returns_cached_instances() -> None:
    """all_tools() returns a fresh list view but the StructuredTool instances
    are cached at module import (Issue #16A)."""
    a = all_tools()
    b = all_tools()
    assert a is not b
    for ta, tb in zip(a, b, strict=True):
        assert ta is tb


def test_semantic_search_tool_invokes_underlying(monkeypatch) -> None:
    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("app.agent.tools._semantic_search", _fake)
    tools = {t.name: t for t in all_tools()}
    tools["semantic_search"].invoke({"query": "italian", "k": 3})
    assert captured == {"query": "italian", "filters": None, "k": 3}


def test_nearby_tool_invokes_underlying(monkeypatch) -> None:
    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("app.agent.tools._nearby", _fake)
    tools = {t.name: t for t in all_tools()}
    tools["nearby"].invoke({"place_id": "p1", "radius_m": 500, "k": 4})
    assert captured == {"place_id": "p1", "radius_m": 500, "filters": None, "k": 4}


def test_get_details_tool_invokes_underlying(monkeypatch) -> None:
    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("app.agent.tools._get_details", _fake)
    tools = {t.name: t for t in all_tools()}
    tools["get_details"].invoke({"place_id": "p1"})
    assert captured == {"place_id": "p1"}


def test_kg_traverse_tool_delegates_to_graph(monkeypatch) -> None:
    """The W2 stub is gone; the tool now delegates to app.tools.graph.kg_traverse
    (lazy-imported inside the wrapper body)."""
    captured: dict = {}

    def _fake(place_id: str, relation_type: str = "SIMILAR_VECTOR", k: int = 5):
        captured.update(place_id=place_id, relation_type=relation_type, k=k)
        return []

    monkeypatch.setattr("app.tools.graph.kg_traverse", _fake)
    tools = {t.name: t for t in all_tools()}
    result = tools["kg_traverse"].invoke({"place_id": "p1", "relation_type": "NEAR", "k": 3})
    assert result == []
    assert captured == {"place_id": "p1", "relation_type": "NEAR", "k": 3}
    assert "NOT YET AVAILABLE" not in (tools["kg_traverse"].description or "")


def test_args_schema_includes_all_params() -> None:
    schema = _args_schema_for(semantic_search)
    fields = schema.model_fields
    assert set(fields.keys()) == {"query", "filters", "k"}


def test_args_schema_raises_on_missing_annotation() -> None:
    def _bad(query, k: int = 5):  # `query` has no annotation
        return []

    with pytest.raises(TypeError, match="missing a type annotation"):
        _args_schema_for(_bad)
