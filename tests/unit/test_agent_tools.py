from __future__ import annotations

from langchain_core.tools import BaseTool

from app.agent.tools import (
    _args_schema_for,
    all_tools,
    get_details,
    kg_traverse,
    nearby,
    semantic_search,
)


def test_all_tools_returns_expected_names() -> None:
    tools = all_tools()
    assert {t.name for t in tools} == {
        "semantic_search",
        "nearby",
        "get_details",
        "kg_traverse",
    }
    for tool in tools:
        assert isinstance(tool, BaseTool)
        assert tool.description  # tool docstrings flow through to LLM


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


def test_kg_traverse_tool_returns_unavailable_stub() -> None:
    tools = {t.name: t for t in all_tools()}
    result = tools["kg_traverse"].invoke({"place_id": "p1"})
    assert result == {
        "available": False,
        "reason": "knowledge graph not yet built; use semantic_search instead",
    }


def test_args_schema_skips_ctx() -> None:
    schema = _args_schema_for(semantic_search)
    fields = schema.model_fields
    assert "ctx" not in fields
    assert "query" in fields
    assert "filters" in fields
    assert "k" in fields


def test_pydantic_ai_function_signatures_have_ctx() -> None:
    """The Pydantic AI tool functions must accept a ctx parameter so that the
    Pydantic AI side can pass in RunContext when used directly."""
    import inspect

    for fn in (semantic_search, nearby, get_details, kg_traverse):
        sig = inspect.signature(fn)
        assert "ctx" in sig.parameters
        first = next(iter(sig.parameters))
        assert first == "ctx"
