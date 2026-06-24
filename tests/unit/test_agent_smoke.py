"""Smoke tests for the agent package — modules import cleanly, public objects
instantiate, and the graph compiles end-to-end."""

from __future__ import annotations

import importlib

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class NoopLLM(BaseChatModel):
    """Minimal LLM stand-in: returns a single empty AIMessage with no tool calls."""

    @property
    def _llm_type(self) -> str:
        return "noop"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok", tool_calls=[]))]
        )

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize(
    "module",
    [
        "app.agent",
        "app.agent.state",
        "app.agent.prompts",
        "app.agent.tools",
        "app.agent.graph",
        "app.agent.planning",
    ],
)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_all_tools_instantiates() -> None:
    from app.agent.tools import all_tools

    tools = all_tools()
    assert len(tools) == 5


def test_retrieval_tools_expose_slot_index_schema() -> None:
    from app.agent.tools import all_tools

    tools = {tool.name: tool for tool in all_tools()}

    for name in ("semantic_search", "nearby"):
        field = tools[name].args_schema.model_fields["slot_index"]
        assert field.default is None


def test_retrieval_tools_ignore_slot_index_for_underlying_calls(monkeypatch) -> None:
    from app.agent.tools import all_tools

    captured: dict[str, dict] = {}

    def fake_semantic_search(**kwargs):
        captured["semantic_search"] = kwargs
        return []

    def fake_nearby(**kwargs):
        captured["nearby"] = kwargs
        return []

    monkeypatch.setattr("app.agent.tools.semantic_search_impl", fake_semantic_search)
    monkeypatch.setattr("app.agent.tools.nearby_impl", fake_nearby)

    tools = {tool.name: tool for tool in all_tools()}
    assert tools["semantic_search"].invoke({"query": "omakase", "slot_index": 0}) == []
    assert tools["nearby"].invoke({"place_id": "ChIJtest_p1_aaaaaaaa", "slot_index": 1}) == []

    assert captured["semantic_search"] == {"query": "omakase", "filters": None, "k": 8}
    assert captured["nearby"] == {
        "place_id": "ChIJtest_p1_aaaaaaaa",
        "radius_m": 800,
        "filters": None,
        "k": 8,
    }


async def test_build_agent_graph_compiles_and_runs_happy_path() -> None:
    from app.agent.graph import build_agent_graph
    from app.agent.state import ItineraryState

    graph = build_agent_graph(NoopLLM(), max_steps=2)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
    assert out["final_reply"] == "ok"


def test_state_to_cards_smoke() -> None:
    """Smoke: state_to_cards always returns a list of PlaceCard-shaped dicts."""
    from app.agent.io import state_to_cards
    from app.agent.state import ItineraryState

    cards = state_to_cards(ItineraryState(final_reply="hi"))
    assert cards == []
