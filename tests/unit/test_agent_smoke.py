"""Smoke tests for the agent package — modules import cleanly, public objects
instantiate, and the graph compiles end-to-end."""

from __future__ import annotations

import importlib

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class _NoopLLM(BaseChatModel):
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


async def test_build_agent_graph_compiles_and_runs_happy_path() -> None:
    from app.agent.graph import build_agent_graph
    from app.agent.state import ItineraryState

    graph = build_agent_graph(_NoopLLM(), max_steps=2)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
    assert out["final_reply"] == "ok"


def test_state_to_cards_smoke() -> None:
    """Smoke: state_to_cards always returns a list of PlaceCard-shaped dicts."""
    from app.agent.io import state_to_cards
    from app.agent.state import ItineraryState

    cards = state_to_cards(ItineraryState(final_reply="hi"))
    assert cards == []
