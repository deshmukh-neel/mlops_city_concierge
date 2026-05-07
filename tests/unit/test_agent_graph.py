"""Integration of the graph with a fake LLM that emits a scripted sequence
of tool calls. Verifies plan->act->critique loops correctly and terminates."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.graph import build_agent_graph
from app.agent.state import ItineraryState
from app.tools.retrieval import PlaceHit


class _ScriptedLLM(BaseChatModel):
    """Test double that returns scripted AIMessages in order."""

    scripted: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.scripted:
            raise RuntimeError("scripted responses exhausted")
        msg = self.scripted.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedLLM:
        return self


def _make_fake(scripted: list[AIMessage]) -> _ScriptedLLM:
    return _ScriptedLLM(scripted=list(scripted))


def test_graph_terminates_on_no_tool_call() -> None:
    fake = _make_fake([AIMessage(content="here is your plan", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = graph.invoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
    assert out["final_reply"] == "here is your plan"


def test_graph_executes_tool_and_continues(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **kw: [
            PlaceHit(
                place_id="p1",
                name="X",
                source="google_places",
                similarity=0.9,
                latitude=None,
                longitude=None,
                rating=4.5,
                price_level="PRICE_LEVEL_MODERATE",
                business_status="OPERATIONAL",
                primary_type="restaurant",
                formatted_address="123 Main",
                snippet=None,
            )
        ],
    )
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "1",
                        "args": {"query": "italian", "k": 3},
                    }
                ],
            ),
            AIMessage(content="found one place", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = graph.invoke(ItineraryState(messages=[HumanMessage(content="italian please")]))
    assert out["step_count"] == 1
    assert "semantic_search" in out["scratch"]
    assert out["scratch"]["semantic_search"][0]["args"] == {"query": "italian", "k": 3}
    assert out["done"] is True


def test_graph_respects_max_steps() -> None:
    looping = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "id": str(i),
                    "args": {"query": "x"},
                }
            ],
        )
        for i in range(20)
    ]
    fake = _make_fake(looping)
    graph = build_agent_graph(fake, max_steps=3)
    # Patch the underlying retrieval so the tool calls don't hit the DB.
    import app.agent.tools as tools_mod

    tools_mod._semantic_search = lambda **kw: []  # type: ignore[assignment]

    out = graph.invoke(ItineraryState(messages=[HumanMessage(content="x")]))
    assert out["step_count"] == 3
    assert out["done"] is True
    assert out["final_reply"]


def test_graph_handles_unknown_tool_name(monkeypatch) -> None:
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "not_a_real_tool",
                        "id": "1",
                        "args": {},
                    }
                ],
            ),
            AIMessage(content="recovered", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = graph.invoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
    assert out["final_reply"] == "recovered"


def test_graph_records_tool_exception_in_scratch(monkeypatch) -> None:
    def _boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.agent.tools._semantic_search", _boom)

    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "1",
                        "args": {"query": "italian"},
                    }
                ],
            ),
            AIMessage(content="apologies, retrieval failed", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = graph.invoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    scratch = out["scratch"]["semantic_search"][0]
    assert "error" in scratch["result"]
    assert "db down" in scratch["result"]["error"]
    assert out["done"] is True
