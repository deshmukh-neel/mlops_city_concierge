"""Functional end-to-end tests for self-correction.

These exercise the full graph with a scripted LLM that *actually revises* in
response to critique hints — closer to how a real LLM would behave than the
single-bounce tests in test_agent_self_correct.py.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.graph import build_agent_graph
from app.agent.state import ItineraryState
from app.tools.retrieval import PlaceHit


class _Scripted(BaseChatModel):
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
            raise RuntimeError("scripted exhausted")
        return ChatResult(generations=[ChatGeneration(message=self.scripted.pop(0))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> _Scripted:
        return self


def _hit(place_id: str = "p1") -> PlaceHit:
    return PlaceHit(
        place_id=place_id,
        name=place_id.upper(),
        source="google_places",
        similarity=0.9,
        latitude=37.78,
        longitude=-122.41,
        rating=4.5,
        price_level="PRICE_LEVEL_MODERATE",
        business_status="OPERATIONAL",
        primary_type="restaurant",
        formatted_address="123 Main",
        snippet=None,
    )


async def test_agent_drops_filter_on_critique_and_succeeds(monkeypatch) -> None:
    """First call returns []; critique emits drop_filter; LLM retries without
    the restrictive filter; second call returns a real hit; agent finalizes.

    The point: the [critique:step] HumanMessage actually steers the LLM's
    next tool call. We verify the second call's args reflect the dropped
    filter."""
    calls: list[dict[str, Any]] = []

    def _search(**kw: Any) -> list[PlaceHit]:
        calls.append({"query": kw.get("query"), "filters": kw.get("filters")})
        return [] if len(calls) == 1 else [_hit()]

    monkeypatch.setattr("app.agent.tools._semantic_search", _search)

    fake = _Scripted(
        scripted=[
            # First plan: search with a restrictive filter -> empty.
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "1",
                        "args": {
                            "query": "italian",
                            "filters": {"price_level_max": 1, "neighborhood": "Mission"},
                        },
                    }
                ],
            ),
            # Second plan (after [critique:step] empty_results, drop_filter):
            # the LLM drops the price_level_max constraint.
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "2",
                        "args": {
                            "query": "italian",
                            "filters": {"neighborhood": "Mission"},
                        },
                    }
                ],
            ),
            # Final: finish.
            AIMessage(content="found one place: P1", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=6)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="italian in mission")]))

    assert out["done"] is True
    assert out["final_reply"] == "found one place: P1"
    # Two tool calls happened, the second one with relaxed filters.
    assert len(calls) == 2
    assert calls[0]["filters"].price_level_max == 1
    assert calls[1]["filters"].price_level_max is None
    # Exactly one hint emitted.
    hints = out["revision_hints"]
    assert len(hints) == 1
    assert hints[0].reason == "empty_results"


async def test_step_and_itinerary_critiques_compose(monkeypatch) -> None:
    """Step critique on bad search + itinerary critique on bad commit, both
    in one run. Verifies that revision_counts tracks them as separate
    categories (each gets its own MAX_REVISIONS_PER_REASON budget)."""
    monkeypatch.setattr(
        "app.agent.graph.itinerary_violations",
        lambda _state: [],
    )
    # First call empty, second succeeds.
    calls = []

    def _search(**kw):
        calls.append(kw)
        return [] if len(calls) == 1 else [_hit("p1")]

    monkeypatch.setattr("app.agent.tools._semantic_search", _search)

    fake = _Scripted(
        scripted=[
            # 1) empty search
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "1",
                        "args": {"query": "x", "filters": {"price_level_max": 1}},
                    }
                ],
            ),
            # 2) revised search succeeds
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "2",
                        "args": {"query": "x", "filters": {}},
                    }
                ],
            ),
            # 3) commit_itinerary
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "commit_itinerary",
                        "id": "3",
                        "args": {
                            "stops": [
                                {
                                    "place_id": "p1",
                                    "name": "P1",
                                    "source": "google_places",
                                    "rationale": "good fit",
                                }
                            ]
                        },
                    }
                ],
            ),
            # 4) finalize summary
            AIMessage(content="here is your plan", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=8)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))

    assert out["done"] is True
    # The empty_results hint fired exactly once.
    assert out["revision_counts"].get("empty_results") == 1
    # One stop committed, no itinerary violations triggered (stub returned []).
    assert len(out["stops"]) == 1
    assert out["stops"][0].place_id == "p1"
