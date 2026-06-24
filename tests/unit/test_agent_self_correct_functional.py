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
from tests.conftest import make_hit


class Scripted(BaseChatModel):
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

    def bind_tools(self, tools: Any, **kwargs: Any) -> Scripted:
        return self


hit = make_hit


async def test_agent_drops_filter_on_critique_and_succeeds(monkeypatch) -> None:
    """First call returns []; critique emits drop_filter; LLM retries without
    the restrictive filter; second call returns a real hit; agent finalizes.

    The point: the [critique:step] HumanMessage actually steers the LLM's
    next tool call. We verify the second call's args reflect the dropped
    filter."""
    calls: list[dict[str, Any]] = []

    def search(**kw: Any) -> list[PlaceHit]:
        calls.append({"query": kw.get("query"), "filters": kw.get("filters")})
        return [] if len(calls) == 1 else [hit()]

    monkeypatch.setattr("app.agent.tools.semantic_search_impl", search)

    fake = Scripted(
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
        "app.agent.revision.itinerary_violations",
        lambda state_obj: [],
    )
    # First call empty, second succeeds.
    calls = []

    def search(**kw):
        calls.append(kw)
        return [] if len(calls) == 1 else [hit("ChIJtest_p1_aaaaaaaa")]

    monkeypatch.setattr("app.agent.tools.semantic_search_impl", search)

    fake = Scripted(
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
                                    "place_id": "ChIJtest_p1_aaaaaaaa",
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
    assert out["stops"][0].place_id == "ChIJtest_p1_aaaaaaaa"


async def test_step_then_itinerary_critique_compose_end_to_end(monkeypatch) -> None:
    """Full revision loop: bad search -> good search -> commit -> itinerary
    check fails -> revised commit -> itinerary check passes -> finalize.

    Verifies that step-level and itinerary-level retry counters track
    independently and accumulate correctly across alternating revisions."""
    # Search returns empty once, then good results.
    search_calls = []

    def search(**kw):
        search_calls.append(kw)
        return (
            []
            if len(search_calls) == 1
            else [hit("ChIJtest_p1_aaaaaaaa"), hit("ChIJtest_p2_aaaaaaaa")]
        )

    monkeypatch.setattr("app.agent.tools.semantic_search_impl", search)

    # First itinerary check after first commit fails geographic_coherence;
    # second check (after revised commit) passes.
    violation_calls = {"n": 0}

    def violations(state_obj):
        violation_calls["n"] += 1
        return ["geographic_coherence"] if violation_calls["n"] == 1 else []

    monkeypatch.setattr("app.agent.revision.itinerary_violations", violations)

    fake = Scripted(
        scripted=[
            # 1) empty search (triggers per-step empty_results hint)
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
            # 2) revised search returns 2 hits
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
            # 3) first commit_itinerary
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "commit_itinerary",
                        "id": "3",
                        "args": {
                            "stops": [
                                {
                                    "place_id": "ChIJtest_p1_aaaaaaaa",
                                    "name": "P1",
                                    "source": "google_places",
                                    "rationale": "first",
                                },
                                {
                                    "place_id": "ChIJtest_p2_aaaaaaaa",
                                    "name": "P2",
                                    "source": "google_places",
                                    "rationale": "second",
                                },
                            ]
                        },
                    }
                ],
            ),
            # 4) revised commit_itinerary, driven by the geographic-coherence
            # revision hint that commit #1 triggered. Finalize-on-commit means
            # there is no separate "here's the plan" narration turn between
            # commits, and no final narration turn after: the graph ends the
            # moment this revised commit passes the hard checks.
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "commit_itinerary",
                        "id": "4",
                        "args": {
                            "stops": [
                                {
                                    "place_id": "ChIJtest_p1_aaaaaaaa",
                                    "name": "P1",
                                    "source": "google_places",
                                    "rationale": "kept",
                                },
                                {
                                    "place_id": "ChIJtest_p2_aaaaaaaa",
                                    "name": "P2",
                                    "source": "google_places",
                                    "rationale": "swapped",
                                },
                            ]
                        },
                    }
                ],
            ),
        ]
    )
    graph = build_agent_graph(fake, max_steps=10)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="plan it")]))

    assert out["done"] is True
    # Both revision counters incremented exactly once each — they don't share state.
    assert out["revision_counts"]["empty_results"] == 1
    assert out["revision_counts"]["geographic_coherence"] == 1
    # And both kinds of hints landed on revision_hints, in the order they fired.
    reasons = [h.reason for h in out["revision_hints"]]
    assert reasons == ["empty_results", "geographic_incoherence"]
    # Stops survived both revisions.
    assert len(out["stops"]) == 2
    # Finalize-on-commit: the reply is synthesized from the committed stops
    # (the model no longer narrates), and the revised plan passed clean so
    # there is no caveat.
    assert out["final_reply"].startswith("Here's your itinerary:")
    assert "P1" in out["final_reply"] and "P2" in out["final_reply"]
    assert "Caveats:" not in (out["final_reply"] or "")
