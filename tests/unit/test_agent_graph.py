"""Integration of the graph with a fake LLM that emits a scripted sequence
of tool calls. Verifies plan->act->critique loops correctly and terminates.

tests/unit/fixtures/reason_04_prune_baseline.json is the REASON-06 byte-identity
baseline for the gpt-4o-mini (empty additional_kwargs, no _reasoning_state) path
through `_prune_for_llm` + `NoOpAdapter().replay_reasoning_state(...)`. The
fixture was generated at Phase 8 commit time from this very test file's input
list. Regenerate ONLY on an INTENTIONAL change to `_prune_for_llm` or
`NoOpAdapter` via:

    poetry run python tests/unit/test_agent_graph.py --regen-reason-04-fixture

and document the change in the fixture commit message AND the Phase 8 SUMMARY
(D-08-15).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import psycopg2
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.adapters import NoOpAdapter
from app.agent.commit import commit_stops, enrich_stops_with_booking
from app.agent.graph import _prune_for_llm, build_agent_graph
from app.agent.state import ItineraryState, Stop, UserConstraints
from app.tools.booking import BookingProposal
from app.tools.directions import DirectionsLeg, DirectionsResult
from app.tools.retrieval import PlaceDetails, PlaceHit


def _details_for(place_id: str, name: str | None = None) -> PlaceDetails:
    """Minimal PlaceDetails fixture for booking-enrichment tests. Booking
    construction itself is mocked via propose_booking_from_details; only the
    place_id needs to be load-bearing here."""
    return PlaceDetails(
        place_id=place_id,
        name=name or f"Place {place_id}",
        source="google_places",
        similarity=0.0,
    )


def _rich_details_for(place_id: str) -> PlaceDetails:
    """PlaceDetails carrying address/rating/price for card-field tests."""
    return PlaceDetails(
        place_id=place_id,
        name=f"Place {place_id}",
        primary_type="restaurant",
        source="google_places",
        similarity=0.0,
        formatted_address=f"{place_id} Main St, San Francisco",
        rating=4.4,
        price_level="PRICE_LEVEL_MODERATE",
        latitude=37.785,
        longitude=-122.404,
    )


def _patch_details_many_for(place_ids: list[str]):
    """Patch get_details_many to return a minimal PlaceDetails for each id.
    Returns the patcher so tests can also assert call_count if they care."""
    return patch(
        "app.agent.commit.get_details_many",
        return_value={pid: _details_for(pid) for pid in place_ids},
    )


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


async def test_graph_terminates_on_no_tool_call() -> None:
    fake = _make_fake([AIMessage(content="here is your plan", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
    assert out["final_reply"] == "here is your plan"


async def test_graph_retries_unneeded_stop_count_clarification(monkeypatch) -> None:
    """If the caller has already parsed an explicit stop count, the model
    should not be allowed to end the request by asking for that same count."""
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [
            PlaceHit(
                place_id="ChIJtest_p1_aaaaaaaa",
                name="Place p1",
                source="google_places",
                similarity=0.9,
            )
        ],
    )
    fake = _make_fake(
        [
            AIMessage(content="How many stops would you like?", tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "search-1",
                        "args": {"query": "romantic dinner in Japantown"},
                    }
                ],
            ),
            AIMessage(content="planning from results", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)

    out = await graph.ainvoke(
        ItineraryState(
            messages=[HumanMessage(content="plan a 3-stop date night")],
            constraints=UserConstraints(num_stops=3),
        )
    )

    assert out["done"] is True
    assert out["final_reply"] == "planning from results"
    assert "semantic_search" in out["scratch"]


async def test_graph_retries_finalize_without_stops_regardless_of_wording(monkeypatch) -> None:
    """#3C: the retry trigger is structural, not lexical. When num_stops is set
    and the model finalizes with no stops committed, nudge it — regardless of
    what the AI's text message says. The old string-match heuristic missed
    phrasings like "I'd love to know the number of places..." Drop the
    heuristic; the structural condition is sufficient and unambiguous."""
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [
            PlaceHit(
                place_id="ChIJtest_p1_aaaaaaaa",
                name="Place p1",
                source="google_places",
                similarity=0.9,
            )
        ],
    )
    fake = _make_fake(
        [
            # Non-stereotyped phrasing — old heuristic would miss this.
            AIMessage(content="To narrow it down, what vibe are you going for?", tool_calls=[]),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "search-1",
                        "args": {"query": "romantic dinner in Japantown"},
                    }
                ],
            ),
            AIMessage(content="planning from results", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(
        ItineraryState(
            messages=[HumanMessage(content="plan a 3-stop date night")],
            constraints=UserConstraints(num_stops=3),
        )
    )
    # Retry fired, model produced a tool call on its second turn, graph ended cleanly.
    assert out["done"] is True
    assert "semantic_search" in out["scratch"]


async def test_graph_does_not_retry_when_num_stops_unset(monkeypatch) -> None:
    """No nudge when there's no explicit count — the model is allowed to
    clarify ambiguous requests."""
    fake = _make_fake([AIMessage(content="How many stops?", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(
        ItineraryState(messages=[HumanMessage(content="plan something")]),
    )
    # Without num_stops, the clarifying question is the legitimate final reply.
    assert out["done"] is True
    assert out["final_reply"] == "How many stops?"


async def test_graph_retry_is_one_shot(monkeypatch) -> None:
    """If after the nudge the model STILL finalizes empty, the retry doesn't
    fire a second time — the conversation ends with whatever the model said."""
    fake = _make_fake(
        [
            AIMessage(content="What's your budget?", tool_calls=[]),
            AIMessage(content="Could you confirm your neighborhood?", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(
        ItineraryState(
            messages=[HumanMessage(content="plan a 3-stop date night")],
            constraints=UserConstraints(num_stops=3),
        )
    )
    assert out["done"] is True
    # The second clarifying turn becomes the final reply — retry did not loop.
    assert out["final_reply"] == "Could you confirm your neighborhood?"


async def test_graph_injects_explicit_stop_count_context() -> None:
    # Two messages: the first is the model's empty-finalize, which under #3C
    # triggers the structural retry nudge (num_stops set + no stops yet); the
    # second is the model's response to that nudge. We only care that the
    # system prompt contained the deterministic stop-count context.
    fake = _make_fake(
        [
            AIMessage(content="ok", tool_calls=[]),
            AIMessage(content="planning", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(
        ItineraryState(
            messages=[HumanMessage(content="plan a 3-stop date night")],
            constraints=UserConstraints(num_stops=3),
        )
    )

    system_messages = [m for m in out["messages"] if isinstance(m, SystemMessage)]
    assert len(system_messages) == 1
    assert "explicitly requested 3 stops" in system_messages[0].content


async def test_graph_executes_tool_and_continues(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **kw: [
            PlaceHit(
                place_id="ChIJtest_p1_aaaaaaaa",
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
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="italian please")]))
    assert out["step_count"] == 1
    assert "semantic_search" in out["scratch"]
    assert out["scratch"]["semantic_search"][0]["args"] == {"query": "italian", "k": 3}
    assert out["done"] is True


async def test_graph_respects_max_steps(monkeypatch) -> None:
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
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="x")]))
    assert out["step_count"] == 3
    assert out["done"] is True
    assert out["final_reply"]


async def test_graph_finalizes_on_commit_even_if_llm_keeps_calling_tools(
    monkeypatch, mocker
) -> None:
    """Root-cause regression: a model that commits a valid itinerary and then
    keeps calling tools (instead of voluntarily emitting a tool-call-free
    final message) must still finalize on the successful commit. Before the
    fix, the graph looped back to `plan` after commit and burned every step
    until `short_circuit_max_steps` overwrote the good plan with the canned
    "I hit the planning step limit." message. gpt-4o-mini does this
    deterministically on multi-stop queries (3/3 live repros)."""
    # Hard checks pass so the commit is accepted as final (no revision loop).
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    commit_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "commit_itinerary",
                "id": "commit-1",
                "args": {
                    "stops": [
                        {
                            "place_id": "ChIJtest_p0_aaaaaaaa",
                            "name": "Place p0",
                            "rationale": "omakase",
                            "source": "google_places",
                        }
                    ]
                },
            }
        ],
    )
    # After committing, the broken model keeps "improving" forever.
    keep_searching = [
        AIMessage(
            content="",
            tool_calls=[{"name": "semantic_search", "id": f"s{i}", "args": {"query": "x"}}],
        )
        for i in range(20)
    ]
    fake = _make_fake([commit_call, *keep_searching])
    graph = build_agent_graph(fake, max_steps=8)

    # p0 must be grounded via a prior tool result or commit_stops rejects it
    # (mirrors production: the agent searches before it commits).
    grounded_scratch = {
        "semantic_search": [
            {
                "args": {},
                "result": [
                    PlaceHit(
                        place_id="ChIJtest_p0_aaaaaaaa",
                        name="Place p0",
                        source="google_places",
                        similarity=0.9,
                    )
                ],
                "step": 0,
                "id": "s0",
            }
        ]
    }
    with _patch_details_many_for(["ChIJtest_p0_aaaaaaaa"]):
        out = await graph.ainvoke(
            ItineraryState(
                messages=[HumanMessage(content="omakase date night, 1 stop")],
                constraints=UserConstraints(num_stops=1),
                scratch=grounded_scratch,
            )
        )

    assert out["done"] is True
    assert out["stops"], "the committed stop must survive"
    assert out["stops"][0].place_id == "ChIJtest_p0_aaaaaaaa"
    # The bug's signature: step limit reached + canned error despite a good plan.
    assert out["step_count"] < 8, "must finalize on commit, not exhaust steps"
    assert "step limit" not in (out["final_reply"] or "").lower()


async def test_graph_handles_unknown_tool_name(monkeypatch) -> None:
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
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
    assert out["final_reply"] == "recovered"


async def test_graph_records_tool_exception_in_scratch(monkeypatch) -> None:
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
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    scratch = out["scratch"]["semantic_search"][0]
    assert "error" in scratch["result"]
    assert "db down" in scratch["result"]["error"]
    assert out["done"] is True

    # The exception must also surface to the LLM via the ToolMessage content,
    # not just to scratch — otherwise the model has no way to react.
    from langchain_core.messages import ToolMessage as _ToolMessage

    tool_messages = [m for m in out["messages"] if isinstance(m, _ToolMessage)]
    assert tool_messages, "act() must append a ToolMessage even on tool failure"
    assert "db down" in tool_messages[-1].content


async def test_plan_does_not_double_insert_system_prompt(monkeypatch) -> None:
    """If the caller already supplied a SystemMessage, plan() must not stack a
    second one on top of it."""
    from langchain_core.messages import SystemMessage

    fake = _make_fake([AIMessage(content="hi", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(
        ItineraryState(
            messages=[
                SystemMessage(content="custom system prompt"),
                HumanMessage(content="hello"),
            ]
        )
    )
    system_messages = [m for m in out["messages"] if isinstance(m, SystemMessage)]
    assert len(system_messages) == 1
    assert system_messages[0].content == "custom system prompt"


async def test_act_handles_parallel_tool_calls(monkeypatch) -> None:
    """Modern OpenAI/Gemini fan out multiple tool calls in one AIMessage. Both
    must run, both ToolMessages append, and step_count increments by 1."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])
    monkeypatch.setattr("app.agent.tools._nearby", lambda **_kw: [])

    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "ChIJtest_a_aaaaaaaaa",
                        "args": {"query": "x"},
                    },
                    {
                        "name": "nearby",
                        "id": "ChIJtest_b_aaaaaaaaa",
                        "args": {"place_id": "ChIJtest_p1_aaaaaaaa"},
                    },
                ],
            ),
            AIMessage(content="done", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))

    from langchain_core.messages import ToolMessage as _ToolMessage

    tool_messages = [m for m in out["messages"] if isinstance(m, _ToolMessage)]
    assert {tm.tool_call_id for tm in tool_messages} == {
        "ChIJtest_a_aaaaaaaaa",
        "ChIJtest_b_aaaaaaaaa",
    }
    assert out["step_count"] == 1
    assert "semantic_search" in out["scratch"]
    assert "nearby" in out["scratch"]


def test_prune_for_llm_keeps_short_history_intact() -> None:
    msgs: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "semantic_search", "id": "1", "args": {"query": "x"}}],
        ),
        ToolMessage(content="[]", tool_call_id="1"),
        AIMessage(content="done", tool_calls=[]),
    ]
    assert _prune_for_llm(msgs) == msgs


def test_prune_for_llm_drops_oldest_tool_exchanges() -> None:
    """With more than 2 tool-issuing AIMessages, the older ones lose their
    tool_calls and their ToolMessages are dropped — but the rest survives."""
    msgs: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(
            content="searching",
            tool_calls=[
                {"name": "semantic_search", "id": "ChIJtest_a_aaaaaaaaa", "args": {"q": "1"}}
            ],
        ),
        ToolMessage(content="r1", tool_call_id="ChIJtest_a_aaaaaaaaa"),
        AIMessage(
            content="searching again",
            tool_calls=[
                {"name": "semantic_search", "id": "ChIJtest_b_aaaaaaaaa", "args": {"q": "2"}}
            ],
        ),
        ToolMessage(content="r2", tool_call_id="ChIJtest_b_aaaaaaaaa"),
        AIMessage(
            content="searching once more",
            tool_calls=[{"name": "semantic_search", "id": "c", "args": {"q": "3"}}],
        ),
        ToolMessage(content="r3", tool_call_id="c"),
        AIMessage(content="done", tool_calls=[]),
    ]
    pruned = _prune_for_llm(msgs)
    tool_messages = [m for m in pruned if isinstance(m, ToolMessage)]
    # The oldest ToolMessage ("r1") is dropped.
    assert {tm.content for tm in tool_messages} == {"r2", "r3"}
    # The oldest AIMessage that issued tool_calls keeps its content but loses
    # the tool_calls so the LLM doesn't see an unanswered call.
    ai_with_tools = [m for m in pruned if isinstance(m, AIMessage) and m.tool_calls]
    assert len(ai_with_tools) == 2  # the two most recent


def test_prune_for_llm_preserves_additional_kwargs_on_stub() -> None:
    """D-08-07: the pre-cutoff stub constructor preserves `additional_kwargs`
    from the original AIMessage. Without this, a `_reasoning_state` payload
    stashed on an AIMessage from > _RECENT_TOOL_EXCHANGES_KEPT turns ago would
    be silently dropped by the pruner before the adapter's
    `replay_reasoning_state` could see it (REASON-04 precondition for Plan 03).
    """
    msgs: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(
            content="searching",
            tool_calls=[{"name": "semantic_search", "id": "a1", "args": {"q": "1"}}],
            additional_kwargs={"reasoning_content": "carried-over"},
        ),
        ToolMessage(content="r1", tool_call_id="a1"),
        AIMessage(
            content="more",
            tool_calls=[{"name": "semantic_search", "id": "a2", "args": {"q": "2"}}],
        ),
        ToolMessage(content="r2", tool_call_id="a2"),
        AIMessage(
            content="more again",
            tool_calls=[{"name": "semantic_search", "id": "a3", "args": {"q": "3"}}],
        ),
        ToolMessage(content="r3", tool_call_id="a3"),
        AIMessage(content="done", tool_calls=[]),
    ]
    pruned = _prune_for_llm(msgs)
    # The stub replaces the oldest tool-issuing AIMessage at index 1
    # (immediately after HumanMessage("hi")). It must keep content +
    # additional_kwargs but lose tool_calls.
    assert pruned[1].content == "searching"
    assert pruned[1].tool_calls == []
    assert pruned[1].additional_kwargs.get("reasoning_content") == "carried-over"


# ---------- Phase 8 Plan 05: REASON-06 byte-identity regression (D-08-15) ----------


# Type tag mapping for message classes — single source of truth used both by
# the regen-fixture script (CLI flag below) and the test itself. Mirrors the
# fixture's "type" field exactly.
_MESSAGE_TYPE_TAG = {
    SystemMessage: "system",
    HumanMessage: "human",
    AIMessage: "ai",
    ToolMessage: "tool",
}


def _serialize_messages_for_fixture(msgs: list[BaseMessage]) -> list[dict[str, Any]]:
    """JSON-serializable representation of a message list (D-08-15).

    Each entry: {type, content, tool_calls, tool_call_id, additional_kwargs}.
    `tool_calls` is None for messages without them (HumanMessage, ToolMessage,
    SystemMessage, AIMessage with empty tool_calls); `tool_call_id` is None
    for non-ToolMessages; `additional_kwargs` is always a dict (possibly empty).

    Used on BOTH sides of the byte-identity check: the fixture is generated
    from `_serialize_messages_for_fixture(pipeline_output)` and the runtime
    output is compared via `_serialize_messages_for_fixture(pipeline_output)`,
    so the helper itself cannot accidentally introduce a comparison-side bias.
    """
    out: list[dict[str, Any]] = []
    for m in msgs:
        type_tag = _MESSAGE_TYPE_TAG.get(type(m), type(m).__name__.lower())
        tool_calls = getattr(m, "tool_calls", None)
        # AIMessages always have a tool_calls attribute (possibly []); to keep
        # the fixture clean for stub/non-tool AIMessages we normalize empty
        # tool_calls to None (matches the JSON intent: "no tool_calls here").
        if isinstance(tool_calls, list) and not tool_calls:
            tool_calls = None
        out.append(
            {
                "type": type_tag,
                "content": m.content,
                "tool_calls": tool_calls,
                "tool_call_id": getattr(m, "tool_call_id", None),
                "additional_kwargs": dict(getattr(m, "additional_kwargs", {}) or {}),
            }
        )
    return out


def _reason_04_input_messages() -> list[BaseMessage]:
    """Realistic refinement-turn message list with 3 tool-issuing AIMessages
    (D-08-15). _RECENT_TOOL_EXCHANGES_KEPT=2 forces the OLDEST to be stubbed,
    exercising the pre-cutoff branch patched by Plan 02. All AIMessages have
    empty `additional_kwargs` — the gpt-4o-mini case where NoOpAdapter is
    observationally a no-op and the Plan-02 kwargs-preservation patch is
    invisible.
    """
    return [
        SystemMessage(content="You are City Concierge..."),
        HumanMessage(content="lunch in mission, 3 stops"),
        AIMessage(
            content="searching",
            tool_calls=[{"name": "semantic_search", "id": "c1", "args": {"q": "lunch mission"}}],
        ),
        ToolMessage(content="[result-1]", tool_call_id="c1"),
        AIMessage(
            content="more searches",
            tool_calls=[{"name": "nearby", "id": "c2", "args": {"place_id": "ChIJX"}}],
        ),
        ToolMessage(content="[result-2]", tool_call_id="c2"),
        AIMessage(
            content="trying again",
            tool_calls=[
                {"name": "semantic_search", "id": "c3", "args": {"q": "lunch mission cheap"}}
            ],
        ),
        ToolMessage(content="[result-3]", tool_call_id="c3"),
        AIMessage(content="done", tool_calls=[]),
    ]


def _reason_04_pipeline_output(msgs: list[BaseMessage]) -> list[BaseMessage]:
    """Run the input list through the post-Phase-8 pipeline that REASON-06
    guards: `_prune_for_llm` then `NoOpAdapter().replay_reasoning_state(...,
    None)`. State=None because the input list carries no `_reasoning_state`
    marker — this is the gpt-4o-mini case where NoOpAdapter.replay is a
    mathematical identity (D-08-15).
    """
    pruned = _prune_for_llm(msgs)
    return NoOpAdapter().replay_reasoning_state(pruned, None)


def test_reason_04_noop_adapter_byte_identical_to_pre_phase8() -> None:
    """REASON-06 byte-identity regression for the gpt-4o-mini (empty
    additional_kwargs, no `_reasoning_state`) path. Fixture at
    `tests/unit/fixtures/reason_04_prune_baseline.json`. To regenerate:

        poetry run python tests/unit/test_agent_graph.py --regen-reason-04-fixture

    Regenerate ONLY when an INTENTIONAL change to `_prune_for_llm` or
    `NoOpAdapter` requires the baseline to move; document the change in the
    fixture commit message AND in the Phase 8 SUMMARY (D-08-15).

    Combined with Plan 02's `additional_kwargs` preservation test and Plan 03's
    full `tests/unit/test_agent_graph.py` sweep, this enforces REASON-06 at
    the hardest possible level — byte-identical output for the locked v2.0 prod
    anchor's message shape.
    """
    msgs = _reason_04_input_messages()
    out = _reason_04_pipeline_output(msgs)
    serialized = _serialize_messages_for_fixture(out)

    fixture_path = Path(__file__).parent / "fixtures" / "reason_04_prune_baseline.json"
    expected = json.loads(fixture_path.read_text())

    assert serialized == expected, (
        f"Byte-identity regression on the gpt-4o-mini path (REASON-06): "
        f"runtime output diverged from "
        f"tests/unit/fixtures/reason_04_prune_baseline.json.\n"
        f"  runtime: {serialized!r}\n"
        f"  fixture: {expected!r}\n"
        f"If this divergence is INTENTIONAL, regenerate the fixture via "
        f"`poetry run python tests/unit/test_agent_graph.py "
        f"--regen-reason-04-fixture` and document the change in the commit "
        f"message + Phase 8 SUMMARY (D-08-15)."
    )


# ---------- Phase 8 Plan 03: ProviderAdapter wiring (D-08-04..06, D-08-16) ----------


class _RecordingLLM(BaseChatModel):
    """Test double that records the inbound messages list for each `_generate`
    call, then returns the next scripted AIMessage in order. Used by the
    replay test to assert the adapter injected `_reasoning_state` into the
    most-recent AIMessage of the outbound payload before `ainvoke`.
    """

    scripted: list[AIMessage]
    recorded_inputs: list[list[BaseMessage]] = []  # noqa: RUF012 — pydantic mutable default

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Capture a shallow copy so subsequent mutation by the graph
        # (or the adapter) on the same list spine doesn't taint history.
        self.recorded_inputs.append(list(messages))
        if not self.scripted:
            raise RuntimeError("scripted responses exhausted")
        msg = self.scripted.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> _RecordingLLM:
        return self


async def test_build_agent_graph_provider_default_is_noop_adapter() -> None:
    """D-08-04 + D-08-08: omitting `provider=` routes through `NoOpAdapter`.
    Observable: after a one-turn loop, the inbound AIMessage's
    `additional_kwargs` does NOT have `_reasoning_state` set (NoOp.capture
    returns None, so the post-ainvoke writer never fires).
    """
    fake = _make_fake([AIMessage(content="here is your plan", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)  # default provider
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))

    ai_messages = [m for m in out["messages"] if isinstance(m, AIMessage)]
    assert ai_messages, "expected at least one AIMessage on output"
    # NoOpAdapter.capture returns None → no kwarg written.
    for ai in ai_messages:
        assert "_reasoning_state" not in ai.additional_kwargs, (
            f"unexpected _reasoning_state on default-provider AIMessage: {ai.additional_kwargs}"
        )


async def test_plan_captures_reasoning_state_via_adapter(monkeypatch) -> None:
    """D-08-05 + D-08-06: when an adapter's `capture_reasoning_state` returns
    a payload, `plan()` writes it onto the just-returned AIMessage's
    `additional_kwargs["_reasoning_state"]`. Use the test-only
    `MockReasoningAdapter` patched into `ADAPTERS["scripted"]`.
    """
    from app.agent.adapters import ADAPTERS, MockReasoningAdapter

    marker = {"provider": "test_capture", "reasoning_content": "captured"}
    monkeypatch.setitem(ADAPTERS, "scripted", MockReasoningAdapter(payload=marker))

    fake = _make_fake([AIMessage(content="here is your plan", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4, provider="scripted")
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))

    ai_messages = [m for m in out["messages"] if isinstance(m, AIMessage)]
    assert ai_messages, "expected at least one AIMessage on output"
    # The most-recent AIMessage carries the captured marker.
    assert ai_messages[-1].additional_kwargs.get("_reasoning_state") == marker


async def test_plan_replays_reasoning_state_into_outbound(monkeypatch) -> None:
    """D-08-05 + REASON-05 precursor: across two `plan()` turns, the captured
    `_reasoning_state` from turn 1 is replayed into turn 2's outbound payload
    BEFORE `ainvoke`. Uses `_RecordingLLM` to capture the input messages list
    and `MockReasoningAdapter` to do both the capture and the replay.
    """
    from app.agent.adapters import ADAPTERS, MockReasoningAdapter

    marker = {"provider": "test_replay", "reasoning_content": "injected"}
    monkeypatch.setitem(ADAPTERS, "scripted", MockReasoningAdapter(payload=marker))

    # Turn 1: LLM emits a tool call so the graph loops back into plan() for turn 2.
    # Turn 2: LLM emits a final AIMessage (no tool calls) so the graph terminates.
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "semantic_search", "id": "rs1", "args": {"query": "x"}}],
        ),
        AIMessage(content="done", tool_calls=[]),
    ]
    recording = _RecordingLLM(scripted=list(scripted), recorded_inputs=[])
    # Patch the tool so act() succeeds without DB.
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    graph = build_agent_graph(recording, max_steps=4, provider="scripted")
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True

    # Two plan() turns recorded.
    assert len(recording.recorded_inputs) >= 2, (
        f"expected ≥ 2 recorded LLM invocations, got {len(recording.recorded_inputs)}"
    )
    turn2_input = recording.recorded_inputs[1]
    # The most-recent AIMessage in turn 2's outbound MUST carry the injected
    # marker (MockReasoningAdapter.replay walks reverse and tags the last AIMessage).
    ai_in_turn2 = [m for m in turn2_input if isinstance(m, AIMessage)]
    assert ai_in_turn2, "turn 2 outbound should contain at least one AIMessage from turn 1"
    assert ai_in_turn2[-1].additional_kwargs.get("_reasoning_state") == marker


def _state_with_grounded(place_ids: list[str], party_size: int = 2) -> ItineraryState:
    """Build a state where the given place_ids appear in scratch, so
    commit_stops considers them grounded."""
    hits = [
        PlaceHit(place_id=pid, name=f"Place {pid}", source="google_places", similarity=0.9)
        for pid in place_ids
    ]
    return ItineraryState(
        scratch={
            "semantic_search": [
                {"args": {}, "result": hits, "step": 0, "id": "ChIJtest_s1_aaaaaaaa"}
            ]
        },
        constraints=UserConstraints(party_size=party_size, when=datetime(2026, 5, 7, 19, 0)),
    )


def test_commit_stops_enriches_with_booking() -> None:
    """Auto-enrichment must stamp booking_url + booking_provider on every
    committed stop, without the LLM calling a tool. Now batched: one
    get_details_many fetches details for all stops in a single round-trip,
    then per-stop URL construction is pure."""
    state = _state_with_grounded(["ChIJtest_p1_aaaaaaaa", "ChIJtest_p2_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "first",
            "source": "google_places",
        },
        {
            "place_id": "ChIJtest_p2_aaaaaaaa",
            "name": "Place p2",
            "rationale": "second",
            "source": "google_places",
        },
    ]

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id,
            provider="resy",
            booking_url=f"https://resy.com/{details.place_id}?seats={party_size}",
        )

    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa", "ChIJtest_p2_aaaaaaaa"]) as mock_get,
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        committed, payload = commit_stops(state, raw_stops)

    assert payload["committed"] == ["ChIJtest_p1_aaaaaaaa", "ChIJtest_p2_aaaaaaaa"]
    assert all(s.booking_url is not None for s in committed)
    assert all(s.booking_provider == "resy" for s in committed)
    assert "ChIJtest_p1_aaaaaaaa" in committed[0].booking_url
    assert "seats=2" in committed[0].booking_url
    # The whole point of this refactor: ONE DB call for the whole commit, not N.
    assert mock_get.call_count == 1


def test_commit_stops_skips_enrichment_for_place_id_missing_from_db() -> None:
    """If get_details_many returns no row for a committed place_id (race
    condition: deletion between scratch grounding and enrichment, or a stale
    id), the stop is committed without a booking link instead of crashing.
    Same recoverable semantic the old ValueError("unknown place_id") had."""
    state = _state_with_grounded(["ChIJtest_p1_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "x",
            "source": "google_places",
        },
    ]
    with patch("app.agent.commit.get_details_many", return_value={}):
        committed, payload = commit_stops(state, raw_stops)

    assert payload["committed"] == ["ChIJtest_p1_aaaaaaaa"]
    assert committed[0].booking_url is None
    assert committed[0].booking_provider is None


def test_commit_stops_enrichment_swallows_psycopg_db_blip() -> None:
    """A transient DB error during the batched read shouldn't kill the whole
    commit — the user still gets the planned stops, just without booking
    links. The error is caught at the single point of DB contact."""
    state = _state_with_grounded(["ChIJtest_p1_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "x",
            "source": "google_places",
        },
    ]
    with patch(
        "app.agent.commit.get_details_many",
        side_effect=psycopg2.OperationalError("connection blip"),
    ):
        committed, payload = commit_stops(state, raw_stops)

    assert payload["committed"] == ["ChIJtest_p1_aaaaaaaa"]
    assert committed[0].booking_url is None
    assert committed[0].booking_provider is None


def test_commit_stops_enrichment_propagates_programmer_errors() -> None:
    """Bugs in URL construction (TypeError, AttributeError, etc.) must NOT
    be silently swallowed — that's how regressions ship to prod undetected.
    Only the documented recoverable cases (missing-from-DB, DB blip) are caught."""
    state = _state_with_grounded(["ChIJtest_p1_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "x",
            "source": "google_places",
        },
    ]
    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa"]),
        patch(
            "app.agent.commit.propose_booking_from_details",
            side_effect=TypeError("propose_booking_from_details() got an unexpected kwarg"),
        ),
        pytest.raises(TypeError),
    ):
        commit_stops(state, raw_stops)


def test_commit_stops_re_commit_is_idempotent() -> None:
    """The W3 critique loop drives a second commit_itinerary call when the
    first plan fails a check. Re-committing the same place_id must produce the
    same booking link — same inputs, same output. Locks the contract so a
    future 'skip if already enriched' optimization can't silently break it."""
    state = _state_with_grounded(["ChIJtest_p1_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "first",
            "source": "google_places",
        },
    ]

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id,
            provider="resy",
            booking_url=f"https://resy.com/{details.place_id}?seats={party_size}",
        )

    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        first_committed, _ = commit_stops(state, raw_stops)
        second_committed, _ = commit_stops(state, raw_stops)

    assert first_committed[0].booking_url == second_committed[0].booking_url
    assert first_committed[0].booking_provider == second_committed[0].booking_provider


def test_commit_stops_per_stop_independence_when_one_id_missing() -> None:
    """A 2-stop commit where one place_id is missing from the DB result must
    still ship the other stop's booking link. The for-loop in
    enrich_stops_with_booking skips per-stop on missing details rather than
    bailing on the whole commit."""
    state = _state_with_grounded(["ChIJtest_p1_aaaaaaaa", "ChIJtest_p2_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "first",
            "source": "google_places",
        },
        {
            "place_id": "ChIJtest_p2_aaaaaaaa",
            "name": "Place p2",
            "rationale": "second",
            "source": "google_places",
        },
    ]

    # get_details_many returns details for p2 but NOT p1 — same shape as the
    # DB filtering p1 out (e.g. the row was deleted mid-commit).
    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id,
            provider="opentable",
            booking_url=f"https://opentable.com/{details.place_id}",
        )

    with (
        patch(
            "app.agent.commit.get_details_many",
            return_value={"ChIJtest_p2_aaaaaaaa": _details_for("ChIJtest_p2_aaaaaaaa")},
        ),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        committed, _ = commit_stops(state, raw_stops)

    # p1 missing from DB → skipped, no booking fields populated.
    assert committed[0].place_id == "ChIJtest_p1_aaaaaaaa"
    assert committed[0].booking_url is None
    assert committed[0].booking_provider is None
    # p2 was independent of p1 and succeeded.
    assert committed[1].place_id == "ChIJtest_p2_aaaaaaaa"
    assert committed[1].booking_url == "https://opentable.com/ChIJtest_p2_aaaaaaaa"
    assert committed[1].booking_provider == "opentable"


def test_enrich_stops_with_booking_mutates_in_place() -> None:
    """Direct coverage of the public helper. Future constraint-edit flows
    will call enrich_stops_with_booking on already-built Stop objects without
    going through commit_stops; the in-place-mutation contract is what makes
    that re-enrichment cheap."""
    stops = [
        Stop(place_id="ChIJtest_p1_aaaaaaaa", name="A", rationale="r", source="google_places"),
        Stop(place_id="ChIJtest_p2_aaaaaaaa", name="B", rationale="r", source="google_places"),
    ]
    state = ItineraryState(
        constraints=UserConstraints(party_size=4, when=datetime(2026, 5, 7, 19, 0)),
    )

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id,
            provider="tock",
            booking_url=f"https://exploretock.com/{details.place_id}?size={party_size}",
        )

    original_ids = [id(s) for s in stops]
    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa", "ChIJtest_p2_aaaaaaaa"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    # Same Stop objects (mutated, not replaced).
    assert [id(s) for s in stops] == original_ids
    # Both got party_size=4 from constraints (no per-stop arrival_time).
    assert all(s.booking_provider == "tock" for s in stops)
    assert all("size=4" in (s.booking_url or "") for s in stops)


def test_enrich_populates_card_fields_from_details() -> None:
    """Authoritative display fields flow from PlaceDetails onto Stop."""
    stops = [Stop(place_id="ChIJtest_p1_aaaaaaaa", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(place_id=details.place_id, provider="tock", booking_url="https://x")

    with (
        patch(
            "app.agent.commit.get_details_many",
            return_value={"ChIJtest_p1_aaaaaaaa": _rich_details_for("ChIJtest_p1_aaaaaaaa")},
        ),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert stops[0].name == "Place ChIJtest_p1_aaaaaaaa"
    assert stops[0].primary_type == "restaurant"
    assert stops[0].address == "ChIJtest_p1_aaaaaaaa Main St, San Francisco"
    assert stops[0].rating == 4.4
    assert stops[0].price_level == 2


def test_enrich_backfills_coordinates_when_stop_has_none() -> None:
    """The LLM commits stops without coordinates (optional in the prompt),
    so without this backfill every stop is lat=lng=None -> the frontend's
    `routable` filter drops them all -> no map pins, no route line. The DB
    details already carry lat/lng; enrichment must copy them onto the stop."""
    stops = [Stop(place_id="ChIJtest_p1_aaaaaaaa", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(place_id=details.place_id, provider="tock", booking_url="https://x")

    with (
        patch(
            "app.agent.commit.get_details_many",
            return_value={"ChIJtest_p1_aaaaaaaa": _rich_details_for("ChIJtest_p1_aaaaaaaa")},
        ),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert stops[0].latitude == 37.785
    assert stops[0].longitude == -122.404


def test_enrich_does_not_overwrite_llm_supplied_coordinates() -> None:
    """If the model DID ground a coordinate from a tool result and committed
    it, that wins — enrichment only fills a missing (None) coordinate."""
    stops = [
        Stop(
            place_id="ChIJtest_p1_aaaaaaaa",
            name="A",
            rationale="r",
            source="google_places",
            latitude=37.111,
            longitude=-122.999,
        )
    ]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(place_id=details.place_id, provider="tock", booking_url="https://x")

    with (
        patch(
            "app.agent.commit.get_details_many",
            return_value={"ChIJtest_p1_aaaaaaaa": _rich_details_for("ChIJtest_p1_aaaaaaaa")},
        ),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert stops[0].latitude == 37.111
    assert stops[0].longitude == -122.999


def test_enrich_card_fields_set_even_when_no_booking_time() -> None:
    """Regression guard: the no-time path skips booking but must NOT skip
    address/rating/price. These do not depend on `when`."""
    stops = [Stop(place_id="ChIJtest_p1_aaaaaaaa", name="A", rationale="r", source="google_places")]
    state = ItineraryState(constraints=UserConstraints(party_size=2))  # when=None

    with (
        patch(
            "app.agent.commit.get_details_many",
            return_value={"ChIJtest_p1_aaaaaaaa": _rich_details_for("ChIJtest_p1_aaaaaaaa")},
        ),
        patch("app.agent.commit.propose_booking_from_details") as mock_build,
    ):
        enrich_stops_with_booking(stops, state)

    mock_build.assert_not_called()  # no booking without a time
    assert stops[0].booking_url is None
    assert stops[0].address == "ChIJtest_p1_aaaaaaaa Main St, San Francisco"
    assert stops[0].rating == 4.4
    assert stops[0].price_level == 2


def test_enrich_card_fields_none_when_details_missing() -> None:
    """place_id missing from DB at enrichment time -> fields stay None
    (same degradation as booking links)."""
    stops = [Stop(place_id="ChIJtest_p1_aaaaaaaa", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    with patch("app.agent.commit.get_details_many", return_value={}):
        enrich_stops_with_booking(stops, state)

    assert stops[0].address is None
    assert stops[0].rating is None
    assert stops[0].price_level is None


def test_enrich_skipped_when_no_time_anywhere() -> None:
    """Neither stop.arrival_time nor constraints.when is set → enrichment must
    skip the stop, NOT inject datetime.now(). The wall-clock fallback would
    embed a timestamp in the URL that's meaningless to the user and breaks
    re-commit idempotency (same inputs, different URL each call)."""
    stops = [Stop(place_id="ChIJtest_p1_aaaaaaaa", name="A", rationale="r", source="google_places")]
    state = ItineraryState(constraints=UserConstraints(party_size=2))  # when=None

    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa"]),
        patch("app.agent.commit.propose_booking_from_details") as mock_build,
    ):
        enrich_stops_with_booking(stops, state)

    # Crucially, the URL builder was NOT called — the no-time path skips entirely.
    mock_build.assert_not_called()
    assert stops[0].booking_url is None
    assert stops[0].booking_provider is None


def test_enrich_idempotent_when_no_time_set() -> None:
    """The no-time skip preserves re-commit idempotency: same inputs in,
    same (absent) booking link out, both calls."""
    stops = [Stop(place_id="ChIJtest_p1_aaaaaaaa", name="A", rationale="r", source="google_places")]
    state = ItineraryState(constraints=UserConstraints(party_size=2))

    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa"]),
        patch("app.agent.commit.propose_booking_from_details") as mock_build,
    ):
        enrich_stops_with_booking(stops, state)
        first_url = stops[0].booking_url
        enrich_stops_with_booking(stops, state)
        second_url = stops[0].booking_url

    assert first_url is None
    assert second_url is None
    assert mock_build.call_count == 0


def test_enrich_uses_constraints_when_when_arrival_time_missing() -> None:
    """If a stop has no arrival_time but constraints.when IS set, the constraint
    fills in. (Counterpoint to the skip test above — make sure we don't skip
    too aggressively.)"""
    stops = [Stop(place_id="ChIJtest_p1_aaaaaaaa", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    captured: list[datetime] = []

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        captured.append(when)
        return BookingProposal(place_id=details.place_id, provider="resy", booking_url="https://x")

    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert captured == [datetime(2026, 5, 7, 19, 0)]


@pytest.mark.parametrize("party_size_in,expected_in_call", [(None, 2), (0, 2), (4, 4), (1, 1)])
def test_enrich_party_size_defaulting(party_size_in: int | None, expected_in_call: int) -> None:
    """party_size None or 0 → defaults to 2 (you can't book a table for 0).
    Other positive values pass through unchanged. Locks the `or 2` semantics
    so a future refactor doesn't accidentally allow 0-party bookings or change
    the default."""
    stops = [
        Stop(
            place_id="ChIJtest_p1_aaaaaaaa",
            name="A",
            rationale="r",
            source="google_places",
            arrival_time=datetime(2026, 5, 7, 19, 0),
        )
    ]
    state = ItineraryState(constraints=UserConstraints(party_size=party_size_in))

    captured: list[int] = []

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        captured.append(party_size)
        return BookingProposal(place_id=details.place_id, provider="resy", booking_url="https://x")

    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert captured == [expected_in_call]


def test_enrich_stops_with_booking_uses_arrival_time_per_stop() -> None:
    """When stops have different arrival_times, each stop's URL must reflect
    its OWN time, not constraints.when. Otherwise a 3-stop itinerary's late
    stops get a 'date' for the first stop's time slot."""
    stops = [
        Stop(
            place_id="ChIJtest_p1_aaaaaaaa",
            name="dinner",
            rationale="r",
            source="google_places",
            arrival_time=datetime(2026, 5, 7, 19, 0),
        ),
        Stop(
            place_id="ChIJtest_p2_aaaaaaaa",
            name="drinks",
            rationale="r",
            source="google_places",
            arrival_time=datetime(2026, 5, 7, 21, 30),
        ),
    ]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    captured: list[tuple[str, datetime]] = []

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        captured.append((details.place_id, when))
        return BookingProposal(
            place_id=details.place_id, provider="resy", booking_url="https://resy.com/x"
        )

    with (
        _patch_details_many_for(["ChIJtest_p1_aaaaaaaa", "ChIJtest_p2_aaaaaaaa"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert captured == [
        ("ChIJtest_p1_aaaaaaaa", datetime(2026, 5, 7, 19, 0)),
        ("ChIJtest_p2_aaaaaaaa", datetime(2026, 5, 7, 21, 30)),
    ]


def _committed_state(n_stops: int, *, with_coords: bool) -> ItineraryState:
    base = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = []
    for i in range(n_stops):
        stops.append(
            Stop(
                # Pad with 'a' so the literal satisfies the 06-01 Task 3
                # Google-Place-ID-format validator (>= 20 chars).
                place_id=f"ChIJtest_p{i}_aaaaaaaa",
                name=f"P{i}",
                source="google_places",
                rationale="",
                arrival_time=base if i == 0 else None,
                planned_duration_min=60,
                latitude=37.77 + i * 0.01 if with_coords else None,
                longitude=-122.41 if with_coords else None,
            )
        )
    return ItineraryState(stops=stops, done=True, final_reply="Here is your plan.")


async def test_retime_node_present_and_routed(monkeypatch) -> None:
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    assert "retime" in graph.get_graph().nodes


async def test_retime_at_most_one_directions_call(monkeypatch, mocker) -> None:
    calls = {"n": 0}

    async def _counting_route_legs(stops, mode="walk"):
        calls["n"] += 1
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=600, distance_m=800.0)] * (len(stops) - 1),
            total_duration_s=600 * (len(stops) - 1),
            mode=mode,
            source="google",
        )

    mocker.patch("app.agent.graph.route_legs", _counting_route_legs)
    # The swap_closed_stops node (after retime) runs its own closure check;
    # stub it to "all open" so we measure retime's route_legs call only.
    mocker.patch(
        "app.agent.swap._per_stop_closure_status",
        side_effect=lambda stops: [False] * len(stops),
    )
    # Prevent critique from hitting the DB (place_ids p0-p2 don't exist in
    # places_raw in the test environment, which would trigger a revision loop
    # and exhaust the scripted LLM when the full suite runs after any test
    # that activates a real DB pool via load_dotenv in ingest_places_sf.py).
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    await graph.ainvoke(_committed_state(3, with_coords=True))
    assert calls["n"] == 1


async def test_retime_passthrough_when_not_routable(monkeypatch, mocker) -> None:
    route = mocker.patch("app.agent.graph.route_legs")
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(_committed_state(1, with_coords=True))
    route.assert_not_called()
    assert out["stops"][0].arrival_time == datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)


async def test_retime_passthrough_when_coordless(monkeypatch, mocker) -> None:
    route = mocker.patch("app.agent.graph.route_legs")
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    await graph.ainvoke(_committed_state(3, with_coords=False))
    route.assert_not_called()


async def test_graph_includes_swap_closed_stops_node(monkeypatch) -> None:
    """The compiled graph routes retime -> swap_closed_stops -> END so the
    closure-aware swap pass runs after real-time arrival_times land. Replaces
    the deleted retime+caveat tests — temporal_coherence handling is now the
    swap node's job, not retime's."""
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    assert "swap_closed_stops" in graph.get_graph().nodes


async def test_graph_does_not_import_final_with_caveats_in_retime() -> None:
    """Regression: closure handling lives in app/agent/swap.py now. The
    retime node must not call _final_with_caveats on temporal_coherence —
    the caveat text users saw on closure was the worst-of-both-worlds path
    (broken plan + ugly warning). The swap node replaces it."""
    import inspect

    from app.agent import graph as graph_mod

    src = inspect.getsource(graph_mod)
    assert "_final_with_caveats" not in src, (
        "retime() must not call _final_with_caveats on temporal_coherence; "
        "closure handling lives in app/agent/swap.py now."
    )


async def test_act_does_not_mutate_aimessage_tool_call_args_across_steps(
    mocker,
) -> None:
    """Regression for "TypeError: Object of type SearchFilters is not JSON
    serializable" surfaced live on turn 3 of the omakase flow.

    Failure path: `act()` used to do `tc["args"] = _inject_closure_exclusions(...)`,
    stuffing a Pydantic SearchFilters into the tool_call args dict that
    LangChain stores inside AIMessage.tool_calls. On the NEXT plan() pass,
    langchain serializes the AIMessage for OpenAI's API and `json.dumps`
    blows up.

    Verify that across a multi-step graph turn — where the same AIMessage
    survives into a follow-up plan() — its tool_call args stay JSON-safe.
    """
    import json as _json
    from datetime import datetime

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.agent.graph import build_agent_graph
    from app.agent.state import ClosureContext, ItineraryState

    captured: dict = {"step_2_args": None, "step_2_dumps_ok": False}

    class _MultiStepLLM(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "multistep"

        def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop=None,
            run_manager=None,
            **kwargs,
        ) -> ChatResult:
            # First call: issue a semantic_search tool call that will get
            # closure exclusions injected. Subsequent calls: capture the
            # AIMessage's tool_call args and finalize.
            issued_tool_calls = sum(
                1 for m in messages if isinstance(m, AIMessage) and m.tool_calls
            )
            if issued_tool_calls == 0:
                msg = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "semantic_search",
                            "args": {"query": "ramen", "filters": {"min_rating": 4.0}},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                # Second plan() — find the prior issuing AIMessage and inspect
                # its tool_call args. This is the exact serialization path
                # langchain.openai takes when re-sending the conversation.
                for m in messages:
                    if isinstance(m, AIMessage) and m.tool_calls:
                        captured["step_2_args"] = m.tool_calls[0]["args"]
                        try:
                            _json.dumps(m.tool_calls[0]["args"])
                            captured["step_2_dumps_ok"] = True
                        except TypeError:
                            captured["step_2_dumps_ok"] = False
                        break
                msg = AIMessage(content="done", tool_calls=[])
            return ChatResult(generations=[ChatGeneration(message=msg)])

    mocker.patch("app.tools.retrieval.semantic_search", return_value=[])

    state = ItineraryState(
        messages=[HumanMessage(content="start")],
        closure_context=[
            ClosureContext(
                place_id="ChIJtest_closed_aaaa",
                place_name="Closed",
                family="bar",
                attempted_arrival=datetime(2026, 5, 19, 20, 0),
                outcome="auto_swapped",
                insert_after_place_id=None,
                insert_before_place_id=None,
                stop_index_hint=0,
            )
        ],
    )

    graph = build_agent_graph(_MultiStepLLM(), max_steps=4)
    await graph.ainvoke(state)

    assert captured["step_2_args"] is not None, "second plan() never ran"
    # `filters` MUST be a plain dict — Pydantic SearchFilters would break the
    # OpenAI API re-serialization.
    filters = captured["step_2_args"].get("filters")
    assert isinstance(filters, dict), (
        f"AIMessage.tool_calls[0]['args']['filters'] must be a dict for "
        f"langchain JSON serialization, got {type(filters).__name__}"
    )
    assert captured["step_2_dumps_ok"], (
        "AIMessage.tool_calls[0]['args'] must be json.dumps-safe — langchain "
        "re-serializes the message on the next OpenAI call."
    )


async def test_retime_noop_when_first_stop_has_no_arrival(monkeypatch, mocker) -> None:
    """chain_arrival_times raises if stops[0].arrival_time is None (possible
    on a max-steps short-circuit with committed-but-untimed stops). retime
    must swallow it and no-op, never propagate out of /chat."""

    async def _ok(stops, mode="walk"):
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=600, distance_m=800.0)] * (len(stops) - 1),
            total_duration_s=600 * (len(stops) - 1),
            mode=mode,
            source="google",
        )

    mocker.patch("app.agent.graph.route_legs", _ok)
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    st = _committed_state(2, with_coords=True)
    # Clear arrival_time on the first stop (simulates untimed commit).
    cleared = [s.model_copy(update={"arrival_time": None}) for s in st.stops]
    st = st.model_copy(update={"stops": cleared})
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(st)  # must NOT raise
    # No-op: stops unchanged (still None on stop 0), reply unchanged.
    assert out["stops"][0].arrival_time is None
    assert out["final_reply"] == "Here is your plan."


# ---------- Phase 8 Plan 05: __main__ regen-fixture entrypoint (D-08-15) ----------
#
# `--regen-reason-04-fixture` regenerates tests/unit/fixtures/reason_04_prune_baseline.json
# from the SAME pipeline the byte-identity regression test asserts against. Run
# this ONLY when an INTENTIONAL change to `_prune_for_llm` or `NoOpAdapter`
# requires the baseline to move; document the change in the fixture commit
# message AND in the Phase 8 SUMMARY. The regen path is gated behind an
# explicit CLI flag (T-08-13 mitigation: no accidental drift).
if __name__ == "__main__":
    import sys

    if "--regen-reason-04-fixture" in sys.argv:
        msgs = _reason_04_input_messages()
        out = _reason_04_pipeline_output(msgs)
        serialized = _serialize_messages_for_fixture(out)

        fixture_path = Path(__file__).parent / "fixtures" / "reason_04_prune_baseline.json"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(serialized, indent=2, sort_keys=True) + "\n")
        print(f"regen-reason-04-fixture: wrote {len(serialized)} entries to {fixture_path}")
    else:
        print(
            "test_agent_graph.py: no recognised CLI flag. "
            "Available flags: --regen-reason-04-fixture",
            file=sys.stderr,
        )
        sys.exit(2)
