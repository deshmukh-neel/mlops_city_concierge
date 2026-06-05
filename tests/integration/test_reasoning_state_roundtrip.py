"""Phase 8 conformance harness for the `ProviderAdapter` reasoning-state contract.

What this file proves (and why it is quarantined):

- **REASON-02**: the contract is shape-agnostic across the four target state shapes
  (OpenAI ``reasoning_content`` str, Anthropic signed ``thinking_blocks``, DeepSeek
  ``reasoning_content`` str, Gemini ``thought_signature`` bytes). One parametrize
  case per shape; each drives ``graph.ainvoke`` end-to-end and asserts the marker
  payload survives in ``AIMessage.additional_kwargs['_reasoning_state']`` of the
  next turn's outbound payload.
- **REASON-03**: the harness runs in CI as a quarantined integration test via the
  ``reasoning_conformance`` pytest marker (registered in ``pyproject.toml``) and the
  dedicated ``make test-reasoning-conformance`` target. Default ``make test``
  excludes it via ``addopts = "... -m 'not reasoning_conformance'"`` so it never
  destabilizes prod merges (Phase 10 / BASE-03 promotes it to a required gate).
- **REASON-05**: the architectural decision gate for v2.1. ``test_reason_05_*``
  builds the REAL ``build_agent_graph`` with the REAL LangGraph reducer and asserts
  the captured ``_reasoning_state`` marker survives end-to-end through
  ``graph.ainvoke``. If it fails, ``08-REASON-05-BLOCKER.md`` is materialized and
  v2.1 replans around a custom imperative loop (D-08-11; Phase 6 D-06-09 part-2
  precedent — phase still ships).

Quarantine mechanics (D-08-14):

- Module-level ``pytestmark`` assignment to the ``reasoning_conformance`` marker
  (defined below) — every test in this file inherits the marker. NOT
  ``pytest.mark.skipif(APP_ENV != 'integration')`` because the harness uses no
  real DB / network — only ``RecordingLLM`` + ``MockReasoningAdapter`` + the
  real LangGraph reducer.
- Run with::

      make test-reasoning-conformance
      # or equivalently
      poetry run pytest -m reasoning_conformance -v
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.agent.adapters import ADAPTERS, MockReasoningAdapter
from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter
from app.agent.graph import build_agent_graph
from app.agent.state import ItineraryState

# D-08-14: marker quarantines this file from default `make test`. The
# `reasoning_conformance` marker is registered in pyproject.toml and excluded
# via `addopts = "... -m 'not reasoning_conformance'"` so this module is only
# collected by `make test-reasoning-conformance`.
pytestmark = pytest.mark.reasoning_conformance


class RecordingLLM(BaseChatModel):
    """Test-only ``BaseChatModel`` that records its inbound messages list on every
    ``_generate`` call and pops scripted ``AIMessage`` responses in order.

    Mirrors the ``ScriptedChatModel`` pattern from ``app/llm_factory.py:107-160``
    (Pydantic ``Field(default_factory=...)`` for mutable state, ``_llm_type``
    property, ``_generate`` that pops from a scripted list, ``bind_tools`` returning
    self). The added wrinkle is ``recorded_inputs``: each call snapshots
    ``list(messages)`` so the conformance assertion can read the EXACT outbound
    payload the graph handed to the LLM at that turn.

    When the scripted list is exhausted, returns ``AIMessage(content="done")`` so
    the agent loop can terminate cleanly without ``RuntimeError`` deadlocks.
    """

    scripted: list[AIMessage] = Field(default_factory=list)
    recorded_inputs: list[list[BaseMessage]] = Field(default_factory=list)

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
        # Snapshot the inbound list spine so subsequent adapter / graph mutation
        # on the same list does not taint history. The marker assertion reads
        # the most-recent recorded entry to inspect what the LLM actually saw.
        self.recorded_inputs.append(list(messages))
        msg = self.scripted.pop(0) if self.scripted else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> RecordingLLM:
        # The agent graph calls .bind_tools(...) on the LLM. Return self so the
        # binding is a no-op (mirror ScriptedChatModel + the unit-level
        # _RecordingLLM in tests/unit/test_agent_graph.py).
        return self


# D-08-13 / CONTEXT.md specifics: the four exact dict literals that REASON-02
# acceptance pivots on. One parametrize case per target shape. Phase 9 swaps
# the Mock for real adapters one shape at a time; the harness body is invariant.
FOUR_SHAPE_PAYLOADS: list[dict[str, Any]] = [
    {"provider": "openai", "reasoning_content": "foo"},
    {
        "provider": "anthropic",
        "thinking_blocks": [
            {"type": "thinking", "signature": "abc", "thinking": "..."},
        ],
    },
    {"provider": "deepseek", "reasoning_content": "bar"},
    {"provider": "gemini", "thought_signature": b"\x00\x01\x02"},
]


@pytest.mark.parametrize("payload", FOUR_SHAPE_PAYLOADS)
async def test_reason_02_four_shape_roundtrip(
    payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REASON-02: the contract is shape-agnostic across the four target shapes.

    For each parametrize case we patch ``ADAPTERS["scripted"]`` with a
    ``MockReasoningAdapter(payload=payload)`` so ``replay_reasoning_state`` tags
    the most-recent outbound ``AIMessage`` and ``capture_reasoning_state`` echoes
    the same payload back. Driving a 2-turn loop through ``graph.ainvoke`` proves
    every shape (str / signed dict / bytes) survives capture → reducer → replay.

    Turn 1: scripted ``AIMessage`` with a ``semantic_search`` tool_call so the
    graph loops back into ``plan()`` for turn 2 (and the reducer actually has to
    accumulate messages across turns).
    Turn 2: scripted ``AIMessage(content="done")`` so the graph terminates.

    Assertion: the most-recent recorded LLM input contains an ``AIMessage`` whose
    ``additional_kwargs["_reasoning_state"]`` equals the parametrized payload.
    """
    monkeypatch.setitem(ADAPTERS, "scripted", MockReasoningAdapter(payload=payload))

    # Patch the underlying retrieval helper so act() succeeds without a real DB.
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "semantic_search", "id": "rs1", "args": {"query": "x"}},
            ],
        ),
        AIMessage(content="done", tool_calls=[]),
    ]
    recording_llm = RecordingLLM(scripted=list(scripted))
    graph = build_agent_graph(recording_llm, max_steps=4, provider="scripted")

    await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="find a bar")]))

    assert len(recording_llm.recorded_inputs) >= 2, (
        f"expected ≥ 2 recorded LLM invocations across the two turns, "
        f"got {len(recording_llm.recorded_inputs)}"
    )
    turn2_input = recording_llm.recorded_inputs[-1]
    ai_messages = [m for m in turn2_input if isinstance(m, AIMessage)]
    assert ai_messages, (
        f"turn 2's outbound should contain at least one AIMessage from turn 1; "
        f"got types {[type(m).__name__ for m in turn2_input]}"
    )
    # The most-recent AIMessage carries the marker — MockReasoningAdapter.replay
    # walks `outbound` in reverse and tags the last AIMessage.
    assert ai_messages[-1].additional_kwargs.get("_reasoning_state") == payload, (
        f"shape={payload.get('provider')!r} marker dropped — saw "
        f"additional_kwargs={ai_messages[-1].additional_kwargs!r}"
    )


# D-08-11 rubric — if this test FAILS, the executor materializes
# `.planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/
# 08-REASON-05-BLOCKER.md`, marks this test
# `@pytest.mark.xfail(strict=False, reason="REASON-05 blocker — see 08-REASON-05-BLOCKER.md")`,
# and commits both. Phase 8 still ships per D-06-09 part 2 precedent.
async def test_reason_05_graph_invoke_preserves_reasoning_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REASON-05 architectural decision gate: ``graph.ainvoke`` preserves the
    captured ``_reasoning_state`` ``additional_kwargs`` across LangGraph's
    ``add_messages`` reducer.

    Critical difference vs Plan 03's unit tests: this test exercises the FULL
    ``graph.ainvoke`` round-trip — NOT a direct ``plan()`` call — so the
    ``add_messages`` reducer is exactly what's under test. If the reducer
    re-instantiates ``AIMessage``s and drops the ``additional_kwargs`` field, this
    assertion fires and v2.1 replans around a custom imperative loop (D-08-11).

    Mechanics (D-08-09, D-08-10, D-08-12):

    - Initial state contains the verbatim D-08-12 turn-1 ``AIMessage`` (content +
      ``semantic_search`` tool_call + ``additional_kwargs={"reasoning_content":
      "thinking about bars..."}``) so the reducer has something concrete to
      preserve.
    - ``ADAPTERS["scripted"]`` is patched to ``MockReasoningAdapter(payload=marker)``
      so both capture and replay route through the contract.
    - ``RecordingLLM`` is seeded with ONE scripted terminating ``AIMessage`` so the
      loop terminates after the tool result + one more plan step.
    - The assertion is ``additional_kwargs``-level (NOT a system-message marker,
      NOT a content-string marker) per D-08-10 — the gate exists specifically to
      test that the kwargs-preservation path (D-08-06) works end-to-end through
      the reducer. A marker injected as system-message or content edit would pass
      even if kwargs were lost, defeating the gate's purpose.
    """
    # D-08-09 / D-08-10 / D-08-12: canonical OpenAI-shape marker.
    marker: dict[str, Any] = {
        "provider": "openai",
        "reasoning_content": "thinking about bars...",
    }
    monkeypatch.setitem(ADAPTERS, "scripted", MockReasoningAdapter(payload=marker))

    # Patch the underlying retrieval helper so act() succeeds without a real DB.
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    # D-08-12: the verbatim turn-1 AIMessage shape — content + semantic_search
    # tool_call + raw provider reasoning kwarg. This is what RecordingLLM emits
    # AS the turn-1 response (NOT a seed in the initial state); plan() captures
    # state from it via the adapter, the reducer carries the captured marker
    # across to turn-2 via additional_kwargs["_reasoning_state"].
    turn1_ai = AIMessage(
        content="searching for bars",
        tool_calls=[
            {
                "name": "semantic_search",
                "args": {"query": "bar in mission", "k": 3},
                "id": "call_test_1",
            },
        ],
        additional_kwargs={"reasoning_content": "thinking about bars..."},
    )

    # RecordingLLM scripts BOTH turns:
    #   turn-1 → the D-08-12 AIMessage with a tool_call (so the graph loops
    #             through act() → critique() → plan() again);
    #   turn-2 → a terminating AIMessage(content="done") with no tool_calls.
    # plan() captures state on the turn-1 AIMessage; the reducer must preserve
    # the capture's additional_kwargs into the turn-2 plan() inbound payload.
    recording_llm = RecordingLLM(
        scripted=[turn1_ai, AIMessage(content="done", tool_calls=[])],
    )

    # Build via the REAL build_agent_graph + REAL LangGraph reducer.
    graph = build_agent_graph(recording_llm, max_steps=4, provider="scripted")

    await graph.ainvoke(
        ItineraryState(messages=[HumanMessage(content="find a bar in mission")]),
    )

    # Two plan() invocations expected: turn-1 (System+Human in, tool-call AIMessage
    # out) → act() runs the tool → critique() routes back → turn-2 (the curated
    # message list after the tool exchange goes in, "done" out).
    assert len(recording_llm.recorded_inputs) >= 2, (
        f"expected ≥ 2 recorded LLM invocations across turn-1 + turn-2, "
        f"got {len(recording_llm.recorded_inputs)}"
    )

    # D-08-10: assertion is `additional_kwargs`-level — ANY AIMessage in the
    # most-recent recorded input must carry the marker. The MockReasoningAdapter
    # tags the most-recent AIMessage on replay; if the reducer preserves
    # additional_kwargs end-to-end the marker shows up here.
    last_input = recording_llm.recorded_inputs[-1]
    ai_messages_in_last = [m for m in last_input if isinstance(m, AIMessage)]
    assert ai_messages_in_last, (
        "expected at least one AIMessage in the most-recent LLM input; "
        f"saw types {[type(m).__name__ for m in last_input]}"
    )
    survived = any(
        m.additional_kwargs.get("_reasoning_state") == marker for m in ai_messages_in_last
    )
    assert survived, (
        "REASON-05 gate: `additional_kwargs['_reasoning_state']` did NOT survive "
        "graph.ainvoke's add_messages reducer. Materialize "
        "08-REASON-05-BLOCKER.md and xfail this test per D-08-11. "
        f"Recorded last-input AIMessage additional_kwargs: "
        f"{[m.additional_kwargs for m in ai_messages_in_last]!r}"
    )


# Phase 9 / PROV-01 (D-09-03 Path B) — sibling test that swaps the REAL
# `OpenAIReasoningAdapter` into `ADAPTERS["scripted"]` and proves the openai
# `reasoning_content` shape (matching `FOUR_SHAPE_PAYLOADS[0]`) survives the
# `graph.ainvoke` round-trip end-to-end. The original
# `test_reason_02_four_shape_roundtrip[payload0]` is unchanged (D-08-13 +
# canonical_refs lock it); this is a NEW test that asserts the real adapter's
# wire shape vs the Mock's `_reasoning_state` storage convention.
async def test_reason_02_openai_real_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REASON-02 + PROV-01: real `OpenAIReasoningAdapter` round-trips
    `additional_kwargs["reasoning_content"]` (the provider-native key) through
    `graph.ainvoke`.

    Mechanics:
    - `monkeypatch.setitem(ADAPTERS, "scripted", OpenAIReasoningAdapter())` so
      the real adapter is the one `plan()` closes over (no `MockReasoningAdapter`).
    - The turn-1 scripted `AIMessage` carries
      `additional_kwargs={"reasoning_content": "thinking..."}` — the shape the
      `OpenAIReasoningChatModel` subclass would emit for `gpt-5-mini` in prod.
    - We assert the turn-2 input's most-recent `AIMessage` carries the same
      `additional_kwargs["reasoning_content"]` value (NOT `_reasoning_state`;
      graph.py stashes capture output at `_reasoning_state`, but
      `OpenAIReasoningAdapter.replay_reasoning_state` writes the
      provider-native `reasoning_content` key on the most-recent AIMessage so
      the next outbound Responses-API request carries it back to OpenAI).
    """
    monkeypatch.setitem(ADAPTERS, "scripted", OpenAIReasoningAdapter())

    # Patch the underlying retrieval helper so act() succeeds without a real DB.
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    turn1_ai = AIMessage(
        content="searching for bars",
        tool_calls=[
            {
                "name": "semantic_search",
                "args": {"query": "bar in mission", "k": 3},
                "id": "call_test_openai_real",
            },
        ],
        additional_kwargs={"reasoning_content": "thinking..."},
    )
    recording_llm = RecordingLLM(
        scripted=[turn1_ai, AIMessage(content="done", tool_calls=[])],
    )
    graph = build_agent_graph(recording_llm, max_steps=4, provider="scripted")

    await graph.ainvoke(
        ItineraryState(messages=[HumanMessage(content="find a bar in mission")]),
    )

    assert len(recording_llm.recorded_inputs) >= 2, (
        "expected ≥ 2 recorded LLM invocations across turn-1 + turn-2, "
        f"got {len(recording_llm.recorded_inputs)}"
    )

    last_input = recording_llm.recorded_inputs[-1]
    ai_messages_in_last = [m for m in last_input if isinstance(m, AIMessage)]
    assert ai_messages_in_last, (
        "expected at least one AIMessage in the most-recent LLM input; "
        f"saw types {[type(m).__name__ for m in last_input]}"
    )
    # The real adapter writes the provider-native key (NOT _reasoning_state).
    # walks outbound in reverse → most-recent AIMessage gets tagged.
    assert ai_messages_in_last[-1].additional_kwargs.get("reasoning_content") == "thinking...", (
        "OpenAIReasoningAdapter replay did NOT write `reasoning_content` onto "
        "the most-recent outbound AIMessage's additional_kwargs. "
        f"Recorded last-input AIMessage additional_kwargs: "
        f"{[m.additional_kwargs for m in ai_messages_in_last]!r}"
    )
