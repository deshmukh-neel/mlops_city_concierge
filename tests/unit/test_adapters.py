"""Unit tests for the Phase 9 `ProviderAdapter` implementations.

Plan 09-01 / PROV-01: `OpenAIReasoningAdapter` capture/replay covered in
isolation (no graph wiring, no LLM call). Mirrors the test style of
`tests/unit/test_agent_graph.py` — sync where possible, instantiate the
adapter directly, assert behavior on synthesized `AIMessage`s.

The conformance harness in
`tests/integration/test_reasoning_state_roundtrip.py` covers the same
adapter end-to-end through `graph.ainvoke` + the LangGraph reducer; this
file covers the adapter contract in isolation per `feedback_test_layering`.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter

# ─── OpenAIReasoningAdapter — PROV-01 (D-09-03 Path B per probe verdict) ─────


def test_openai_reasoning_adapter_capture_returns_payload_when_kwarg_present() -> None:
    """Test 1: ``capture_reasoning_state`` reads ``additional_kwargs['reasoning_content']``
    and wraps it in the canonical ``FOUR_SHAPE_PAYLOADS[0]`` shape.

    Plan 09-01 §<behavior> Test 1: matches
    ``{"provider": "openai", "reasoning_content": "thinking..."}`` exactly.
    """
    adapter = OpenAIReasoningAdapter()
    msg = AIMessage(
        content="x",
        additional_kwargs={"reasoning_content": "thinking..."},
    )

    payload = adapter.capture_reasoning_state(msg)

    assert payload == {"provider": "openai", "reasoning_content": "thinking..."}


def test_openai_reasoning_adapter_capture_returns_none_when_kwarg_missing() -> None:
    """Test 2: ``capture_reasoning_state`` returns ``None`` for non-reasoning
    messages (the D-08-02 contract default — adapter returns ``None`` when the
    message carries no provider-native reasoning state)."""
    adapter = OpenAIReasoningAdapter()
    msg = AIMessage(content="x")

    assert adapter.capture_reasoning_state(msg) is None


def test_openai_reasoning_adapter_replay_writes_kwarg_on_most_recent_ai_message() -> None:
    """Test 3: ``replay_reasoning_state`` walks ``outbound`` in reverse and writes
    ``additional_kwargs['reasoning_content']`` (the provider-native key, NOT the
    ``_reasoning_state`` key — graph.py owns that key per PATTERNS.md §Shared
    Patterns) on the most-recent ``AIMessage``.
    """
    adapter = OpenAIReasoningAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="a"),
        AIMessage(content="b"),
    ]
    state = {"provider": "openai", "reasoning_content": "r"}

    result = adapter.replay_reasoning_state(outbound, state)

    # Same list spine.
    assert result is outbound
    # Most-recent AIMessage (the last one in the list) got tagged.
    assert outbound[-1].additional_kwargs.get("reasoning_content") == "r"
    # The earlier AIMessage was NOT tagged (walks in reverse, breaks on first AIMessage).
    assert outbound[1].additional_kwargs.get("reasoning_content") is None


def test_openai_reasoning_adapter_replay_returns_outbound_unchanged_when_state_none() -> None:
    """Test 4: ``replay_reasoning_state`` returns ``outbound`` byte-identical
    when ``state is None`` (D-08-02 contract — no mutation when nothing to
    replay).
    """
    adapter = OpenAIReasoningAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="a", additional_kwargs={"existing": "kept"}),
    ]

    result = adapter.replay_reasoning_state(outbound, None)

    assert result is outbound
    assert outbound[-1].additional_kwargs == {"existing": "kept"}


def test_openai_reasoning_adapter_capture_does_not_mutate_input_message() -> None:
    """Test 5 (T-09-01-T3 mutation safety): ``capture_reasoning_state`` returns a
    NEW dict and does NOT mutate the input ``AIMessage``'s
    ``additional_kwargs``. Compares ``dict(msg.additional_kwargs)`` before and
    after the call.
    """
    adapter = OpenAIReasoningAdapter()
    original_kwargs = {"reasoning_content": "thinking..."}
    msg = AIMessage(content="x", additional_kwargs=original_kwargs)
    before = dict(msg.additional_kwargs)

    payload = adapter.capture_reasoning_state(msg)
    # Mutate the payload to prove capture returned a fresh dict.
    assert payload is not None
    payload["reasoning_content"] = "TAMPERED"

    after = dict(msg.additional_kwargs)
    assert after == before, (
        "capture_reasoning_state mutated the input message's additional_kwargs "
        f"(T3 mitigation failed): before={before!r}, after={after!r}"
    )
    # Sanity: the original kwarg untouched even after we tampered with payload.
    assert msg.additional_kwargs["reasoning_content"] == "thinking..."
