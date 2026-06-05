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

from app.agent.adapters.anthropic import AnthropicAdapter
from app.agent.adapters.deepseek import DeepSeekReasonerAdapter
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


# ─── DeepSeekReasonerAdapter — PROV-02 (D-09-04 model-level carve-out) ───────


def test_deepseek_reasoner_adapter_capture_returns_payload_when_kwarg_present() -> None:
    """PROV-02 Test 1: ``capture_reasoning_state`` reads
    ``additional_kwargs['reasoning_content']`` (populated natively by
    ``ChatDeepSeek`` for ``deepseek-reasoner`` per the
    ``langchain-deepseek>=1.0.0,<2.0.0`` contract) and wraps it in the
    canonical ``FOUR_SHAPE_PAYLOADS[2]`` shape — matches
    ``{"provider": "deepseek", "reasoning_content": "bar"}`` exactly.
    """
    adapter = DeepSeekReasonerAdapter()
    msg = AIMessage(
        content="x",
        additional_kwargs={"reasoning_content": "bar"},
    )

    payload = adapter.capture_reasoning_state(msg)

    assert payload == {"provider": "deepseek", "reasoning_content": "bar"}


def test_deepseek_reasoner_adapter_capture_returns_none_when_kwarg_missing() -> None:
    """PROV-02 Test 2: ``capture_reasoning_state`` returns ``None`` for non-reasoning
    messages — D-08-02 contract default. Also covers the ``deepseek-chat`` path
    where ``extra_body={"thinking": {"type": "disabled"}}`` means the kwarg is
    never populated; behavior on that path is byte-identical to NoOpAdapter."""
    adapter = DeepSeekReasonerAdapter()
    msg = AIMessage(content="x")

    assert adapter.capture_reasoning_state(msg) is None


def test_deepseek_reasoner_adapter_replay_writes_kwarg_on_most_recent_ai_message() -> None:
    """PROV-02 Test 3: ``replay_reasoning_state`` walks ``outbound`` in reverse and
    writes ``additional_kwargs['reasoning_content']`` (the provider-native key,
    NOT the ``_reasoning_state`` key — graph.py owns that key per PATTERNS.md
    §Shared Patterns) on the most-recent ``AIMessage``. This is what makes the
    next outbound DeepSeek request carry reasoning_content back to the API and
    avoid the 400 "reasoning_content must be passed back" failure.
    """
    adapter = DeepSeekReasonerAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="a"),
        AIMessage(content="b"),
    ]
    state = {"provider": "deepseek", "reasoning_content": "r"}

    result = adapter.replay_reasoning_state(outbound, state)

    # Same list spine.
    assert result is outbound
    # Most-recent AIMessage (the last one in the list) got tagged.
    assert outbound[-1].additional_kwargs.get("reasoning_content") == "r"
    # The earlier AIMessage was NOT tagged (walks in reverse, breaks on first AIMessage).
    assert outbound[1].additional_kwargs.get("reasoning_content") is None


def test_deepseek_reasoner_adapter_replay_returns_outbound_unchanged_when_state_none() -> None:
    """PROV-02 Test 4: ``replay_reasoning_state`` returns ``outbound`` byte-identical
    when ``state is None`` (D-08-02 contract — no mutation when nothing to
    replay).
    """
    adapter = DeepSeekReasonerAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="a", additional_kwargs={"existing": "kept"}),
    ]

    result = adapter.replay_reasoning_state(outbound, None)

    assert result is outbound
    assert outbound[-1].additional_kwargs == {"existing": "kept"}


def test_deepseek_reasoner_adapter_capture_does_not_mutate_input_message() -> None:
    """PROV-02 Test 5 (T-09-02-T3 mutation safety): ``capture_reasoning_state``
    returns a NEW dict and does NOT mutate the input ``AIMessage``'s
    ``additional_kwargs``. Compares ``dict(msg.additional_kwargs)`` before and
    after the call.
    """
    adapter = DeepSeekReasonerAdapter()
    original_kwargs = {"reasoning_content": "bar"}
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
    assert msg.additional_kwargs["reasoning_content"] == "bar"


# ─── AnthropicAdapter — PROV-03 (D-09-06 + PATTERNS.md ASYMMETRY CALLOUT) ─────
#
# Critical asymmetry: AnthropicAdapter reads + writes `message.content` (a
# heterogeneous list of blocks including signed `thinking_blocks`), NOT
# `message.additional_kwargs`. The signature inside each thinking block MUST
# round-trip byte-identical or Anthropic's API returns 400 on the next request
# — that contract is what these tests defend.


def test_anthropic_adapter_capture_returns_payload_from_thinking_blocks_in_content() -> None:
    """PROV-03 Test 1 (capture, list content with thinking blocks):
    ``capture_reasoning_state`` filters ``message.content`` for blocks where
    ``type == "thinking"`` and wraps the surviving blocks in the canonical
    ``FOUR_SHAPE_PAYLOADS[1]`` shape — matches
    ``{"provider": "anthropic", "thinking_blocks": [...]}`` exactly.
    """
    adapter = AnthropicAdapter()
    msg = AIMessage(
        content=[
            {"type": "thinking", "signature": "abc", "thinking": "..."},
            {"type": "text", "text": "the visible reply"},
        ],
    )

    payload = adapter.capture_reasoning_state(msg)

    assert payload == {
        "provider": "anthropic",
        "thinking_blocks": [{"type": "thinking", "signature": "abc", "thinking": "..."}],
    }


def test_anthropic_adapter_capture_returns_none_when_content_is_str() -> None:
    """PROV-03 Test 2 (capture, str content): ``capture_reasoning_state``
    returns ``None`` when ``message.content`` is a plain string — non-reasoning
    Anthropic responses (or fallback shapes) surface as str content, not block
    lists, and there are no ``thinking_blocks`` to round-trip.
    """
    adapter = AnthropicAdapter()
    msg = AIMessage(content="just a string reply")

    assert adapter.capture_reasoning_state(msg) is None


def test_anthropic_adapter_capture_returns_none_when_no_thinking_blocks_in_list() -> None:
    """PROV-03 Test 3 (capture, list content without thinking blocks):
    ``capture_reasoning_state`` returns ``None`` when the content list has only
    non-thinking blocks (text-only Anthropic responses, or any block-list
    response with thinking disabled).
    """
    adapter = AnthropicAdapter()
    msg = AIMessage(content=[{"type": "text", "text": "no thinking here"}])

    assert adapter.capture_reasoning_state(msg) is None


def test_anthropic_adapter_replay_prepends_thinking_blocks_to_list_content() -> None:
    """PROV-03 Test 4 (replay onto list content): ``replay_reasoning_state``
    walks ``outbound`` in reverse and PREPENDS the captured ``thinking_blocks``
    to the most-recent AIMessage's ``content`` list. Per Anthropic's API
    contract the signature field MUST round-trip byte-identical — this test
    asserts the prepended block's signature equals what was captured.
    """
    adapter = AnthropicAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content=[{"type": "text", "text": "old reply"}]),
    ]
    state = {
        "provider": "anthropic",
        "thinking_blocks": [{"type": "thinking", "signature": "abc", "thinking": "..."}],
    }

    result = adapter.replay_reasoning_state(outbound, state)

    # Same list spine.
    assert result is outbound
    # Most-recent AIMessage now carries the thinking block PREPENDED.
    msg_content = outbound[-1].content
    assert isinstance(msg_content, list)
    assert len(msg_content) == 2
    assert msg_content[0] == {"type": "thinking", "signature": "abc", "thinking": "..."}
    assert msg_content[1] == {"type": "text", "text": "old reply"}
    # Byte-identical signature round-trip (the Anthropic-API-enforced contract).
    assert msg_content[0]["signature"] == "abc"


def test_anthropic_adapter_replay_is_idempotent_when_thinking_blocks_already_present() -> None:
    """PROV-03 live-run idempotency fix (2026-06-05): when the target AIMessage
    ALREADY has thinking blocks in its content list (the normal case for
    Anthropic — langchain-anthropic surfaces them on .content directly), the
    replay MUST NOT prepend duplicates. Inside the agent's plan() loop,
    `capture` runs after each ainvoke (stashes blocks on additional_kwargs)
    and `replay` runs before the NEXT ainvoke — so within a single agent
    turn that AIMessage is seen by both. Unconditional prepending produced
    `content=[thinking, thinking, text, tool_use]` and Anthropic 400'd with
    'thinking or redacted_thinking blocks in the latest assistant message
    cannot be modified' (request_id req_011CbkoAUyWgQkkoYCe3D5yj).

    The fix: detect that thinking blocks are already present in content and
    leave the message untouched — the existing blocks are the wire-correct
    signed originals from Anthropic's response.
    """
    adapter = AnthropicAdapter()
    original_thinking = {"type": "thinking", "signature": "abc", "thinking": "..."}
    text_block = {"type": "text", "text": "the visible reply"}
    tool_use_block = {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content=[original_thinking, text_block, tool_use_block]),
    ]
    state = {
        "provider": "anthropic",
        "thinking_blocks": [{"type": "thinking", "signature": "abc", "thinking": "..."}],
    }

    adapter.replay_reasoning_state(outbound, state)

    msg_content = outbound[-1].content
    assert isinstance(msg_content, list)
    # Length unchanged — no duplicate prepended.
    assert len(msg_content) == 3, (
        f"replay duplicated thinking blocks; content has {len(msg_content)} "
        f"blocks but should still have 3 (the original Anthropic response). "
        "This is the PROV-03 live-run 400 root cause."
    )
    # Original blocks intact in original order (value-equality; the adapter
    # MAY leave the list spine untouched or rebind it — either is acceptable
    # provided the content remains the wire-correct original).
    assert msg_content[0] == original_thinking
    assert msg_content[1] == text_block
    assert msg_content[2] == tool_use_block
    # Critical wire-contract assertion: the signature on the (only) thinking
    # block is byte-identical to the original. Anthropic's API rejects any
    # mutation, so even a duplicate-and-restore round-trip would 400.
    assert msg_content[0]["signature"] == "abc"
    # No "thinking" block appears more than once.
    thinking_count = sum(
        1 for b in msg_content if isinstance(b, dict) and b.get("type") == "thinking"
    )
    assert thinking_count == 1, (
        f"expected exactly 1 thinking block in content, found {thinking_count}"
    )


def test_anthropic_adapter_replay_promotes_str_content_to_list_with_thinking_blocks() -> None:
    """PROV-03 Test 5 (replay onto str content): when the most-recent
    AIMessage's ``content`` is a string (e.g. earlier non-thinking turn),
    ``replay_reasoning_state`` PROMOTES it to a list: ``[<thinking_block>,
    {"type": "text", "text": <existing str>}]``. This keeps the outbound
    payload in the block-list shape Anthropic expects when thinking_blocks
    are present.
    """
    adapter = AnthropicAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="str reply"),
    ]
    state = {
        "provider": "anthropic",
        "thinking_blocks": [{"type": "thinking", "signature": "abc", "thinking": "..."}],
    }

    adapter.replay_reasoning_state(outbound, state)

    msg_content = outbound[-1].content
    assert isinstance(msg_content, list)
    assert len(msg_content) == 2
    assert msg_content[0] == {"type": "thinking", "signature": "abc", "thinking": "..."}
    assert msg_content[1] == {"type": "text", "text": "str reply"}


def test_anthropic_adapter_replay_returns_outbound_unchanged_when_state_none() -> None:
    """PROV-03 Test 6: ``replay_reasoning_state`` returns ``outbound``
    byte-identical when ``state is None`` (D-08-02 contract — no mutation when
    nothing to replay).
    """
    adapter = AnthropicAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="unchanged"),
    ]

    result = adapter.replay_reasoning_state(outbound, None)

    assert result is outbound
    assert outbound[-1].content == "unchanged"


def test_anthropic_adapter_capture_does_not_mutate_input_message_content() -> None:
    """PROV-03 Test 7 (T-09-03-T3 mutation safety): ``capture_reasoning_state``
    returns a NEW dict whose ``thinking_blocks`` value is a fresh list of
    shallow-copied dicts. Mutating the returned payload MUST NOT reach back
    into the input ``message.content`` (which would otherwise be a shared
    reference into the same provider message — fatal because Anthropic
    enforces signature byte-identity on the wire).
    """
    adapter = AnthropicAdapter()
    original_content = [
        {"type": "thinking", "signature": "abc", "thinking": "..."},
        {"type": "text", "text": "visible"},
    ]
    msg = AIMessage(content=list(original_content))
    before_content = list(msg.content)

    payload = adapter.capture_reasoning_state(msg)
    assert payload is not None
    # Tamper with the returned thinking_blocks — the input's content list
    # MUST remain byte-identical to before the call.
    payload["thinking_blocks"][0]["signature"] = "TAMPERED"

    after_content = list(msg.content)
    assert after_content == before_content, (
        "capture_reasoning_state leaked a shared reference into the input "
        f"message.content list (T3 mitigation failed): "
        f"before={before_content!r}, after={after_content!r}"
    )
    # And the original signature on the message stays "abc" (NOT "TAMPERED").
    assert msg.content[0]["signature"] == "abc"
