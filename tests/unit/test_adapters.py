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
from app.agent.adapters.gemini import GeminiAdapter
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


# ─── GeminiAdapter — PROV-04 (D-09-08 EXPERIMENTAL — no merge gate) ──────────
#
# Critical asymmetry vs PROV-01 / PROV-02 / PROV-03: the Gemini reasoning
# payload is `bytes` (an opaque `thought_signature`), NOT a `str`
# `reasoning_content` (PROV-01, PROV-02) and NOT a heterogeneous block list
# (PROV-03). The capture path tries `additional_kwargs["thought_signature"]`
# FIRST (matching the Phase 8 fixture FOUR_SHAPE_PAYLOADS[3] and the
# MockReasoningAdapter conformance harness) and falls back to scanning
# `AIMessage.tool_calls[i]` per CONTEXT.md <specifics> PROV-04 (lcgg 4.x
# library-version surfacing uncertainty). The bytes payload MUST round-trip
# byte-identical or Gemini's next request loses its reasoning anchor —
# this file's tests defend that contract.


def test_gemini_adapter_capture_returns_payload_when_additional_kwargs_signature_present() -> None:
    """PROV-04 Test 1 (capture, additional_kwargs primary path): ``capture_reasoning_state``
    reads ``additional_kwargs["thought_signature"]`` (bytes) and wraps it in
    the canonical ``FOUR_SHAPE_PAYLOADS[3]`` shape — matches
    ``{"provider": "gemini", "thought_signature": b"\\x00\\x01\\x02"}`` exactly.
    """
    adapter = GeminiAdapter()
    msg = AIMessage(
        content="x",
        additional_kwargs={"thought_signature": b"\x00\x01\x02"},
    )

    payload = adapter.capture_reasoning_state(msg)

    assert payload == {"provider": "gemini", "thought_signature": b"\x00\x01\x02"}


def test_gemini_adapter_capture_returns_payload_when_tool_calls_carry_signature() -> None:
    """PROV-04 Test 2 (capture, tool_calls fallback path): when
    ``additional_kwargs`` lacks ``thought_signature`` but a tool_call carries
    it, ``capture_reasoning_state`` STILL captures the bytes — falls back to
    scanning ``message.tool_calls`` (CONTEXT.md <specifics> PROV-04: lcgg 4.x
    might surface the field on individual tool_calls rather than at the
    top-level kwarg).

    Note: langchain_core's ``AIMessage`` constructor coerces ``tool_calls`` via
    a strict ``ToolCall`` TypedDict that rejects extra keys (``thought_signature``).
    We bypass the constructor coercion by building the message with a valid
    base tool_call and then assigning the augmented list directly to the
    attribute, mirroring how ``langchain-google-genai`` 4.x would populate
    the field if it chose the per-tool-call surfacing path.
    """
    adapter = GeminiAdapter()
    msg = AIMessage(content="x")
    # Direct attribute assignment bypasses TypedDict coercion — this matches
    # the runtime shape lcgg 4.x produces when it surfaces thought_signature
    # at the per-tool-call level rather than on additional_kwargs.
    msg.tool_calls = [
        {
            "name": "search",
            "args": {"query": "x"},
            "id": "tc_1",
            "thought_signature": b"\xaa\xbb",
        },
    ]

    payload = adapter.capture_reasoning_state(msg)

    # First bytes signature found in tool_calls; PROV-04 ships single-signature
    # (per-call alignment deferred to a future v2.2 / Phase 10 follow-up).
    assert payload == {"provider": "gemini", "thought_signature": b"\xaa\xbb"}


def test_gemini_adapter_capture_returns_none_when_no_signature_present() -> None:
    """PROV-04 Test 3 (capture, no signature anywhere): ``capture_reasoning_state``
    returns ``None`` when neither ``additional_kwargs`` nor any tool_call
    carries a bytes ``thought_signature`` — the D-08-02 contract default.
    Covers non-reasoning Gemini responses (or fallback shapes from any
    Gemini model where thinking is disabled).
    """
    adapter = GeminiAdapter()
    msg = AIMessage(content="x")

    assert adapter.capture_reasoning_state(msg) is None


def test_gemini_adapter_capture_returns_none_when_signature_is_not_bytes() -> None:
    """PROV-04 Test 3b (capture, non-bytes type guard): ``capture_reasoning_state``
    returns ``None`` when the kwarg is present but the value is NOT bytes
    (e.g. somebody base64-decoded it to str upstream — the documented
    foot-gun in memory ``project_gemini3_thought_signatures``). Defensive
    isinstance check; bytes-only is the wire contract.
    """
    adapter = GeminiAdapter()
    msg = AIMessage(
        content="x",
        additional_kwargs={"thought_signature": "not-bytes-str"},
    )

    assert adapter.capture_reasoning_state(msg) is None


def test_gemini_adapter_replay_writes_signature_on_most_recent_ai_message() -> None:
    """PROV-04 Test 4 (replay): ``replay_reasoning_state`` walks ``outbound``
    in reverse and writes ``additional_kwargs["thought_signature"]`` (the
    provider-native key — NOT the ``_reasoning_state`` key, which graph.py
    owns per PATTERNS.md §Shared Patterns) on the most-recent ``AIMessage``.
    This is what makes the next outbound Gemini request carry the
    thought_signature back to the API.
    """
    adapter = GeminiAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="a"),
        AIMessage(content="b"),
    ]
    state = {"provider": "gemini", "thought_signature": b"\x00\x01\x02"}

    result = adapter.replay_reasoning_state(outbound, state)

    # Same list spine.
    assert result is outbound
    # Most-recent AIMessage (the last one in the list) got tagged with bytes.
    assert outbound[-1].additional_kwargs.get("thought_signature") == b"\x00\x01\x02"
    # The earlier AIMessage was NOT tagged (walks in reverse, breaks on first
    # AIMessage encountered).
    assert outbound[1].additional_kwargs.get("thought_signature") is None


def test_gemini_adapter_replay_returns_outbound_unchanged_when_state_none() -> None:
    """PROV-04 Test 5 (replay, None state): ``replay_reasoning_state`` returns
    ``outbound`` byte-identical when ``state is None`` (D-08-02 contract —
    no mutation when nothing to replay).
    """
    adapter = GeminiAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="a", additional_kwargs={"existing": "kept"}),
    ]

    result = adapter.replay_reasoning_state(outbound, None)

    assert result is outbound
    assert outbound[-1].additional_kwargs == {"existing": "kept"}


def test_gemini_adapter_capture_does_not_mutate_input_message() -> None:
    """PROV-04 Test 6 (T-09-04-T3 mutation safety): ``capture_reasoning_state``
    returns a NEW dict and does NOT mutate the input ``AIMessage``'s
    ``additional_kwargs``. Mutating the returned payload's bytes reference
    is impossible (bytes is immutable in Python) but mutating the payload
    dict itself (e.g. replacing the signature) MUST NOT reach back into the
    originating message.
    """
    adapter = GeminiAdapter()
    original_signature = b"\x00\x01\x02"
    original_kwargs = {"thought_signature": original_signature}
    msg = AIMessage(content="x", additional_kwargs=original_kwargs)
    before = dict(msg.additional_kwargs)

    payload = adapter.capture_reasoning_state(msg)
    # Tamper with the returned payload — the input's additional_kwargs MUST
    # remain byte-identical to before the call.
    assert payload is not None
    payload["thought_signature"] = b"\xff\xff"

    after = dict(msg.additional_kwargs)
    assert after == before, (
        "capture_reasoning_state mutated the input message's additional_kwargs "
        f"(T3 mitigation failed): before={before!r}, after={after!r}"
    )
    # Sanity: the original kwarg untouched even after we tampered with payload.
    assert msg.additional_kwargs["thought_signature"] == b"\x00\x01\x02"


def test_gemini_adapter_capture_to_replay_round_trip_preserves_bytes_byte_for_byte() -> None:
    """PROV-04 Test 7 (T-09-04-T7 bytes round-trip): the captured + replayed
    bytes payload survives byte-identical through the adapter contract.

    Mechanics: capture from a fresh AIMessage carrying a bytes signature;
    replay the captured payload onto a fresh outbound list; confirm the
    written bytes match the original byte-for-byte (NOT just value-equal —
    `isinstance(written, bytes) and written == original`). This guards
    against any accidental base64-encoding-as-str regression.
    """
    adapter = GeminiAdapter()
    original = b"\x00\x01\x02\x03\xff\xfe"
    source = AIMessage(content="x", additional_kwargs={"thought_signature": original})

    payload = adapter.capture_reasoning_state(source)
    assert payload is not None
    assert isinstance(payload["thought_signature"], bytes)
    assert payload["thought_signature"] == original

    outbound = [HumanMessage(content="h"), AIMessage(content="next")]
    adapter.replay_reasoning_state(outbound, payload)

    replayed = outbound[-1].additional_kwargs.get("thought_signature")
    assert isinstance(replayed, bytes), (
        f"replayed thought_signature MUST be bytes (not str / base64-decoded); "
        f"got type={type(replayed).__name__}"
    )
    assert replayed == original, (
        f"bytes round-trip drifted: original={original!r}, replayed={replayed!r}"
    )
