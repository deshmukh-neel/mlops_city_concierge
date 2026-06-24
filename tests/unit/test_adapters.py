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

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.adapters import ADAPTERS, ProviderAdapter
from app.agent.adapters.anthropic import AnthropicAdapter
from app.agent.adapters.deepseek import DeepSeekReasonerAdapter
from app.agent.adapters.gemini import GeminiAdapter
from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter

PROBE_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "probe_provider_capture.py"


def load_probe_script():  # type: ignore[return]
    """Load probe_provider_capture.py as a module (it lives outside a package)."""
    spec = importlib.util.spec_from_file_location("probe_provider_capture", PROBE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PROBE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_provider_capture"] = module
    spec.loader.exec_module(module)
    return module


# D-10-11: fixture directory; populated by `make probe-providers`
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "provider_payloads"


def adapter_for(provider: str) -> ProviderAdapter:
    """Dispatch a provider string to the registered ADAPTERS instance."""
    return ADAPTERS[provider]


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


def test_anthropic_adapter_replay_logs_when_target_signature_differs_from_captured(
    caplog,
) -> None:
    """WR-04 observability (2026-06-05): when the idempotency guard skips
    replay because the target AIMessage already has thinking blocks BUT
    the existing block signatures DIFFER from the captured payload's
    signatures, the silent-discard path emits a debug log. Behavior is
    unchanged — target wins, captured payload is discarded — but the log
    surfaces "signature-set mismatch" so a future bug-hunt has telemetry.

    Real-world trigger: a revision step or graph reducer mis-order
    replaces the latest AIMessage with one carrying different signed
    blocks between capture and replay. Without the log, the captured
    payload vanishes with no observable signal.
    """
    import logging

    adapter = AnthropicAdapter()
    target_thinking = {"type": "thinking", "signature": "TARGET_SIG", "thinking": "t"}
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content=[target_thinking, {"type": "text", "text": "reply"}]),
    ]
    # Captured payload carries a DIFFERENT signature than the target's
    # existing thinking block.
    state = {
        "provider": "anthropic",
        "thinking_blocks": [
            {"type": "thinking", "signature": "CAPTURED_SIG", "thinking": "c"},
        ],
    }

    with caplog.at_level(logging.DEBUG, logger="app.agent.adapters.anthropic"):
        adapter.replay_reasoning_state(outbound, state)

    # Behavior contract: target wins — content unchanged (no prepend, no
    # mutation of the target's wire-correct signed block).
    msg_content = outbound[-1].content
    assert isinstance(msg_content, list)
    assert msg_content[0]["signature"] == "TARGET_SIG"
    # Observability contract: the debug log surfaces both signature sets.
    debug_messages = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "app.agent.adapters.anthropic" and rec.levelno == logging.DEBUG
    ]
    assert any("TARGET_SIG" in m and "CAPTURED_SIG" in m for m in debug_messages), (
        "WR-04 debug log missing: expected a record naming both "
        "TARGET_SIG (existing) and CAPTURED_SIG (discarded). "
        f"Got debug records: {debug_messages!r}"
    )


def test_anthropic_adapter_replay_does_not_log_when_signatures_match(
    caplog,
) -> None:
    """WR-04 false-positive guard (2026-06-05): when the target's existing
    thinking-block signatures MATCH the captured payload's signatures
    (the normal in-loop case where capture-then-replay sees the same
    AIMessage), the idempotency skip is benign and the debug log MUST
    NOT fire — otherwise prod logs would fill with spurious "discarded"
    messages every agent turn.
    """
    import logging

    adapter = AnthropicAdapter()
    same_thinking = {"type": "thinking", "signature": "SAME_SIG", "thinking": "t"}
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content=[same_thinking, {"type": "text", "text": "reply"}]),
    ]
    state = {
        "provider": "anthropic",
        "thinking_blocks": [{"type": "thinking", "signature": "SAME_SIG", "thinking": "t"}],
    }

    with caplog.at_level(logging.DEBUG, logger="app.agent.adapters.anthropic"):
        adapter.replay_reasoning_state(outbound, state)

    discard_records = [
        rec
        for rec in caplog.records
        if rec.name == "app.agent.adapters.anthropic" and "discarded" in rec.getMessage().lower()
    ]
    assert not discard_records, (
        "WR-04 debug log fired for matching-signature replay (false positive). "
        f"Got: {[r.getMessage() for r in discard_records]!r}"
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


def test_gemini_adapter_capture_returns_none_when_tool_calls_carry_signature() -> None:
    """PROV-04 Test 2 (WR-02 regression guard, 2026-06-05): the obsolete
    Path 3 scanned ``message.tool_calls`` for a bytes ``thought_signature``
    key and captured it under the synthetic-shape payload. The Wave 4 live
    probe confirmed lcgg 4.x surfaces per-call signatures EXCLUSIVELY at
    ``additional_kwargs[FUNCTION_CALL_THOUGHT_SIGNATURES_KEY]`` (Path 1),
    never on individual tool_call dicts. Path 3 was dead code with an
    asymmetric round-trip (captured from tool_calls but replayed to
    additional_kwargs), so WR-02 removed it. This test pins the new
    behavior: a per-call tool_call signature is IGNORED on capture, so
    Path 1 / Path 2 are the only paths in play.
    """
    adapter = GeminiAdapter()
    msg = AIMessage(content="x")
    # Direct attribute assignment bypasses TypedDict coercion — this matches
    # the runtime shape lcgg 4.x produces. After WR-02, this signature is
    # ignored because the live probe confirmed lcgg does not place
    # signatures here.
    msg.tool_calls = [
        {
            "name": "search",
            "args": {"query": "x"},
            "id": "tc_1",
            "thought_signature": b"\xaa\xbb",
        },
    ]

    payload = adapter.capture_reasoning_state(msg)

    assert payload is None


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


# ─── GeminiAdapter — live-wire shape coverage (post-probe 2026-06-05) ────────
#
# The live `gemini-3.1-pro-preview` probe via `langchain-google-genai==4.x`
# surfaced the REAL wire shape for function-call thought signatures:
#
#     additional_kwargs["__gemini_function_call_thought_signatures__"]
#         : dict[tool_call_id_str, base64_signature_str]
#
# This is the path real Gemini traffic flows through; the bytes-at-top-level
# shape from the Phase 8 fixture never appears in production. The adapter
# handles BOTH (the Phase 8 fixture path gates REASON-02 conformance) — the
# tests below cover the real-wire path explicitly.


def test_gemini_adapter_capture_returns_payload_for_real_lcgg_function_call_map() -> None:
    """PROV-04 Test 8 (live-wire capture path): when ``additional_kwargs``
    carries ``__gemini_function_call_thought_signatures__`` as a dict mapping
    tool_call IDs to base64 signature strings (the real lcgg 4.x wire
    shape, as confirmed by a 2026-06-05 live probe against
    ``gemini-3.1-pro-preview``), ``capture_reasoning_state`` extracts the
    map verbatim and wraps it in
    ``{"provider": "gemini", "function_call_thought_signatures": <dict>}``.
    """
    adapter = GeminiAdapter()
    fc_map = {
        "tc_abc": "EtMCCtACAQw51sf3BTHReBemRCHD6WOWV43KPXSO==",
        "tc_def": "ErICCq8CAQw51scjOGUwKROU3m3IFKNMqErMqrETCQ==",
    }
    msg = AIMessage(
        content="x",
        additional_kwargs={"__gemini_function_call_thought_signatures__": fc_map},
    )

    payload = adapter.capture_reasoning_state(msg)

    assert payload is not None
    assert payload["provider"] == "gemini"
    assert payload["function_call_thought_signatures"] == fc_map


def test_gemini_adapter_capture_does_not_mutate_real_lcgg_function_call_map() -> None:
    """PROV-04 Test 9 (T-09-04-T3 mutation safety on live-wire path): mutating
    the captured ``function_call_thought_signatures`` payload MUST NOT reach
    back into the input ``message.additional_kwargs`` map. The shallow-copy
    in capture is what defends this contract.
    """
    adapter = GeminiAdapter()
    original_map = {"tc_abc": "sig_str"}
    msg = AIMessage(
        content="x",
        additional_kwargs={"__gemini_function_call_thought_signatures__": original_map},
    )
    before = dict(msg.additional_kwargs["__gemini_function_call_thought_signatures__"])

    payload = adapter.capture_reasoning_state(msg)
    assert payload is not None
    # Tamper with the captured map; the input's map MUST remain unchanged.
    payload["function_call_thought_signatures"]["tc_abc"] = "TAMPERED"
    payload["function_call_thought_signatures"]["tc_new"] = "INJECTED"

    after = dict(msg.additional_kwargs["__gemini_function_call_thought_signatures__"])
    assert after == before, (
        f"capture leaked a shared reference into the input message's "
        f"function_call_thought_signatures map (T3 mitigation failed): "
        f"before={before!r}, after={after!r}"
    )


def test_gemini_adapter_replay_writes_real_lcgg_function_call_map_on_most_recent_ai() -> None:
    """PROV-04 Test 10 (live-wire replay path): when the captured payload
    carries ``function_call_thought_signatures``, ``replay_reasoning_state``
    writes the dict back to ``additional_kwargs[
    "__gemini_function_call_thought_signatures__"]`` on the most-recent
    outbound AIMessage. lcgg 4.x's outbound serializer reads from this key
    and re-attaches each base64 signature to its corresponding FunctionCall
    so the next live Gemini request preserves the reasoning anchor.
    """
    adapter = GeminiAdapter()
    outbound = [
        HumanMessage(content="h"),
        AIMessage(content="a"),
        AIMessage(content="b"),
    ]
    fc_map = {"tc_xyz": "base64_sig_str"}
    state = {"provider": "gemini", "function_call_thought_signatures": fc_map}

    result = adapter.replay_reasoning_state(outbound, state)

    assert result is outbound
    # Most-recent AIMessage carries the function-call map at the lcgg key.
    assert (
        outbound[-1].additional_kwargs.get("__gemini_function_call_thought_signatures__") == fc_map
    )
    # Earlier AIMessage was NOT tagged.
    assert outbound[1].additional_kwargs.get("__gemini_function_call_thought_signatures__") is None


def test_gemini_adapter_real_lcgg_path_takes_priority_over_synthetic_path() -> None:
    """PROV-04 Test 11 (priority ordering): when BOTH wire shapes are present
    on the same AIMessage (synthetic edge case — unlikely in production but
    possible if a synthetic test path leaks the fixture key through), the
    real-wire ``function_call_thought_signatures`` dict captures first.
    Real traffic should never carry both; this test documents the
    deterministic priority for future maintainers.
    """
    adapter = GeminiAdapter()
    msg = AIMessage(
        content="x",
        additional_kwargs={
            "__gemini_function_call_thought_signatures__": {"tc_1": "sig_str"},
            "thought_signature": b"\x00\x01\x02",
        },
    )

    payload = adapter.capture_reasoning_state(msg)

    assert payload is not None
    # Real-wire path wins: payload carries the dict, NOT the bytes.
    assert "function_call_thought_signatures" in payload
    assert payload["function_call_thought_signatures"] == {"tc_1": "sig_str"}
    assert "thought_signature" not in payload


def test_gemini_adapter_real_lcgg_round_trip_preserves_dict_keys_and_values() -> None:
    """PROV-04 Test 12 (live-wire round-trip): the captured + replayed
    ``function_call_thought_signatures`` dict survives equal through the
    adapter contract. Keys (tool_call IDs) and values (base64 strings) MUST
    be preserved exactly so lcgg's outbound serializer can re-align each
    signature with its corresponding FunctionCall.
    """
    adapter = GeminiAdapter()
    original_map = {
        "tc_aaa": "ErICCq8CAQw51scjOGUwKROU3m3IFKNMqErMqrETCQ==",
        "tc_bbb": "EtMCCtACAQw51sf3BTHReBemRCHD6WOWV43KPXSO==",
        "tc_ccc": "EtACAQw5xyzABC123==",
    }
    source = AIMessage(
        content="x",
        additional_kwargs={"__gemini_function_call_thought_signatures__": original_map},
    )
    payload = adapter.capture_reasoning_state(source)
    assert payload is not None

    outbound = [HumanMessage(content="h"), AIMessage(content="next")]
    adapter.replay_reasoning_state(outbound, payload)

    replayed = outbound[-1].additional_kwargs.get("__gemini_function_call_thought_signatures__")
    assert isinstance(replayed, dict), (
        "replayed function_call_thought_signatures MUST be a dict (lcgg's "
        f"outbound serializer reads it as a mapping); got type={type(replayed).__name__}"
    )
    assert replayed == original_map, (
        f"function_call_thought_signatures round-trip drifted: "
        f"original={original_map!r}, replayed={replayed!r}"
    )


# ─── EVAL-05: parametrized real-wire fixture-loading tests ───────────────────
#
# These tests AUGMENT the synthetic cases above — they load the JSON fixtures
# written by `make probe-providers` and verify that each provider's adapter
# does not crash against the real wire shape (D-10-12).
#
# Existing synthetic tests document the contract and run without files.
# These tests close the live-shape gap (D-09-09 Gemini lcgg key miss, 4 live
# Anthropic bugs). When no fixture is present (e.g. in CI without keys), the
# test SKIPS gracefully — CI never needs to run `make probe-providers`.


@pytest.mark.parametrize("provider", ["openai", "deepseek", "anthropic", "gemini"])
def test_adapter_capture_on_real_wire_fixture(provider: str) -> None:
    """EVAL-05 / D-10-12: Load checked-in real-wire fixture and verify the
    provider's adapter capture_reasoning_state does not crash.

    Existing synthetic dict tests document the contract and run without files;
    this test closes the live-shape gap (D-09-09 Gemini lcgg key miss, 4 live
    Anthropic bugs). SKIPS when no fixture is present (CI-safe).
    """
    fixture_path = FIXTURE_DIR / f"{provider}.json"
    if not fixture_path.exists():
        pytest.skip(f"No fixture for {provider} — run `make probe-providers` first")

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Reconstruct an AIMessage from the fixture's captured fields.
    # For Anthropic: the adapter reads message.content (block list) so
    # reconstruct list-content when content_shape indicates blocks.
    additional_kwargs = payload.get("additional_kwargs_values", {})
    response_metadata = payload.get("response_metadata", {})
    content_shape: str = payload.get("content_shape", "str (len=0)")

    if provider == "anthropic" and content_shape.startswith("list"):
        # Anthropic thinking-enabled responses use heterogeneous block lists.
        # Reconstruct a minimal block list matching the real shape.
        content: object = [{"type": "text", "text": "probe response"}]
    else:
        content = "probe response"

    msg = AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata,
    )

    adapter = adapter_for(provider)
    # Must not raise; result is either None or a dict with a "provider" key.
    result = adapter.capture_reasoning_state(msg)
    if result is not None:
        assert "provider" in result, (
            f"capture_reasoning_state for '{provider}' returned a result without "
            f"a 'provider' key: {result!r}"
        )


# ─── WR-10: probe fixture type-fidelity tests ─────────────────────────────────
#
# The probe script's redact function called str(value) on additional_kwargs
# values, collapsing dict/list/number values to Python repr strings like
# "{'k': 'v'}".  WR-10 fixes this by routing additional_kwargs values through
# the same json.loads(redact(json.dumps(..., default=str))) pattern already
# used for response_metadata/usage_metadata/tool_calls.
#
# These tests verify:
#   1. A dict-valued additional_kwargs entry survives as a real dict (not str repr).
#   2. Secrets in additional_kwargs values are still redacted after the fix.
#   3. The adapter fixture test still passes/skips correctly (no regression).
#
# Tests use a synthetic fixture in tmp_path — no live keys needed (CI-safe).


@pytest.fixture
def probe():  # type: ignore[return]
    """The probe_provider_capture module under test."""
    return load_probe_script()


def test_wr10_dict_additional_kwargs_value_preserved_as_dict(probe, tmp_path: Path) -> None:
    """WR-10: a dict-valued additional_kwargs entry must survive redaction as a dict.

    Before the fix, redact(message.additional_kwargs[k]) calls str(value) on a
    dict, producing "{'reasoning': {'tokens': 42}}" — a Python repr string, not
    JSON.  After the fix, json.loads(redact(json.dumps(value, default=str)))
    round-trips it as {"reasoning": {"tokens": 42}}.
    """
    # Simulate the fixture-build path: apply the type-faithful redaction to a dict value.
    structured_value = {"reasoning": {"tokens": 42, "model": "test-model"}}
    # WR-10 fix: json.loads(redact(json.dumps(v, default=str)))
    redacted = json.loads(probe.redact(json.dumps(structured_value, default=str)))
    assert isinstance(redacted, dict), (
        f"WR-10: dict additional_kwargs value must survive as dict, got {type(redacted).__name__!r}: "
        f"{redacted!r}"
    )
    assert redacted == structured_value, (
        f"WR-10: dict value must round-trip faithfully; got {redacted!r}"
    )


def test_wr10_list_additional_kwargs_value_preserved_as_list(probe) -> None:
    """WR-10: a list-valued additional_kwargs entry must survive redaction as a list."""
    list_value = [{"type": "thinking", "thinking": "step 1"}, {"type": "text", "text": "done"}]
    redacted = json.loads(probe.redact(json.dumps(list_value, default=str)))
    assert isinstance(redacted, list), (
        f"WR-10: list additional_kwargs value must survive as list, got {type(redacted).__name__!r}"
    )
    assert len(redacted) == 2


def test_wr10_secrets_in_additional_kwargs_are_still_redacted(probe, monkeypatch) -> None:
    """WR-10: type-fidelity must not weaken redaction — secrets in dict values are removed.

    Plants a fake secret in an env var and embeds it in a dict-valued additional_kwargs
    entry.  The WR-10 redaction path must strip it.
    """
    fake_secret = "sk-" + "A" * 30  # matches SECRET_PATTERNS OpenAI pattern
    # Plant the secret inside a nested dict value.
    structured_value = {"nested": {"api_key": fake_secret, "safe_field": "hello"}}
    # Apply the WR-10 redaction path.
    redacted_str = probe.redact(json.dumps(structured_value, default=str))
    assert fake_secret not in redacted_str, (
        "WR-10: secret in additional_kwargs dict value must be redacted"
    )
    assert "<REDACTED>" in redacted_str, (
        "WR-10: redaction marker must appear after stripping the secret"
    )
    # The round-trip must still produce a dict (type preserved despite redaction).
    redacted = json.loads(redacted_str)
    assert isinstance(redacted, dict), "WR-10: redacted value must still be a dict"


def test_wr10_synthetic_fixture_roundtrips_dict_value(probe, tmp_path: Path) -> None:
    """WR-10: a synthetic type-faithful fixture with a dict additional_kwargs value
    can be written and re-loaded, with the dict type preserved (not stringified).

    This exercises the full fixture-build shape used by test_adapter_capture_on_real_wire_fixture.
    """
    structured_value = {"reasoning_effort": "high", "tokens": 128}
    # Apply the WR-10 redaction (same code path the fixed probe uses).
    add_kwargs_values = {
        "reasoning_config": json.loads(probe.redact(json.dumps(structured_value, default=str)))
    }

    # Write a synthetic fixture to tmp_path.
    fixture = {
        "provider": "openai",
        "model": "gpt-5-mini",
        "library_version": "0.0.0",
        "probe_query": "search for a bar in mission",
        "additional_kwargs_keys": ["reasoning_config"],
        "additional_kwargs_values": add_kwargs_values,
        "response_metadata": {},
        "content_shape": "str (len=12)",
        "usage_metadata": None,
        "tool_calls": [],
    }
    fixture_file = tmp_path / "openai.json"
    fixture_file.write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    # Re-load the fixture (same as test_adapter_capture_on_real_wire_fixture does).
    payload = json.loads(fixture_file.read_text(encoding="utf-8"))
    additional_kwargs = payload.get("additional_kwargs_values", {})

    # The dict value must be a real dict, not a str repr.
    assert "reasoning_config" in additional_kwargs
    assert isinstance(additional_kwargs["reasoning_config"], dict), (
        f"WR-10: additional_kwargs_values['reasoning_config'] must be dict after fixture round-trip, "
        f"got {type(additional_kwargs['reasoning_config']).__name__!r}"
    )
    assert additional_kwargs["reasoning_config"] == structured_value


# ─── Multi-replay (REPLAY-01) — per-adapter conformance ──────────────────────
#
# D-14-04: All four adapters get additive multi-replay coverage even though
# only three models run in the experiment. The generic ABC default covers
# OpenAI and DeepSeek (both use additional_kwargs["reasoning_content"]);
# Anthropic and Gemini tests verify their wire-format asymmetry is preserved
# through per-message multi-replay.
#
# Three tests per adapter:
#   1. Per-message injection (flag-on): each AIMessage receives its own state.
#   2. Skip on no state: AIMessages without _reasoning_state are untouched.
#   3. Flag-off non-interference: the existing single-message
#      replay_reasoning_state is byte-identical — documents the contract that
#      enables per-adapter revertability (Phase 9 precedent).
#
# All tests instantiate adapters directly with synthesized AIMessages.
# No graph, no LLM, no DB.


# ── OpenAIReasoningAdapter ────────────────────────────────────────────────────


def test_openai_reasoning_adapter_multi_replay_injects_per_message_state() -> None:
    """REPLAY-01: replay_reasoning_state_multi writes per-message _reasoning_state
    onto each in-window AIMessage's additional_kwargs["reasoning_content"].

    Verifies that msg1 receives r1 AND msg2 receives r2 — distinct values per
    message, not the most-recent-only behavior of the single-message path.
    """
    adapter = OpenAIReasoningAdapter()
    msg1 = AIMessage(
        content="first turn",
        additional_kwargs={"_reasoning_state": {"provider": "openai", "reasoning_content": "r1"}},
    )
    msg2 = AIMessage(
        content="second turn",
        additional_kwargs={"_reasoning_state": {"provider": "openai", "reasoning_content": "r2"}},
    )
    outbound = [HumanMessage(content="h"), msg1, msg2]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    # Both messages received their own distinct reasoning_content (per-message injection).
    assert msg1.additional_kwargs.get("reasoning_content") == "r1"
    assert msg2.additional_kwargs.get("reasoning_content") == "r2"


def test_openai_reasoning_adapter_multi_replay_skips_messages_without_state() -> None:
    """REPLAY-01: messages with no _reasoning_state are left untouched.

    An AIMessage that carries no _reasoning_state key must NOT get a
    reasoning_content injected — it stays None.
    """
    adapter = OpenAIReasoningAdapter()
    msg_no_state = AIMessage(content="no state")
    outbound = [HumanMessage(content="h"), msg_no_state]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    assert msg_no_state.additional_kwargs.get("reasoning_content") is None


def test_openai_reasoning_adapter_multi_replay_flag_off_path_unchanged() -> None:
    """REPLAY-01 flag-off non-interference: the existing single-message
    replay_reasoning_state signature is UNTOUCHED. This documents the
    contract that keeps the Phase-13 plateau byte-identical when the flag is
    off. Per-adapter revertability: removing the ABC default restores flag-off
    byte-identity without any other changes.
    """
    adapter = OpenAIReasoningAdapter()
    outbound = [HumanMessage(content="h"), AIMessage(content="a"), AIMessage(content="b")]
    state = {"provider": "openai", "reasoning_content": "r"}

    result = adapter.replay_reasoning_state(outbound, state)

    # Same list spine; most-recent AIMessage got reasoning_content.
    assert result is outbound
    assert outbound[-1].additional_kwargs.get("reasoning_content") == "r"
    # The earlier AIMessage was NOT tagged (single-message path only targets most-recent).
    assert outbound[1].additional_kwargs.get("reasoning_content") is None


# ── DeepSeekReasonerAdapter ───────────────────────────────────────────────────


def test_deepseek_reasoner_adapter_multi_replay_injects_per_message_state() -> None:
    """REPLAY-01: replay_reasoning_state_multi writes per-message _reasoning_state
    onto each in-window AIMessage's additional_kwargs["reasoning_content"].

    DeepSeek uses the same additional_kwargs wire shape as OpenAI — the ABC
    generic default via replay_reasoning_state applies correctly per message.
    """
    adapter = DeepSeekReasonerAdapter()
    msg1 = AIMessage(
        content="first turn",
        additional_kwargs={
            "_reasoning_state": {"provider": "deepseek", "reasoning_content": "ds_r1"}
        },
    )
    msg2 = AIMessage(
        content="second turn",
        additional_kwargs={
            "_reasoning_state": {"provider": "deepseek", "reasoning_content": "ds_r2"}
        },
    )
    outbound = [HumanMessage(content="h"), msg1, msg2]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    # Both messages received their own distinct reasoning_content.
    assert msg1.additional_kwargs.get("reasoning_content") == "ds_r1"
    assert msg2.additional_kwargs.get("reasoning_content") == "ds_r2"


def test_deepseek_reasoner_adapter_multi_replay_skips_messages_without_state() -> None:
    """REPLAY-01: messages with no _reasoning_state are left untouched.

    An AIMessage carrying no _reasoning_state must NOT receive reasoning_content
    — covers deepseek-chat (non-reasoner) messages that may be in-window but
    carry no captured state.
    """
    adapter = DeepSeekReasonerAdapter()
    msg_no_state = AIMessage(content="no state")
    outbound = [HumanMessage(content="h"), msg_no_state]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    assert msg_no_state.additional_kwargs.get("reasoning_content") is None


def test_deepseek_reasoner_adapter_multi_replay_flag_off_path_unchanged() -> None:
    """REPLAY-01 flag-off non-interference: the existing single-message
    replay_reasoning_state signature is UNTOUCHED — documents the contract
    that enables per-adapter revertability (Phase 9 precedent).
    """
    adapter = DeepSeekReasonerAdapter()
    outbound = [HumanMessage(content="h"), AIMessage(content="a"), AIMessage(content="b")]
    state = {"provider": "deepseek", "reasoning_content": "r"}

    result = adapter.replay_reasoning_state(outbound, state)

    assert result is outbound
    assert outbound[-1].additional_kwargs.get("reasoning_content") == "r"
    assert outbound[1].additional_kwargs.get("reasoning_content") is None


# ── AnthropicAdapter ──────────────────────────────────────────────────────────
#
# Critical asymmetry: AnthropicAdapter writes onto message.content (block list),
# NOT additional_kwargs. The ABC generic default calls the single-message
# replay_reasoning_state per message, which in turn uses the content-list
# injection path — so the multi-replay test must assert content-list state,
# not additional_kwargs["reasoning_content"].
#
# The idempotency guard (PROV-03 live-run fix 2026-06-05) applies per message:
# if a target AIMessage already has thinking blocks in its content, the replay
# skips that message's inject step.


def test_anthropic_adapter_multi_replay_injects_per_message_thinking_blocks() -> None:
    """REPLAY-01 Anthropic: replay_reasoning_state_multi injects thinking blocks
    onto EACH targeted AIMessage's content list via content-list injection (NOT
    additional_kwargs), respecting the PROV-03 wire-format asymmetry.

    Both msg1 (str content) and msg2 (list content without thinking) receive
    their own captured thinking_blocks prepended/promoted into content.
    """
    adapter = AnthropicAdapter()
    # msg1: str content — will be promoted to block list.
    msg1 = AIMessage(
        content="first str reply",
        additional_kwargs={
            "_reasoning_state": {
                "provider": "anthropic",
                "thinking_blocks": [{"type": "thinking", "signature": "sig1", "thinking": "t1"}],
            }
        },
    )
    # msg2: list content with no thinking blocks — thinking blocks will be prepended.
    msg2 = AIMessage(
        content=[{"type": "text", "text": "second list reply"}],
        additional_kwargs={
            "_reasoning_state": {
                "provider": "anthropic",
                "thinking_blocks": [{"type": "thinking", "signature": "sig2", "thinking": "t2"}],
            }
        },
    )
    outbound = [HumanMessage(content="h"), msg1, msg2]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    # msg1: str was promoted to [thinking_block, text_block].
    assert isinstance(msg1.content, list)
    assert len(msg1.content) == 2
    assert msg1.content[0] == {"type": "thinking", "signature": "sig1", "thinking": "t1"}
    assert msg1.content[1] == {"type": "text", "text": "first str reply"}
    # msg2: thinking block prepended to existing list content.
    assert isinstance(msg2.content, list)
    assert len(msg2.content) == 2
    assert msg2.content[0] == {"type": "thinking", "signature": "sig2", "thinking": "t2"}
    assert msg2.content[1] == {"type": "text", "text": "second list reply"}


def test_anthropic_adapter_multi_replay_idempotency_guard_per_message() -> None:
    """REPLAY-01 Anthropic idempotency: the idempotency guard applies per message
    in the multi-replay path. A message already carrying thinking blocks in its
    content is not given duplicate blocks — the existing blocks win.

    This is the PROV-03 live-run idempotency fix applied through the multi path:
    the ABC generic default calls the single-message replay per message, which
    contains the idempotency guard, so duplicates are impossible.
    """
    adapter = AnthropicAdapter()
    existing_thinking = {"type": "thinking", "signature": "existing_sig", "thinking": "already"}
    # msg already has thinking blocks in content — idempotency guard should skip replay.
    msg_with_blocks = AIMessage(
        content=[existing_thinking, {"type": "text", "text": "reply"}],
        additional_kwargs={
            "_reasoning_state": {
                "provider": "anthropic",
                "thinking_blocks": [
                    {"type": "thinking", "signature": "existing_sig", "thinking": "already"}
                ],
            }
        },
    )
    outbound = [HumanMessage(content="h"), msg_with_blocks]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    # Content unchanged — no duplicate thinking block prepended.
    assert isinstance(msg_with_blocks.content, list)
    assert len(msg_with_blocks.content) == 2, (
        f"idempotency guard did not fire: content has {len(msg_with_blocks.content)} "
        "blocks but should still have 2 (no duplicate thinking block)"
    )
    assert msg_with_blocks.content[0] == existing_thinking
    # Exactly one thinking block — guard prevented duplication.
    thinking_count = sum(
        1 for b in msg_with_blocks.content if isinstance(b, dict) and b.get("type") == "thinking"
    )
    assert thinking_count == 1


def test_anthropic_adapter_multi_replay_flag_off_path_unchanged() -> None:
    """REPLAY-01 flag-off non-interference: the existing single-message
    replay_reasoning_state writes content-list injection on the most-recent
    AIMessage ONLY — multi-replay does not alter this signature.
    """
    adapter = AnthropicAdapter()
    msg_old = AIMessage(content=[{"type": "text", "text": "old turn"}])
    msg_recent = AIMessage(content="recent str reply")
    outbound = [HumanMessage(content="h"), msg_old, msg_recent]
    state = {
        "provider": "anthropic",
        "thinking_blocks": [{"type": "thinking", "signature": "abc", "thinking": "..."}],
    }

    result = adapter.replay_reasoning_state(outbound, state)

    assert result is outbound
    # Most-recent AIMessage (msg_recent) was promoted from str → block list.
    assert isinstance(msg_recent.content, list)
    assert len(msg_recent.content) == 2
    assert msg_recent.content[0]["type"] == "thinking"
    # Earlier msg_old was NOT touched.
    assert msg_old.content == [{"type": "text", "text": "old turn"}]


# ── GeminiAdapter ─────────────────────────────────────────────────────────────
#
# GeminiAdapter has two wire shapes; each multi-replay test covers one shape:
#   - Real lcgg 4.x: additional_kwargs["__gemini_function_call_thought_signatures__"] (dict)
#   - Synthetic fixture: additional_kwargs["thought_signature"] (bytes)
# The multi-replay test verifies the correct provider key is injected per message
# via the ABC generic default (which delegates to the single-message replay per
# message, which writes to the correct wire-shape key).


def test_gemini_adapter_multi_replay_injects_per_message_real_lcgg_path() -> None:
    """REPLAY-01 Gemini (real-wire path): replay_reasoning_state_multi injects
    the correct provider key (__gemini_function_call_thought_signatures__) per
    message for the real lcgg 4.x wire shape.

    Both msg1 and msg2 receive their own captured fc_map at the lcgg key — per
    message, not just the most-recent.
    """
    adapter = GeminiAdapter()
    fc_map1 = {"tc_aaa": "sig_str_1"}
    fc_map2 = {"tc_bbb": "sig_str_2"}
    msg1 = AIMessage(
        content="first turn",
        additional_kwargs={
            "_reasoning_state": {
                "provider": "gemini",
                "function_call_thought_signatures": fc_map1,
            }
        },
    )
    msg2 = AIMessage(
        content="second turn",
        additional_kwargs={
            "_reasoning_state": {
                "provider": "gemini",
                "function_call_thought_signatures": fc_map2,
            }
        },
    )
    outbound = [HumanMessage(content="h"), msg1, msg2]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    # Both messages received their own fc_map at the correct lcgg key.
    assert msg1.additional_kwargs.get("__gemini_function_call_thought_signatures__") == fc_map1
    assert msg2.additional_kwargs.get("__gemini_function_call_thought_signatures__") == fc_map2


def test_gemini_adapter_multi_replay_skips_messages_without_state() -> None:
    """REPLAY-01 Gemini: messages with no _reasoning_state are left untouched.

    An AIMessage with no _reasoning_state must NOT get any thought signature
    key injected — covers non-tool-calling Gemini turns in-window.
    """
    adapter = GeminiAdapter()
    msg_no_state = AIMessage(content="no state")
    outbound = [HumanMessage(content="h"), msg_no_state]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    assert msg_no_state.additional_kwargs.get("__gemini_function_call_thought_signatures__") is None
    assert msg_no_state.additional_kwargs.get("thought_signature") is None


def test_gemini_adapter_multi_replay_flag_off_path_unchanged() -> None:
    """REPLAY-01 flag-off non-interference: the existing single-message
    replay_reasoning_state writes the lcgg key to the most-recent AIMessage ONLY —
    multi-replay does not alter this signature. Documents the non-interference
    contract that keeps the Phase-13 plateau byte-identical when the flag is off.
    """
    adapter = GeminiAdapter()
    msg_old = AIMessage(content="a")
    msg_recent = AIMessage(content="b")
    outbound = [HumanMessage(content="h"), msg_old, msg_recent]
    fc_map = {"tc_xyz": "base64_sig"}
    state = {"provider": "gemini", "function_call_thought_signatures": fc_map}

    result = adapter.replay_reasoning_state(outbound, state)

    assert result is outbound
    # Most-recent AIMessage received the fc_map at the lcgg key.
    assert msg_recent.additional_kwargs.get("__gemini_function_call_thought_signatures__") == fc_map
    # Earlier AIMessage was NOT tagged (single-message path only targets most-recent).
    assert msg_old.additional_kwargs.get("__gemini_function_call_thought_signatures__") is None
