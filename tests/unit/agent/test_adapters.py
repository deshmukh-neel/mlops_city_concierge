"""Unit tests for the Phase 8 reasoning-state adapter contract.

Covers D-08-01, D-08-02, D-08-03, D-08-08, and the test-only segregation of
MockReasoningAdapter (D-08-09). The contract surface (ABC + opaque dict +
NoOp default + Mock for tests + registry) is what Phase 9 sub-phases extend
one provider at a time.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.adapters import (
    ADAPTERS,
    AnthropicAdapter,
    DeepSeekReasonerAdapter,
    GeminiAdapter,
    MockReasoningAdapter,
    NoOpAdapter,
    OpenAIReasoningAdapter,
    ProviderAdapter,
    StatePayload,
)
from app.llm_factory import SUPPORTED_PROVIDERS


def test_provider_adapter_defines_two_abstract_methods() -> None:
    """D-08-02: exactly two abstract methods — capture + replay.

    Not one combined hook, not a three-method capture/serialize/replay.
    """
    assert hasattr(ProviderAdapter, "capture_reasoning_state")
    assert hasattr(ProviderAdapter, "replay_reasoning_state")
    # ABC enforcement: the two methods are abstract, so direct
    # instantiation must fail.
    abstract_methods = getattr(ProviderAdapter, "__abstractmethods__", frozenset())
    assert "capture_reasoning_state" in abstract_methods
    assert "replay_reasoning_state" in abstract_methods
    assert len(abstract_methods) == 2


def test_state_payload_is_dict_of_str_to_any() -> None:
    """D-08-03: StatePayload is an opaque dict[str, Any] type alias.

    No Union, no discriminator — adding a fifth shape in a future phase is
    a registry addition only, not a shared-type rewrite.
    """
    # The alias is structural; instances built from it are plain dicts.
    payload: StatePayload = {"provider": "openai", "reasoning_content": "x"}
    assert isinstance(payload, dict)


def test_noop_adapter_capture_returns_none() -> None:
    """NoOpAdapter.capture_reasoning_state returns None for every input."""
    adapter = NoOpAdapter()
    assert adapter.capture_reasoning_state(AIMessage(content="x")) is None
    # Non-empty additional_kwargs also yields None — NoOp ignores everything.
    msg_with_kwargs = AIMessage(
        content="y",
        additional_kwargs={"reasoning_content": "should be ignored"},
    )
    assert adapter.capture_reasoning_state(msg_with_kwargs) is None


def test_noop_adapter_replay_returns_outbound_unchanged() -> None:
    """NoOpAdapter.replay_reasoning_state returns outbound unchanged.

    Identity preservation is acceptable; the list contents must not be
    mutated. State payload (if any) is ignored.
    """
    adapter = NoOpAdapter()
    outbound = [AIMessage(content="x")]
    replayed = adapter.replay_reasoning_state(
        outbound, {"provider": "openai", "reasoning_content": "y"}
    )
    # Same list object (no copy required) and contents unchanged.
    assert replayed is outbound
    assert replayed[-1].additional_kwargs == {}


def test_noop_adapter_replay_with_none_state_returns_outbound_unchanged() -> None:
    """NoOpAdapter.replay treats state=None as a no-op."""
    adapter = NoOpAdapter()
    outbound = [AIMessage(content="x")]
    replayed = adapter.replay_reasoning_state(outbound, None)
    assert replayed is outbound
    assert replayed[-1].additional_kwargs == {}


def test_adapters_registry_keys_match_supported_providers() -> None:
    """D-08-08: ADAPTERS keys MUST equal SUPPORTED_PROVIDERS, no drift.

    Phase 8 shipped every value as a NoOpAdapter — zero behavior change vs
    Phase 7. Phase 9 sub-phases swap individual entries in place:
    - PROV-01 (Plan 09-01) swaps `openai` → `OpenAIReasoningAdapter`.
    - PROV-02 (Plan 09-02) swaps `deepseek` → `DeepSeekReasonerAdapter`.
    - PROV-03 (Plan 09-03) adds `anthropic` to SUPPORTED_PROVIDERS and swaps
      it to `AnthropicAdapter` in the same plan.
    - PROV-04 (Plan 09-04) swaps `gemini` → `GeminiAdapter` (EXPERIMENTAL per
      D-09-08 — no merge gate; logged-not-gated empirical median).

    Post-PROV-04 invariant: all four reasoning providers (openai, deepseek,
    anthropic, gemini) are wired to their real adapters; the remaining
    entries (`kimi` PROV-FUT-02 library-blocked, `scripted` CI/test-only)
    stay on NoOpAdapter. This test enforces that "key set = full
    SUPPORTED_PROVIDERS coverage" + "every reasoning-capable provider is
    wired off NoOp + every non-reasoning provider stays on NoOp".
    """
    assert set(ADAPTERS.keys()) == set(SUPPORTED_PROVIDERS)
    # Phase 9 / PROV-01: openai key is now wired to OpenAIReasoningAdapter.
    assert isinstance(ADAPTERS["openai"], OpenAIReasoningAdapter)
    # Phase 9 / PROV-02: deepseek key is now wired to DeepSeekReasonerAdapter.
    assert isinstance(ADAPTERS["deepseek"], DeepSeekReasonerAdapter)
    # Phase 9 / PROV-03: anthropic was added to SUPPORTED_PROVIDERS by Plan
    # 09-03 (first-time wiring per D-09-05) and immediately swapped to the
    # real AnthropicAdapter (D-09-06 carve-out: thinking ENABLED + temp=1.0).
    assert isinstance(ADAPTERS["anthropic"], AnthropicAdapter)
    # Phase 9 / PROV-04 (Plan 09-04): gemini key is now wired to the real
    # GeminiAdapter (EXPERIMENTAL — no merge gate per D-09-08; the bytes
    # `thought_signature` payload round-trips through `additional_kwargs`
    # mirroring `FOUR_SHAPE_PAYLOADS[3]` in the conformance harness).
    assert isinstance(ADAPTERS["gemini"], GeminiAdapter)
    # Providers that PROV-01..04 did NOT swap stay on NoOpAdapter (D-08-08
    # spirit). After PROV-04 lands, that's `kimi` (PROV-FUT-02 library-blocked)
    # and `scripted` (CI/test only — never has reasoning state).
    for provider in SUPPORTED_PROVIDERS:
        if provider in ("openai", "deepseek", "anthropic", "gemini"):
            continue
        assert isinstance(ADAPTERS[provider], NoOpAdapter), (
            f"ADAPTERS[{provider!r}] was unexpectedly swapped off NoOpAdapter "
            f"by Plan 09-01..09-04; only `openai`, `deepseek`, `anthropic`, "
            f"and `gemini` should be swapped post-PROV-04. Got: "
            f"{type(ADAPTERS[provider]).__name__}"
        )


def test_mock_reasoning_adapter_captures_stored_payload() -> None:
    """MockReasoningAdapter.capture returns the stored payload regardless of message."""
    payload: StatePayload = {"provider": "openai", "reasoning_content": "marker"}
    adapter = MockReasoningAdapter(payload=payload)
    # Both empty and populated AIMessages yield the same stored payload.
    assert adapter.capture_reasoning_state(AIMessage(content="anything")) == payload
    assert (
        adapter.capture_reasoning_state(
            AIMessage(content="other", additional_kwargs={"reasoning_content": "diff"})
        )
        == payload
    )


def test_mock_reasoning_adapter_replay_tags_most_recent_ai_message() -> None:
    """D-08-09: MockReasoningAdapter.replay tags the most-recent AIMessage with the payload.

    Sets additional_kwargs['_reasoning_state'] on the AIMessage so the
    conformance harness can detect that the kwarg survived the reducer.
    """
    payload: StatePayload = {"provider": "openai", "reasoning_content": "marker"}
    adapter = MockReasoningAdapter(payload=payload)
    outbound = [AIMessage(content="x")]
    replayed = adapter.replay_reasoning_state(outbound, payload)
    assert replayed[-1].additional_kwargs.get("_reasoning_state") == payload


def test_mock_reasoning_adapter_replay_with_none_state_is_noop() -> None:
    """MockReasoningAdapter.replay treats state=None as a no-op (no marker injected)."""
    payload: StatePayload = {"provider": "openai", "reasoning_content": "marker"}
    adapter = MockReasoningAdapter(payload=payload)
    outbound = [AIMessage(content="x")]
    replayed = adapter.replay_reasoning_state(outbound, None)
    assert replayed[-1].additional_kwargs.get("_reasoning_state") is None


def test_mock_reasoning_adapter_replay_tags_last_ai_message_when_multiple() -> None:
    """When multiple AIMessages are in outbound, the most-recent one (reverse order) gets tagged."""
    payload: StatePayload = {"provider": "openai", "reasoning_content": "marker"}
    adapter = MockReasoningAdapter(payload=payload)
    older = AIMessage(content="old")
    newer = AIMessage(content="new")
    outbound = [older, newer]
    replayed = adapter.replay_reasoning_state(outbound, payload)
    # Newer (last) AIMessage gets the marker; older one stays untouched.
    assert replayed[1].additional_kwargs.get("_reasoning_state") == payload
    assert "_reasoning_state" not in replayed[0].additional_kwargs


def test_mock_reasoning_adapter_not_registered_in_prod_registry() -> None:
    """D-08-09: MockReasoningAdapter is exported for tests but NOT in ADAPTERS.

    Verifies test-only segregation — the production registry must never
    accidentally route through the mock.
    """
    assert not any(isinstance(v, MockReasoningAdapter) for v in ADAPTERS.values())
