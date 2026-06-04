"""Unit tests for the Phase 8 reasoning-state adapter contract.

Covers D-08-01, D-08-02, D-08-03, D-08-08, and the test-only segregation of
MockReasoningAdapter (D-08-09). The contract surface (ABC + opaque dict +
NoOp default + Mock for tests + registry) is what Phase 9 sub-phases extend
one provider at a time.
"""

from __future__ import annotations

from app.agent.adapters import (
    ADAPTERS,
    MockReasoningAdapter,
    NoOpAdapter,
    ProviderAdapter,
    StatePayload,
)
from langchain_core.messages import AIMessage

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

    Every value in Phase 8 is a NoOpAdapter — zero behavior change vs
    Phase 7. Phase 9 sub-phases swap individual entries.
    """
    assert set(ADAPTERS.keys()) == set(SUPPORTED_PROVIDERS)
    assert all(isinstance(v, NoOpAdapter) for v in ADAPTERS.values())


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
