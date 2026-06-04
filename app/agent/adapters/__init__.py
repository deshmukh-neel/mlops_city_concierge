"""Reasoning-state provider-adapter contract surface (Phase 8 / v2.1).

Mirrors the register-then-dispatch idiom from `app/llm_factory.py:59-186`
(D-08-04): a module-level `ADAPTERS` dict keyed by provider string is the
single source of truth for which adapter `build_agent_graph` closes over.

Phase 8 ships ONLY the contract:
- `ProviderAdapter` ABC with exactly two abstract methods (D-08-02).
- `StatePayload` opaque dict alias — no discriminator, no Union (D-08-03).
- `NoOpAdapter` registered for every value of `SUPPORTED_PROVIDERS` so the
  gpt-5-mini matrix cell stays at the Phase 7 measurement (D-08-08).
- `MockReasoningAdapter` exported for the conformance harness; explicitly
  NOT registered in `ADAPTERS` (D-08-09).

Phase 9 sub-phases each add one file in this subpackage (e.g.
`openai_gpt5.py`, `anthropic.py`, `deepseek.py`, `gemini.py`) and swap the
corresponding `ADAPTERS` entry so each provider lands independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from app.llm_factory import SUPPORTED_PROVIDERS

# Opaque payload type — provider-internal shape, validated only by the
# adapter that produced it. Mirrors the `conversation_state` opaque-pattern
# already used end-to-end through `/chat`. (D-08-03)
StatePayload = dict[str, Any]


class ProviderAdapter(ABC):
    """Two-method contract for round-tripping per-provider reasoning state.

    Subclasses implement both methods; the agent loop (`plan()` in
    `app/agent/graph.py`) calls `replay` immediately before `ainvoke` and
    `capture` immediately after. The captured `StatePayload` is stashed on
    the just-returned AIMessage's `additional_kwargs["_reasoning_state"]`
    (D-08-06) so it survives the LangGraph `add_messages` reducer between
    turns. (D-08-02)
    """

    @abstractmethod
    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        """Extract provider-specific reasoning state from an inbound AIMessage.

        Returns `None` for non-reasoning providers (or when the message
        carries no reasoning state).
        """

    @abstractmethod
    def replay_reasoning_state(
        self, outbound: list[BaseMessage], state: StatePayload | None
    ) -> list[BaseMessage]:
        """Inject previously-captured reasoning state into the next outbound payload.

        Returns `outbound` unchanged when `state is None`. Implementations
        must not mutate the list spine; in-place edits on individual
        messages' `additional_kwargs` are acceptable per D-08-06.
        """


class NoOpAdapter(ProviderAdapter):
    """Default adapter — captures nothing, replays nothing.

    Registered for every value in `SUPPORTED_PROVIDERS` in Phase 8 so the
    matrix cell behavior is byte-identical to Phase 7 (D-08-08). Phase 9
    sub-phases swap individual `ADAPTERS` entries for real adapters.
    """

    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        return None

    def replay_reasoning_state(
        self, outbound: list[BaseMessage], state: StatePayload | None
    ) -> list[BaseMessage]:
        return outbound


class MockReasoningAdapter(ProviderAdapter):
    """Test-only adapter exported for the conformance harness (D-08-09).

    Constructed with a fixed `payload`; `capture` returns the payload
    regardless of the inbound message; `replay` tags the most-recent
    `AIMessage` in `outbound` by setting
    `additional_kwargs["_reasoning_state"] = state` so the harness can
    assert the marker survives `graph.ainvoke`'s reducer end-to-end.

    Explicitly NOT registered in `ADAPTERS`; production code must never
    route through this class. The conformance test imports it directly.
    """

    def __init__(self, payload: StatePayload) -> None:
        self.payload = payload

    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        return self.payload

    def replay_reasoning_state(
        self, outbound: list[BaseMessage], state: StatePayload | None
    ) -> list[BaseMessage]:
        if state is None:
            return outbound
        # Walk in reverse to find the most-recent AIMessage. The harness
        # asserts the marker lands on that specific message because the
        # REASON-05 gate is about kwarg survival through the reducer.
        for msg in reversed(outbound):
            if isinstance(msg, AIMessage):
                msg.additional_kwargs["_reasoning_state"] = state
                break
        return outbound


# Registry keyed by provider string. Dict-comprehension-driven from
# `SUPPORTED_PROVIDERS` so adding a sixth provider in `llm_factory.py`
# automatically extends the registry without a parallel edit here.
# (D-08-08; mirrors `llm_factory.py` register-then-dispatch shape per D-08-04.)
ADAPTERS: dict[str, ProviderAdapter] = {p: NoOpAdapter() for p in SUPPORTED_PROVIDERS}


__all__ = [
    "ADAPTERS",
    "MockReasoningAdapter",
    "NoOpAdapter",
    "ProviderAdapter",
    "StatePayload",
]
