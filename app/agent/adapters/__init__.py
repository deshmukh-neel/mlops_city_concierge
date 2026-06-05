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

# Phase 9 / PROV-01 (D-09-03 Path B): swap the openai entry to the real
# OpenAIReasoningAdapter. Import is deferred to AFTER the ABC + registry are
# defined to avoid a circular-import deadlock (openai_gpt5.py does
# ``from app.agent.adapters import ProviderAdapter, StatePayload``). The
# dispatch decision for which OpenAI models actually benefit from the
# adapter lives in ``app/llm_factory.py`` (only ``chat_model.startswith("gpt-5")``
# wires up ``OpenAIReasoningChatModel``; gpt-4o-mini stays on plain ChatOpenAI
# so the v2.0 anchor cannot regress per CLAUDE.md). For gpt-4o-mini the
# adapter is a kwarg-reader: it returns None because plain ChatOpenAI never
# populates ``additional_kwargs["reasoning_content"]``, so behavior on the
# anchor path is byte-identical to NoOpAdapter (D-08-08 spirit preserved).
#
# D-09-07 PROV-05 Option A — cell-by-cell mutation (not the explicit-literal
# Option B). Plan 09-03 will add "anthropic" to SUPPORTED_PROVIDERS; a
# hard-coded literal here would KeyError before that change lands. Plan 09-04
# is the right moment to consolidate to Option B.
from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter  # noqa: E402

ADAPTERS["openai"] = OpenAIReasoningAdapter()

# Phase 9 / PROV-02 (D-09-04): swap the deepseek entry to the real
# DeepSeekReasonerAdapter. ``ChatDeepSeek`` for ``deepseek-reasoner``
# populates ``additional_kwargs["reasoning_content"]`` natively (per the
# documented ``langchain-deepseek>=1.0.0,<2.0.0`` contract), so no subclass
# is required — the factory's model-level conditional in
# ``_DEEPSEEK_REASONER_THINKING_ENABLED`` is what flips thinking ON so the
# model actually emits the field. For ``deepseek-chat`` / ``deepseek-v4-pro``
# the wrapper never populates the kwarg (thinking is disabled by policy), so
# this adapter returns ``None`` on capture and behavior is byte-identical
# to NoOpAdapter on the non-reasoning DeepSeek path (D-08-08 spirit
# preserved; matches the OpenAIReasoningAdapter rule on the gpt-4o-mini
# anchor path).
from app.agent.adapters.deepseek import DeepSeekReasonerAdapter  # noqa: E402

ADAPTERS["deepseek"] = DeepSeekReasonerAdapter()

# Phase 9 / PROV-03 (D-09-05 + D-09-06): swap the anthropic entry to the real
# ``AnthropicAdapter``. ``anthropic`` joined ``SUPPORTED_PROVIDERS`` in
# Plan 09-03 (PROV-03 first-time wiring); the dict-comp above auto-extends
# the registry with a NoOp entry, and this line replaces it.
#
# ASYMMETRY vs the other Phase-9 adapters: ``AnthropicAdapter`` reads + writes
# ``message.content`` (heterogeneous block list including signed
# ``thinking_blocks``), NOT ``message.additional_kwargs``. See
# ``app/agent/adapters/anthropic.py`` top-of-file docstring for the rationale
# (Anthropic surfaces reasoning state on the content block list directly per
# the Claude messages API + ``langchain-anthropic`` passthrough). The signed
# blocks MUST round-trip byte-identical or the API 400s — the unit test
# ``test_anthropic_adapter_replay_prepends_thinking_blocks_to_list_content``
# asserts the signature equality, and Anthropic's API enforces it on the
# wire as a second layer of defense.
from app.agent.adapters.anthropic import AnthropicAdapter  # noqa: E402

ADAPTERS["anthropic"] = AnthropicAdapter()


__all__ = [
    "ADAPTERS",
    "AnthropicAdapter",
    "DeepSeekReasonerAdapter",
    "MockReasoningAdapter",
    "NoOpAdapter",
    "OpenAIReasoningAdapter",
    "ProviderAdapter",
    "StatePayload",
]
