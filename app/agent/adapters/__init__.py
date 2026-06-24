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

    def replay_reasoning_state_multi(self, outbound: list[BaseMessage]) -> list[BaseMessage]:
        """Replay per-message _reasoning_state for every in-window AIMessage.

        REPLAY-01 (D-14-03): iterates outbound, and for each AIMessage that
        carries a ``_reasoning_state`` in its ``additional_kwargs``, calls
        the existing single-message ``replay_reasoning_state`` on the
        sub-list up to and including that message.

        Generic default: iterate all AIMessages in outbound and apply
        per-message injection via the existing ``replay_reasoning_state``
        contract. Per-adapter overrides only where wire format demands.
        Flag-off path (the existing ``replay_reasoning_state``) is UNTOUCHED.
        """
        for i, m in enumerate(outbound):
            if isinstance(m, AIMessage):
                per_msg_state = m.additional_kwargs.get("_reasoning_state")
                if per_msg_state is not None:
                    self.replay_reasoning_state(outbound[: i + 1], per_msg_state)
        return outbound


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

    def replay_reasoning_state_multi(self, outbound: list[BaseMessage]) -> list[BaseMessage]:
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


# Registry keyed by provider string. Phase 9 / Plan 09-04 (PROV-04 / D-09-07
# Option B consolidation): now that all four reasoning providers are wired to
# real adapters (PROV-01..04), the dict-comprehension + per-cell mutations
# pattern from Plans 09-01..09-03 is consolidated into a single explicit
# literal here. The literal reads cleaner than `{p: NoOpAdapter() for p in
# SUPPORTED_PROVIDERS}` followed by four override lines, and makes the
# per-provider adapter choice greppable at one site.
#
# PROV-05 / D-09-07 isolation rule: this file (`__init__.py`) is the ONLY
# file in the `app/agent/adapters/` subpackage permitted to import across
# sibling adapter files. Each individual `<provider>.py` imports only from
# `app.agent.adapters` base + `langchain_core` + stdlib — so a per-provider
# revert (e.g. `git revert <hash of Plan 09-04>`) drops the import + literal
# entry here without leaving dangling references in sibling files.
#
# Per-adapter wiring rationale (see each adapter's module docstring for the
# full provider-specific design notes):
#   - openai (PROV-01 / D-09-03 Path B): OpenAIReasoningAdapter pairs with
#     OpenAIReasoningChatModel for the gpt-5 family. gpt-4o-mini stays on
#     plain ChatOpenAI in `app/llm_factory.py` so the v2.0 anchor cannot
#     regress per CLAUDE.md; on the anchor path the adapter returns None on
#     capture and behavior is byte-identical to NoOpAdapter (D-08-08 spirit).
#   - gemini (PROV-04 / D-09-08 / D-09-09): GeminiAdapter round-trips the
#     bytes `thought_signature` payload (asymmetric vs PROV-01/02's str
#     `reasoning_content` and PROV-03's signed `thinking_blocks` list).
#     EXPERIMENTAL — no merge gate; empirical median is logged-not-gated.
#     Critique-loop fix deferred per `project_w10_migration_necessary_not_sufficient`.
#   - deepseek (PROV-02 / D-09-04): DeepSeekReasonerAdapter round-trips
#     `additional_kwargs["reasoning_content"]` for the `deepseek-reasoner`
#     model; non-reasoning DeepSeek (`deepseek-chat`, `deepseek-v4-pro`)
#     paths stay byte-identical to NoOpAdapter behavior on capture (the
#     factory's `DEEPSEEK_REASONER_THINKING_ENABLED` carve-out is what
#     flips thinking ON only for the reasoner family).
#   - kimi (PROV-FUT-02, library-blocked): stays on NoOpAdapter per memory
#     `project_agent_loses_reasoning_state_all_providers` — `langchain-moonshot`
#     does not expose `reasoning_content` at the library boundary, so the
#     adapter would be a no-op anyway.
#   - anthropic (PROV-03 / D-09-05 + D-09-06): AnthropicAdapter reads + writes
#     `message.content` (heterogeneous block list including signed
#     `thinking_blocks`), NOT `message.additional_kwargs`. The signed
#     blocks MUST round-trip byte-identical or Anthropic's API 400s.
#   - scripted (CI / test only): NoOpAdapter — the scripted provider never
#     emits reasoning state.
#
# Imports for the four real adapters are placed AFTER the ABC + base classes
# above to avoid a circular-import deadlock (each adapter does
# ``from app.agent.adapters import ProviderAdapter, StatePayload`` at
# module-load time).
from app.agent.adapters.anthropic import AnthropicAdapter  # noqa: E402
from app.agent.adapters.deepseek import DeepSeekReasonerAdapter  # noqa: E402
from app.agent.adapters.gemini import GeminiAdapter  # noqa: E402
from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter  # noqa: E402

ADAPTERS: dict[str, ProviderAdapter] = {
    "openai": OpenAIReasoningAdapter(),  # PROV-01 (Path B subclass; gpt-5 family only in factory)
    "gemini": GeminiAdapter(),  # PROV-04 (EXPERIMENTAL — no merge gate per D-09-08)
    "deepseek": DeepSeekReasonerAdapter(),  # PROV-02 (model-level carve-out, reasoner family)
    "kimi": NoOpAdapter(),  # PROV-FUT-02 (library-blocked per `project_agent_loses_reasoning_state_all_providers`)
    "anthropic": AnthropicAdapter(),  # PROV-03 (asymmetric: reads/writes `message.content`)
    "scripted": NoOpAdapter(),  # CI/test only — never has reasoning state
}

# Audit-time invariant: the literal MUST cover every provider in
# `SUPPORTED_PROVIDERS` (D-08-08 — Phase 8 contract surface). A drift here
# (e.g. someone adds a new provider to llm_factory.py but forgets to wire it
# into ADAPTERS) would yield a KeyError at graph-build time when
# `ADAPTERS.get(provider, NoOpAdapter())` runs. The assertion lives in the
# test `test_adapters_registry_keys_match_supported_providers` rather than
# at import time so import-time cost stays at zero.


__all__ = [
    "ADAPTERS",
    "AnthropicAdapter",
    "DeepSeekReasonerAdapter",
    "GeminiAdapter",
    "MockReasoningAdapter",
    "NoOpAdapter",
    "OpenAIReasoningAdapter",
    "ProviderAdapter",
    "StatePayload",
]
