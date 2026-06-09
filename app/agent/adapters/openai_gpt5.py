"""OpenAI gpt-5 family reasoning-state adapter (Phase 9 / PROV-01).

Round-trips OpenAI Responses-API reasoning items across agent turns so the
gpt-5 family does NOT lose its reasoning trace between tool-call rounds.

**D-09-03 Path B (probe verdict: "subclass required")**

The Phase 7 probe (`scripts/probe_gpt5_capture.py`, artifact at
`.planning/phases/09-.../09-PROV-01-PROBE.md`) confirmed that
``langchain-openai==1.2.2``'s Chat Completions wrapper for ``gpt-5-mini`` does
NOT surface reasoning content on ``AIMessage.additional_kwargs`` — the API
returns only a ``reasoning_tokens`` counter, never the text. The companion
subclass ``OpenAIReasoningChatModel`` in ``app/llm_factory.py`` switches the
gpt-5 family onto the **Responses API** (which DOES expose reasoning items as
content blocks) and lifts a copy of those blocks into
``AIMessage.additional_kwargs["reasoning_content"]`` so this adapter can read
them through the documented Phase 8 contract.

**Asymmetry callout (D-09-07 / PATTERNS.md):** this adapter reads and writes
the provider-native key ``additional_kwargs["reasoning_content"]``. The
``_reasoning_state`` key is graph.py's storage convention and is set by
``graph.py`` AFTER ``capture_reasoning_state`` returns — adapters do NOT
write to ``_reasoning_state`` themselves. (``MockReasoningAdapter`` is the
single exception; the conformance harness asserts the key end-to-end via
that adapter.)

**PROV-05 / D-09-07 isolation rule:** imports ONLY from
``app.agent.adapters`` base + ``langchain_core`` + stdlib. Never from a
sibling adapter file (e.g. ``deepseek.py``, ``anthropic.py``, ``gemini.py``).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from app.agent.adapters import ProviderAdapter, StatePayload


class OpenAIReasoningAdapter(ProviderAdapter):
    """``ProviderAdapter`` for the OpenAI gpt-5 family (D-09-03 Path B).

    ``capture_reasoning_state`` reads
    ``message.additional_kwargs.get("reasoning_content")`` (populated by
    ``OpenAIReasoningChatModel`` in ``app/llm_factory.py``) and wraps it in
    the canonical ``{"provider": "openai", "reasoning_content": <value>}``
    shape that matches ``FOUR_SHAPE_PAYLOADS[0]`` in the Phase 8 conformance
    harness.

    ``replay_reasoning_state`` walks ``outbound`` in reverse to find the
    most-recent ``AIMessage`` and writes the captured reasoning content back
    onto its ``additional_kwargs["reasoning_content"]`` so the next outbound
    Responses API request carries the same reasoning state to OpenAI.
    """

    PROVIDER_KEY = "openai"

    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        reasoning = message.additional_kwargs.get("reasoning_content")
        if reasoning is None:
            return None
        # Return a fresh dict — never alias the message's own kwargs container
        # (T-09-01-T3 mutation safety: callers may mutate the returned payload
        # without affecting the originating message).
        return {"provider": self.PROVIDER_KEY, "reasoning_content": reasoning}

    def replay_reasoning_state(
        self, outbound: list[BaseMessage], state: StatePayload | None
    ) -> list[BaseMessage]:
        if state is None:
            return outbound
        reasoning = state.get("reasoning_content")
        if reasoning is None:
            return outbound
        # Walk in reverse to find the most-recent AIMessage. The replay target
        # is whatever the next outbound LLM call will see as the LAST assistant
        # turn — that's the message that needs to carry reasoning state forward.
        for msg in reversed(outbound):
            if isinstance(msg, AIMessage):
                msg.additional_kwargs["reasoning_content"] = reasoning
                break
        return outbound


__all__ = ["OpenAIReasoningAdapter"]
