"""DeepSeek reasoner-family reasoning-state adapter (Phase 9 / PROV-02).

Round-trips ``deepseek-reasoner``'s ``reasoning_content`` field across agent
turns so the model does NOT lose its reasoning trace between tool-call rounds.

**D-09-04 (model-level carve-out, no subclass required)**

Per the DeepSeek docs and the ``langchain-deepseek>=1.0.0,<2.0.0`` pin's
documented contract, ``ChatDeepSeek`` for ``deepseek-reasoner`` populates
``AIMessage.additional_kwargs["reasoning_content"]`` directly — unlike the
gpt-5 family (Plan 09-01 PROV-01 Path B), no library-level reshape is needed.
The factory-level carve-out flips ``extra_body={"thinking": {"type":
"enabled"}}`` on ``deepseek-reasoner`` (see ``DEEPSEEK_REASONER_THINKING_ENABLED``
in ``app/llm_factory.py``) so the API actually emits the reasoning text;
this adapter then captures + replays it via the documented additional_kwargs
contract.

**Asymmetry callout (D-09-07 / PATTERNS.md):** this adapter reads and writes
the provider-native key ``additional_kwargs["reasoning_content"]``. The
``_reasoning_state`` key is graph.py's storage convention and is set by
``graph.py`` AFTER ``capture_reasoning_state`` returns — adapters do NOT
write to ``_reasoning_state`` themselves. (``MockReasoningAdapter`` is the
single exception; the conformance harness asserts the key end-to-end via
that adapter.)

**PROV-05 / D-09-07 isolation rule:** imports ONLY from
``app.agent.adapters`` base + ``langchain_core`` + stdlib. Never from a
sibling adapter file (e.g. ``openai_gpt5.py``, ``anthropic.py``,
``gemini.py``). Plan 09-05 ``revert`` simulation depends on this isolation.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from app.agent.adapters import ProviderAdapter, StatePayload


class DeepSeekReasonerAdapter(ProviderAdapter):
    """``ProviderAdapter`` for the DeepSeek reasoner family (D-09-04).

    ``capture_reasoning_state`` reads
    ``message.additional_kwargs.get("reasoning_content")`` (populated by
    ``ChatDeepSeek`` for ``deepseek-reasoner`` per the langchain-deepseek
    contract) and wraps it in the canonical
    ``{"provider": "deepseek", "reasoning_content": <value>}`` shape that
    matches ``FOUR_SHAPE_PAYLOADS[2]`` in the Phase 8 conformance harness.

    ``replay_reasoning_state`` walks ``outbound`` in reverse to find the
    most-recent ``AIMessage`` and writes the captured reasoning content back
    onto its ``additional_kwargs["reasoning_content"]`` so the next outbound
    DeepSeek request carries the same reasoning state to the API — DeepSeek
    requires this round-trip on the assistant tool_call message of the NEXT
    request (per project memory ``project_agent_loses_reasoning_state_all_providers``
    documenting the 400 "reasoning_content ... must be passed back" failure
    on agents that drop it between turns).

    Symmetric to ``OpenAIReasoningAdapter`` in shape: both read/write
    ``additional_kwargs["reasoning_content"]``. The difference is upstream
    — OpenAI needed a ``ChatOpenAI`` subclass to lift Responses-API blocks
    into the kwarg; DeepSeek's wrapper populates it natively when the model
    is reasoner-family + thinking is enabled.
    """

    PROVIDER_KEY = "deepseek"

    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        reasoning = message.additional_kwargs.get("reasoning_content")
        if reasoning is None:
            return None
        # Return a fresh dict — never alias the message's own kwargs container
        # (T-09-02-T3 mutation safety: callers may mutate the returned payload
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
        # turn — that's the message that needs to carry reasoning state forward
        # so DeepSeek's API does not 400 on "reasoning_content must be passed
        # back".
        for msg in reversed(outbound):
            if isinstance(msg, AIMessage):
                msg.additional_kwargs["reasoning_content"] = reasoning
                break
        return outbound


__all__ = ["DeepSeekReasonerAdapter"]
