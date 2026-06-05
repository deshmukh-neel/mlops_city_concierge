"""Gemini 3 reasoning-state adapter (Phase 9 / PROV-04, EXPERIMENTAL).

Round-trips Gemini 3's opaque ``thought_signature`` bytes across agent turns so
the model does NOT lose its reasoning trace between tool-call rounds.

**D-09-08 / D-09-09 design points**

PROV-04 is EXPERIMENTAL — no merge gate. The adapter is built against the
Phase 8 fixture payload ``{"provider": "gemini", "thought_signature":
b"\\x00\\x01\\x02"}`` (``FOUR_SHAPE_PAYLOADS[3]`` in
``tests/integration/test_reasoning_state_roundtrip.py``). The bytes payload
asymmetry vs PROV-01 / PROV-02 (str ``reasoning_content``) and PROV-03 (list
of signed ``thinking_blocks``) is what validates that the Phase 8 contract
scales to a fourth shape with zero ABC churn (REASON-02 acceptance).

**Capture path (primary + fallback)**

CONTEXT.md §``<specifics>`` PROV-04 flags a library-version uncertainty: in
``langchain-google-genai>=4.2.0`` the ``thought_signature`` MAY surface on
``AIMessage.additional_kwargs`` (matching the Phase 8 fixture) OR on
individual ``AIMessage.tool_calls[i]`` items. The adapter handles BOTH:

1. **Primary:** ``message.additional_kwargs.get("thought_signature")``. If
   that returns ``bytes``, wrap it in the canonical payload and return.
2. **Fallback:** scan ``message.tool_calls`` (if present) for the first
   ``tool_call`` dict carrying a bytes ``thought_signature`` key. PROV-04
   ships single-signature; if Gemini 3 ever emits per-tool-call signatures
   and per-call replay matters for the next request, a future v2.2 / Phase
   10 follow-up extends the payload to a list of signatures with index
   alignment. The fallback is in place so the adapter works under either
   library surfacing without iteration cost.

**Replay path**

Walk ``outbound`` in reverse, find the most-recent ``AIMessage``, and write
the captured bytes back to its ``additional_kwargs["thought_signature"]``.
We do NOT attempt to redistribute the signature back across ``tool_calls`` —
the ``additional_kwargs`` round-trip is what the Phase 8 fixture asserts,
and ``langchain-google-genai`` reads from ``additional_kwargs`` at the
outbound boundary via the same library passthrough that surfaced the
signature on capture. Bytes stay bytes; no base64 encoding (a documented
foot-gun from prior W10-era exploration — ``project_gemini3_thought_signatures``).

**Critique-loop deferral (D-09-08 + memory cross-link)**

PROV-04's empirical ``refinement_minimal_edit`` median is LOGGED-NOT-GATED
because the Gemini 3 critique-loop fix (``LOW_SIMILARITY_THRESHOLD=0.55`` in
``app/agent/revision.py:21``) is deferred per project memory
``project_w10_migration_necessary_not_sufficient``. The adapter delivers the
state-preservation half of the PROV-04 charter; the decisiveness half is a
prompt/critique change that cross-cuts every provider and is owned by a
future critique-loop phase (v2.2 or Phase 10.5). The PROV-04 SUMMARY.md
carries this forward for the milestone-archive audit.

**T-09-04-T3 / T-09-04-T7 mutation safety**

``capture_reasoning_state`` returns a FRESH dict; the original message's
``additional_kwargs`` / ``tool_calls`` are never mutated. Python's ``bytes``
is immutable, so reference-identity preservation is the same as
byte-identity preservation — but the defensive copy ``bytes(...)`` is
applied anyway so callers cannot inadvertently alias an internal payload.
The unit test asserts captured bytes match replayed bytes byte-for-byte
end-to-end through the conformance harness (REASON-02 acceptance).

**PROV-05 / D-09-07 isolation rule**

Imports ONLY from ``app.agent.adapters`` base + ``langchain_core`` + stdlib.
Never from a sibling adapter file (e.g. ``openai_gpt5.py``, ``deepseek.py``,
``anthropic.py``). Plan 09-05 ``revert`` simulation depends on this
isolation — a per-provider revert MUST NOT leave dangling imports in
sibling files.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage

from app.agent.adapters import ProviderAdapter, StatePayload


class GeminiAdapter(ProviderAdapter):
    """``ProviderAdapter`` for the Gemini 3 family (D-09-08 / D-09-09).

    ``capture_reasoning_state`` reads
    ``message.additional_kwargs.get("thought_signature")`` first; if absent,
    falls back to scanning ``message.tool_calls`` for a bytes
    ``thought_signature`` (per CONTEXT.md ``<specifics>`` PROV-04 library
    surfacing uncertainty). Returns the canonical
    ``{"provider": "gemini", "thought_signature": <bytes>}`` shape that
    matches ``FOUR_SHAPE_PAYLOADS[3]`` in the Phase 8 conformance harness.

    ``replay_reasoning_state`` walks ``outbound`` in reverse to find the
    most-recent ``AIMessage`` and writes the captured bytes back onto its
    ``additional_kwargs["thought_signature"]`` so the next outbound Gemini
    request carries the signature back to the API. PROV-04 ships
    single-signature replay; per-tool-call alignment is a future
    extension if/when the library surface demands it.

    PROV-04 is EXPERIMENTAL — empirical median is logged-not-gated (D-09-08).
    Decisiveness / critique-loop fix is deferred per memory
    ``project_w10_migration_necessary_not_sufficient``; this adapter delivers
    the state-preservation half of the PROV-04 charter.
    """

    PROVIDER_KEY = "gemini"

    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        # ── Primary path: additional_kwargs["thought_signature"] ────────────
        # The Phase 8 fixture (D-09-09) is shaped this way; the
        # MockReasoningAdapter conformance harness asserts the
        # additional_kwargs path end-to-end. Most likely the
        # langchain-google-genai 4.x wrapper surfaces it here too.
        signature = message.additional_kwargs.get("thought_signature")
        if isinstance(signature, bytes):
            # Defensive copy: bytes is immutable so identity == equality, but
            # the defensive bytes(signature) makes T-09-04-T3 mutation safety
            # mechanically obvious to a reviewer — the captured payload is
            # never aliased to the message's own kwargs container.
            return {
                "provider": self.PROVIDER_KEY,
                "thought_signature": bytes(signature),
            }

        # ── Fallback path: AIMessage.tool_calls[i]["thought_signature"] ─────
        # CONTEXT.md <specifics> PROV-04: lcgg 4.x might surface the field
        # on individual tool_calls rather than additional_kwargs. Scan in
        # order and capture the FIRST bytes signature found. PROV-04 ships
        # single-signature; per-call alignment is a future extension if
        # the library actually emits per-call signatures whose order matters
        # for the next request (today's contract is single-signature per
        # turn).
        tool_calls = getattr(message, "tool_calls", None) or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            sig = tc.get("thought_signature")
            if isinstance(sig, bytes):
                return {
                    "provider": self.PROVIDER_KEY,
                    "thought_signature": bytes(sig),
                }

        return None

    def replay_reasoning_state(
        self, outbound: list[BaseMessage], state: StatePayload | None
    ) -> list[BaseMessage]:
        if state is None:
            return outbound
        signature = state.get("thought_signature")
        if not isinstance(signature, bytes):
            return outbound
        # Walk in reverse to find the most-recent AIMessage. The replay target
        # is whatever the next outbound LLM call will see as the LAST
        # assistant turn — that's the message that needs to carry the
        # thought_signature forward so Gemini's API does not reject the next
        # request as missing the prior reasoning anchor. Bytes stay bytes
        # (no base64 encoding — historical foot-gun documented in memory
        # project_gemini3_thought_signatures).
        for msg in reversed(outbound):
            if isinstance(msg, AIMessage):
                msg.additional_kwargs["thought_signature"] = signature
                break
        return outbound


__all__ = ["GeminiAdapter"]
