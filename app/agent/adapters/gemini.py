"""Gemini 3 reasoning-state adapter (Phase 9 / PROV-04, EXPERIMENTAL).

Round-trips Gemini 3's opaque function-call ``thought_signature`` payload
across agent turns so the model does NOT lose its reasoning trace between
tool-call rounds.

**D-09-08 / D-09-09 design points (post-live-probe update 2026-06-05)**

PROV-04 is EXPERIMENTAL — no merge gate. The adapter was DESIGNED against
the Phase 8 fixture payload ``{"provider": "gemini", "thought_signature":
b"\\x00\\x01\\x02"}`` (``FOUR_SHAPE_PAYLOADS[3]`` in
``tests/integration/test_reasoning_state_roundtrip.py``) — but a live
``gemini-3.1-pro-preview`` probe via ``langchain-google-genai==4.x``
surfaced a SECOND wire shape that the synthetic fixture does NOT cover:

::

    additional_kwargs["__gemini_function_call_thought_signatures__"]
        : dict[tool_call_id_str, base64_signature_str]

This is the REAL on-the-wire shape lcgg 4.x stores function-call signatures
in (see ``langchain_google_genai/chat_models.py:127`` —
``_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY``). The signatures are stored
as **base64-encoded strings** (lcgg decodes them to bytes only at the
``FunctionCall(thought_signature=...)`` serialization boundary, ~ line 731).
The Phase 8 fixture's bytes-at-top-level shape is the IDEALIZED shape and
still gates REASON-02 conformance via the synthetic harness; the
real-wire shape is what production traffic carries.

PROV-04 supports TWO capture paths:

1. **Primary capture path (real lcgg 4.x):**
   ``additional_kwargs["__gemini_function_call_thought_signatures__"]``
   dict. Per-tool-call base64 strings keyed by tool_call ID. Captured as
   ``{"provider": "gemini", "function_call_thought_signatures": <dict>}``.
   On replay, the dict is written back to the same key on the most-recent
   outbound AIMessage so lcgg 4.x's serializer re-attaches each signature
   to its corresponding outgoing FunctionCall.

2. **Synthetic fixture / Phase 8 conformance path:**
   ``additional_kwargs["thought_signature"]`` bytes (D-09-09 fixture).
   Captured as ``{"provider": "gemini", "thought_signature": <bytes>}``
   matching ``FOUR_SHAPE_PAYLOADS[3]``. On replay, the bytes are written
   back to the same key on the most-recent outbound AIMessage. This path
   keeps the REASON-02 conformance harness passing byte-for-byte while
   the primary path handles real wire traffic.

A prior Path 3 scanned ``message.tool_calls`` for per-call bytes signatures
as a speculative lcgg surfacing variant. The Wave 4 live probe confirmed
lcgg surfaces per-call signatures EXCLUSIVELY at Path 1's key — Path 3 was
dead code and its replay was asymmetric (captured from tool_calls but
written back to additional_kwargs), so WR-02 removed it 2026-06-05.

The two capture paths are tried in priority order; the FIRST one that
yields a non-empty payload wins. Replay symmetrically writes back to
whichever key the captured payload identifies (function-call map dict OR
thought_signature bytes), so a captured payload always round-trips back
to the same wire shape.

**Critique-loop deferral (D-09-08 + memory cross-link)**

PROV-04's empirical ``refinement_minimal_edit`` median is LOGGED-NOT-GATED
because the Gemini 3 critique-loop fix (``LOW_SIMILARITY_THRESHOLD=0.55`` in
``app/agent/revision.py:21``) is deferred per project memory
``project_w10_migration_necessary_not_sufficient``. The adapter delivers the
state-preservation half of the PROV-04 charter; the decisiveness half is a
prompt/critique change that cross-cuts every provider and is owned by a
future critique-loop phase (v2.2 or Phase 10.5).

**T-09-04-T3 / T-09-04-T7 mutation safety**

``capture_reasoning_state`` returns a FRESH dict. For the real lcgg-shape
path, the captured dict is a shallow copy of the message's
``__gemini_function_call_thought_signatures__`` mapping (so downstream
mutation cannot reach back into the message's own kwargs). For the
synthetic-shape path, ``bytes(...)`` defensive copy makes T-09-04-T3
mechanically obvious to a reviewer (bytes is immutable so identity ==
equality, but the explicit copy documents intent). Round-trip
byte-identity is asserted by the unit + conformance tests.

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

# Mirror lcgg 4.x's internal key
# (langchain_google_genai/chat_models.py:127 — `_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY`)
# WITHOUT importing the library (D-09-07 isolation rule). String-literal pin
# is acceptable because the key is part of lcgg's stable public-ish wire
# format — changing it would be a breaking change for any consumer that
# already rounds-trips signatures (which is what this adapter does).
FUNCTION_CALL_THOUGHT_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"

# Phase 8 fixture key (D-09-09) — the synthetic-shape path that gates
# REASON-02 conformance. The fixture-payload bytes never appear in real
# lcgg 4.x output, but the harness drives the adapter through this key
# end-to-end to assert shape-agnostic survival through the LangGraph
# reducer.
SYNTHETIC_FIXTURE_KEY = "thought_signature"


class GeminiAdapter(ProviderAdapter):
    """``ProviderAdapter`` for the Gemini 3 family (D-09-08 / D-09-09).

    Two capture paths in priority order:

    1. **Real lcgg 4.x wire shape:**
       ``additional_kwargs["__gemini_function_call_thought_signatures__"]``
       — ``dict[tool_call_id_str, base64_signature_str]``. lcgg's
       outbound serializer reads from this key to re-attach signatures to
       outgoing FunctionCalls. Captured as
       ``{"provider": "gemini", "function_call_thought_signatures": <dict>}``.

    2. **Phase 8 fixture / synthetic shape:**
       ``additional_kwargs["thought_signature"]`` — ``bytes`` (D-09-09).
       Captured as ``{"provider": "gemini", "thought_signature": <bytes>}``
       matching ``FOUR_SHAPE_PAYLOADS[3]``.

    (A previous Path 3 scanned ``message.tool_calls`` for per-call bytes
    signatures; WR-02 removed it 2026-06-05 after the Wave 4 live probe
    confirmed lcgg never surfaces signatures there — it was dead code with
    an asymmetric capture-vs-replay round-trip.)

    Replay re-attaches the captured payload to the most-recent outbound
    ``AIMessage`` at the SAME wire-shape key that the capture identified —
    so the round-trip wire shape is preserved exactly. No base64
    encoding / decoding inside the adapter — strings stay strings and
    bytes stay bytes; lcgg 4.x owns the base64 ↔ raw-bytes conversion at
    its own serialization boundary.

    PROV-04 is EXPERIMENTAL — empirical median is logged-not-gated
    (D-09-08). Decisiveness / critique-loop fix is deferred per memory
    ``project_w10_migration_necessary_not_sufficient``; this adapter
    delivers the state-preservation half of the PROV-04 charter.
    """

    PROVIDER_KEY = "gemini"

    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        # ── Path 1: real lcgg 4.x wire shape (live-probe-confirmed) ─────────
        # `__gemini_function_call_thought_signatures__` is a dict of
        # per-tool-call base64 signature strings. lcgg's outbound
        # serializer reads from this key when re-emitting FunctionCalls.
        # The live `gemini-3.1-pro-preview` probe (2026-06-05) surfaced
        # this as the real wire shape — the Phase 8 fixture bytes-shape
        # never appears in production traffic.
        fc_signatures = message.additional_kwargs.get(FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
        if isinstance(fc_signatures, dict) and fc_signatures:
            # Shallow-copy the dict and its values (values are str, already
            # immutable) so downstream mutation of the captured payload
            # cannot reach back into the message's own kwargs container
            # (T-09-04-T3 mutation safety).
            return {
                "provider": self.PROVIDER_KEY,
                "function_call_thought_signatures": dict(fc_signatures),
            }

        # ── Path 2: Phase 8 fixture / synthetic shape ───────────────────────
        # The conformance harness drives the adapter through this key. Real
        # lcgg 4.x never populates `additional_kwargs["thought_signature"]`
        # at the top level (signatures live on the function-call map or on
        # content-block `extras` — not at the AIMessage kwarg root). This
        # path keeps REASON-02 conformance asserting byte-for-byte
        # survival through the reducer while Path 1 handles real traffic.
        signature = message.additional_kwargs.get(SYNTHETIC_FIXTURE_KEY)
        if isinstance(signature, bytes):
            # Defensive copy: bytes is immutable so identity == equality, but
            # the explicit bytes(signature) makes T-09-04-T3 mutation safety
            # mechanically obvious to a reviewer.
            return {
                "provider": self.PROVIDER_KEY,
                "thought_signature": bytes(signature),
            }

        # ── Path 3 (REMOVED 2026-06-05, WR-02) ──────────────────────────────
        # A previous Path 3 scanned `message.tool_calls` for a `thought_signature`
        # bytes value and captured it under the synthetic-shape key. The Wave 4
        # live probe against the pinned `langchain-google-genai>=4.0.0,<5.0.0`
        # confirmed lcgg surfaces per-call signatures EXCLUSIVELY at
        # `additional_kwargs[FUNCTION_CALL_THOUGHT_SIGNATURES_KEY]` (Path 1),
        # never on individual tool_call dicts. Capture-from-tool_calls + replay-
        # to-additional_kwargs was an asymmetric round-trip that would silently
        # drop the signature on the wire — dead code in a security-adjacent
        # context (state round-trip drives provider 400s) is a regression
        # hazard. Path 1 (real lcgg) + Path 2 (synthetic fixture) suffice.

        return None

    def replay_reasoning_state(
        self, outbound: list[BaseMessage], state: StatePayload | None
    ) -> list[BaseMessage]:
        if state is None:
            return outbound
        # Determine which wire shape the payload identifies. Prefer the
        # real-wire function-call map (Path 1) over the synthetic bytes
        # (Path 2) when both are present in the payload — defensive
        # ordering for future payload-shape evolution.
        fc_signatures = state.get("function_call_thought_signatures")
        signature = state.get("thought_signature")
        if not isinstance(fc_signatures, dict) and not isinstance(signature, bytes):
            return outbound

        # Walk in reverse to find the most-recent AIMessage. The replay
        # target is whatever the next outbound LLM call will see as the
        # LAST assistant turn — that's the message that needs to carry the
        # signature(s) forward so Gemini's API does not lose its prior
        # reasoning anchor on the next request.
        for msg in reversed(outbound):
            if isinstance(msg, AIMessage):
                if isinstance(fc_signatures, dict):
                    # Real-wire path: write back to lcgg's internal map key
                    # so its outbound serializer re-attaches each signature
                    # to its corresponding FunctionCall.
                    msg.additional_kwargs[FUNCTION_CALL_THOUGHT_SIGNATURES_KEY] = dict(
                        fc_signatures
                    )
                if isinstance(signature, bytes):
                    # Synthetic-shape path: write back to the Phase 8 fixture
                    # key. Bytes stay bytes — no base64 encoding (historical
                    # foot-gun per memory project_gemini3_thought_signatures).
                    msg.additional_kwargs[SYNTHETIC_FIXTURE_KEY] = signature
                break
        return outbound


__all__ = ["GeminiAdapter"]
