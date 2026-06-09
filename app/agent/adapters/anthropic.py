"""Anthropic Claude reasoning-state adapter (Phase 9 / PROV-03).

Round-trips Claude's signed ``thinking_blocks`` across agent turns so the
model does NOT lose its reasoning trace between tool-call rounds.

**ASYMMETRY CALLOUT vs the other 3 Phase-9 adapters (PATTERNS.md §
"app/agent/adapters/anthropic.py (PROV-03) — ASYMMETRY CALLOUT" and
CONTEXT.md §<specifics> PROV-03 Anthropic ``thinking_blocks`` shape):**

``AnthropicAdapter`` reads and writes ``message.content`` (a heterogeneous
list of blocks, including ``{"type": "thinking", "signature": "...",
"thinking": "..."}`` blocks alongside ``{"type": "text", ...}`` blocks).
This is **different** from ``OpenAIReasoningAdapter`` /
``DeepSeekReasonerAdapter`` / ``GeminiAdapter``, which all read and write
``message.additional_kwargs``. Do NOT cross-apply the additional_kwargs
pattern here — Anthropic's API surfaces reasoning state on the content
block list directly and ``langchain-anthropic`` passes that list through
onto ``AIMessage.content`` unchanged.

**Byte-identical signature contract:** the ``signature`` field inside each
thinking block MUST round-trip byte-identical or Anthropic's API returns
400 on the next request ("signature mismatch" / equivalent). The
double-defense:

1. ``capture_reasoning_state`` shallow-copies each thinking block dict so
   downstream mutation of the captured payload cannot reach back into the
   originating ``message.content`` list (T-09-03-T3 mutation safety).
2. ``replay_reasoning_state`` writes the captured blocks back to the
   most-recent outbound AIMessage's ``.content`` (prepending if content
   is already a list; promoting str content to a list-of-blocks shape so
   the outbound payload matches what Anthropic expects when thinking_blocks
   are present).
3. The Anthropic API itself enforces signature byte-identity on the wire,
   so any drift would 400 — the unit + conformance tests verify the
   contract before the wire does.

**PROV-05 / D-09-07 isolation rule:** imports ONLY from
``app.agent.adapters`` base + ``langchain_core`` + stdlib. Never from a
sibling adapter file (e.g. ``openai_gpt5.py``, ``deepseek.py``,
``gemini.py``). Plan 09-05 ``revert`` simulation depends on this isolation.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage

from app.agent.adapters import ProviderAdapter, StatePayload

_log = logging.getLogger(__name__)


class AnthropicAdapter(ProviderAdapter):
    """``ProviderAdapter`` for Claude (D-09-06 carve-out: thinking ENABLED).

    ``capture_reasoning_state`` reads ``message.content`` and filters for
    blocks where ``type == "thinking"``, wrapping the surviving blocks in
    the canonical ``{"provider": "anthropic", "thinking_blocks": [...]}``
    shape that matches ``FOUR_SHAPE_PAYLOADS[1]`` in the Phase 8
    conformance harness.

    ``replay_reasoning_state`` walks ``outbound`` in reverse to find the
    most-recent ``AIMessage`` and writes the captured thinking blocks back
    onto its ``.content``. If content is already a list, the blocks are
    PREPENDED (preserving any non-thinking blocks already present). If
    content is a string, it is PROMOTED to a list of blocks: the captured
    thinking blocks plus a ``{"type": "text", "text": <original str>}``
    block — this is the shape Anthropic expects on the wire when
    ``thinking_blocks`` are part of the assistant turn.

    Asymmetric to ``OpenAIReasoningAdapter`` and
    ``DeepSeekReasonerAdapter``: those two read/write
    ``additional_kwargs["reasoning_content"]``. ``AnthropicAdapter`` reads
    and writes ``message.content``. PATTERNS.md §ASYMMETRY CALLOUT
    documents the rationale (Anthropic surfaces reasoning state on the
    content block list, not on additional_kwargs).
    """

    PROVIDER_KEY = "anthropic"

    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        content = message.content
        if not isinstance(content, list):
            return None
        thinking_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        if not thinking_blocks:
            return None
        # T-09-03-T3 mitigation: shallow-copy each thinking block dict so
        # downstream mutation of the captured payload (or of the list itself)
        # cannot reach back into the originating message.content. The
        # signature field MUST round-trip byte-identical to what was
        # captured — keeping the captured blocks isolated from the input
        # is what makes that contract defensible.
        return {
            "provider": self.PROVIDER_KEY,
            "thinking_blocks": [dict(b) for b in thinking_blocks],
        }

    def replay_reasoning_state(
        self, outbound: list[BaseMessage], state: StatePayload | None
    ) -> list[BaseMessage]:
        if state is None:
            return outbound
        blocks = state.get("thinking_blocks")
        if not blocks:
            return outbound
        # Walk in reverse to find the most-recent AIMessage. The replay
        # target is whatever the next outbound LLM call will see as the
        # LAST assistant turn — that's the message that needs to carry
        # the signed thinking_blocks forward so Anthropic's API does not
        # 400 on the next request.
        for msg in reversed(outbound):
            if isinstance(msg, AIMessage):
                existing = msg.content
                # ─────────────────────────────────────────────────────────────
                # PROV-03 live-run idempotency fix (2026-06-05): Anthropic
                # returns AIMessages with thinking_blocks ALREADY ON .content
                # (langchain-anthropic surfaces them as content blocks, not
                # additional_kwargs — that's the documented asymmetry). The
                # graph's plan() loop calls `capture` after each ainvoke and
                # `replay` BEFORE each ainvoke. Inside a single agent turn,
                # the same AIMessage gets seen by both — capture stashes its
                # thinking_blocks on additional_kwargs["_reasoning_state"];
                # replay was unconditionally prepending them back onto
                # .content, DUPLICATING the blocks (`content` became
                # [thinking, thinking, text, tool_use]). Anthropic's API
                # then 400s on the next request:
                #
                #   400 — messages.1.content.1: `thinking` or
                #   `redacted_thinking` blocks in the latest assistant message
                #   cannot be modified. These blocks must remain as they were
                #   in the original response.
                #   (request_id req_011CbkoAUyWgQkkoYCe3D5yj)
                #
                # The fix: skip the prepend when the target AIMessage already
                # carries thinking blocks in its content list — they are
                # already the wire-correct, byte-identical signed blocks from
                # the original Anthropic response, so no replay is needed.
                # When content has no thinking blocks (str shape from a pruned
                # AIMessage outside the keep-window, or a list with only
                # text/tool_use blocks), the original replay logic applies so
                # the signature survives the prune cutoff.
                # ─────────────────────────────────────────────────────────────
                if isinstance(existing, list):
                    already_has_thinking = any(
                        isinstance(b, dict) and b.get("type") == "thinking" for b in existing
                    )
                    if already_has_thinking:
                        # WR-04 observability: under normal agent-loop flow,
                        # the target's existing thinking blocks ARE the
                        # captured ones (capture stashes additional_kwargs;
                        # the wire blocks remain on content unchanged). If
                        # the signature sets differ, something between
                        # capture and replay rewrote the AIMessage (e.g. a
                        # revision step replaced content with a different
                        # signed reply, or the graph reducer mis-ordered
                        # turns). Behavior is unchanged — target wins — but
                        # a debug log gives future bug-hunts telemetry to
                        # spot silent payload-discard.
                        existing_sigs = sorted(
                            sig
                            for b in existing
                            if isinstance(b, dict) and b.get("type") == "thinking"
                            for sig in [b.get("signature")]
                            if isinstance(sig, str)
                        )
                        captured_sigs = sorted(
                            sig
                            for b in blocks
                            if isinstance(b, dict)
                            for sig in [b.get("signature")]
                            if isinstance(sig, str)
                        )
                        if existing_sigs != captured_sigs:
                            _log.debug(
                                "AnthropicAdapter.replay: target AIMessage already has "
                                "thinking blocks with signatures %r but captured payload "
                                "carried %r — skipping replay (target wins; captured "
                                "payload discarded). If you see this in production, the "
                                "agent loop rewrote an AIMessage between capture and "
                                "replay — verify the graph reducer ordering.",
                                existing_sigs,
                                captured_sigs,
                            )
                        # Wire-correct blocks already present; do NOT prepend
                        # duplicates. Anthropic's signature contract is
                        # satisfied by the original response's blocks.
                        break
                    # Prepend the captured blocks; preserve any non-thinking
                    # blocks already in the list (e.g. tool_use blocks the
                    # graph appended for the agent loop). This branch fires
                    # when prune dropped the thinking blocks but
                    # additional_kwargs survived.
                    msg.content = list(blocks) + existing
                elif isinstance(existing, str):
                    # Promote str content to a list-of-blocks shape so the
                    # outbound payload matches Anthropic's expected wire
                    # format when thinking_blocks are present. Common after
                    # _prune_for_llm converts an older AIMessage's list
                    # content to str via str(m.content).
                    msg.content = list(blocks) + [{"type": "text", "text": existing}]
                else:
                    # Unexpected content shape (None, dict, etc.) — set to
                    # just the blocks so the next request at least carries
                    # the signature forward. (langchain-anthropic always
                    # surfaces content as str or list in practice; this
                    # branch is defensive.)
                    msg.content = list(blocks)
                break
        return outbound


__all__ = ["AnthropicAdapter"]
