"""Critique / revision-diagnosis logic for the agent graph.

Split out of graph.py (FUTURE_WATCH: app/agent/ directory size). Public entry
points (critique_step, critique_final_with_stops, short_circuit_max_steps,
finalize_as_is) are called by the graph's `critique` node. Everything else is
module-internal (underscore-prefixed); white-box tests import those by their
private name, matching the existing test pattern.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agent.critique import CRITIQUE_ITINERARY, CRITIQUE_STEP, CRITIQUE_VIBE, vibe
from app.agent.critique.checks import is_rationale_aligned, itinerary_violations
from app.agent.state import ItineraryState, RevisionAction, RevisionHint

LOW_SIMILARITY_THRESHOLD = 0.55
MAX_REVISIONS_PER_REASON = 2


def _can_retry(state: ItineraryState, reason: str) -> bool:
    return state.revision_counts.get(reason, 0) < MAX_REVISIONS_PER_REASON


def _bumped_counts(state: ItineraryState, reason: str) -> dict[str, int]:
    counts = dict(state.revision_counts)
    counts[reason] = counts.get(reason, 0) + 1
    return counts


def _scratch_entries_for_last_round(
    state: ItineraryState,
) -> list[tuple[str, dict[str, Any]]]:
    """Return (tool_name, scratch_entry) pairs for every tool_call in the most
    recent issuing AIMessage, in tool_call order.

    Pairings are matched by tool_call id when present, falling back to the
    highest-step entry for that tool name (handles legacy entries that
    pre-date id tracking)."""
    last_tool_idx = None
    for i in range(len(state.messages) - 1, -1, -1):
        if isinstance(state.messages[i], ToolMessage):
            last_tool_idx = i
            break
    if last_tool_idx is None:
        return []
    issuing_ai: AIMessage | None = None
    for i in range(last_tool_idx - 1, -1, -1):
        m = state.messages[i]
        if isinstance(m, AIMessage) and m.tool_calls:
            issuing_ai = m
            break
    if issuing_ai is None or not issuing_ai.tool_calls:
        return []
    pairs: list[tuple[str, dict[str, Any]]] = []
    for tc in issuing_ai.tool_calls:
        name = tc["name"]
        entries = state.scratch.get(name) or []
        if not entries:
            continue
        match = next((e for e in entries if e.get("id") == tc["id"]), None)
        if match is None:
            match = max(entries, key=lambda e: e.get("step", -1))
        pairs.append((name, match))
    return pairs


def _most_restrictive_filter(filters: dict[str, Any] | None) -> str:
    """Deterministic priority order for which filter to drop first."""
    if not filters:
        return "none"
    for f in ("open_at", "price_level_max", "min_rating", "neighborhood", "types_any"):
        if filters.get(f) is not None:
            return f
    return "none"


def _diagnose_one(
    tool_name: str,
    entry: dict[str, Any],
    low_similarity_count: int = 0,
) -> RevisionHint | None:
    """Inspect a single tool call+result pair. Returns a hint or None.

    `low_similarity_count` is the current value of `state.revision_counts["low_similarity"]`
    — used to gate the `neighborhood_no_match` escalation. We only flip from
    "rephrase your query" to "ask the user" AFTER the model has burned its
    rephrase budget; that way poor-query attempts in a data-rich neighborhood
    (e.g. "date night dinner" at sim 0.29 in Hayes Valley, when "dinner Hayes
    Valley" at sim 0.58 would work fine) get a chance to fix themselves first.
    """
    result = entry.get("result")
    args = entry.get("args") or {}

    if isinstance(result, dict) and "error" in result:
        return RevisionHint(
            reason="tool_error",
            detail=str(result["error"]),
            suggested_action="try_different_tool",
            target={"tool": tool_name},
        )

    if isinstance(result, list):
        if not result:
            return RevisionHint(
                reason="empty_results",
                detail=f"No matches for {args}.",
                suggested_action="drop_filter",
                target={"filter": _most_restrictive_filter(args.get("filters"))},
            )
        if all(getattr(h, "business_status", None) != "OPERATIONAL" for h in result):
            return RevisionHint(
                reason="all_closed",
                detail="Every result is closed or permanently_closed.",
                suggested_action="broaden_query",
                target={"filter": "business_status"},
            )
        # low_similarity only makes sense for the vector tool. `nearby`,
        # `kg_traverse` and `get_details` return similarity=0.0 BY DESIGN
        # (geographic / graph / lookup — no cosine score), so flagging them
        # emitted a bogus "Top similarity 0.00, broaden_query" critique that
        # derailed models following the prompt's anchored-`nearby` pattern.
        if tool_name == "semantic_search":
            top = result[0]
            sim = getattr(top, "similarity", 0.0) or 0.0
            if sim < LOW_SIMILARITY_THRESHOLD:
                # When the user pinned a neighborhood and even the best in-
                # neighborhood result is weak, looping `broaden_query` won't
                # help — the dataset literally doesn't have a strong match in
                # that neighborhood (e.g. only 3 Sushi Restaurants in the
                # Mission, top sim 0.51 for "omakase" — the real omakase
                # restaurants are in SoMa/Japantown). Escalate to the user
                # rather than burning step budget rephrasing the query.
                filters = args.get("filters") or {}
                neighborhood = filters.get("neighborhood") if isinstance(filters, dict) else None
                # Only escalate to "ask the user" AFTER the model has used its
                # rephrase budget for low_similarity. In data-rich neighborhoods
                # (e.g. Hayes Valley), a bad query like "date night dinner"
                # scores 0.29 but "dinner Hayes Valley" scores 0.58 — let the
                # broaden_query loop catch self-fixable bad queries first.
                if neighborhood and low_similarity_count >= MAX_REVISIONS_PER_REASON:
                    return RevisionHint(
                        reason="neighborhood_no_match",
                        detail=(
                            f"After {low_similarity_count} rephrase attempts, "
                            f"top similarity in {neighborhood} is still "
                            f"{sim:.2f} — below threshold {LOW_SIMILARITY_THRESHOLD}. "
                            f"The dataset likely doesn't have strong matches "
                            f"for this category in {neighborhood}."
                        ),
                        suggested_action="clarify_with_user",
                        target={"neighborhood": neighborhood, "query": args.get("query")},
                    )
                return RevisionHint(
                    reason="low_similarity",
                    detail=(
                        f"Top similarity {sim:.2f} below threshold {LOW_SIMILARITY_THRESHOLD}."
                    ),
                    suggested_action="broaden_query",
                    target={"query": args.get("query")},
                )
    return None


def _diagnose_last_tool_result(state: ItineraryState) -> RevisionHint | None:
    """Diagnose every tool_call in the most recent issuing AIMessage and
    return the first hint in tool_call order. Returns None if every call was
    healthy. The agent revises one issue per round; later issues, if any,
    will be diagnosed on the next round."""
    low_similarity_count = state.revision_counts.get("low_similarity", 0)
    for tool_name, entry in _scratch_entries_for_last_round(state):
        hint = _diagnose_one(tool_name, entry, low_similarity_count=low_similarity_count)
        if hint is not None:
            return hint
    return None


def _first_misaligned_stop_index(state: ItineraryState) -> int:
    """Return the index of the first stop that fails is_rationale_aligned.

    Used by the rationale_stop_alignment branch of `_hint_for_violation` to
    identify which stop the model should rewrite. Shares the per-stop boolean
    rule with `rationale_stop_alignment` in `app/agent/critique/checks.py` via
    the public `is_rationale_aligned` helper (DRY — plan 04-05 ADVISORY 3).

    Returns 0 as a defensive fallback when either (a) state.stops is empty (the
    dispatcher should never call this with no violations, but it must not
    raise) or (b) every stop is aligned (also unreachable in practice — the
    dispatcher only fires on violations).
    """
    for i, stop in enumerate(state.stops):
        if not is_rationale_aligned(stop):
            return i
    return 0


def _hint_for_violation(reason: str, state: ItineraryState) -> RevisionHint:
    """Map a check name from itinerary_violations() to a structured hint."""
    if reason == "geographic_coherence":
        return RevisionHint(
            reason="geographic_incoherence",
            detail="One or more consecutive stops exceed the per-leg walking budget.",
            suggested_action="tighten_radius",
            target={"stops": list(range(1, len(state.stops)))},
        )
    if reason == "temporal_coherence":
        return RevisionHint(
            reason="temporal_incoherence",
            detail="A stop is closed at its planned arrival time.",
            suggested_action="shift_arrival_time",
            target={},
        )
    if reason == "walking_budget_respected":
        return RevisionHint(
            reason="walking_budget_exceeded",
            detail="Total walking exceeds the user's budget.",
            suggested_action="rebalance_walking_budget",
            target={},
        )
    if reason == "stop_count_satisfied":
        # stop_count_satisfied only fails when num_stops is set (the check
        # returns 1.0 unconditionally otherwise), so requested is guaranteed
        # non-None whenever this branch executes. Narrow it explicitly so the
        # type checker can prove the < below is safe.
        requested = state.constraints.num_stops
        if requested is None:
            raise RuntimeError("stop_count_satisfied invariant: num_stops must be set")
        actual = len(state.stops)
        detail = (
            f"The committed itinerary has {actual} stops, but the user requested {requested} stops."
        )
        # Direction is load-bearing — telling the model to "add" when it needs
        # to "remove" is actively misleading. The reason is also distinct from
        # the generic constraint_unmet_in_final so debugging and revision-hint
        # logs can tell a count mismatch apart from a per-stop constraint fail.
        action: RevisionAction = "add_missing_stops" if actual < requested else "remove_extra_stops"
        return RevisionHint(
            reason="stop_count_mismatch",
            detail=detail,
            suggested_action=action,
            target={"requested_stops": requested, "actual_stops": actual},
        )
    if reason == "no_hallucinated_place_ids":
        return RevisionHint(
            reason="hallucinated_place_id",
            detail="One or more place_ids do not exist in places_raw.",
            suggested_action="swap_stop",
            target={},
        )
    if reason == "rationale_stop_alignment":
        # Locate the offending stop using the SAME per-stop rule that the
        # scorer uses (is_rationale_aligned). The model rewrites the rationale
        # to describe the committed place; suggested_action="rewrite_rationale"
        # matches the REVISION_GUIDANCE text ("do NOT swap the stop — only the
        # rationale text is misaligned"). Budget gating uses the CHECK name
        # ("rationale_stop_alignment") so this reason's retry budget is
        # independent of constraint_unmet_in_final.
        offending_index = _first_misaligned_stop_index(state)
        return RevisionHint(
            reason="rationale_misaligned",
            detail=(
                f"Stop {offending_index + 1}'s rationale doesn't describe the "
                f"committed place's primary_type or name."
            ),
            suggested_action="rewrite_rationale",
            target={"stop_index": offending_index},
        )
    return RevisionHint(
        reason="constraint_unmet_in_final",
        detail="Final stops do not satisfy the shared user constraints.",
        suggested_action="swap_stop",
        target={},
    )


def _final_with_caveats(last_content: str, violations: list[str]) -> str:
    """Compose a final reply that lists what didn't quite work. Better than
    silently shipping a bad plan."""
    caveats = (
        "\n\nCaveats: I couldn't fully satisfy "
        + ", ".join(violations)
        + (" after revisions. You may want to adjust the plan.")
    )
    return (last_content or "") + caveats


def _last_ai_content(last: BaseMessage | None) -> str:
    return last.content if isinstance(last, AIMessage) and isinstance(last.content, str) else ""


def summarize_stops(state: ItineraryState) -> str:
    """Deterministic user-facing summary built from committed stops.

    The graph finalizes the moment a commit passes the hard checks (so the
    model can't over-loop), which means the model never narrates the plan
    itself. Synthesize the summary from the stops instead — same arrival +
    duration contract the OUTPUT FORMAT prompt asked the model for, minus the
    extra LLM turn and its latency.
    """
    lines = []
    for i, s in enumerate(state.stops, 1):
        when = s.arrival_time.strftime("%-I:%M %p").lstrip("0") if s.arrival_time else None
        timing = f" — arrive {when}, ~{s.planned_duration_min} min" if when else ""
        rationale = f" {s.rationale}" if s.rationale else ""
        lines.append(f"{i}. {s.name}{timing}.{rationale}".rstrip())
    body = "\n".join(lines)
    return f"Here's your itinerary:\n{body}" if body else ""


def short_circuit_max_steps(state: ItineraryState) -> dict[str, Any]:
    if state.final_reply:
        return {"done": True, "final_reply": state.final_reply}
    # Defense-in-depth: the finalize-on-commit routing should make reaching
    # the step ceiling with committed stops impossible. But if a future model
    # finds a new way to over-loop, a committed plan must never surface to the
    # user as "I hit the step limit" — render the stops we actually have.
    if state.stops:
        return {"done": True, "final_reply": summarize_stops(state)}
    return {
        "done": True,
        "final_reply": "I hit the planning step limit. Here is the best plan I had so far.",
    }


def finalize_as_is(state: ItineraryState, last: BaseMessage | None) -> dict[str, Any]:
    return {"done": True, "final_reply": state.final_reply or _last_ai_content(last)}


def critique_final_with_stops(
    state: ItineraryState,
    last: BaseMessage | None,
    judge_llm: BaseChatModel | None,
) -> dict[str, Any]:
    """Run deterministic checks; if any fail, drive a revision (or ship with
    caveats once budgets are exhausted). If they all pass, run the optional
    cheap-LLM vibe check; if that fails, drive a revision. Otherwise finalize."""
    last_content = _last_ai_content(last)
    violations = itinerary_violations(state)
    if violations:
        actionable = next((v for v in violations if _can_retry(state, v)), None)
        if actionable is not None:
            hint = _hint_for_violation(actionable, state)
            return {
                "revision_hints": [*state.revision_hints, hint],
                "revision_counts": _bumped_counts(state, actionable),
                "messages": [
                    HumanMessage(
                        content=(
                            f"{CRITIQUE_ITINERARY} {actionable}: {hint.detail} "
                            f"Suggested action: {hint.suggested_action}. "
                            f"Revise the affected stop(s) and re-call commit_itinerary."
                        )
                    )
                ],
                "done": False,
            }
        return {"done": True, "final_reply": _final_with_caveats(last_content, violations)}

    score = vibe.vibe_check(state, judge_llm)
    if score is not None and score < vibe.VIBE_THRESHOLD and _can_retry(state, "vibe_mismatch"):
        hint = RevisionHint(
            reason="vibe_mismatch",
            detail=f"Cross-stop vibe coherence scored {score:.1f}/5.",
            suggested_action="swap_stop",
            target={},
        )
        return {
            "revision_hints": [*state.revision_hints, hint],
            "revision_counts": _bumped_counts(state, "vibe_mismatch"),
            "messages": [
                HumanMessage(
                    content=(
                        f"{CRITIQUE_VIBE} cross-stop vibe coherence scored "
                        f"{score:.1f}/5. Swap whichever stop feels off and "
                        f"re-call commit_itinerary."
                    )
                )
            ],
            "done": False,
        }

    # last_content is empty when we finalized on a commit (the last message
    # is the commit tool call, not a model narration) — synthesize instead.
    return {
        "done": True,
        "final_reply": state.final_reply or last_content or summarize_stops(state),
    }


def critique_step(state: ItineraryState) -> dict[str, Any]:
    """React to the last tool result. Emit a per-step hint if it was empty,
    all-closed, low-similarity, or errored. Bounded by retry budget."""
    hint = _diagnose_last_tool_result(state)
    if hint is None or not _can_retry(state, hint.reason):
        return {}
    return {
        "revision_hints": [*state.revision_hints, hint],
        "revision_counts": _bumped_counts(state, hint.reason),
        "messages": [
            HumanMessage(
                content=(
                    f"{CRITIQUE_STEP} {hint.reason}: {hint.detail} "
                    f"Suggested next action: {hint.suggested_action} on {hint.target}."
                )
            )
        ],
    }
