"""LangGraph StateGraph for the agent. Three nodes: plan, act, critique.

plan      LLM proposes the next action: a tool call or `final`.
act       Executes the tool. Tool result -> state.scratch[<tool>][<step>].
critique  Deterministic + LLM check; can request `revise` (W3 expands this).

Edge logic:
  plan -> act (if tool call) | END (if final)
  act  -> critique
  critique -> plan (if revise or more work) | END (if good)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.agent.critique import vibe
from app.agent.critique.checks import itinerary_violations
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import ItineraryState, RevisionHint, Stop
from app.agent.tools import COMMIT_ITINERARY_TOOL_NAME, all_tools

LOW_SIMILARITY_THRESHOLD = 0.55
MAX_REVISIONS_PER_REASON = 2


def _grounded_place_ids(scratch: dict[str, Any]) -> set[str]:
    """All place_ids the agent has actually seen via prior tool results."""
    grounded: set[str] = set()
    for entries in scratch.values():
        for entry in entries:
            result = entry.get("result")
            if isinstance(result, list):
                for hit in result:
                    pid = getattr(hit, "place_id", None) or (
                        hit.get("place_id") if isinstance(hit, dict) else None
                    )
                    if pid:
                        grounded.add(pid)
            elif result is not None:
                pid = getattr(result, "place_id", None) or (
                    result.get("place_id") if isinstance(result, dict) else None
                )
                if pid:
                    grounded.add(pid)
    return grounded


def _commit_stops(state: ItineraryState, raw_stops: Any) -> tuple[list[Stop], dict[str, Any]]:
    """Validate and coerce LLM-supplied stops into Stop models.

    Returns (committed_stops, tool_result_payload). The payload is what the
    LLM sees back as the tool result; rejected place_ids surface there so the
    model can self-correct in W3.
    """
    if not isinstance(raw_stops, list):
        return [], {"error": "stops must be a list"}
    grounded = _grounded_place_ids(state.scratch)
    committed: list[Stop] = []
    rejected: list[dict[str, Any]] = []
    for raw in raw_stops:
        if not isinstance(raw, dict):
            rejected.append({"reason": "stop must be an object", "value": str(raw)})
            continue
        pid = raw.get("place_id")
        if not pid or pid not in grounded:
            rejected.append({"place_id": pid, "reason": "place_id not seen via prior tool result"})
            continue
        try:
            committed.append(Stop(**raw))
        except Exception as e:  # noqa: BLE001
            rejected.append({"place_id": pid, "reason": f"invalid stop: {e}"})
    return committed, {
        "committed": [s.place_id for s in committed],
        "rejected": rejected,
    }


_RECENT_TOOL_EXCHANGES_KEPT = 2


def _prune_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Curate the message list sent to the LLM to keep token cost bounded.

    Strategy: keep all SystemMessages and HumanMessages, plus the AIMessages.
    For ToolMessages — which dominate token cost as the agent loops — keep
    only those paired with the last `_RECENT_TOOL_EXCHANGES_KEPT` AIMessages
    that issued tool_calls. Earlier tool results are dropped together with
    their issuing AIMessage's tool_calls, so the LLM never sees an unanswered
    tool_call (which would violate the OpenAI/Gemini conversation contract).
    """
    if not messages:
        return messages

    # Find indices of AIMessages that issued tool_calls.
    tool_caller_indices = [
        i for i, m in enumerate(messages) if isinstance(m, AIMessage) and m.tool_calls
    ]
    if len(tool_caller_indices) <= _RECENT_TOOL_EXCHANGES_KEPT:
        return messages

    # Cutoff: keep messages from this index onward in their original form.
    keep_from = tool_caller_indices[-_RECENT_TOOL_EXCHANGES_KEPT]

    # Before the cutoff: drop tool_calls from AIMessages and drop ToolMessages
    # entirely. After: keep as-is.
    pruned: list[BaseMessage] = []
    for i, m in enumerate(messages):
        if i >= keep_from:
            pruned.append(m)
            continue
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage) and m.tool_calls:
            # Replace with a content-only AIMessage so we don't strand the
            # LLM thinking it issued tool_calls that were never answered.
            pruned.append(
                AIMessage(content=m.content if isinstance(m.content, str) else str(m.content))
            )
            continue
        pruned.append(m)
    return pruned


def _serialize_tool_result(result: Any) -> str:
    """Emit JSON the LLM can parse, regardless of underlying tool return type.

    Pydantic models -> model_dump(mode='json'). Lists of pydantic models too.
    Dicts/primitives pass through json.dumps with default=str so non-JSON
    types like datetime degrade gracefully.
    """
    if isinstance(result, BaseModel):
        return json.dumps(result.model_dump(mode="json"))
    if isinstance(result, list) and result and isinstance(result[0], BaseModel):
        return json.dumps([m.model_dump(mode="json") for m in result])
    return json.dumps(result, default=str)


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


def _diagnose_one(tool_name: str, entry: dict[str, Any]) -> RevisionHint | None:
    """Inspect a single tool call+result pair. Returns a hint or None."""
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
        top = result[0]
        sim = getattr(top, "similarity", 0.0) or 0.0
        if sim < LOW_SIMILARITY_THRESHOLD:
            return RevisionHint(
                reason="low_similarity",
                detail=f"Top similarity {sim:.2f} below threshold {LOW_SIMILARITY_THRESHOLD}.",
                suggested_action="broaden_query",
                target={"query": args.get("query")},
            )
    return None


def _diagnose_last_tool_result(state: ItineraryState) -> RevisionHint | None:
    """Diagnose every tool_call in the most recent issuing AIMessage and
    return the first hint in tool_call order. Returns None if every call was
    healthy. The agent revises one issue per round; later issues, if any,
    will be diagnosed on the next round."""
    for tool_name, entry in _scratch_entries_for_last_round(state):
        hint = _diagnose_one(tool_name, entry)
        if hint is not None:
            return hint
    return None


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
    if reason == "no_hallucinated_place_ids":
        return RevisionHint(
            reason="hallucinated_place_id",
            detail="One or more place_ids do not exist in places_raw.",
            suggested_action="swap_stop",
            target={},
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


def build_agent_graph(
    llm: BaseChatModel,
    max_steps: int = 8,
    judge_llm: BaseChatModel | None = None,
):
    """Construct the agent graph.

    `judge_llm` is the cheap model used for vibe coherence scoring. If None
    and EVAL_VIBE_CRITIQUE_ENABLED=true, one is constructed via
    vibe.make_judge() at graph-build time. If construction fails (missing
    creds, unknown provider) the vibe pass is silently skipped.
    """
    tools = all_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}
    if judge_llm is None and vibe.is_enabled():
        judge_llm = vibe.make_judge()

    async def plan(state: ItineraryState) -> dict[str, Any]:
        messages_in: list[BaseMessage] = list(state.messages)
        if state.step_count == 0 and not any(isinstance(m, SystemMessage) for m in messages_in):
            messages_in = [
                SystemMessage(SYSTEM_PROMPT.format(max_steps=max_steps)),
                *messages_in,
            ]
        # Send the LLM a curated view: keep only the most recent ToolMessage
        # per tool name so token cost stays linear in tool *kinds*, not in
        # tool *calls*. Full history is preserved on state.messages for tracing.
        messages_for_llm = _prune_for_llm(messages_in)
        ai = await llm_with_tools.ainvoke(messages_for_llm)
        new_messages: list[BaseMessage] = []
        if len(messages_in) > len(state.messages):
            new_messages.append(messages_in[0])
        new_messages.append(ai)
        return {"messages": new_messages}

    async def act(state: ItineraryState) -> dict[str, Any]:
        ai = state.messages[-1]
        if not isinstance(ai, AIMessage) or not ai.tool_calls:
            return {}

        new_messages: list[BaseMessage] = []
        scratch_updates: dict[str, list[dict[str, Any]]] = {}
        committed_stops: list[Stop] | None = None

        for tc in ai.tool_calls:
            if tc["name"] == COMMIT_ITINERARY_TOOL_NAME:
                stops, payload = _commit_stops(state, tc["args"].get("stops"))
                committed_stops = stops
                new_messages.append(
                    ToolMessage(content=_serialize_tool_result(payload), tool_call_id=tc["id"])
                )
                scratch_updates.setdefault(tc["name"], []).append(
                    {
                        "args": tc["args"],
                        "result": payload,
                        "step": state.step_count,
                        "id": tc["id"],
                    }
                )
                continue

            tool = tool_by_name.get(tc["name"])
            if tool is None:
                new_messages.append(
                    ToolMessage(
                        content=f"unknown tool {tc['name']}",
                        tool_call_id=tc["id"],
                    )
                )
                continue
            try:
                # psycopg2 + OpenAI are sync; offload to a worker thread so the
                # event loop stays responsive while the tool blocks on I/O.
                result: Any = await asyncio.to_thread(tool.invoke, tc["args"])
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
            new_messages.append(
                ToolMessage(content=_serialize_tool_result(result), tool_call_id=tc["id"])
            )
            scratch_updates.setdefault(tc["name"], []).append(
                {
                    "args": tc["args"],
                    "result": result,
                    "step": state.step_count,
                    "id": tc["id"],
                }
            )

        merged_scratch = dict(state.scratch)
        for name, entries in scratch_updates.items():
            merged_scratch[name] = [*merged_scratch.get(name, []), *entries]

        update: dict[str, Any] = {
            "messages": new_messages,
            "scratch": merged_scratch,
            "step_count": state.step_count + 1,
        }
        if committed_stops is not None:
            update["stops"] = committed_stops
        return update

    def critique(state: ItineraryState) -> dict[str, Any]:
        """Two-pass critique:

        1. If the LLM is finalizing (no-tool-call AI message) AND we have
           committed stops, run the deterministic itinerary checks. On any
           violation we still have retries left for, inject a hint as a
           HumanMessage and route back to plan. On exhaustion, ship with
           caveats.
        2. Otherwise (we just came from `act`), inspect the last tool result
           and emit a per-step hint if it was empty / all-closed / low-sim /
           errored. Bounded by MAX_REVISIONS_PER_REASON per failure category.
        """
        update: dict[str, Any] = {}

        if state.step_count >= max_steps:
            update["done"] = True
            update["final_reply"] = state.final_reply or (
                "I hit the planning step limit. Here is the best plan I had so far."
            )
            return update

        last = state.messages[-1] if state.messages else None
        finalizing = isinstance(last, AIMessage) and not last.tool_calls
        last_content = (
            last.content if isinstance(last, AIMessage) and isinstance(last.content, str) else ""
        )

        if finalizing and state.stops:
            violations = itinerary_violations(state)
            if violations:
                actionable = next((v for v in violations if _can_retry(state, v)), None)
                if actionable is not None:
                    hint = _hint_for_violation(actionable, state)
                    update["revision_hints"] = [*state.revision_hints, hint]
                    update["revision_counts"] = _bumped_counts(state, actionable)
                    update["messages"] = [
                        HumanMessage(
                            content=(
                                f"[critique:itinerary] {actionable}: {hint.detail} "
                                f"Suggested action: {hint.suggested_action}. "
                                f"Revise the affected stop(s) and re-call commit_itinerary."
                            )
                        )
                    ]
                    update["done"] = False
                    return update
                # Exhausted retries on at least one violation; ship with caveats.
                update["done"] = True
                update["final_reply"] = _final_with_caveats(last_content, violations)
                return update

            # Deterministic checks passed; one cheap-LLM vibe pass before shipping.
            score = vibe.vibe_check(state, judge_llm)
            if (
                score is not None
                and score < vibe.VIBE_THRESHOLD
                and _can_retry(state, "vibe_mismatch")
            ):
                hint = RevisionHint(
                    reason="vibe_mismatch",
                    detail=f"Cross-stop vibe coherence scored {score:.1f}/5.",
                    suggested_action="swap_stop",
                    target={},
                )
                update["revision_hints"] = [*state.revision_hints, hint]
                update["revision_counts"] = _bumped_counts(state, "vibe_mismatch")
                update["messages"] = [
                    HumanMessage(
                        content=(
                            f"[critique:vibe] cross-stop vibe coherence scored "
                            f"{score:.1f}/5. Swap whichever stop feels off and "
                            f"re-call commit_itinerary."
                        )
                    )
                ]
                update["done"] = False
                return update

            update["done"] = True
            update["final_reply"] = state.final_reply or last_content
            return update

        if finalizing:
            # No stops committed yet (e.g. clarifying question). Finalize as-is.
            update["done"] = True
            update["final_reply"] = state.final_reply or last_content
            return update

        # Per-step deterministic critique: react to the last tool result.
        hint = _diagnose_last_tool_result(state)
        if hint is not None and _can_retry(state, hint.reason):
            update["revision_hints"] = [*state.revision_hints, hint]
            update["revision_counts"] = _bumped_counts(state, hint.reason)
            update["messages"] = [
                HumanMessage(
                    content=(
                        f"[critique:step] {hint.reason}: {hint.detail} "
                        f"Suggested next action: {hint.suggested_action} on {hint.target}."
                    )
                )
            ]
        return update

    def route_after_plan(state: ItineraryState) -> Literal["act", "critique"]:
        last = state.messages[-1]
        return "act" if isinstance(last, AIMessage) and last.tool_calls else "critique"

    def route_after_critique(state: ItineraryState) -> str:
        return END if state.done else "plan"

    g = StateGraph(ItineraryState)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("critique", critique)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "critique": "critique"})
    g.add_edge("act", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"plan": "plan", END: END})
    return g.compile()
