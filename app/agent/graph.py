"""LangGraph StateGraph for the agent. Three nodes: plan, act, critique.

plan      LLM proposes the next action: a tool call or `final`.
act       Executes the tool. Tool result -> state.scratch[<tool>][<step>].
critique  Deterministic + LLM check; can request `revise` (W3 expands this).

Edge logic:
  plan -> act (if tool call) | END (if final)
  act  -> critique
  critique -> plan (if revise or more work) | END (if good)

Stop-commit + booking enrichment lives in app.agent.commit; critique/
revision-diagnosis branch logic lives in app.agent.revision (split out of
graph.py per FUTURE_WATCH: app/agent/ directory size).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.agent.adapters import ADAPTERS, NoOpAdapter, ProviderAdapter
from app.agent.commit import commit_stops
from app.agent.critique import vibe
from app.agent.planning import chain_arrival_times
from app.agent.prompts import SYSTEM_PROMPT, current_datetime_str, rule8_viability_addendum
from app.agent.revision import (
    critique_final_with_stops,
    critique_step,
    finalize_as_is,
    short_circuit_max_steps,
)
from app.agent.state import ItineraryState, Stop
from app.agent.swap import _inject_closure_exclusions, swap_closed_stops
from app.agent.tools import COMMIT_ITINERARY_TOOL_NAME, all_tools
from app.tools.directions import route_legs
from app.tools.filters import SearchFilters, family_of

logger = logging.getLogger(__name__)


def _inject_primary_type_family(
    tool_name: str,
    args: dict[str, Any],
    requested_primary_types: list[str],
) -> dict[str, Any]:
    """Phase 4 D-04-04: write `filters.primary_type_family` for the slot when
    the model cooperated by emitting `slot_index`.

    Mirrors `_inject_closure_exclusions` (app/agent/swap.py:456-510) exactly
    in shape and JSON-safety invariants:

      - Returns a NEW args dict (never mutates the input).
      - The result's `filters` value is ALWAYS a plain dict, NEVER a Pydantic
        `SearchFilters` instance — the caller stores effective_args inside
        AIMessage scratch, and `json.dumps(args)` must not crash on the next
        plan() step (project memory `aimessage_tool_call_args_json_safe.md`).
      - Noop on every defensive branch (D-04-06 trust boundary, fail-open on
        unmappable types) so the model can decline cooperation without the
        graph silently substituting a guess.

    Routing rules:

      - tool_name NOT in (semantic_search, nearby) -> noop. kg_traverse has no
        `filters` arg; commit_itinerary takes a stops list; get_details has
        only place_id. None of these accept primary_type_family.
      - requested_primary_types empty -> noop (the user didn't structure their
        request as per-slot categories).
      - slot_index missing / None / not int -> noop (D-04-06 trust boundary;
        the model declined to declare which slot this retrieval is for).
      - slot_index < 0 or >= len(requested_primary_types) -> noop (defensive
        against bad model output).
      - family_of(requested_primary_types[slot_index]) is None -> noop
        (fail-open consistent with the codebase pattern; unmappable keyword).

    Otherwise: build `effective_args` with `filters.primary_type_family` set
    to the family derived from the slot. When the model already wrote a
    different `primary_type_family` into filters, the graph OVERWRITES it —
    explicit enforcement per D-04-04 (T-04-03-05).
    """
    if tool_name not in ("semantic_search", "nearby"):
        return dict(args)
    if not requested_primary_types:
        return dict(args)
    slot_index = args.get("slot_index")
    # `bool` is a subclass of `int` — reject it explicitly so True/False
    # don't sneak in as indices.
    if slot_index is None or not isinstance(slot_index, int) or isinstance(slot_index, bool):
        return dict(args)
    if slot_index < 0 or slot_index >= len(requested_primary_types):
        return dict(args)
    target_family = family_of(requested_primary_types[slot_index])
    if target_family is None:
        return dict(args)

    new_args = dict(args)
    existing_filters = new_args.get("filters")
    if existing_filters is None:
        base: dict[str, Any] = {}
    elif isinstance(existing_filters, SearchFilters):
        base = existing_filters.model_dump(exclude_none=True)
    elif isinstance(existing_filters, dict):
        # LangChain delivers `filters` as a plain dict in tool_call args.
        # Round-trip through SearchFilters once for defensive validation
        # (rejects unknown fields) then re-emit as a dict so the result
        # stays JSON-serializable.
        base = SearchFilters.model_validate(existing_filters).model_dump(exclude_none=True)
    else:
        # Unknown filters shape — fall back to dict() if possible, else empty.
        base = dict(existing_filters) if hasattr(existing_filters, "__iter__") else {}
    base["primary_type_family"] = target_family
    new_args["filters"] = base
    return new_args


_RECENT_TOOL_EXCHANGES_KEPT = 2
_EXPLICIT_STOP_COUNT_RETRY_KEY = "explicit_num_stops_clarification"


def _constraints_context(state: ItineraryState) -> str:
    """Human-readable deterministic constraints appended to the system prompt.

    Two sources contribute:
    1. An explicit user-stated stop count (preserved across multi-turn /chat).
    2. closure_context — every outcome contributes, so refinement turns never
       re-suggest a place we've already learned is closed. Even after the
       user accepts a drive alternative, the original closed source must
       stay excluded.
    """
    parts: list[str] = []
    if state.constraints.num_stops is not None:
        parts.append(
            f"- The user explicitly requested {state.constraints.num_stops} stops. "
            "Do not ask how many stops they want; plan exactly that many stops."
        )
    if state.closure_context:
        names = ", ".join(c.place_name for c in state.closure_context)
        parts.append(
            f"- Earlier in this conversation, these places were closed at the planned "
            f"arrival time and should NOT be re-suggested: {names}. "
            f"Their place_ids are also excluded from your search-result candidates."
        )
    if not parts:
        return ""
    return "\n\nDETERMINISTIC REQUEST CONTEXT:\n" + "\n".join(parts)


def _retry_unnecessary_stop_count_clarification(
    state: ItineraryState,
) -> dict[str, Any] | None:
    """One-shot nudge when the user gave an explicit count but the model
    finalized without any stops. The trigger is structural — no stops
    committed plus no tool calls (the call site is the `finalizing` branch
    of critique) — not lexical. An earlier string-match version missed
    non-stereotyped clarifying phrasings like "what vibe are you going for?"
    while still gating retries to one per turn via revision_counts."""
    num_stops = state.constraints.num_stops
    if num_stops is None:
        return None
    if state.stops:
        return None
    if state.revision_counts.get(_EXPLICIT_STOP_COUNT_RETRY_KEY, 0) > 0:
        return None
    counts = dict(state.revision_counts)
    counts[_EXPLICIT_STOP_COUNT_RETRY_KEY] = 1
    return {
        "revision_counts": counts,
        "messages": [
            HumanMessage(
                content=(
                    f"The user already specified {num_stops} stops. "
                    "Do not ask how many stops; use retrieval tools and plan exactly "
                    f"{num_stops} stops now."
                )
            )
        ],
        "done": False,
    }


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
            # D-08-07: preserve additional_kwargs (e.g. _reasoning_state)
            # across the cutoff window so adapter capture/replay can survive.
            pruned.append(
                AIMessage(
                    content=m.content if isinstance(m.content, str) else str(m.content),
                    additional_kwargs=m.additional_kwargs,
                )
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


def build_agent_graph(
    llm: BaseChatModel,
    max_steps: int = 8,
    judge_llm: BaseChatModel | None = None,
    *,
    provider: str = "openai",
):
    """Construct the agent graph.

    `judge_llm` is the cheap model used for vibe coherence scoring. If None
    and EVAL_VIBE_CRITIQUE_ENABLED=true, one is constructed via
    vibe.make_judge() at graph-build time. If construction fails (missing
    creds, unknown provider) the vibe pass is silently skipped.

    `provider` (keyword-only, default "openai") selects the
    `ProviderAdapter` that `plan()` closes over for reasoning-state
    round-trip. Resolved ONCE via the ADAPTERS registry (with NoOpAdapter
    fallback) at graph-build time (D-08-04, D-08-16). Unknown providers
    fall back to `NoOpAdapter` rather than raise — defensive default for
    Phase 8 where every registered entry is `NoOpAdapter` anyway (D-08-08);
    Phase 9 sub-phases may add stricter validation when real adapters land.
    """
    tools = all_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}
    # D-08-04 / D-08-16: resolve the ProviderAdapter once at graph-build time
    # and close it over plan(). Phase 8 ships NoOpAdapter for every provider
    # in SUPPORTED_PROVIDERS (D-08-08), so this is byte-identical to pre-Phase-8
    # behavior on the gpt-4o-mini path.
    adapter: ProviderAdapter = ADAPTERS.get(provider, NoOpAdapter())
    if judge_llm is None and vibe.is_enabled():
        judge_llm = vibe.make_judge()

    # Phase 13 / DEC arm-flag reads — resolved ONCE at graph-build time and
    # closed over the inner functions. With all three flags unset/0, behavior
    # is byte-identical to the baseline path (flag-off is the default state).
    #
    # FORCED_COMMIT_STEP (int, default 0 = off): A2 arm — at step N, if the
    #   model has not committed AND every slot has a viable candidate, synthesize
    #   a commit from best-so-far and route it through the normal commit path.
    #   Default value documented here: 6 (max_steps=8, leaves headroom for
    #   revision loops); the firing condition reads the env value, never hardcodes.
    #
    # VIABILITY_CONTRACT_ENABLED (bool, default off): A1 arm — appends the
    #   rule8_viability_addendum to the system prompt so the model sees the exact
    #   cosine threshold that determines viability.
    #
    # PARALLEL_TOOL_EXECUTION_ENABLED (bool, default off): A3 arm — runs all
    #   tool calls in one act() step concurrently via asyncio.gather with
    #   results appended in ORIGINAL tool_call order.
    _forced_commit_step: int = int(os.environ.get("FORCED_COMMIT_STEP", "0") or "0")
    _viability_contract_enabled: bool = os.environ.get(
        "VIABILITY_CONTRACT_ENABLED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    _parallel_tool_execution_enabled: bool = os.environ.get(
        "PARALLEL_TOOL_EXECUTION_ENABLED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    # Pre-compute the prompt addendum at build time (pure string; empty when flag off).
    _viability_prompt_addendum: str = rule8_viability_addendum(_viability_contract_enabled)

    async def plan(state: ItineraryState) -> dict[str, Any]:
        messages_in: list[BaseMessage] = list(state.messages)
        if state.step_count == 0 and not any(isinstance(m, SystemMessage) for m in messages_in):
            messages_in = [
                SystemMessage(
                    SYSTEM_PROMPT.format(
                        max_steps=max_steps,
                        current_datetime=current_datetime_str(),
                    )
                    + _viability_prompt_addendum
                    + _constraints_context(state)
                ),
                *messages_in,
            ]
        # Send the LLM a curated view: keep only the most recent ToolMessage
        # per tool name so token cost stays linear in tool *kinds*, not in
        # tool *calls*. Full history is preserved on state.messages for tracing.
        messages_for_llm = _prune_for_llm(messages_in)

        # D-08-05 / D-08-06: POST-PRUNE reasoning-state replay. Read the most
        # recent AIMessage's _reasoning_state kwarg (stashed by the previous
        # turn's capture, preserved across the _RECENT_TOOL_EXCHANGES_KEPT
        # cutoff by D-08-07's additional_kwargs forwarding in _prune_for_llm).
        # The adapter decides how to inject it; NoOpAdapter returns the list
        # unchanged so this is byte-identical for non-reasoning providers.
        captured_state = None
        for m in reversed(messages_for_llm):
            if isinstance(m, AIMessage):
                captured_state = m.additional_kwargs.get("_reasoning_state")
                break
        messages_for_llm = adapter.replay_reasoning_state(messages_for_llm, captured_state)

        # D-12-01: record LLM-call wall time for INST-04 latency decomposition.
        _llm_start = time.monotonic()
        ai = await llm_with_tools.ainvoke(messages_for_llm)
        _llm_elapsed = time.monotonic() - _llm_start

        # D-08-05 / D-08-06: capture the new turn's reasoning state and stash
        # it on the just-returned AIMessage's additional_kwargs. Storage lives
        # on the AIMessage (NOT on ItineraryState, NOT in a module-level dict)
        # so it survives the LangGraph add_messages reducer between turns —
        # load-bearing for the REASON-05 conformance gate in Plan 04.
        state_payload = adapter.capture_reasoning_state(ai)
        if state_payload is not None:
            ai.additional_kwargs["_reasoning_state"] = state_payload

        new_messages: list[BaseMessage] = []
        if len(messages_in) > len(state.messages):
            new_messages.append(messages_in[0])
        new_messages.append(ai)

        # Append a step_telemetry entry. tool_exec_seconds is 0.0 here (plan has
        # no tool execution); act() will patch the entry with the tool time.
        #
        # WR-07: step_count increments only in act(), so revision loops
        # (plan -> critique -> plan, e.g. a finalize rejected by
        # critique_final_with_stops or the stop-count clarification retry) run
        # plan() more than once at the SAME step_count. Merge those into the
        # existing trailing entry (summing llm_call_seconds) so step_telemetry
        # keeps exactly one entry per step index — Phase 13 consumers join on
        # "step" against viable_candidates_per_step and would double-count
        # llm_call_seconds on duplicate entries.
        new_telemetry = list(state.step_telemetry)
        if new_telemetry and new_telemetry[-1].get("step") == state.step_count:
            merged = dict(new_telemetry[-1])
            merged["llm_call_seconds"] = merged.get("llm_call_seconds", 0.0) + _llm_elapsed
            new_telemetry[-1] = merged
        else:
            new_telemetry.append(
                {
                    "step": state.step_count,
                    "llm_call_seconds": _llm_elapsed,
                    "tool_exec_seconds": 0.0,
                    "tool_calls_this_step": 0,
                }
            )
        return {"messages": new_messages, "step_telemetry": new_telemetry}

    async def act(state: ItineraryState) -> dict[str, Any]:
        ai = state.messages[-1]
        if not isinstance(ai, AIMessage) or not ai.tool_calls:
            return {}

        new_messages: list[BaseMessage] = []
        scratch_updates: dict[str, list[dict[str, Any]]] = {}
        committed_stops: list[Stop] | None = None

        # D-12-01: measure sequential tool-execution wall time for INST-04.
        _tool_start = time.monotonic()
        tool_calls_this_step = 0

        for tc in ai.tool_calls:
            if tc["name"] == COMMIT_ITINERARY_TOOL_NAME:
                stops, payload = commit_stops(state, tc["args"].get("stops"))
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
                tool_calls_this_step += 1
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
            # Belt-and-suspenders: merge closure-context exclusions AND
            # per-slot primary_type_family into the tool args at the SQL
            # layer. The prompt guidance in _constraints_context is an
            # optimization; this is enforcement.
            #
            # CRITICAL: never reassign `tc["args"]`. `tc` references a dict
            # INSIDE the AIMessage stored on state.messages — the next plan()
            # step re-serializes that AIMessage for OpenAI's API via
            # `json.dumps`, and a Pydantic SearchFilters instance there
            # crashes with "Object of type SearchFilters is not JSON
            # serializable". Compute `effective_args` locally instead so the
            # injected args drive `tool.invoke` and the scratch record while
            # the AIMessage stays untouched.
            #
            # Phase 4 D-04-04: chain the primary_type_family helper on top of
            # closure-exclusions. Both helpers emit JSON-safe dicts so the
            # chain composition stays JSON-safe end-to-end. The slot_index
            # strip then drops the marker arg before tool.invoke because the
            # underlying retrieval functions don't take it (and the strip is
            # a NO-OP for tool calls without a slot_index key, e.g.,
            # kg_traverse — see ADVISORY 4).
            if tc["name"] in ("semantic_search", "nearby", "kg_traverse"):
                effective_args = _inject_closure_exclusions(
                    tc["name"], tc["args"], state.closure_context
                )
                effective_args = _inject_primary_type_family(
                    tc["name"],
                    effective_args,
                    list(state.constraints.requested_primary_types),
                )
                effective_args = {k: v for k, v in effective_args.items() if k != "slot_index"}
            else:
                effective_args = tc["args"]
            try:
                # psycopg2 + OpenAI are sync; offload to a worker thread so the
                # event loop stays responsive while the tool blocks on I/O.
                result: Any = await asyncio.to_thread(tool.invoke, effective_args)
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
            new_messages.append(
                ToolMessage(content=_serialize_tool_result(result), tool_call_id=tc["id"])
            )
            scratch_updates.setdefault(tc["name"], []).append(
                {
                    "args": effective_args,
                    "result": result,
                    "step": state.step_count,
                    "id": tc["id"],
                }
            )
            tool_calls_this_step += 1

        _tool_elapsed = time.monotonic() - _tool_start

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

        # Patch the current step's telemetry entry (written by plan()) with actual
        # tool-execution time and tool-call count. If the entry is missing (edge
        # case where act() runs without a preceding plan() telemetry write),
        # append a fresh entry. All values are plain int/float (D-12-01 JSON-safe).
        telemetry = list(state.step_telemetry)
        if telemetry and telemetry[-1]["step"] == state.step_count:
            last = dict(telemetry[-1])
            last["tool_exec_seconds"] = _tool_elapsed
            last["tool_calls_this_step"] = tool_calls_this_step
            telemetry[-1] = last
        else:
            telemetry.append(
                {
                    "step": state.step_count,
                    "llm_call_seconds": 0.0,
                    "tool_exec_seconds": _tool_elapsed,
                    "tool_calls_this_step": tool_calls_this_step,
                }
            )
        update["step_telemetry"] = telemetry
        return update

    def critique(state: ItineraryState) -> dict[str, Any]:
        """Route to the appropriate critique branch based on state.

        Branches: max_steps short-circuit → finalizing-with-stops (itinerary
        + vibe) → finalizing-without-stops (clarifying question) → post-act
        per-step diagnose. Each branch is a pure function returning the
        LangGraph update dict."""
        if state.step_count >= max_steps:
            return short_circuit_max_steps(state)

        last = state.messages[-1] if state.messages else None
        finalizing = isinstance(last, AIMessage) and not last.tool_calls

        # A successful commit_itinerary IS a finalization signal. Without
        # this, termination depends on the model voluntarily emitting a
        # tool-call-free message after committing — gpt-4o-mini doesn't, so
        # it burns every step "improving" the plan until short_circuit
        # overwrites it with the canned step-limit error (3/3 live repros).
        # Run the same deterministic gauntlet as a model-driven finalize:
        # if hard checks pass it ends here; if they fail, that path drives
        # the normal revision loop (done=False + revision HumanMessage), so
        # legitimate self-correction is preserved.
        committed_this_step = any(
            entry.get("step") == state.step_count - 1
            for entry in state.scratch.get(COMMIT_ITINERARY_TOOL_NAME, [])
        )
        if (finalizing or committed_this_step) and state.stops:
            return critique_final_with_stops(state, last, judge_llm)
        if finalizing:
            retry = _retry_unnecessary_stop_count_clarification(state)
            if retry is not None:
                return retry
            return finalize_as_is(state, last)
        return critique_step(state)

    async def retime(state: ItineraryState) -> dict[str, Any]:
        """Reconcile final arrival_times against real Google Directions once.

        Self-guards to a no-op on every non-routable path (not done, <2
        stops with coords). route_legs is natively async (httpx) — awaited
        directly here, NOT via asyncio.to_thread (contrast act(), which
        wraps sync DB tools in a worker thread)."""
        if not state.done or not state.stops:
            return {}
        coords = [
            (s.latitude, s.longitude)
            for s in state.stops
            if s.latitude is not None and s.longitude is not None
        ]
        if len(coords) < 2 or len(coords) != len(state.stops):
            # Mixed/absent coords: the haversine arrival_time already on the
            # stops is the best we have. Leave it.
            return {}

        try:
            result = await route_legs(coords, mode="walk")
            leg_min = [leg.duration_s / 60 for leg in result.legs]
            retimed = chain_arrival_times(state.stops, leg_min)
        except Exception:  # noqa: BLE001
            # route_legs never raises, but chain_arrival_times can (e.g.
            # stops[0].arrival_time unset on a max-steps short-circuit).
            # Can't retime — leave the existing stops/reply untouched.
            return {}

        # Closure detection on the real-time arrivals is now the responsibility
        # of the swap_closed_stops node (next in the graph), which silently
        # swaps walking-distance alternatives and asks the user about anything
        # else. retime just supplies the retimed stops.
        return {"stops": retimed}

    def route_after_plan(state: ItineraryState) -> Literal["act", "critique"]:
        last = state.messages[-1]
        return "act" if isinstance(last, AIMessage) and last.tool_calls else "critique"

    def route_after_critique(state: ItineraryState) -> str:
        # Every finalized plan flows through `retime` (was END). retime
        # self-guards routability and returns {} when there's nothing to do.
        return "retime" if state.done else "plan"

    g = StateGraph(ItineraryState)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("critique", critique)
    g.add_node("retime", retime)
    g.add_node("swap_closed_stops", swap_closed_stops)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "critique": "critique"})
    g.add_edge("act", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"plan": "plan", "retime": "retime"})
    g.add_edge("retime", "swap_closed_stops")
    g.add_edge("swap_closed_stops", END)
    return g.compile()
