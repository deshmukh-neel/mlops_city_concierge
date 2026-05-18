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
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.agent.commit import commit_stops
from app.agent.critique import vibe
from app.agent.critique.checks import CRITIQUE_THRESHOLDS, temporal_coherence
from app.agent.planning import chain_arrival_times
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.revision import (
    _final_with_caveats,
    critique_final_with_stops,
    critique_step,
    finalize_as_is,
    short_circuit_max_steps,
)
from app.agent.state import ItineraryState, Stop
from app.agent.tools import COMMIT_ITINERARY_TOOL_NAME, all_tools
from app.tools.directions import route_legs

logger = logging.getLogger(__name__)


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

        update: dict[str, Any] = {"stops": retimed}

        # Re-run ONLY the open-at-arrival check on the real times. Other
        # checks (geographic/walking/hallucination) are coord/id-based and
        # unaffected by re-timing, so re-running them would be wasted work.
        # temporal_coherence is sync psycopg2 I/O — offload to a thread so
        # the event loop stays responsive (same pattern as act()).
        probe = state.model_copy(update={"stops": retimed})
        try:
            score = await asyncio.to_thread(temporal_coherence, probe)
        except Exception:  # noqa: BLE001
            # Fails open exactly like itinerary_violations(): a DB blip must
            # not block /chat. Ship the re-timed plan without the re-check.
            return update

        # Only append a caveat if the revision loop hasn't already shipped
        # one (the END->retime rewire routes the caveats-exhausted path
        # through here too; _final_with_caveats is not idempotent, so a
        # second call would duplicate the identical "Caveats:" paragraph).
        if score < CRITIQUE_THRESHOLDS["temporal_coherence"]:
            existing = state.final_reply or ""
            if "Caveats:" not in existing:
                update["final_reply"] = _final_with_caveats(existing, ["temporal_coherence"])
        return update

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
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "critique": "critique"})
    g.add_edge("act", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"plan": "plan", "retime": "retime"})
    g.add_edge("retime", END)
    return g.compile()
