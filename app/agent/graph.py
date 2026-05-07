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

import json
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import ItineraryState, PlaceCard, Stop
from app.agent.tools import COMMIT_ITINERARY_TOOL_NAME, all_tools


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


def build_agent_graph(llm: BaseChatModel, max_steps: int = 8):
    tools = all_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    def plan(state: ItineraryState) -> dict[str, Any]:
        messages_in: list[BaseMessage] = list(state.messages)
        if state.step_count == 0 and not any(isinstance(m, SystemMessage) for m in messages_in):
            messages_in = [
                SystemMessage(SYSTEM_PROMPT.format(max_steps=max_steps)),
                *messages_in,
            ]
        ai = llm_with_tools.invoke(messages_in)
        # If we prepended a SystemMessage, surface it through the reducer too so
        # downstream nodes see a consistent history.
        new_messages: list[BaseMessage] = []
        if len(messages_in) > len(state.messages):
            new_messages.append(messages_in[0])
        new_messages.append(ai)
        return {"messages": new_messages}

    def act(state: ItineraryState) -> dict[str, Any]:
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
                    {"args": tc["args"], "result": payload, "step": state.step_count}
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
                result: Any = tool.invoke(tc["args"])
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
            new_messages.append(
                ToolMessage(content=_serialize_tool_result(result), tool_call_id=tc["id"])
            )
            scratch_updates.setdefault(tc["name"], []).append(
                {"args": tc["args"], "result": result, "step": state.step_count}
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
        # W3 fills this in. For W2, do the bare minimum: check loop bound and
        # whether the last AI message ended without a tool call.
        last = state.messages[-1] if state.messages else None
        update: dict[str, Any] = {}
        if isinstance(last, AIMessage) and not last.tool_calls:
            update["done"] = True
            update["final_reply"] = state.final_reply or (
                last.content if isinstance(last.content, str) else str(last.content)
            )
        if state.step_count >= max_steps and not (state.done or update.get("done")):
            update["done"] = True
            if not state.final_reply and not update.get("final_reply"):
                update["final_reply"] = (
                    "I hit the planning step limit. Here is the best plan I had so far."
                )
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


def state_to_response(state: ItineraryState, rag_label: str) -> dict:
    """Convert ItineraryState into the {reply, places, ragLabel} contract that
    frontend/src/api/chat.js expects."""
    cards = [
        PlaceCard(
            place_id=s.place_id,
            name=s.name,
            primary_type=s.primary_type,
            arrival_time=s.arrival_time,
            rationale=s.rationale,
        ).model_dump(mode="json")
        for s in state.stops
    ]
    return {
        "reply": state.final_reply or "",
        "places": cards,
        "ragLabel": rag_label,
    }
