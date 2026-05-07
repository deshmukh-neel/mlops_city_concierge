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

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import ItineraryState, PlaceCard
from app.agent.tools import all_tools


def build_agent_graph(llm: BaseChatModel, max_steps: int = 8):
    tools = all_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    def plan(state: ItineraryState) -> ItineraryState:
        if state.step_count == 0 and not any(isinstance(m, SystemMessage) for m in state.messages):
            state.messages.insert(0, SystemMessage(SYSTEM_PROMPT.format(max_steps=max_steps)))
        ai = llm_with_tools.invoke(state.messages)
        state.messages.append(ai)
        return state

    def act(state: ItineraryState) -> ItineraryState:
        ai = state.messages[-1]
        if not isinstance(ai, AIMessage) or not ai.tool_calls:
            return state
        for tc in ai.tool_calls:
            tool = tool_by_name.get(tc["name"])
            if tool is None:
                state.messages.append(
                    ToolMessage(
                        content=f"unknown tool {tc['name']}",
                        tool_call_id=tc["id"],
                    )
                )
                continue
            try:
                result = tool.invoke(tc["args"])
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
            state.messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            state.scratch.setdefault(tc["name"], []).append(
                {
                    "args": tc["args"],
                    "result": result,
                    "step": state.step_count,
                }
            )
        state.step_count += 1
        return state

    def critique(state: ItineraryState) -> ItineraryState:
        # W3 fills this in. For W2, do the bare minimum: check loop bound and
        # whether the last AI message ended without a tool call.
        last = state.messages[-1] if state.messages else None
        if isinstance(last, AIMessage) and not last.tool_calls:
            state.done = True
            state.final_reply = state.final_reply or (
                last.content if isinstance(last.content, str) else str(last.content)
            )
        if state.step_count >= max_steps and not state.done:
            state.done = True
            if not state.final_reply:
                state.final_reply = (
                    "I hit the planning step limit. Here is the best plan I had so far."
                )
        return state

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
