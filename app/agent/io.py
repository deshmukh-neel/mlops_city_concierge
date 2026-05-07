"""Conversion between the agent's internal types and the wire contract.

Inbound: ChatMessage (HTTP) -> list[BaseMessage] for the graph.
Outbound: ItineraryState.stops -> list[PlaceCard] dicts for the frontend.

The agent module owns the agent shapes; HTTP-layer concerns like rag_label
stay in app.main and are stamped onto the response there.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.state import ItineraryState, PlaceCard


class _RoleAndContent(Protocol):
    @property
    def role(self) -> str: ...
    @property
    def content(self) -> str: ...


def messages_from_history(history: Iterable[_RoleAndContent]) -> list[BaseMessage]:
    """Map ChatMessage role/content pairs onto LangChain BaseMessages."""
    out: list[BaseMessage] = []
    for m in history:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        else:
            out.append(AIMessage(content=m.content))
    return out


def state_to_cards(state: ItineraryState) -> list[dict[str, Any]]:
    """Project committed stops into the PlaceCard shape the frontend renders."""
    return [
        PlaceCard(
            place_id=s.place_id,
            name=s.name,
            primary_type=s.primary_type,
            arrival_time=s.arrival_time,
            rationale=s.rationale,
        ).model_dump(mode="json")
        for s in state.stops
    ]
