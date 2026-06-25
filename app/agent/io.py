"""Conversion between agent state and the HTTP-facing wire contract."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.state import ItineraryState, PlaceCard, Stop


class RoleAndContent(Protocol):
    @property
    def role(self) -> str: ...
    @property
    def content(self) -> str: ...


def messages_from_history(history: Iterable[RoleAndContent]) -> list[BaseMessage]:
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
            address=s.address,
            rating=s.rating,
            price_level=s.price_level,
            primary_type=s.primary_type,
            latitude=s.latitude,
            longitude=s.longitude,
            arrival_time=s.arrival_time,
            rationale=s.rationale,
            booking_url=s.booking_url,
            booking_provider=s.booking_provider,
        ).model_dump(mode="json")
        for s in state.stops
    ]


REFINEMENT_PREAMBLE: str = (
    "REFINEMENT TURN — the user is editing one stop in the itinerary below. "
    "The fenced JSON block carries the prior committed plan: each entry has "
    "a 1-indexed `slot`, the `place_id` of that stop, and its planned "
    "`arrival_time`. The user's next message names what to change. "
    "Reuse the `place_id` and `slot` index of every stop you are not changing "
    "exactly as listed; only the slot named by the user gets a new `place_id`. "
    "Produce the updated itinerary by calling the `commit_itinerary` tool "
    "with the full stop list."
)


def build_refinement_prompt_message(committed_stops: list[Stop]) -> HumanMessage:
    """Build the structured-plan message used for a refinement turn."""
    if not committed_stops:
        raise ValueError("committed_stops must be non-empty")

    plan_payload: dict[str, Any] = {
        "current_plan": [
            {
                "slot": i + 1,
                "place_id": s.place_id,
                "arrival_time": s.arrival_time.isoformat() if s.arrival_time else None,
            }
            for i, s in enumerate(committed_stops)
        ]
    }
    json_block = json.dumps(plan_payload, ensure_ascii=False)
    content = f"{REFINEMENT_PREAMBLE}\n\n```json\n{json_block}\n```"
    return HumanMessage(content=content)
