"""Conversion between the agent's internal types and the wire contract.

Inbound: ChatMessage (HTTP) -> list[BaseMessage] for the graph.
Outbound: ItineraryState.stops -> list[PlaceCard] dicts for the frontend.
Refinement: list[Stop] -> structured-plan HumanMessage via
``build_refinement_prompt_message`` (Phase 6 / plan 06-05) — SHARED between
``/chat`` (production) and ``evaluate_multi_turn_case`` (eval runner, plan
06-06) so the two surfaces build BYTE-IDENTICAL refinement messages.

The agent module owns the agent shapes; HTTP-layer concerns like rag_label
stay in app.main and are stamped onto the response there.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.state import ItineraryState, PlaceCard, Stop


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


# Phase 6 / plan 06-05 — refinement structured-plan HumanMessage builder.
#
# This helper is the SHARED point of byte-identity between the production
# `/chat` injection block (plan 06-05 Task 2a) and the eval runner's
# multi-turn threading (plan 06-06). Both surfaces call THIS function with
# the same `committed_stops` list, so a hypothetical wording drift in the
# preamble cannot desync the two surfaces — PATTERNS.md Critical Caveat #5.
#
# HIGH-4 strategy (a) — prompt-injection mitigation via field whitelisting:
# the JSON payload surfaced to the model contains ONLY `slot`, `place_id`,
# and `arrival_time` per entry. Client-supplied display strings (`name`,
# `primary_type`, `rationale`, `source`, `address`) are dropped at the
# helper boundary BEFORE `json.dumps`. The model has the `place_id` (the
# byte-equal contract anchor) and `arrival_time` (downstream-timing
# anchor); `name`/`primary_type` are server-derivable from `place_id`
# against `places_raw` and are not needed for the preserve-byte-equal
# contract. The SYSTEM_PROMPT addendum (plan 06-05 Task 2a) tells the
# model to look at `place_id` (not `name`) for preservation, so dropping
# the display strings does not weaken the contract.
#
# The preamble wording is deliberately *not* hedging on the commit
# directive ("consider whether to commit", "carefully think about
# committing") — PATTERNS.md Caveat #8 pins this so the refinement turn
# does not pull against `commit_itinerary`'s decisiveness directive.
_REFINEMENT_PREAMBLE: str = (
    "REFINEMENT TURN — the user is editing one stop in the itinerary below. "
    "The fenced JSON block carries the prior committed plan: each entry has "
    "a 1-indexed `slot`, the `place_id` of that stop, and its planned "
    "`arrival_time`. The user's next message names what to change. "
    "Produce the updated itinerary by calling the `commit_itinerary` tool "
    "with the full stop list."
)


def build_refinement_prompt_message(committed_stops: list[Stop]) -> HumanMessage:
    """Build the Phase 6 structured-plan ``HumanMessage`` for a refinement turn.

    Returns a ``HumanMessage`` whose ``.content`` is a ``str`` (NEVER a
    Pydantic object or dict — see ``project_aimessage_tool_call_args_json_safe``).
    The content is a hybrid prose preamble + fenced JSON block: the preamble
    teaches the model what the structured plan is for, and the JSON block
    carries the byte-equal ``place_id`` anchors the model must preserve.

    HIGH-4 strategy (a) — prompt-injection mitigation: only ``slot``,
    ``place_id``, and ``arrival_time`` per entry are surfaced. Client-supplied
    display strings (``name``, ``primary_type``, ``rationale``, ``source``,
    ``address``) are dropped at the helper boundary.

    Slots are 1-indexed to match user prose ("make stop 2 cheaper") and the
    YAML ``expected_refinement.target_slot: 2`` convention from D-06-08.

    Raises ``ValueError`` on empty ``committed_stops``. The ``/chat`` handler
    block (plan 06-05 Task 2a) guards on ``incoming.committed_stops`` being
    non-empty before calling, but the helper enforces the contract too so
    misuse from the eval runner (plan 06-06) is loud.
    """
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
    content = f"{_REFINEMENT_PREAMBLE}\n\n```json\n{json_block}\n```"
    return HumanMessage(content=content)
