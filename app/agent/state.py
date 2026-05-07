"""Structured state passed through every node of the agent graph.

Keeping state as a Pydantic model (not just messages) lets the LLM revise
specific stops or constraints without regenerating prose, and lets the
critique node do deterministic checks (geographic coherence, hours).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class UserConstraints(BaseModel):
    """Parsed/inferred constraints from the user message.

    These are SHARED across all stops: a "$$$" budget applies to dinner AND
    drinks AND dessert. Per-stop constraints (cuisine, vibe) live on `Stop`.
    """

    party_size: int | None = None
    budget_per_person_max: int | None = None  # USD
    price_level_max: int | None = None  # 0-4 (Google)
    min_rating: float | None = None
    min_user_rating_count: int = 50  # quality floor (W1 default)
    when: datetime | None = None  # arrival time for stop 1
    neighborhood: str | None = None
    vibes: list[str] = Field(default_factory=list)  # free-text tags
    must_be_open: bool = True
    num_stops: int | None = Field(
        default=None,
        description=(
            "Total stops the user wants. None = ask. Default 3 if user "
            "says 'plan a date/evening' without a count."
        ),
    )
    walking_budget_m: int = Field(
        default=2400,
        description=(
            "Total walking distance budget across the whole itinerary, "
            "in meters. Default ~30 min total walking at 80 m/min."
        ),
    )


DEFAULT_STOP_DURATION_MIN: dict[str, int] = {
    "restaurant": 90,
    "fine_dining_restaurant": 120,
    "cafe": 45,
    "coffee_shop": 45,
    "bar": 60,
    "wine_bar": 60,
    "cocktail_bar": 60,
    "bakery": 30,
    "ice_cream_shop": 30,
    "museum": 120,
    "art_gallery": 60,
    "park": 30,
    "tourist_attraction": 60,
}
DEFAULT_STOP_DURATION_MIN_FALLBACK = 60


def default_duration_for(primary_type: str | None) -> int:
    if not primary_type:
        return DEFAULT_STOP_DURATION_MIN_FALLBACK
    return DEFAULT_STOP_DURATION_MIN.get(primary_type.lower(), DEFAULT_STOP_DURATION_MIN_FALLBACK)


class Stop(BaseModel):
    place_id: str
    name: str
    arrival_time: datetime | None = None
    planned_duration_min: int = DEFAULT_STOP_DURATION_MIN_FALLBACK
    rationale: str
    source: str  # 'google_places' | 'editorial'
    latitude: float | None = None
    longitude: float | None = None
    primary_type: str | None = None


class ItineraryState(BaseModel):
    """The single piece of state passed through every graph node."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    constraints: UserConstraints = Field(default_factory=UserConstraints)
    stops: list[Stop] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    done: bool = False
    final_reply: str | None = None
    awaiting_stops_count: bool = Field(
        default=False,
        description=(
            "True after the agent asked the user how many stops they "
            "want. The /chat handler echoes the question and the next "
            "turn parses the answer back into constraints.num_stops."
        ),
    )
    walked_meters_so_far: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PlaceCard(BaseModel):
    """Frontend-facing place shape. Matches what frontend/src/api/chat.js
    renders. Derived from ItineraryState.stops at response time."""

    place_id: str
    name: str
    address: str | None = None
    rating: float | None = None
    price_level: int | None = None
    primary_type: str | None = None
    arrival_time: datetime | None = None
    rationale: str
    booking_url: str | None = None  # populated by W4
