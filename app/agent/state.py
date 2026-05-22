"""Structured state passed through every node of the agent graph.

Keeping state as a Pydantic model (not just messages) lets the LLM revise
specific stops or constraints without regenerating prose, and lets the
critique node do deterministic checks (geographic coherence, hours).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from app.tools.booking_types import Provider as BookingProvider

RevisionReason = Literal[
    "empty_results",
    "all_closed",
    "low_similarity",
    "constraint_violation",
    "tool_error",
    "geographic_incoherence",
    "temporal_incoherence",
    "walking_budget_exceeded",
    "constraint_unmet_in_final",
    "stop_count_mismatch",
    "hallucinated_place_id",
    "vibe_mismatch",
    "rationale_misaligned",
]

RevisionAction = Literal[
    "drop_filter",
    "expand_radius",
    "broaden_query",
    "clarify_with_user",
    "try_different_tool",
    "swap_stop",
    "tighten_radius",
    "shift_arrival_time",
    "rebalance_walking_budget",
    "add_missing_stops",
    "remove_extra_stops",
]


class RevisionHint(BaseModel):
    """A structured cue from `critique` to `plan` describing what went wrong
    and what action would likely fix it. Stored on state for tracing and to
    bound retries per failure category."""

    reason: RevisionReason
    detail: str
    suggested_action: RevisionAction
    target: dict[str, Any] = Field(default_factory=dict)


class UserConstraints(BaseModel):
    """Parsed/inferred constraints from the user message.

    These are SHARED across all stops: a "$$$" budget applies to dinner AND
    drinks AND dessert. Per-stop constraints (cuisine, vibe) live on `Stop`.
    """

    party_size: int | None = None
    budget_per_person_max: int | None = None  # USD
    price_level_max: int | None = None  # 0-4 (Google)
    min_rating: float | None = None
    min_user_rating_count: int | None = None  # only set when user expresses it
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
    requested_primary_types: list[str] = Field(
        default_factory=list,
        description=(
            "Per-slot expected Google primary_type values when the user "
            "names category slots (e.g., 'omakase, then drinks, then "
            "dessert'). Default [] preserves free-text behavior (D-01 / "
            "D-03 contract: category_compliance scorer abstains when this "
            "is empty). Read by app.agent.critique.checks.category_compliance "
            "(EVAL-01) and by Phase 4's primary_type_family enforcement."
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


_PRICE_LEVEL_RANK: dict[str, int] = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


def price_level_to_rank(value: str | None) -> int | None:
    """Map Google's price_level enum string to the 0..4 rank PlaceCard expects.

    Mirrors the SQL `price_level_rank()` function (alembic c428add573d7) so the
    card's integer tier matches the rank used in SearchFilters comparisons.
    Unknown / unspecified / None all collapse to None.
    """
    if value is None:
        return None
    return _PRICE_LEVEL_RANK.get(value)


class Stop(BaseModel):
    place_id: str
    name: str
    address: str | None = None
    rating: float | None = None
    price_level: int | None = None  # 0..4, mapped via price_level_to_rank
    arrival_time: datetime | None = None
    planned_duration_min: int = DEFAULT_STOP_DURATION_MIN_FALLBACK
    rationale: str
    source: str  # 'google_places' | 'editorial'
    latitude: float | None = None
    longitude: float | None = None
    primary_type: str | None = None
    booking_url: str | None = None
    booking_provider: BookingProvider | None = None


ClosureOutcome = Literal[
    "auto_swapped",
    "user_accepted_drive",
    "user_declined_dropped",
    "pending_user_decision",
    "queued_user_decision",
]


class ClosureContext(BaseModel):
    """One closure event recorded during a /chat conversation.

    Persisted across turns via the opaque `conversation_state` round-trip on
    /chat. Placement anchors (`insert_after_place_id` / `insert_before_place_id`)
    are durable against neighbor drops/inserts because they reference
    neighboring stops' place_ids rather than indices; `stop_index_hint` is the
    last-resort fallback used only when both anchors are absent from current
    stops. Resolution order is documented in
    `app.agent.swap._resolve_insert_position`.
    """

    schema_version: int = 1
    place_id: str
    place_name: str
    family: str
    attempted_arrival: datetime
    outcome: ClosureOutcome
    insert_after_place_id: str | None = None
    insert_before_place_id: str | None = None
    stop_index_hint: int
    proposed_alternative: Stop | None = None
    proposed_distance_m: float | None = None


MAX_CLOSURE_CONTEXT_ENTRIES = 10


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
    revision_hints: list[RevisionHint] = Field(default_factory=list)
    revision_counts: dict[str, int] = Field(default_factory=dict)
    closure_context: list[ClosureContext] = Field(default_factory=list)

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
    latitude: float | None = None
    longitude: float | None = None
    arrival_time: datetime | None = None
    rationale: str
    booking_url: str | None = None
    booking_provider: BookingProvider | None = None
