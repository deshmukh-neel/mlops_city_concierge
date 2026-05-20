from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.agent.io import state_to_cards
from app.agent.state import (
    DEFAULT_STOP_DURATION_MIN,
    DEFAULT_STOP_DURATION_MIN_FALLBACK,
    ClosureContext,
    ItineraryState,
    PlaceCard,
    Stop,
    UserConstraints,
    default_duration_for,
)


def test_itinerary_state_defaults() -> None:
    state = ItineraryState()
    assert state.stops == []
    assert state.messages == []
    assert state.scratch == {}
    assert state.step_count == 0
    assert state.done is False
    assert state.final_reply is None
    assert state.awaiting_stops_count is False
    assert state.walked_meters_so_far == 0.0
    assert isinstance(state.constraints, UserConstraints)
    assert state.constraints.min_user_rating_count is None
    assert state.constraints.walking_budget_m == 2400


def test_stop_round_trips_via_model_dump() -> None:
    stop = Stop(
        place_id="p1",
        name="Trick Dog",
        rationale="iconic SF cocktail bar",
        source="google_places",
        primary_type="cocktail_bar",
        latitude=37.7,
        longitude=-122.4,
        planned_duration_min=60,
        arrival_time=datetime(2026, 5, 6, 19, 0, tzinfo=timezone.utc),
    )
    payload = stop.model_dump(mode="json")
    rebuilt = Stop(**payload)
    assert rebuilt.place_id == stop.place_id
    assert rebuilt.arrival_time is not None


def test_default_duration_for_known_and_unknown_types() -> None:
    assert default_duration_for("restaurant") == DEFAULT_STOP_DURATION_MIN["restaurant"]
    assert default_duration_for("RESTAURANT") == DEFAULT_STOP_DURATION_MIN["restaurant"]
    assert default_duration_for("unknown_type") == DEFAULT_STOP_DURATION_MIN_FALLBACK
    assert default_duration_for(None) == DEFAULT_STOP_DURATION_MIN_FALLBACK
    assert default_duration_for("") == DEFAULT_STOP_DURATION_MIN_FALLBACK


def test_state_to_cards_empty_stops() -> None:
    state = ItineraryState(final_reply="No matches found.")
    assert state_to_cards(state) == []


def test_state_to_cards_with_stops() -> None:
    state = ItineraryState(
        stops=[
            Stop(
                place_id="p1",
                name="X",
                rationale="r",
                source="google_places",
                primary_type="restaurant",
            )
        ],
        final_reply="Try X.",
    )
    cards = state_to_cards(state)
    assert len(cards) == 1
    card = cards[0]
    assert card["place_id"] == "p1"
    assert card["name"] == "X"
    assert card["rationale"] == "r"
    assert card["primary_type"] == "restaurant"
    # PlaceCard fields not derived from Stop should be None / default.
    assert card["address"] is None


@pytest.mark.parametrize("provider", ["resy", "tock", "opentable", "google_maps", "unknown", None])
def test_stop_accepts_every_valid_booking_provider(provider: str | None) -> None:
    """The closed BookingProvider literal must accept exactly the five providers
    used by app.tools.booking, plus None."""
    stop = Stop(
        place_id="p1",
        name="X",
        rationale="r",
        source="google_places",
        booking_provider=provider,
    )
    assert stop.booking_provider == provider


def test_stop_rejects_unknown_booking_provider() -> None:
    """End-to-end typing contract: a typo on Stop fails the same way it would
    fail on BookingProposal — Stop and BookingProposal share the literal."""
    with pytest.raises(ValueError):
        Stop(
            place_id="p1",
            name="X",
            rationale="r",
            source="google_places",
            booking_provider="resyy",  # type: ignore[arg-type]
        )


def test_place_card_rejects_unknown_booking_provider() -> None:
    """The frontend label table is keyed off booking_provider; a mistyped
    value would silently render as 'no label'. Lock the contract here."""
    with pytest.raises(ValueError):
        PlaceCard(
            place_id="p1",
            name="X",
            rationale="r",
            booking_provider="yelp",  # type: ignore[arg-type]
        )


def test_place_card_serialization_keys() -> None:
    card = PlaceCard(place_id="p1", name="X", rationale="r")
    payload = card.model_dump(mode="json")
    expected_keys = {
        "place_id",
        "name",
        "address",
        "rating",
        "price_level",
        "primary_type",
        "latitude",
        "longitude",
        "arrival_time",
        "rationale",
        "booking_url",
        "booking_provider",
    }
    assert set(payload.keys()) == expected_keys


# ─── ClosureContext + ItineraryState.closure_context (closure-aware swap) ───

_SF = ZoneInfo("America/Los_Angeles")


def test_closure_context_minimal_fields_validate() -> None:
    """Pending entry with no proposal — used when nearby() returns no candidate."""
    ctx = ClosureContext(
        place_id="ChIJ_closed",
        place_name="Mochill Mochidonut",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 2, tzinfo=_SF),
        outcome="pending_user_decision",
        insert_after_place_id="ChIJ_stop1",
        insert_before_place_id=None,
        stop_index_hint=2,
        proposed_alternative=None,
        proposed_distance_m=None,
    )
    assert ctx.schema_version == 1
    assert ctx.outcome == "pending_user_decision"
    assert ctx.proposed_alternative is None


def test_closure_context_with_proposal_validates() -> None:
    sophies = Stop(
        place_id="ChIJ_sophies",
        name="Sophie's Crepes",
        rationale="closest open dessert",
        source="google_places",
        latitude=37.7849,
        longitude=-122.4093,
        primary_type="Dessert Shop",
        planned_duration_min=30,
    )
    ctx = ClosureContext(
        place_id="ChIJ_closed",
        place_name="Mochill Mochidonut",
        family="dessert",
        attempted_arrival=datetime(2026, 5, 19, 20, 2, tzinfo=_SF),
        outcome="pending_user_decision",
        insert_after_place_id="ChIJ_stop1",
        insert_before_place_id=None,
        stop_index_hint=2,
        proposed_alternative=sophies,
        proposed_distance_m=4800.0,
    )
    assert ctx.proposed_alternative is not None
    assert ctx.proposed_alternative.place_id == "ChIJ_sophies"
    assert ctx.proposed_distance_m == 4800.0


def test_itinerary_state_default_closure_context_empty() -> None:
    state = ItineraryState()
    assert state.closure_context == []


def test_itinerary_state_accepts_closure_context_list() -> None:
    state = ItineraryState(
        closure_context=[
            ClosureContext(
                place_id="p",
                place_name="X",
                family="bar",
                attempted_arrival=datetime(2026, 5, 19, 20, 0, tzinfo=_SF),
                outcome="auto_swapped",
                insert_after_place_id=None,
                insert_before_place_id=None,
                stop_index_hint=0,
                proposed_alternative=None,
                proposed_distance_m=None,
            )
        ]
    )
    assert len(state.closure_context) == 1
    assert state.closure_context[0].outcome == "auto_swapped"
