from __future__ import annotations

from datetime import datetime, timezone

from app.agent.io import state_to_cards
from app.agent.state import (
    DEFAULT_STOP_DURATION_MIN,
    DEFAULT_STOP_DURATION_MIN_FALLBACK,
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
    assert state.constraints.min_user_rating_count == 50
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
        "arrival_time",
        "rationale",
        "booking_url",
    }
    assert set(payload.keys()) == expected_keys
