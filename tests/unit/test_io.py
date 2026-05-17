"""Unit tests for app.agent.io projection."""

from __future__ import annotations

from app.agent.io import state_to_cards
from app.agent.state import ItineraryState, Stop


def test_state_to_cards_carries_address_rating_price() -> None:
    state = ItineraryState(
        stops=[
            Stop(
                place_id="p1",
                name="Tartine",
                address="600 Guerrero St, San Francisco",
                rating=4.5,
                price_level=2,
                rationale="great pastries",
                source="google_places",
            )
        ]
    )

    cards = state_to_cards(state)

    assert len(cards) == 1
    assert cards[0]["address"] == "600 Guerrero St, San Francisco"
    assert cards[0]["rating"] == 4.5
    assert cards[0]["price_level"] == 2


def test_state_to_cards_defaults_missing_fields_to_none() -> None:
    state = ItineraryState(
        stops=[Stop(place_id="p1", name="X", rationale="r", source="google_places")]
    )

    cards = state_to_cards(state)

    assert cards[0]["address"] is None
    assert cards[0]["rating"] is None
    assert cards[0]["price_level"] is None
