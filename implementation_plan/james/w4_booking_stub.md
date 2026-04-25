# W4 — Booking handoff stub (`propose_booking`)

**Branch:** `feature/agent-w4-booking-stub`
**Depends on:** W2
**Unblocks:** future "real automation" PR (not in this plan)

## Goal

Add a `propose_booking` tool that the agent calls when the user asks to book a stop in their itinerary. It does **not** automate bookings. It returns a deep-link to the venue's booking page (Resy / Tock / OpenTable / Google Maps fallback) with the user's party size and time pre-filled where the provider's URL scheme supports it. The frontend renders this as a button on the place card.

This locks in the *tool surface* the agent uses, so a later PR can swap the stub for Playwright-based automation behind `BOOKING_AUTOMATION_ENABLED` without touching the agent or the frontend.

After this PR:
- Every place card the agent emits can have a `booking_url`.
- The agent calls `propose_booking` after finalizing an itinerary, not before (it's a leaf action).
- Zero credentials stored. Zero ToS risk.

## Files

### New: `app/tools/booking.py`

```python
"""Booking handoff. Detect provider from website_uri or maps_uri; produce a
deep-link with party size + time pre-filled where supported. Fall back to
Google Maps if no provider is detected.

NOT AUTOMATION. The user still taps to confirm on the provider's site.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from urllib.parse import urlencode, urlparse

from pydantic import BaseModel

from app.tools.retrieval import get_details

Provider = Literal["resy", "tock", "opentable", "google_maps", "unknown"]


class BookingProposal(BaseModel):
    place_id: str
    provider: Provider
    booking_url: str
    automation_available: bool = False  # future flag
    notes: Optional[str] = None         # human-readable hint, e.g. "no online booking"


def detect_provider(website_uri: Optional[str]) -> Provider:
    if not website_uri:
        return "unknown"
    host = (urlparse(website_uri).hostname or "").lower()
    if "resy.com" in host:
        return "resy"
    if "exploretock.com" in host or "tock.com" in host:
        return "tock"
    if "opentable.com" in host:
        return "opentable"
    return "unknown"


def propose_booking(
    place_id: str,
    when: datetime,
    party_size: int = 2,
) -> BookingProposal:
    details = get_details(place_id)
    if details is None:
        raise ValueError(f"unknown place_id {place_id}")

    provider = detect_provider(details.website_uri)
    url = _build_booking_url(provider, details, when, party_size)

    return BookingProposal(
        place_id=place_id,
        provider=provider,
        booking_url=url,
        notes=_notes_for(provider),
    )


def _build_booking_url(provider: Provider, details, when: datetime, party_size: int) -> str:
    iso_date = when.strftime("%Y-%m-%d")
    iso_time = when.strftime("%H:%M")

    if provider == "resy":
        # Resy: https://resy.com/cities/sf/{venue-slug}?date=2026-04-26&seats=2
        # We don't reliably know the slug — the website_uri usually has it.
        return _append_query(details.website_uri, {"date": iso_date, "seats": party_size})

    if provider == "tock":
        # Tock: https://www.exploretock.com/{venue}?date=...&size=...&time=...
        return _append_query(details.website_uri, {
            "date": iso_date, "size": party_size, "time": iso_time,
        })

    if provider == "opentable":
        # OpenTable: https://www.opentable.com/{venue}?covers=...&dateTime=...
        dt = f"{iso_date}T{iso_time}"
        return _append_query(details.website_uri, {"covers": party_size, "dateTime": dt})

    # Fallback: Google Maps
    return details.maps_uri or f"https://www.google.com/maps/search/?api=1&query={details.name}"


def _append_query(url: str, params: dict) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode(params)}"


def _notes_for(provider: Provider) -> str:
    return {
        "resy":       "Tap to open Resy with your time and party size pre-filled.",
        "tock":       "Tap to open Tock with your reservation pre-filled.",
        "opentable":  "Tap to open OpenTable with your reservation pre-filled.",
        "google_maps":"No online booking detected; opens Google Maps.",
        "unknown":    "Online booking not supported; opens the venue's website.",
    }[provider]
```

### Modify: `app/agent/tools.py`

Wire the tool into the agent:

```python
from app.tools.booking import propose_booking, BookingProposal

@tool("propose_booking")
def propose_booking_tool(place_id: str, when_iso: str, party_size: int = 2) -> BookingProposal:
    """Generate a one-tap booking link for a place. Call this AFTER you've
    finalized which stops the user wants. `when_iso` is an ISO 8601 datetime
    string in local time."""
    from datetime import datetime
    return propose_booking(place_id, datetime.fromisoformat(when_iso), party_size)


def all_tools():
    return [
        semantic_search_tool, nearby_tool, get_details_tool,
        kg_traverse_tool, propose_booking_tool,
    ]
```

### Modify: `app/agent/state.py`

Add `booking_url` to `Stop`:

```python
class Stop(BaseModel):
    # ... existing
    booking_url: Optional[str] = None
    booking_provider: Optional[str] = None
```

### Modify: `app/agent/graph.py`

Update `state_to_response` to project `booking_url` onto `PlaceCard`:

```python
def state_to_response(state: ItineraryState, rag_label: str) -> dict:
    cards = [
        PlaceCard(
            place_id=s.place_id,
            name=s.name,
            arrival_time=s.arrival_time,
            rationale=s.rationale,
            booking_url=s.booking_url,
        ).model_dump(mode="json")
        for s in state.stops
    ]
    return {"reply": state.final_reply or "", "places": cards, "ragLabel": rag_label}
```

### Modify: `app/agent/prompts.py`

Add to the system prompt:

```
BOOKING:
- After you've decided on the final list of stops, call `propose_booking` for
  each stop that the user wants to reserve. Use the user's stated party size,
  defaulting to 2 if unstated. Use the stop's `arrival_time` as `when_iso`.
- Attach the returned `booking_url` to the corresponding stop before returning.
- Never call `propose_booking` for stops the user hasn't committed to.
```

## Tests

### New: `tests/unit/test_booking_tool.py`

```python
from datetime import datetime
from unittest.mock import patch

from app.tools.booking import detect_provider, propose_booking
from app.tools.retrieval import PlaceDetails


@pytest.mark.parametrize("uri,expected", [
    ("https://resy.com/cities/sf/foo", "resy"),
    ("https://www.exploretock.com/bar", "tock"),
    ("https://www.opentable.com/baz", "opentable"),
    ("https://example.com", "unknown"),
    (None, "unknown"),
])
def test_detect_provider(uri, expected):
    assert detect_provider(uri) == expected


def _fake_details(uri, maps_uri="https://maps.google.com/?cid=1"):
    return PlaceDetails(
        place_id="p1", name="Foo", source="google_places", similarity=0.0,
        primary_type="restaurant", formatted_address="...", latitude=0, longitude=0,
        rating=4.5, price_level=3, business_status="OPERATIONAL",
        types=["restaurant"], user_rating_count=100,
        website_uri=uri, maps_uri=maps_uri, editorial_summary=None,
        regular_opening_hours={}, snippet=None,
    )


def test_resy_includes_date_and_seats():
    with patch("app.tools.booking.get_details",
               return_value=_fake_details("https://resy.com/cities/sf/foo")):
        b = propose_booking("p1", datetime(2026, 4, 26, 19, 30), party_size=4)
    assert b.provider == "resy"
    assert "date=2026-04-26" in b.booking_url
    assert "seats=4" in b.booking_url


def test_tock_includes_time_size_date():
    with patch("app.tools.booking.get_details",
               return_value=_fake_details("https://www.exploretock.com/bar")):
        b = propose_booking("p1", datetime(2026, 4, 26, 19, 30), 2)
    assert b.provider == "tock"
    assert "time=19%3A30" in b.booking_url
    assert "size=2" in b.booking_url


def test_unknown_provider_falls_back_to_maps():
    with patch("app.tools.booking.get_details",
               return_value=_fake_details("https://random.cafe",
                                          maps_uri="https://maps.google.com/?cid=42")):
        b = propose_booking("p1", datetime(2026, 4, 26, 19, 30), 2)
    assert b.provider == "unknown"
    assert "maps.google.com" in b.booking_url


def test_unknown_place_id_raises():
    with patch("app.tools.booking.get_details", return_value=None):
        with pytest.raises(ValueError):
            propose_booking("nope", datetime.now(), 2)
```

## Manual verification

```bash
curl -s http://localhost:8000/chat \
  -d '{"message": "book me dinner for 2 tonight at 7:30, italian in north beach", "history": []}' | jq .
```

Expected:
- `places[0].booking_url` is a Resy / Tock / OpenTable URL with `date`/`seats`/`size`/`time` params, OR a Google Maps fallback.
- Tapping the URL in a browser opens the provider's booking flow with fields pre-filled.

Manually verify URLs against 3 real SF places (one Resy, one Tock, one no-provider) by clicking them.

## Risks / open questions

- **URL scheme drift.** Resy/Tock/OpenTable can change their URL params. If any provider stops honoring our query strings, the link still opens — the user just enters the form manually. Acceptable degradation.
- **Provider detection by hostname is shallow.** Some venues link to a chain page (e.g. `https://www.example.com/reservations` that redirects to Resy). We won't detect those. Acceptable; users can still tap and book manually.
- **Multi-stop bookings:** if the user wants both stops booked, the agent emits two `booking_url`s. Acceptable for the stub. Real automation (later) can chain them.
- **Future automation gate:** when we add Playwright, gate it on `BOOKING_AUTOMATION_ENABLED=true` AND `BOOKING_USER_ID == settings.demo_user_id`. Never enabled for real users in this plan.
