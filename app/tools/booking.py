"""Booking handoff. Detect the reservation provider for a place and build a
deep-link with date / party size / time pre-filled where the provider's URL
scheme supports it. Falls back to Google Maps when no provider is detected.

Not automation: the user still confirms the reservation on the provider's
site. See implementation_plan/james/w4_booking_stub.md for the design and the
docs/api/chat_contract.md "Booking" section for the frontend contract.

This module is deliberately a plain library — it is NOT registered as an LLM
tool. URL construction is a deterministic transform of (place_id, when,
party_size); the agent graph auto-enriches committed stops by calling
propose_booking from app/agent/graph.py without involving the LLM.
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlencode, urlparse

from pydantic import BaseModel

from app.tools.booking_types import Provider
from app.tools.retrieval import PlaceDetails, get_details


class BookingProposal(BaseModel):
    place_id: str
    provider: Provider
    booking_url: str
    automation_available: bool = False
    notes: str | None = None


def detect_provider(website_uri: str | None) -> Provider:
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
    """Build a deep-link to the venue's booking flow.

    Raises ValueError if `place_id` is not in the DB. The caller (graph
    auto-enrichment) should swallow this and skip enrichment for that stop —
    a missing booking URL is recoverable; a 500 from a leaf utility is not.
    """
    details = get_details(place_id=place_id)
    if details is None:
        raise ValueError(f"unknown place_id {place_id}")

    detected = detect_provider(details.website_uri)
    url, effective = _build_booking_url(detected, details, when, party_size)
    return BookingProposal(
        place_id=place_id,
        provider=effective,
        booking_url=url,
        notes=_notes_for(effective),
    )


def _build_booking_url(
    detected: Provider,
    details: PlaceDetails,
    when: datetime,
    party_size: int,
) -> tuple[str, Provider]:
    """Return (url, effective_provider). The effective provider may differ from
    the detected one when we fall back. Three-tier fallback when no provider
    deep-link is available: venue website ("unknown"), then google_maps via
    the row's maps_uri, then google_maps via a name search. The frontend keys
    its label off the effective provider so the user sees "Open venue website"
    or "Open in Google Maps" rather than "Open in Unknown".
    """
    iso_date = when.strftime("%Y-%m-%d")
    iso_time = when.strftime("%H:%M")
    website = details.website_uri

    if detected == "resy" and website:
        return _append_query(website, {"date": iso_date, "seats": party_size}), "resy"

    if detected == "tock" and website:
        return (
            _append_query(
                website,
                {"date": iso_date, "size": party_size, "time": iso_time},
            ),
            "tock",
        )

    if detected == "opentable" and website:
        return (
            _append_query(
                website,
                {"covers": party_size, "dateTime": f"{iso_date}T{iso_time}"},
            ),
            "opentable",
        )

    # Three-tier fallback when no provider deep-link is available:
    #   1. The venue's own website (most likely to host a reservations page).
    #   2. Google Maps via the row's maps_uri (deep-link to the pin).
    #   3. Google Maps via a name search (last resort — no website, no pin).
    # We DON'T use maps_uri when a website exists: a map pin strips the
    # reservations page the user actually wants.
    if website:
        return website, "unknown"
    if details.maps_uri:
        return details.maps_uri, "google_maps"
    return (
        f"https://www.google.com/maps/search/?api=1&query={details.name}",
        "google_maps",
    )


def _append_query(url: str, params: dict) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode(params)}"


_NOTES: dict[Provider, str] = {
    "resy": "Tap to open Resy with your date and party size pre-filled.",
    "tock": "Tap to open Tock with your reservation pre-filled.",
    "opentable": "Tap to open OpenTable with your reservation pre-filled.",
    "unknown": (
        "Online booking not detected; opens the venue's website. "
        "You may need to find their reservations page."
    ),
    "google_maps": "No website on file; opens Google Maps.",
}


def _notes_for(provider: Provider) -> str:
    return _NOTES[provider]
