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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

from pydantic import BaseModel

from app.tools.booking_types import Provider
from app.tools.retrieval import PlaceDetails, get_details


class BookingProposal(BaseModel):
    place_id: str
    provider: Provider
    booking_url: str
    automation_available: bool = False
    notes: str | None = None


@dataclass(frozen=True)
class _ProviderSpec:
    """How a single deep-linkable provider is recognized and parameterized.

    `host_match` substrings are checked against the URL hostname (lowercased)
    in `detect_provider`. `params` builds the query dict the venue's booking
    page expects; signature is (iso_date, iso_time, party_size) -> dict.
    """

    host_match: tuple[str, ...]
    params: Callable[[str, str, int], dict[str, object]]
    notes: str


_PROVIDER_SPECS: dict[Provider, _ProviderSpec] = {
    "resy": _ProviderSpec(
        host_match=("resy.com",),
        params=lambda d, _t, n: {"date": d, "seats": n},
        notes="Tap to open Resy with your date and party size pre-filled.",
    ),
    "tock": _ProviderSpec(
        host_match=("exploretock.com", "tock.com"),
        params=lambda d, t, n: {"date": d, "size": n, "time": t},
        notes="Tap to open Tock with your reservation pre-filled.",
    ),
    "opentable": _ProviderSpec(
        host_match=("opentable.com",),
        params=lambda d, t, n: {"covers": n, "dateTime": f"{d}T{t}"},
        notes="Tap to open OpenTable with your reservation pre-filled.",
    ),
}

_FALLBACK_NOTES: dict[Provider, str] = {
    "unknown": (
        "Online booking not detected; opens the venue's website. "
        "You may need to find their reservations page."
    ),
    "google_maps": "No website on file; opens Google Maps.",
}


def detect_provider(website_uri: str | None) -> Provider:
    if not website_uri:
        return "unknown"
    host = (urlparse(website_uri).hostname or "").lower()
    for name, spec in _PROVIDER_SPECS.items():
        if any(needle in host for needle in spec.host_match):
            return name
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

    We DON'T use maps_uri when a website exists: a map pin strips the
    reservations page the user actually wants.
    """
    website = details.website_uri
    spec = _PROVIDER_SPECS.get(detected)
    if spec and website:
        params = spec.params(when.strftime("%Y-%m-%d"), when.strftime("%H:%M"), party_size)
        return _append_query(website, params), detected
    if website:
        return website, "unknown"
    if details.maps_uri:
        return details.maps_uri, "google_maps"
    return (
        f"https://www.google.com/maps/search/?api=1&query={details.name}",
        "google_maps",
    )


def _append_query(url: str, params: dict) -> str:
    """Merge `params` into the URL's query string, replacing existing keys.

    Uses urlsplit/urlunsplit so fragments are preserved, pre-encoded params
    are decoded then re-encoded canonically, and our params overwrite any
    pre-existing same-key params (so e.g. our `date=2026-04-26` wins over a
    `?date=lunch` in the input). Hand-rolled string concat got this wrong on
    fragments (`#section?...` is server-invisible) and same-key collisions.
    """
    parts = urlsplit(url)
    merged = dict(parse_qsl(parts.query, keep_blank_values=True))
    merged.update({k: str(v) for k, v in params.items()})
    return urlunsplit(parts._replace(query=urlencode(merged)))


def _notes_for(provider: Provider) -> str:
    spec = _PROVIDER_SPECS.get(provider)
    return spec.notes if spec is not None else _FALLBACK_NOTES[provider]
