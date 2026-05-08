"""Integration test for booking handoff against a real Postgres + pgvector.

Exercises propose_booking end-to-end: real get_details query against the
places_raw table, real provider detection from the row's website_uri, real
URL construction. Skipped unless APP_ENV=integration is set.

Run with:
    make db-up
    APP_ENV=integration poetry run pytest tests/integration/test_booking.py
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)

from app.tools.booking import propose_booking  # noqa: E402
from app.tools.retrieval import get_details, semantic_search  # noqa: E402


def _any_real_place_id() -> str:
    """Pick any place_id from the live DB so we don't hard-code a fixture
    that drifts. semantic_search is the same path the agent uses."""
    hits = semantic_search(query="restaurant", k=1)
    if not hits:
        pytest.skip("places_raw is empty; seed the DB before running integration tests.")
    return hits[0].place_id


def test_propose_booking_against_real_database() -> None:
    """Real DB lookup + real URL construction. The exact provider depends on
    what's seeded; whichever tier we land on, the assertions for THAT tier's
    contract must hold (params present, URL pointing at the right thing)."""
    place_id = _any_real_place_id()
    when = datetime(2026, 5, 7, 19, 30)
    proposal = propose_booking(place_id, when, party_size=2)

    assert proposal.place_id == place_id
    assert proposal.booking_url
    assert proposal.booking_url.startswith("http")
    assert proposal.provider in {"resy", "tock", "opentable", "google_maps", "unknown"}
    assert proposal.notes is not None

    # Provider-tier-specific contract. Loose set-membership above passes for any
    # URL; these per-tier checks would fail if the URL builder regressed (wrong
    # row, missing params, fallback inverted).
    iso_date = "2026-05-07"
    iso_time = "19:30"
    if proposal.provider == "resy":
        assert f"date={iso_date}" in proposal.booking_url
        assert "seats=2" in proposal.booking_url
    elif proposal.provider == "tock":
        assert f"date={iso_date}" in proposal.booking_url
        assert "size=2" in proposal.booking_url
        # urlencode percent-encodes the colon in HH:MM.
        assert "time=19%3A30" in proposal.booking_url
    elif proposal.provider == "opentable":
        assert "covers=2" in proposal.booking_url
        assert f"dateTime={iso_date}T{iso_time.replace(':', '%3A')}" in proposal.booking_url
    elif proposal.provider == "unknown":
        # Three-tier fallback: when no provider deep-link is detected but the
        # row has a website, we return that website unchanged.
        details = get_details(place_id=place_id)
        assert details is not None
        assert proposal.booking_url == details.website_uri
    elif proposal.provider == "google_maps":
        # Either the row's maps_uri or the name-search fallback. Both go to
        # google.com/maps and carry no booking params (none exist for Maps).
        assert "google.com/maps" in proposal.booking_url


def test_propose_booking_unknown_place_id_raises_against_real_db() -> None:
    """A place_id that isn't in the DB raises ValueError, not a silent None."""
    with pytest.raises(ValueError, match="unknown place_id"):
        propose_booking(
            "definitely-not-a-real-place-id-zzz", datetime(2026, 5, 7, 19, 30), party_size=2
        )
