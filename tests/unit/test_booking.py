from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from app.agent.commit import commit_stops
from app.agent.state import ItineraryState, UserConstraints
from app.tools.booking import BookingProposal, detect_provider, propose_booking
from app.tools.retrieval import PlaceDetails, PlaceHit

# ─── Smoke: models construct ────────────────────────────────────────────────


def test_booking_proposal_constructs_with_required_fields_only() -> None:
    """The proposal model accepts the minimum surface — provider, place_id,
    booking_url. Optional fields default cleanly so callers never have to pass
    `automation_available` or `notes` to build a valid object."""
    p = BookingProposal(place_id="x", provider="resy", booking_url="https://resy.com/x")
    assert p.automation_available is False
    assert p.notes is None


def test_booking_proposal_rejects_unknown_provider() -> None:
    """The Provider literal is the contract with the frontend label table.
    A typo here would surface as a 'no label rendered' bug in the UI."""
    with pytest.raises(ValueError):
        BookingProposal(place_id="x", provider="resyy", booking_url="https://x")  # type: ignore[arg-type]


# ─── Unit / logic: pure functions with mocked DB boundary ──────────────────


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("https://resy.com/cities/sf/foo", "resy"),
        ("https://www.resy.com/cities/sf/foo", "resy"),
        ("https://www.exploretock.com/bar", "tock"),
        ("https://tock.com/bar", "tock"),
        ("https://www.opentable.com/baz", "opentable"),
        ("https://example.com", "unknown"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_provider(uri: str | None, expected: str) -> None:
    assert detect_provider(uri) == expected


def _fake_details(
    website_uri: str | None,
    maps_uri: str | None = "https://maps.google.com/?cid=1",
    name: str = "Foo",
) -> PlaceDetails:
    return PlaceDetails(
        place_id="ChIJtest_p1_aaaaaaaa",
        name=name,
        source="google_places",
        similarity=0.0,
        website_uri=website_uri,
        maps_uri=maps_uri,
    )


def test_resy_url_includes_date_and_seats() -> None:
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details("https://resy.com/cities/sf/foo"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=4)
    assert b.provider == "resy"
    assert "date=2026-04-26" in b.booking_url
    assert "seats=4" in b.booking_url
    assert b.booking_url.startswith("https://resy.com/cities/sf/foo?")


def test_tock_url_includes_date_size_time() -> None:
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details("https://www.exploretock.com/bar"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.provider == "tock"
    # urlencode percent-encodes the colon in 19:30
    assert "time=19%3A30" in b.booking_url
    assert "size=2" in b.booking_url
    assert "date=2026-04-26" in b.booking_url


def test_opentable_url_includes_covers_and_datetime() -> None:
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details("https://www.opentable.com/baz"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=3)
    assert b.provider == "opentable"
    assert "covers=3" in b.booking_url
    # Tee+colon both percent-encoded by urlencode
    assert "dateTime=2026-04-26T19%3A30" in b.booking_url


def test_appends_with_ampersand_when_url_already_has_query() -> None:
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details("https://resy.com/cities/sf/foo?ref=share"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    # Existing ?ref=share preserved, our params appended with &
    assert "?ref=share&" in b.booking_url
    assert b.booking_url.count("?") == 1


def test_preserves_fragment_when_appending_query() -> None:
    """A URL fragment (#section) must come AFTER the query string, not before
    it — otherwise the params end up after the fragment and the server never
    sees them. Hand-rolled `?` concatenation got this wrong."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details("https://resy.com/cities/sf/foo#book"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.booking_url.endswith("#book")
    assert "date=2026-04-26" in b.booking_url
    assert "seats=2" in b.booking_url
    # The ? must come BEFORE the # in a valid URL.
    assert b.booking_url.index("?") < b.booking_url.index("#")


def test_overwrites_existing_param_with_same_key() -> None:
    """If the venue's website already has `?date=lunch`, our `date=2026-...`
    must replace it — most providers honor the FIRST value of a duplicate key
    and would silently ignore our deep-link params otherwise."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details("https://resy.com/cities/sf/foo?date=lunch"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert "date=2026-04-26" in b.booking_url
    assert "date=lunch" not in b.booking_url
    # Single `date=` token, not two.
    assert b.booking_url.count("date=") == 1


def test_handles_trailing_question_mark_cleanly() -> None:
    """`https://x?` with no params should produce `https://x?date=...&seats=...`,
    not `https://x?&date=...` (leading `&` after `?` is malformed)."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details("https://resy.com/cities/sf/foo?"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert "?&" not in b.booking_url
    assert "date=2026-04-26" in b.booking_url


def test_unknown_provider_with_website_returns_website_unchanged() -> None:
    """A venue whose website isn't Resy/Tock/OpenTable still has a website that
    is much more useful than a map pin — it likely hosts a reservations page.
    The website URL is returned as-is (no booking params to inject) under the
    'unknown' provider label."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details(
            website_uri="https://random.cafe",
            maps_uri="https://maps.google.com/?cid=42",
        ),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.provider == "unknown"
    assert b.booking_url == "https://random.cafe"
    # Notes must steer the user toward finding the reservations page.
    assert b.notes is not None and "website" in b.notes.lower()


def test_no_provider_no_website_falls_back_to_maps_uri() -> None:
    """Without a website, the row's maps_uri is the next best deep-link."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details(
            website_uri=None,
            maps_uri="https://maps.google.com/?cid=42",
        ),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.provider == "google_maps"
    assert b.booking_url == "https://maps.google.com/?cid=42"


def test_no_website_no_maps_uri_falls_back_to_maps_search() -> None:
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details(website_uri=None, maps_uri=None, name="Tony's"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.provider == "google_maps"
    assert "google.com/maps/search" in b.booking_url
    # urlencode escapes the apostrophe to %27.
    assert "Tony%27s" in b.booking_url


def test_maps_search_encodes_ampersand_in_venue_name() -> None:
    """Raw f-string interpolation of `&` would terminate the query string and
    break the URL. urlencode must escape it to %26."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details(
            website_uri=None, maps_uri=None, name="Tartine Manufactory & Bakery"
        ),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.provider == "google_maps"
    # The literal `&` must NOT appear inside the query token; only `&` joining
    # api=1 and query=... is allowed.
    assert "Bakery" in b.booking_url
    # Exactly one '&' as a separator between api=1 and query=... — none inside
    # the encoded venue name.
    assert b.booking_url.count("&") == 1
    assert "%26" in b.booking_url


def test_maps_search_encodes_unicode_in_venue_name() -> None:
    """Non-ASCII characters must be percent-encoded; raw interpolation produces
    URLs that some clients reject and others mojibake."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details(website_uri=None, maps_uri=None, name="Café Joséphine"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.provider == "google_maps"
    # Non-ASCII chars survive only as percent-encoded bytes.
    assert "Caf" in b.booking_url
    assert "Joséphine" not in b.booking_url
    assert "%C3%A9" in b.booking_url  # 'é' encoded


def test_maps_search_encodes_plus_in_venue_name() -> None:
    """A literal '+' in a name (e.g. 'M.Y. China + Hakkasan') means space
    after URL-decoding; encoding must escape it to %2B."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details(website_uri=None, maps_uri=None, name="Mr.+Mrs Bun"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.provider == "google_maps"
    # '+' inside the venue name escaped as %2B; raw '+' would be misread as space.
    assert "%2B" in b.booking_url


def test_empty_website_string_treated_as_no_website() -> None:
    """An empty `website_uri` (PostgreSQL NOT NULL with default '') is just as
    useless as None. Don't return an empty URL as a 'website' fallback."""
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details(
            website_uri="",
            maps_uri="https://maps.google.com/?cid=99",
        ),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.provider == "google_maps"
    assert b.booking_url == "https://maps.google.com/?cid=99"


def test_unknown_place_id_raises() -> None:
    with (
        patch("app.tools.booking.get_details", return_value=None),
        pytest.raises(ValueError, match="unknown place_id"),
    ):
        propose_booking("nope", datetime(2026, 4, 26, 19, 30), party_size=2)


def test_notes_match_provider() -> None:
    with patch(
        "app.tools.booking.get_details",
        return_value=_fake_details("https://resy.com/cities/sf/foo"),
    ):
        b = propose_booking("ChIJtest_p1_aaaaaaaa", datetime(2026, 4, 26, 19, 30), party_size=2)
    assert b.notes is not None
    assert "Resy" in b.notes


# ─── Functional: commit -> real URL construction -> Stop fields ──────
#
# These tests drive `commit_stops` end-to-end with the *real*
# propose_booking_from_details implementation. Only the SQL boundary
# (`get_details_many`, called once per commit) is mocked. This verifies the
# composition: provider detection + URL building + commit enrichment all work
# together. Compare to the unit tests above (mock get_details, hit
# propose_booking directly) and the test in test_agent_graph (mock the URL
# builder entirely) — this layer is the in-between.


def _grounded_state(place_ids: list[str], party_size: int = 2) -> ItineraryState:
    hits = [
        PlaceHit(place_id=pid, name=f"Place {pid}", source="google_places", similarity=0.9)
        for pid in place_ids
    ]
    return ItineraryState(
        scratch={
            "semantic_search": [
                {"args": {}, "result": hits, "step": 0, "id": "ChIJtest_s1_aaaaaaaa"}
            ]
        },
        constraints=UserConstraints(party_size=party_size, when=datetime(2026, 5, 7, 19, 0)),
    )


def _patch_commit_details_many(place_id: str, details: PlaceDetails):
    """Patch the commit module's batched DB call to return one details row."""
    return patch(
        "app.agent.commit.get_details_many",
        return_value={place_id: details},
    )


def test_functional_commit_attaches_resy_url_via_real_propose_booking() -> None:
    """Resy detection + URL params + Stop enrichment all wired together."""
    state = _grounded_state(["ChIJtest_p1_aaaaaaaa"], party_size=4)
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "dinner",
            "source": "google_places",
            "arrival_time": datetime(2026, 5, 7, 19, 30).isoformat(),
        }
    ]
    with _patch_commit_details_many(
        "ChIJtest_p1_aaaaaaaa", _fake_details("https://resy.com/cities/sf/foo")
    ):
        committed, _ = commit_stops(state, raw_stops)

    assert len(committed) == 1
    stop = committed[0]
    assert stop.booking_provider == "resy"
    assert stop.booking_url is not None
    assert "date=2026-05-07" in stop.booking_url
    assert "seats=4" in stop.booking_url


def test_functional_falls_back_to_venue_website_for_non_provider_site() -> None:
    """A venue whose website isn't Resy/Tock/OpenTable still gets a usable
    booking_url — the venue's own website (more useful than a map pin, since
    it likely hosts a reservations page)."""
    state = _grounded_state(["ChIJtest_p1_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "x",
            "source": "google_places",
        },
    ]
    with _patch_commit_details_many(
        "ChIJtest_p1_aaaaaaaa",
        _fake_details(
            website_uri="https://random-cafe.example",
            maps_uri="https://maps.google.com/?cid=999",
        ),
    ):
        committed, _ = commit_stops(state, raw_stops)

    stop = committed[0]
    assert stop.booking_provider == "unknown"
    assert stop.booking_url == "https://random-cafe.example"


def test_functional_falls_back_to_maps_when_no_website() -> None:
    """No website at all → Google Maps deep-link from places_raw.maps_uri."""
    state = _grounded_state(["ChIJtest_p1_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "x",
            "source": "google_places",
        },
    ]
    with _patch_commit_details_many(
        "ChIJtest_p1_aaaaaaaa",
        _fake_details(website_uri=None, maps_uri="https://maps.google.com/?cid=999"),
    ):
        committed, _ = commit_stops(state, raw_stops)

    stop = committed[0]
    assert stop.booking_provider == "google_maps"
    assert stop.booking_url == "https://maps.google.com/?cid=999"


def test_functional_uses_constraints_when_arrival_time_missing() -> None:
    """When the LLM commits a stop without arrival_time, enrichment should
    fall back to constraints.when so the URL still has a date param."""
    state = _grounded_state(["ChIJtest_p1_aaaaaaaa"])
    raw_stops = [
        {
            "place_id": "ChIJtest_p1_aaaaaaaa",
            "name": "Place p1",
            "rationale": "x",
            "source": "google_places",
        },
    ]
    with _patch_commit_details_many(
        "ChIJtest_p1_aaaaaaaa", _fake_details("https://www.opentable.com/baz")
    ):
        committed, _ = commit_stops(state, raw_stops)

    stop = committed[0]
    # constraints.when = 2026-05-07 19:00
    assert "dateTime=2026-05-07T19%3A00" in (stop.booking_url or "")
