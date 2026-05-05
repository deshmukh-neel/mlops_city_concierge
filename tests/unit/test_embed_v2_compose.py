"""Unit tests for the v2 embedding text composer.

Verifies the role taxonomy from implementation_plan/james/w0a_embeddings_v2.md:
the v2 string keeps semantic signal (Role 1) and drops structured facts and
URL payloads (Roles 2 & 3).
"""

from __future__ import annotations

from scripts.embed_places_pgvector_v2 import compose_embedding_text_v2


def test_drops_googlemapslinks() -> None:
    record = {
        "name": "X",
        "primary_type": "Coffee Shop",
        "types": ["coffee_shop"],
        "source_json": {
            "googleMapsLinks": {
                "placeUri": "https://maps.google.com/?cid=1&g_mp=...",
                "photosUri": "https://www.google.com/maps/...",
            },
        },
    }
    text = compose_embedding_text_v2(record)
    assert "google.com" not in text
    assert "g_mp=" not in text
    assert "googleMapsLinks" not in text


def test_drops_facts_and_numbers() -> None:
    record = {
        "name": "X",
        "primary_type": "Restaurant",
        "types": ["restaurant"],
        "rating": 4.8,
        "user_rating_count": 254,
        "price_level": "PRICE_LEVEL_INEXPENSIVE",
        "latitude": 37.77,
        "longitude": -122.41,
        "regular_opening_hours": {"weekdayDescriptions": ["Monday: 8AM-4PM"]},
        "business_status": "OPERATIONAL",
        "source_json": {"nationalPhoneNumber": "(415) 555-0100"},
    }
    text = compose_embedding_text_v2(record)
    assert "Rating:" not in text
    assert "User Ratings:" not in text
    assert "Price Level:" not in text
    assert "Latitude:" not in text
    assert "Longitude:" not in text
    assert "Opening Hours:" not in text
    assert "Business Status:" not in text
    assert "Phone:" not in text


def test_summary_extracts_text_only() -> None:
    record = {
        "name": "X",
        "primary_type": "Cafe",
        "types": [],
        "source_json": {
            "generativeSummary": {
                "overview": {
                    "text": "Coffee shop brewing organic beans.",
                    "languageCode": "en-US",
                },
                "disclosureText": {
                    "text": "Summarized with Gemini",
                    "languageCode": "en-US",
                },
                "overviewFlagContentUri": "https://www.google.com/local/...",
            },
        },
    }
    text = compose_embedding_text_v2(record)
    assert "Coffee shop brewing organic beans." in text
    assert "Summarized with Gemini" not in text
    assert "google.com" not in text
    assert "languageCode" not in text


def test_extracts_neighborhood() -> None:
    record = {
        "name": "X",
        "primary_type": "Restaurant",
        "types": [],
        "source_json": {
            "addressComponents": [
                {"types": ["street_number"], "longText": "499"},
                {"types": ["neighborhood", "political"], "longText": "Mission Bay"},
                {"types": ["locality", "political"], "longText": "San Francisco"},
            ],
        },
    }
    assert "Neighborhood: Mission Bay" in compose_embedding_text_v2(record)


def test_extracts_containing_areas() -> None:
    record = {
        "name": "X",
        "primary_type": "Restaurant",
        "types": [],
        "source_json": {
            "addressDescriptor": {
                "areas": [
                    {"displayName": {"text": "Mission Bay"}, "containment": "WITHIN"},
                    {"displayName": {"text": "South of Market"}, "containment": "NEAR"},
                ],
            },
        },
    }
    text = compose_embedding_text_v2(record)
    assert "Containing Areas: Mission Bay, South of Market" in text


def test_extracts_landmarks_without_distances() -> None:
    record = {
        "name": "X",
        "primary_type": "Restaurant",
        "types": [],
        "source_json": {
            "addressDescriptor": {
                "landmarks": [
                    {
                        "displayName": {"text": "Chase Center"},
                        "travelDistanceMeters": 322.04,
                    },
                    {
                        "displayName": {"text": "Crane Cove Park"},
                        "travelDistanceMeters": 453.77,
                    },
                ],
            },
        },
    }
    text = compose_embedding_text_v2(record)
    assert "Chase Center" in text
    assert "Crane Cove Park" in text
    assert "322" not in text
    assert "travelDistance" not in text


def test_caps_landmarks_at_five() -> None:
    record = {
        "name": "X",
        "primary_type": "Restaurant",
        "types": [],
        "source_json": {
            "addressDescriptor": {
                "landmarks": [{"displayName": {"text": f"Landmark {i}"}} for i in range(10)],
            },
        },
    }
    text = compose_embedding_text_v2(record)
    assert "Landmark 0" in text
    assert "Landmark 4" in text
    assert "Landmark 5" not in text


def test_includes_amenity_labels() -> None:
    record = {
        "name": "X",
        "primary_type": "Restaurant",
        "types": [],
        "source_json": {
            "outdoorSeating": True,
            "servesCoffee": True,
            "delivery": False,
            "accessibilityOptions": {"wheelchairAccessibleEntrance": True},
        },
    }
    text = compose_embedding_text_v2(record)
    assert "outdoor seating" in text
    assert "coffee" in text
    assert "delivery" not in text
    assert "wheelchair accessible entrance" in text


def test_includes_reviews_when_present() -> None:
    record = {
        "name": "X",
        "primary_type": "Cafe",
        "types": [],
        "source_json": {
            "reviews": [
                {"text": {"text": "Great espresso and cozy seating."}},
                {"text": {"text": "Friendly staff, slow on weekends."}},
            ],
        },
    }
    text = compose_embedding_text_v2(record)
    assert "Reviews:" in text
    assert "Great espresso and cozy seating." in text
    assert "Friendly staff, slow on weekends." in text


def test_no_empty_lines_when_fields_missing() -> None:
    record = {"name": "X", "primary_type": "", "types": [], "source_json": {}}
    text = compose_embedding_text_v2(record)
    assert "Name: X" in text
    for line in text.splitlines():
        assert not line.endswith(": ")
