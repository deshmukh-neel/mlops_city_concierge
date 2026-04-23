import os

import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE-PLACES-API-KEY")
BASE_URL = "https://places.googleapis.com/v1/places:searchText"

FIELDS = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.types",
        "places.regularOpeningHours",
        "places.websiteUri",
        "places.editorialSummary",
        "places.photos",
        "places.location",
    ]
)


def search_places(query: str, max_results: int = 20) -> list[dict]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_KEY,
        "X-Goog-FieldMask": FIELDS,
    }
    body = {
        "textQuery": query,
        "maxResultCount": max_results,
        "locationBias": {
            "circle": {
                "center": {"latitude": 37.7749, "longitude": -122.4194},
                "radius": 10000.0,  # 10km radius around SF
            }
        },
    }
    response = requests.post(BASE_URL, json=body, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json().get("places", [])


if __name__ == "__main__":
    venues = search_places("wine bars in Hayes Valley San Francisco")
    for v in venues[:3]:
        print(v["displayName"]["text"], "—", v.get("rating"), "★")
