# W0a — Cleaner embeddings on a parallel `place_embeddings_v2` table

**Branch:** `feature/agent-w0a-embeddings-v2`
**Depends on:** nothing (W0a slots before W1)
**Unblocks:** W1 (filter columns rely on v2 fields), W6 (eval the v2 retriever), W7 (KG seeds use v2's `neighborhood` extraction)

## Why this exists

We inspected real Cloud SQL data on 2026-05-04. The current `place_embeddings.embedding_text` mixes three roles into one 1536-d vector:

1. **Semantic signal** — name, primary_type, types, editorial / generative / review summaries, service / dining / food-drink booleans. This is what retrieval *should* match on.
2. **Filterable structured facts** — rating, user_rating_count, price_level, price_range, opening hours text, lat/lng, business_status, phone numbers, formatted_address. These are facts the agent should *filter* on, not match on. Embedding them dilutes the vector.
3. **Action payload** — website URI, `googleMapsLinks` (5 URL variants per place — `placeUri`, `photosUri`, `reviewsUri`, `directionsUri`, `writeAReviewUri`), `flagContentUri`, `reviewsUri` inside summaries, language codes, "Summarized with Gemini" disclosure boilerplate. None of this is meaningful to the embedding model. Worse, it's nearly identical across all places (e.g. the recurring `g_mp=Cidnb29nbGUubWFwcy5wbGFjZXMudjEuUGxhY2VzLlNlYXJjaFRleHQQAhgEIAA` query string), which actively pulls unrelated places closer in vector space.

We also discovered:

- **No raw reviews ingested.** `places.reviews` is missing from the field mask in `scripts/ingest_places_sf.py:70-143`. `reviewSummary` (the AI-generated overview) is present for 3,162 of 5,855 places; raw individual reviews are absent for all 5,855.
- **`addressComponents` neighborhood and `addressDescriptor.areas[]` are unused.** Each place's `source_json` already carries a structured `neighborhood` (e.g. "Mission Bay") and a list of named containing areas. Today we substring-match `formatted_address` instead.
- **`addressDescriptor.landmarks[]` is unused.** Each place includes nearby landmarks with travel-distance metadata — Google literally hands us a graph for free. W7 (knowledge graph) consumes this.

This PR fixes the embedding pipeline **on a separate `place_embeddings_v2` table** so the existing `place_embeddings` is untouched. We compare retrieval quality between v1 and v2 in W6 evals, then flip the app to v2 by env flag once the comparison shows a win.

## What this PR delivers

After merge:

- A new `place_embeddings_v2` table with the same shape as `place_embeddings`.
- A new `scripts/embed_places_pgvector_v2.py` that writes only to v2.
- A rewritten `compose_embedding_text_v2()` that:
  - drops Role 2 / Role 3 fields entirely from the embedded string;
  - extracts only the meaningful `text` field from summary objects (drops disclosure / language / flag-content URLs);
  - adds `neighborhood` from `addressComponents`;
  - adds containing areas' display names from `addressDescriptor.areas[]`;
  - adds nearby landmark names (no distances or place IDs — those go to W7);
  - optionally includes individual reviews if/when `places.reviews` is added to the ingest field mask (deferred — see Risks below).
- An `EMBEDDING_TABLE` env var the app reads at startup. Defaults to `place_embeddings` (v1) so this PR is non-breaking. Switching to `place_embeddings_v2` is a one-line env change.
- A diagnostic script that prints v1 vs v2 chunks side-by-side for the same `place_id`, used in the manual verification step.

## Files

### New: `scripts/db/migrations/000_place_embeddings_v2.sql`

```sql
-- Parallel embeddings table. Same schema as place_embeddings; lives next to it
-- so we can A/B-compare retrieval quality before flipping the app over.
-- Once W0a + W1 + W6 confirm v2 wins, v1 may be dropped in a follow-up PR.

CREATE TABLE IF NOT EXISTS place_embeddings_v2 (
    place_id              TEXT PRIMARY KEY REFERENCES places_raw(place_id) ON DELETE CASCADE,
    embedding             vector(1536) NOT NULL,
    embedding_model       TEXT NOT NULL,
    embedding_text        TEXT NOT NULL,
    embedded_at           TIMESTAMPTZ DEFAULT NOW(),
    source_updated_at     TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_place_embeddings_v2_vector
    ON place_embeddings_v2
    USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE place_embeddings_v2 IS
  'Cleaned embeddings (no URLs, no structured facts, with neighborhood + landmark names). Drives retrieval when EMBEDDING_TABLE=place_embeddings_v2.';
```

### New: `scripts/embed_places_pgvector_v2.py`

This file is a near-clone of `scripts/embed_places_pgvector.py` with three changes: target table is `place_embeddings_v2`, `compose_embedding_text` is replaced by `compose_embedding_text_v2`, and a few helpers are stricter about pulling clean text out of nested JSON.

```python
"""
Generate cleaned embeddings for places_raw rows and upsert into place_embeddings_v2.

This is a fork of scripts/embed_places_pgvector.py with a rewritten
compose_embedding_text. The two scripts intentionally coexist so we can
re-run either one and compare retrieval quality. See implementation_plan/
james/w0a_embeddings_v2.md for the rationale.

Usage:
    python scripts/embed_places_pgvector_v2.py
"""

# Imports + constants identical to v1 except:
TARGET_TABLE = "place_embeddings_v2"


# ---- text composition ------------------------------------------------------

def _summary_object_text(value: object) -> str:
    """Extract the human-readable text from Places v1 summary objects.

    Shape: {"text": {"text": "...", "languageCode": "en-US"},
            "disclosureText": {...}, "flagContentUri": "...", "reviewsUri": "..."}
    We want ONLY the inner text; everything else is noise (URLs, language tags,
    "Summarized with Gemini" boilerplate).
    """
    if not isinstance(value, dict):
        return ""
    inner = value.get("text") or value.get("overview")
    if isinstance(inner, dict):
        text = inner.get("text")
        if isinstance(text, str):
            return text
    if isinstance(inner, str):
        return inner
    return ""


def _neighborhood_from_address_components(source_json: dict) -> str:
    """Extract the structured neighborhood from addressComponents.

    Shape: addressComponents: [{"types": ["neighborhood", "political"],
                                "longText": "Mission Bay", ...}, ...]
    """
    components = source_json.get("addressComponents") or []
    if not isinstance(components, list):
        return ""
    for component in components:
        if not isinstance(component, dict):
            continue
        types = component.get("types") or []
        if "neighborhood" in types:
            text = component.get("longText") or component.get("shortText")
            if isinstance(text, str):
                return text
    return ""


def _containing_area_names(source_json: dict) -> str:
    """Names of containing areas from addressDescriptor.areas[].displayName.

    Shape: addressDescriptor.areas: [{"displayName": {"text": "Mission Bay", ...},
                                       "containment": "WITHIN"}, ...]
    """
    descriptor = source_json.get("addressDescriptor") or {}
    if not isinstance(descriptor, dict):
        return ""
    areas = descriptor.get("areas") or []
    if not isinstance(areas, list):
        return ""
    names: list[str] = []
    for area in areas:
        if not isinstance(area, dict):
            continue
        display_name = area.get("displayName")
        text = _localized_text(display_name)
        if text and text not in names:
            names.append(text)
    return ", ".join(names)


def _nearby_landmark_names(source_json: dict, max_landmarks: int = 5) -> str:
    """Names of nearby landmarks (no distances, no place IDs).

    Distance + place IDs are consumed by W7 (knowledge graph) directly from
    source_json — they are KG inputs, not embedding inputs.
    """
    descriptor = source_json.get("addressDescriptor") or {}
    if not isinstance(descriptor, dict):
        return ""
    landmarks = descriptor.get("landmarks") or []
    if not isinstance(landmarks, list):
        return ""
    names: list[str] = []
    for landmark in landmarks[:max_landmarks]:
        if not isinstance(landmark, dict):
            continue
        text = _localized_text(landmark.get("displayName"))
        if text:
            names.append(text)
    return ", ".join(names)


def compose_embedding_text_v2(record: dict) -> str:
    """Cleaned embedding text. See w0a_embeddings_v2.md for the role taxonomy.

    KEEP (Role 1 — semantic signal):
      name, primary_type, types, editorial_summary, generative summary text,
      review summary text, service / dining / food-drink boolean labels,
      neighborhood, containing areas, nearby landmark names, accessibility /
      parking / payment labels.

    DROP (Role 2 / Role 3 — facts and action payloads, do NOT embed):
      rating, user_rating_count, price_level, price_range, opening hours text,
      lat/lng, business_status, phone numbers, all URLs (websiteUri,
      googleMapsUri, googleMapsLinks, flagContentUri, reviewsUri inside
      summaries), language codes, "Summarized with Gemini" disclosure text.
    """
    source_json = record.get("source_json") or {}
    if not isinstance(source_json, dict):
        source_json = {}

    parts: list[str] = []

    # --- core identification ------------------------------------------------
    parts.append(f"Name: {record.get('name') or ''}")
    parts.append(f"Primary Type: {record.get('primary_type') or ''}")
    types = record.get("types") or []
    if types:
        parts.append(f"Types: {', '.join(types)}")

    # --- geographic context (names only, never numbers) ---------------------
    neighborhood = _neighborhood_from_address_components(source_json)
    if neighborhood:
        parts.append(f"Neighborhood: {neighborhood}")
    containing = _containing_area_names(source_json)
    if containing:
        parts.append(f"Containing Areas: {containing}")
    landmarks = _nearby_landmark_names(source_json)
    if landmarks:
        parts.append(f"Nearby Landmarks: {landmarks}")

    # --- editorial / generative / review prose ------------------------------
    editorial = record.get("editorial_summary") or _summary_object_text(
        source_json.get("editorialSummary")
    )
    if editorial:
        parts.append(f"Editorial Summary: {editorial}")
    generative = _summary_object_text(source_json.get("generativeSummary"))
    if generative:
        parts.append(f"Generative Summary: {generative}")
    review_summary = _summary_object_text(source_json.get("reviewSummary"))
    if review_summary:
        parts.append(f"Review Summary: {review_summary}")

    # --- amenities (boolean labels only — no values, no JSON) ---------------
    service = _enabled_features(source_json, SERVICE_FEATURES)
    if service:
        parts.append(f"Service Options: {service}")
    dining = _enabled_features(source_json, DINING_FEATURES)
    if dining:
        parts.append(f"Dining Features: {dining}")
    food_drink = _enabled_features(source_json, FOOD_DRINK_FEATURES)
    if food_drink:
        parts.append(f"Food and Drink: {food_drink}")
    for label, key in JSON_FLAG_GROUPS.items():
        flags = _json_flags(source_json, key)
        if flags:
            parts.append(f"{label}: {flags}")

    # --- raw reviews (only if ingest started capturing places.reviews) ------
    reviews = source_json.get("reviews") or []
    if isinstance(reviews, list) and reviews:
        review_texts: list[str] = []
        for review in reviews[:5]:
            if not isinstance(review, dict):
                continue
            text = _localized_text(review.get("text"))
            if text:
                review_texts.append(text)
        if review_texts:
            parts.append("Reviews: " + " | ".join(review_texts))

    return "\n".join(parts)
```

The rest of the script — `fetch_rows_to_embed`, `upsert_embedding`, `iter_embedding_batches`, `run` — is identical to v1 except every reference to `place_embeddings` becomes `place_embeddings_v2`.

### Modify: `app/config.py`

Add the env var:

```python
# in Settings:
embedding_table: str = Field(default="place_embeddings", env="EMBEDDING_TABLE")
```

Validate it's one of `{"place_embeddings", "place_embeddings_v2"}` so a typo doesn't silently fail.

### Modify: `app/retriever.py`

Two changes:

1. Read the table name from settings instead of hardcoding `place_embeddings` in the SQL string.
2. Keep everything else identical so `/predict` continues to work against whichever table is selected.

```python
# in PgVectorRetriever._get_relevant_documents:
sql = f"""
SELECT
    e.embedding_text,
    p.place_id, p.name, p.rating, p.formatted_address, p.primary_type,
    1 - (e.embedding <=> %s::vector) AS similarity
FROM {settings.embedding_table} e
JOIN places_raw p ON p.place_id = e.place_id
WHERE e.embedding_model = %s
ORDER BY e.embedding <=> %s::vector
LIMIT %s
"""
```

The table name is validated at config load, so the f-string is not a SQL-injection vector.

### Modify: `Makefile`

Add a target so `make embed-v2` is the canonical way to refresh v2:

```makefile
embed-v2:
	python scripts/embed_places_pgvector_v2.py
```

### New: `scripts/diagnose_chunks.py`

A read-only diagnostic that prints v1 vs v2 chunks side-by-side for inspection.

```python
"""Print v1 vs v2 chunks for the same place_ids. Used to eyeball quality
before/after the cleanup. Read-only; never writes."""

import argparse
import psycopg2
from app.config import resolve_database_url

QUERY = """
SELECT p.name, v1.embedding_text AS v1_text, v2.embedding_text AS v2_text,
       LENGTH(v1.embedding_text) AS v1_len, LENGTH(v2.embedding_text) AS v2_len
FROM places_raw p
LEFT JOIN place_embeddings    v1 ON v1.place_id = p.place_id
LEFT JOIN place_embeddings_v2 v2 ON v2.place_id = p.place_id
WHERE v1.place_id IS NOT NULL AND v2.place_id IS NOT NULL
  AND p.user_rating_count > 100
ORDER BY random()
LIMIT %s
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=5)
    args = p.parse_args()
    with psycopg2.connect(resolve_database_url()) as conn, conn.cursor() as cur:
        cur.execute(QUERY, (args.n,))
        for name, v1, v2, v1_len, v2_len in cur.fetchall():
            print("=" * 80)
            print(f"NAME: {name}    v1={v1_len} chars   v2={v2_len} chars")
            print("--- V1 ---\n" + v1)
            print("--- V2 ---\n" + v2)

if __name__ == "__main__":
    main()
```

## Tests

### New: `tests/unit/test_embed_v2_compose.py`

Pure-function tests on `compose_embedding_text_v2`. No DB, no OpenAI:

```python
def test_drops_googlemapslinks():
    record = {
        "name": "X", "primary_type": "Coffee Shop", "types": ["coffee_shop"],
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


def test_drops_facts_and_numbers():
    record = {
        "name": "X", "primary_type": "Restaurant", "types": ["restaurant"],
        "rating": 4.8, "user_rating_count": 254,
        "price_level": "PRICE_LEVEL_INEXPENSIVE",
        "latitude": 37.77, "longitude": -122.41,
        "regular_opening_hours": {"weekdayDescriptions": ["Monday: 8AM-4PM"]},
        "business_status": "OPERATIONAL",
        "source_json": {"nationalPhoneNumber": "(415) 555-0100"},
    }
    text = compose_embedding_text_v2(record)
    assert "Rating:"        not in text
    assert "User Ratings:"  not in text
    assert "Price Level:"   not in text
    assert "Latitude:"      not in text
    assert "Longitude:"     not in text
    assert "Opening Hours:" not in text
    assert "Business Status:" not in text
    assert "Phone:"         not in text


def test_summary_extracts_text_only():
    record = {
        "name": "X", "primary_type": "Cafe", "types": [],
        "source_json": {
            "generativeSummary": {
                "overview": {"text": "Coffee shop brewing organic beans.",
                             "languageCode": "en-US"},
                "disclosureText": {"text": "Summarized with Gemini",
                                   "languageCode": "en-US"},
                "overviewFlagContentUri": "https://www.google.com/local/...",
            },
        },
    }
    text = compose_embedding_text_v2(record)
    assert "Coffee shop brewing organic beans." in text
    assert "Summarized with Gemini" not in text
    assert "google.com"             not in text
    assert "languageCode"           not in text


def test_extracts_neighborhood():
    record = {
        "name": "X", "primary_type": "Restaurant", "types": [],
        "source_json": {
            "addressComponents": [
                {"types": ["street_number"], "longText": "499"},
                {"types": ["neighborhood", "political"], "longText": "Mission Bay"},
                {"types": ["locality", "political"],     "longText": "San Francisco"},
            ],
        },
    }
    assert "Neighborhood: Mission Bay" in compose_embedding_text_v2(record)


def test_extracts_landmarks_without_distances():
    record = {
        "name": "X", "primary_type": "Restaurant", "types": [],
        "source_json": {
            "addressDescriptor": {
                "landmarks": [
                    {"displayName": {"text": "Chase Center"},
                     "travelDistanceMeters": 322.04},
                    {"displayName": {"text": "Crane Cove Park"},
                     "travelDistanceMeters": 453.77},
                ],
            },
        },
    }
    text = compose_embedding_text_v2(record)
    assert "Chase Center"    in text
    assert "Crane Cove Park" in text
    assert "322"             not in text
    assert "travelDistance"  not in text


def test_no_empty_lines_when_fields_missing():
    record = {"name": "X", "primary_type": "", "types": [], "source_json": {}}
    text = compose_embedding_text_v2(record)
    assert "Name: X" in text
    for line in text.splitlines():
        assert not line.endswith(": ")
```

### Integration (gated on `APP_ENV=integration`)

`tests/integration/test_embed_v2_e2e.py` — run the script against a small fixture, confirm `place_embeddings_v2` populates, confirm chunk lengths are meaningfully smaller than v1 (median <70% of v1).

## Manual verification

```bash
# Apply the new table (idempotent):
psql "$DATABASE_URL" -f scripts/db/migrations/000_place_embeddings_v2.sql

# Populate v2:
make embed-v2

# Eyeball v1 vs v2 for 5 random well-known places:
poetry run python -m scripts.diagnose_chunks -n 5

# Confirm length distribution:
psql "$DATABASE_URL" -c "
  SELECT 'v1' AS table, MAX(LENGTH(embedding_text)), AVG(LENGTH(embedding_text)),
         PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LENGTH(embedding_text))
  FROM place_embeddings
  UNION ALL
  SELECT 'v2', MAX(LENGTH(embedding_text)), AVG(LENGTH(embedding_text)),
         PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LENGTH(embedding_text))
  FROM place_embeddings_v2;
"

# Test retrieval against v2 without redeploying:
EMBEDDING_TABLE=place_embeddings_v2 make dev
curl -s http://localhost:8000/predict \
  -d '{"query": "cozy coffee shop with books", "limit": 5}' | jq .

# Compare to v1:
EMBEDDING_TABLE=place_embeddings make dev
curl -s http://localhost:8000/predict \
  -d '{"query": "cozy coffee shop with books", "limit": 5}' | jq .
```

Expected: v2 chunks roughly 600–900 chars (vs v1 avg ~2,468). v2 retrieval surfaces semantically tighter matches for vibe queries because the URL noise no longer pulls unrelated places together.

## Tracking the change in MLflow

`embedding_table` becomes a logged param on every run created by `scripts/log_model_to_mlflow.py`. See W6 for the full param expansion. Without this, A/B comparisons between v1 and v2 are not apples-to-apples.

## Risks / open questions

- **Re-embed cost.** ~5,855 places × ~600 chars. With `text-embedding-3-small` at $0.02/1M tokens, the full re-embed is well under $1. Trivial.
- **Raw reviews are still missing.** This PR does not add `places.reviews` to the ingest field mask. Doing so would require a re-ingest run (Google Places API cost) plus a re-embed. The `compose_embedding_text_v2` function already handles a `reviews` array if present, so once a future PR adds the field to `ALL_PLACE_FIELDS` in `scripts/ingest_places_sf.py:70-143` and re-runs ingest, `make embed-v2` will pick it up automatically via the existing `source_updated_at` change-detection path.
- **Token-length ceiling.** v1 max is ~3,893 chars (well under 8,191 tokens). v2 chunks are smaller. We will not hit truncation.
- **Single vector per place is unchanged.** This PR does not introduce per-review or per-section chunking. If review texts ever land and any chunk approaches the 8k-token cap, we revisit.
- **HNSW index recall under filter pressure.** Same caveat as v1; addressed in W1's risks section. Not a v2-specific issue.
- **v1 and v2 will drift if ingest changes shape.** Both scripts read from the same `places_raw` so they stay in sync as long as both are re-run after an ingest change. Document this in the PR description.
- **Promotion semantics.** Flipping `EMBEDDING_TABLE=place_embeddings_v2` in production is gated on a W6 eval that shows non-regression on retrieval-quality metrics. No alias mechanism on the embeddings table itself; the env var IS the alias.

## Outcomes (2026-05-04)

Populated `place_embeddings_v2` for the full SF corpus and measured v1 vs v2 chunk lengths on Cloud SQL:

|        | rows  | max len | avg len |
|--------|-------|---------|---------|
| v1     | 5,855 | 3,893   | 2,468   |
| v2     | 5,855 | 1,622   | **873** |

v2 average lands squarely inside the 600–900 char prediction. Spot-checked v1 vs v2 chunks for 3 well-known places (Cull Canyon Recreation Area, The 500 Club, Jasmin's) — every URL, language code, and "Summarized with Gemini" disclosure is gone from v2, and `Neighborhood` / `Containing Areas` / `Nearby Landmarks` are present where the data exists.

### Bugs found while running and fixed in the same PR

While running `make embed-v2` end-to-end against Cloud SQL we hit three pre-existing issues in the embed pipeline. They affected v1 too — fixed once for both:

1. **`ModuleNotFoundError: No module named 'app'`** when running scripts directly. Fixed by adding `scripts/__init__.py` and switching the Make targets to `poetry run python -m scripts.<name>`. Also corrects the manual-verification command above.
2. **Single-batch run** — `run()` only embedded the first `BATCH_SIZE = 1000` places and exited. Replaced with a `while True` loop that re-fetches until empty; the fetch query already excludes up-to-date rows, so the loop terminates naturally.
3. **Per-row commit bottleneck** — capped throughput at ~50 rows/min through the Cloud SQL proxy. Replaced the per-row `upsert_embedding` loop with `psycopg2.extras.execute_values` batching (one round-trip per ~1,000-row OpenAI batch). A full corpus re-embed dropped from ~75 minutes to ~3 minutes. Applied to v2 only; v1 left unchanged since it's being deprecated.

---

**Status:** Merged in [PR #58](https://github.com/deshmukh-neel/mlops_city_concierge/pull/58). `EMBEDDING_TABLE=place_embeddings_v2` promotion is gated on W6 retrieval evals and remains intentionally deferred.
