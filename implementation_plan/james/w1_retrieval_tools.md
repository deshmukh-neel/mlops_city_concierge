# W1 — Unified place view + filterable retrieval tools

**Branch:** `feature/agent-w1-retrieval-tools`

**Depends on:**

- W0a (the `place_embeddings_v2` table + the `EMBEDDING_TABLE` env var that selects which embedding table the view reads from).
- PR #63 / #64 (Alembic now lives in `alembic/versions/`; `make migration MSG="..."` creates new migrations and `make migrate` applies them. Both prod and the local dev DB have been stamped at `5187c6b09b25`).

**Unblocks:** W2, W3, W6, W7 (and indirectly W4)

## Goal

Replace the single-pass, metadata-blind retriever with a set of **tools** the agent can call with structured filters. Hide the underlying tables (`places_raw` + the active embedding table — `place_embeddings` or `place_embeddings_v2` per W0a — plus the future editorial table) behind a SQL view so the retrieval tools never touch source tables directly.

After this PR:
- New `place_documents` view exposes every column the agent might filter on, with a `source` column.
- The view exposes filterable columns the embedding (after W0a's cleanup) no longer carries: `neighborhood`, `servesCocktails`, `outdoorSeating`, `reservable`, `allowsDogs`, `liveMusic`, `goodForGroups`, `goodForChildren`, `parkingOptions`. These are the high-signal Role-2 facts we deliberately removed from the embedding text — they belong in WHERE clauses, not in the vector.
- Three tool functions (`semantic_search`, `nearby`, `get_details`) return typed Pydantic objects.
- A `SearchFilters` model compiles to safe parameterized SQL fragments and now carries quality-floor defaults (`min_user_rating_count = 50`, `business_status = 'OPERATIONAL'`) so the Pasadena Velasco failure mode (5.0 rating, 1 rater) cannot reach the agent without an explicit override.
- `place_is_open` correctly handles overnight periods (close < open the next day) — needed for bars and late-night spots.
- The legacy `PgVectorRetriever` still works for `/predict` so this PR is non-breaking on its own.

## Review decisions (2026-05-06)

The architecture/code-quality/tests/performance review walk-through produced the following deltas to this plan. Where a decision conflicts with text below, this list wins.

### Architecture

- **A3 — view selection.** Tools layer derives the view name from `Settings.embedding_table` via `_VIEW_FOR_TABLE = {"place_embeddings": "place_documents", "place_embeddings_v2": "place_documents_v2"}` in `app/tools/retrieval.py`. Same allowlist-then-f-string pattern as `app/retriever.py:50`. No new env var.
- **A4 — connection helper.** `get_conn()` is a context manager added to `app/db.py` (next to the existing `get_db()` generator). Body opens a fresh `psycopg2.connect(settings.resolved_database_url)` for now; PR #56 will swap the body to use the pool when it lands, with no caller change. `build_embedding(query, settings)` extracted to `app/retriever.py` (or co-located near `get_conn` if cleaner). Legacy `PgVectorRetriever` is untouched.

### Code Quality

- **C5 — view DDL templating.** The migration's `upgrade()` defines a single `_VIEW_SQL_TEMPLATE` Python string with `{view_name}` and `{embedding_table}` placeholders, then calls `op.execute(template.format(...))` twice. Removes ~60 lines of duplicated SELECT clauses. Both substitutions are hardcoded literals from the W1 migration file — no injection surface.
- **C6 — placeholder helpers.** Delete `_row_to_hit` and `_row_to_details`. Construct `PlaceHit(**row)` / `PlaceDetails(**row)` directly.
- **C7 — `_execute()` shape.** Use `psycopg2.extras.RealDictCursor` so the cursor returns dict-shaped rows directly; drop the manual `zip(cols, r)`. Move `from app.db import get_conn` to the top of `app/tools/retrieval.py` (no circular-import risk now that `get_conn` lives in `app/db.py`).
- **C8 — `nearby` SQL bug.** The plan's `HAVING dist_m <= %s` without `GROUP BY` is invalid. Rewrite as a CTE: compute `dist_m` in a `candidates` CTE, filter with `WHERE` in the outer query.

### Tests

- **T9 — smoke test.** Add `tests/unit/test_tools_retrieval_smoke.py` (~10 lines) that imports the module, constructs `SearchFilters()`, `PlaceHit(...)`, `PlaceDetails(...)` with sample values. Catches import-time and Pydantic-schema errors without needing a DB.
- **T10 — `open_at` timezone enforcement.** Add a Pydantic validator on `SearchFilters.open_at` that requires a tz-aware datetime. Cover both cases (naive raises, tz-aware passes) in `test_filters.py`.
- **T11 — integration test helpers.** Define `_period(open_dow, open_h, open_m, close_dow, close_h, close_m) → dict` and `_is_open(hours, at) → bool` in `tests/integration/test_place_is_open.py`. The latter calls `SELECT place_is_open(%s::jsonb, %s)` via `get_conn()`.
- **T12 — view-mapping contract test.** Add a 4-line unit test asserting `set(_VIEW_FOR_TABLE.keys()) == set(ALLOWED_EMBEDDING_TABLES)`. Catches drift between the allowlist and the view map.

### Performance

- **P13 — HNSW + filter recall lever.** `semantic_search` uses `LIMIT k * _OVERFETCH_FACTOR`; `_OVERFETCH_FACTOR = 1` for now. Constant in `app/tools/retrieval.py`. Comment cites W6 as the trigger to bump it.
- **P14 — view-computed booleans.** Leave as-is. ~5,800 rows; cost is microseconds.
- **P15 — `nearby` reads.** (a) Anchor reads `latitude`/`longitude` from `places_raw` directly, not from `place_documents` (skip the embedding JOIN). (b) Drop `pd.embedding` from the neighbor SELECT — we don't need vectors in the result payload.
- **P16 — `LOWER(neighborhood)` index.** Leave as-is. Add a functional index in a follow-up only if W6 surfaces neighborhood-filtered queries as slow.

## Files

### New: Alembic migration `create_place_documents_view`

Generate with `make migration MSG="create place_documents view"` — Alembic creates the file in `alembic/versions/<timestamp>-<rev>_create_place_documents_view.py` (see `5187c6b09b25_create_place_embeddings_v2.py` for the precedent). The body is hand-written SQL via `op.execute()` because Alembic's helpers can't model VIEWs or PL/pgSQL functions.

The migration creates **four** SQL objects in one atomic upgrade:

1. `place_is_open(jsonb, timestamptz)` — PL/pgSQL helper, used by `SearchFilters.open_at`.
2. `neighborhood_of(jsonb)` — PL/pgSQL helper, used by both views. Owned by W1 because we need it now; W7's migration uses `CREATE OR REPLACE FUNCTION` so it can ship later without breakage. Definition matches `_neighborhood_from_address_components` in `scripts/embed_places_pgvector_v2.py`.
3. `place_documents` view — joins `places_raw` + `place_embeddings` (v1).
4. `place_documents_v2` view — joins `places_raw` + `place_embeddings_v2` (v2).

`downgrade()` drops all four in reverse order.

```python
# alembic/versions/<timestamp>-<rev>_create_place_documents_view.py
from alembic import op

revision = "<rev>"
down_revision = "5187c6b09b25"  # create_place_embeddings_v2
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION neighborhood_of(source_json JSONB)
        RETURNS TEXT AS $$
        DECLARE component JSONB;
        BEGIN
          IF source_json IS NULL THEN RETURN ''; END IF;
          FOR component IN
            SELECT * FROM jsonb_array_elements(source_json->'addressComponents')
          LOOP
            IF (component->'types') ? 'neighborhood' THEN
              RETURN COALESCE(component->>'longText', component->>'shortText', '');
            END IF;
          END LOOP;
          RETURN '';
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    op.execute(_PLACE_IS_OPEN_FN)        # see SQL block below
    op.execute(_PLACE_DOCUMENTS_VIEW)    # v1
    op.execute(_PLACE_DOCUMENTS_V2_VIEW) # v2


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS place_documents_v2")
    op.execute("DROP VIEW IF EXISTS place_documents")
    op.execute("DROP FUNCTION IF EXISTS place_is_open(JSONB, TIMESTAMPTZ)")
    op.execute("DROP FUNCTION IF EXISTS neighborhood_of(JSONB)")
```

The SQL bodies for the views and `place_is_open()` are below. They go inline as triple-quoted Python strings; the SQL is identical to the original plan.

```sql
-- Unified place document view. Backed by places_raw + place_embeddings today;
-- a teammate is adding an editorial source (Eater + Infatuation). When that
-- table lands, this view becomes a UNION ALL — agent code does not change.

-- Note: this view reads from whichever embedding table the app selects via the
-- EMBEDDING_TABLE env var (W0a). To keep the view definition stable, we define
-- it once per table name and the migration creates BOTH variants. The retriever
-- helper picks the right view name at query time:
--   place_documents     -> joins place_embeddings   (v1)
--   place_documents_v2  -> joins place_embeddings_v2 (v2 — preferred)
-- This costs one extra view definition and avoids a runtime `format()` on the
-- view name.

CREATE OR REPLACE VIEW place_documents AS
SELECT
    p.place_id,
    p.name,
    p.primary_type,
    p.types,
    p.formatted_address,
    -- structured neighborhood pulled from addressComponents (mirrors
    -- _neighborhood_from_address_components in scripts/embed_places_pgvector_v2.py).
    -- The neighborhood_of() helper is created earlier in this same migration —
    -- W7's migration uses CREATE OR REPLACE FUNCTION with an identical body so
    -- it can ship in any order without breakage.
    neighborhood_of(p.source_json) AS neighborhood,
    p.latitude,
    p.longitude,
    p.rating,
    p.user_rating_count,
    p.price_level,
    p.business_status,
    p.website_uri,
    p.maps_uri,
    p.editorial_summary,
    p.regular_opening_hours,
    p.source_city,
    -- High-signal boolean amenities promoted from source_json. These are the
    -- Role-2 facts W0a deliberately removed from embedding_text — exposing them
    -- here is what makes that removal safe.
    COALESCE((p.source_json->>'servesCocktails')::boolean,  false) AS serves_cocktails,
    COALESCE((p.source_json->>'servesBeer')::boolean,       false) AS serves_beer,
    COALESCE((p.source_json->>'servesWine')::boolean,       false) AS serves_wine,
    COALESCE((p.source_json->>'servesCoffee')::boolean,     false) AS serves_coffee,
    COALESCE((p.source_json->>'servesBreakfast')::boolean,  false) AS serves_breakfast,
    COALESCE((p.source_json->>'servesBrunch')::boolean,     false) AS serves_brunch,
    COALESCE((p.source_json->>'servesLunch')::boolean,      false) AS serves_lunch,
    COALESCE((p.source_json->>'servesDinner')::boolean,     false) AS serves_dinner,
    COALESCE((p.source_json->>'servesVegetarianFood')::boolean, false) AS serves_vegetarian,
    COALESCE((p.source_json->>'outdoorSeating')::boolean,   false) AS outdoor_seating,
    COALESCE((p.source_json->>'reservable')::boolean,       false) AS reservable,
    COALESCE((p.source_json->>'allowsDogs')::boolean,       false) AS allows_dogs,
    COALESCE((p.source_json->>'liveMusic')::boolean,        false) AS live_music,
    COALESCE((p.source_json->>'goodForGroups')::boolean,    false) AS good_for_groups,
    COALESCE((p.source_json->>'goodForChildren')::boolean,  false) AS good_for_children,
    COALESCE((p.source_json->>'goodForWatchingSports')::boolean, false) AS good_for_sports,
    -- Parking flags as a small jsonb so SearchFilters can do
    -- "needs_parking" without modeling each subfield.
    COALESCE(p.source_json->'parkingOptions', '{}'::jsonb) AS parking_options,
    'google_places'::text AS source,
    e.embedding,
    e.embedding_model,
    e.embedding_text,
    e.source_updated_at AS embedded_source_updated_at
FROM places_raw p
JOIN place_embeddings e ON e.place_id = p.place_id;

-- v2 variant — same definition, different join target. Used when
-- EMBEDDING_TABLE=place_embeddings_v2 (W0a). Once v1 is retired this becomes
-- the only view and the suffix can be dropped.
CREATE OR REPLACE VIEW place_documents_v2 AS
SELECT
    p.place_id, p.name, p.primary_type, p.types, p.formatted_address,
    neighborhood_of(p.source_json) AS neighborhood,
    p.latitude, p.longitude, p.rating, p.user_rating_count, p.price_level,
    p.business_status, p.website_uri, p.maps_uri, p.editorial_summary,
    p.regular_opening_hours, p.source_city,
    COALESCE((p.source_json->>'servesCocktails')::boolean,  false) AS serves_cocktails,
    COALESCE((p.source_json->>'servesBeer')::boolean,       false) AS serves_beer,
    COALESCE((p.source_json->>'servesWine')::boolean,       false) AS serves_wine,
    COALESCE((p.source_json->>'servesCoffee')::boolean,     false) AS serves_coffee,
    COALESCE((p.source_json->>'servesBreakfast')::boolean,  false) AS serves_breakfast,
    COALESCE((p.source_json->>'servesBrunch')::boolean,     false) AS serves_brunch,
    COALESCE((p.source_json->>'servesLunch')::boolean,      false) AS serves_lunch,
    COALESCE((p.source_json->>'servesDinner')::boolean,     false) AS serves_dinner,
    COALESCE((p.source_json->>'servesVegetarianFood')::boolean, false) AS serves_vegetarian,
    COALESCE((p.source_json->>'outdoorSeating')::boolean,   false) AS outdoor_seating,
    COALESCE((p.source_json->>'reservable')::boolean,       false) AS reservable,
    COALESCE((p.source_json->>'allowsDogs')::boolean,       false) AS allows_dogs,
    COALESCE((p.source_json->>'liveMusic')::boolean,        false) AS live_music,
    COALESCE((p.source_json->>'goodForGroups')::boolean,    false) AS good_for_groups,
    COALESCE((p.source_json->>'goodForChildren')::boolean,  false) AS good_for_children,
    COALESCE((p.source_json->>'goodForWatchingSports')::boolean, false) AS good_for_sports,
    COALESCE(p.source_json->'parkingOptions', '{}'::jsonb) AS parking_options,
    'google_places'::text AS source,
    e.embedding, e.embedding_model, e.embedding_text,
    e.source_updated_at AS embedded_source_updated_at
FROM places_raw p
JOIN place_embeddings_v2 e ON e.place_id = p.place_id;

-- Comment so future maintainers know to UNION the editorial table here.
COMMENT ON VIEW place_documents IS
  'Unified retrieval surface. When editorial places table lands, redefine as UNION ALL with source = ''editorial''.';
```

Apply with `make migrate` after generating the migration. CI's `migrations` job will round-trip `downgrade base → upgrade head` against an ephemeral pgvector container, so a broken downgrade fails the PR before merge.

### New: `app/tools/__init__.py`

```python
"""Agent-callable tools. Each tool is a deterministic Python function with a
typed Pydantic input/output. The agent graph (W2) registers these as
LangGraph/LangChain tools."""
```

### New: `app/tools/filters.py`

```python
"""Structured filters compiled to parameterized SQL fragments.

Every filter here corresponds to a real column on the place_documents view
(see scripts/db/migrations/001_place_documents_view.sql). A filter that maps
to a JSONB lookup (e.g. open_at) is implemented as a function the LLM does
not need to know about — it just sets `open_at`."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Structured constraints the agent passes to retrieval tools.

    All fields are optional but the quality-floor defaults
    (`min_user_rating_count = 50`, `business_status = 'OPERATIONAL'`) apply
    unless explicitly overridden. Empty SearchFilters() does NOT match
    everything — it matches operational places with at least 50 raters. The
    agent must opt out of the floors deliberately.

    Why the floors: on 2026-05-04 we found a 5.0-rated "Pasadena Velasco Open
    Space" with one rater bubbling to the top of results. A concierge that
    surfaces single-rater places is worse than no concierge.
    """

    price_level_max: Optional[int] = Field(
        default=None, ge=0, le=4,
        description="Max Google price_level. 0=free, 4=very expensive.",
    )
    min_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    min_user_rating_count: Optional[int] = Field(
        default=50, ge=0,
        description="Quality floor. Default 50 to keep single-rater places out. "
                    "Set to 0 to include any number of raters.",
    )
    open_at: Optional[datetime] = Field(
        default=None,
        description="If set, restrict to places open at this local time. "
                    "Used per-stop with planned arrival time, NOT the user's prompt time.",
    )
    neighborhood: Optional[str] = Field(
        default=None,
        description="Exact match against the structured neighborhood column "
                    "(case-insensitive). Falls back to formatted_address ILIKE only "
                    "when no row in the neighborhood column matches.",
    )
    types_any: Optional[list[str]] = Field(
        default=None,
        description="Match if any of these strings appears in types[].",
    )
    business_status: Optional[str] = Field(
        default="OPERATIONAL",
        description="Default OPERATIONAL. Set None to include closed/permanently_closed.",
    )
    source: Optional[str] = Field(
        default=None,
        description="One of 'google_places', 'editorial'. None = any.",
    )

    # ---- Boolean amenity filters (promoted columns from W1's view) ----
    serves_cocktails:    Optional[bool] = None
    serves_beer:         Optional[bool] = None
    serves_wine:         Optional[bool] = None
    serves_coffee:       Optional[bool] = None
    serves_breakfast:    Optional[bool] = None
    serves_brunch:       Optional[bool] = None
    serves_lunch:        Optional[bool] = None
    serves_dinner:       Optional[bool] = None
    serves_vegetarian:   Optional[bool] = None
    outdoor_seating:     Optional[bool] = None
    reservable:          Optional[bool] = None
    allows_dogs:         Optional[bool] = None
    live_music:          Optional[bool] = None
    good_for_groups:     Optional[bool] = None
    good_for_children:   Optional[bool] = None
    good_for_sports:     Optional[bool] = None


def compile_filters(f: SearchFilters) -> tuple[str, list]:
    """Return (sql_where_fragment, params_list).

    The fragment begins with 'AND' (caller prepends a WHERE clause for the
    semantic-search query). Params are positional (psycopg2 / asyncpg style).
    """
    clauses: list[str] = []
    params: list = []

    if f.price_level_max is not None:
        clauses.append("price_level <= %s")
        params.append(f.price_level_max)

    if f.min_rating is not None:
        clauses.append("rating >= %s")
        params.append(f.min_rating)

    if f.min_user_rating_count is not None:
        clauses.append("user_rating_count >= %s")
        params.append(f.min_user_rating_count)

    if f.business_status is not None:
        clauses.append("business_status = %s")
        params.append(f.business_status)

    if f.neighborhood:
        # Prefer the structured neighborhood column added in W1's view.
        # Falls back to formatted_address substring for places with no
        # addressComponents.neighborhood (a few edge cases).
        clauses.append("(LOWER(neighborhood) = LOWER(%s) "
                       "OR formatted_address ILIKE %s)")
        params.append(f.neighborhood)
        params.append(f"%{f.neighborhood}%")

    # Boolean amenity filters — only emit a clause if explicitly set, so
    # `None` means "don't care" and `False` means "must be false".
    _BOOL_COLUMNS = {
        "serves_cocktails":  "serves_cocktails",
        "serves_beer":       "serves_beer",
        "serves_wine":       "serves_wine",
        "serves_coffee":     "serves_coffee",
        "serves_breakfast":  "serves_breakfast",
        "serves_brunch":     "serves_brunch",
        "serves_lunch":      "serves_lunch",
        "serves_dinner":     "serves_dinner",
        "serves_vegetarian": "serves_vegetarian",
        "outdoor_seating":   "outdoor_seating",
        "reservable":        "reservable",
        "allows_dogs":       "allows_dogs",
        "live_music":        "live_music",
        "good_for_groups":   "good_for_groups",
        "good_for_children": "good_for_children",
        "good_for_sports":   "good_for_sports",
    }
    for attr, column in _BOOL_COLUMNS.items():
        value = getattr(f, attr)
        if value is not None:
            clauses.append(f"{column} = %s")
            params.append(value)

    if f.types_any:
        clauses.append("types && %s")  # PG array overlap operator
        params.append(f.types_any)

    if f.source:
        clauses.append("source = %s")
        params.append(f.source)

    if f.open_at is not None:
        # regular_opening_hours JSONB shape (Google Places v1):
        #   { "periods": [ {"open": {"day": 0, "hour": 9, "minute": 0},
        #                    "close": {"day": 0, "hour": 22, "minute": 0}} ], ... }
        # We push the check into SQL via a helper function (defined inline below).
        clauses.append("place_is_open(regular_opening_hours, %s)")
        params.append(f.open_at)

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params
```

The `place_is_open(jsonb, timestamptz)` helper is a tiny PL/pgSQL function — its body goes in the same migration (the `_PLACE_IS_OPEN_FN` constant referenced in the `upgrade()` skeleton above):

```sql
CREATE OR REPLACE FUNCTION place_is_open(hours JSONB, at_ts TIMESTAMPTZ)
RETURNS BOOLEAN AS $$
DECLARE
  dow            INT := EXTRACT(DOW FROM at_ts);  -- 0=Sun, matches Google
  hh             INT := EXTRACT(HOUR FROM at_ts);
  mm             INT := EXTRACT(MINUTE FROM at_ts);
  minutes_of_day INT := hh * 60 + mm;
  period         JSONB;
  open_dow       INT;
  close_dow      INT;
  open_minutes   INT;
  close_minutes  INT;
BEGIN
  IF hours IS NULL OR hours = '{}'::jsonb THEN
    RETURN TRUE;  -- unknown hours: don't exclude
  END IF;
  FOR period IN SELECT * FROM jsonb_array_elements(hours->'periods') LOOP
    open_dow      := (period->'open'->>'day')::int;
    close_dow     := COALESCE((period->'close'->>'day')::int, open_dow);
    open_minutes  := (period->'open'->>'hour')::int * 60
                   + COALESCE((period->'open'->>'minute')::int, 0);
    close_minutes := (period->'close'->>'hour')::int * 60
                   + COALESCE((period->'close'->>'minute')::int, 0);

    -- Same-day period (e.g. 11:00–22:00).
    IF open_dow = close_dow AND open_dow = dow THEN
      IF minutes_of_day BETWEEN open_minutes AND close_minutes THEN
        RETURN TRUE;
      END IF;

    -- Overnight period crossing midnight (e.g. Friday 18:00 → Saturday 02:00).
    -- Match if we are on the open day after open_minutes,
    --        OR on the close day before close_minutes.
    ELSIF open_dow <> close_dow THEN
      IF dow = open_dow  AND minutes_of_day >= open_minutes  THEN
        RETURN TRUE;
      END IF;
      IF dow = close_dow AND minutes_of_day <= close_minutes THEN
        RETURN TRUE;
      END IF;
    END IF;
  END LOOP;
  RETURN FALSE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

Now correctly handles overnight periods (e.g. a bar open Friday 6pm to Saturday 2am). The two cases — same-day and across-midnight — are distinguished by comparing `open_dow` and `close_dow`. Tests below cover both.

### New: `app/tools/retrieval.py`

```python
"""Retrieval tools the agent calls. Reuses the embedding-generation logic from
app/retriever.py:17-37 (do not duplicate). Talks to the `place_documents` view.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.config import get_settings
from app.retriever import build_embedding  # extracted helper (see modify section below)
from app.tools.filters import SearchFilters, compile_filters


class PlaceHit(BaseModel):
    place_id: str
    name: str
    primary_type: Optional[str]
    formatted_address: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    rating: Optional[float]
    price_level: Optional[int]
    business_status: Optional[str]
    source: str
    similarity: float
    snippet: Optional[str]  # excerpt of embedding_text


class PlaceDetails(PlaceHit):
    types: list[str]
    user_rating_count: Optional[int]
    website_uri: Optional[str]
    maps_uri: Optional[str]
    editorial_summary: Optional[str]
    regular_opening_hours: dict


def semantic_search(
    query: str,
    filters: Optional[SearchFilters] = None,
    k: int = 10,
) -> list[PlaceHit]:
    """Vector similarity over place_documents with optional structured filters."""
    settings = get_settings()
    filters = filters or SearchFilters()
    where_fragment, filter_params = compile_filters(filters)
    embedding = build_embedding(query, settings)

    sql = f"""
        SELECT
            place_id, name, primary_type, formatted_address,
            latitude, longitude, rating, price_level, business_status,
            source,
            1 - (embedding <=> %s::vector) AS similarity,
            LEFT(embedding_text, 400) AS snippet
        FROM place_documents
        WHERE embedding_model = %s
        {where_fragment}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    vector_literal = _vector_literal(embedding)
    params = [vector_literal, settings.openai_embedding_model, *filter_params,
              vector_literal, k]
    rows = _execute(sql, params)
    return [PlaceHit(**_row_to_hit(r)) for r in rows]


def nearby(
    place_id: str,
    radius_m: int = 800,
    filters: Optional[SearchFilters] = None,
    k: int = 10,
) -> list[PlaceHit]:
    """Geographic 'within radius_m' search around an anchor place. Optional
    semantic re-rank if the agent passes a query in filters (future)."""
    filters = filters or SearchFilters()
    where_fragment, filter_params = compile_filters(filters)

    sql = f"""
        WITH anchor AS (
            SELECT latitude, longitude FROM place_documents WHERE place_id = %s LIMIT 1
        )
        SELECT
            pd.place_id, pd.name, pd.primary_type, pd.formatted_address,
            pd.latitude, pd.longitude, pd.rating, pd.price_level,
            pd.business_status, pd.source,
            0.0 AS similarity,
            LEFT(pd.embedding_text, 400) AS snippet,
            -- Haversine in meters (good enough at SF scale)
            6371000 * 2 * ASIN(SQRT(
                POWER(SIN(RADIANS(pd.latitude - a.latitude) / 2), 2) +
                COS(RADIANS(a.latitude)) * COS(RADIANS(pd.latitude)) *
                POWER(SIN(RADIANS(pd.longitude - a.longitude) / 2), 2)
            )) AS dist_m
        FROM place_documents pd, anchor a
        WHERE pd.place_id <> %s
        {where_fragment}
        HAVING dist_m <= %s
        ORDER BY dist_m ASC
        LIMIT %s
    """
    params = [place_id, place_id, *filter_params, radius_m, k]
    rows = _execute(sql, params)
    return [PlaceHit(**_row_to_hit(r)) for r in rows]


def get_details(place_id: str) -> Optional[PlaceDetails]:
    sql = """
        SELECT place_id, name, primary_type, types, formatted_address,
               latitude, longitude, rating, user_rating_count, price_level,
               business_status, website_uri, maps_uri, editorial_summary,
               regular_opening_hours, source,
               LEFT(embedding_text, 800) AS snippet, 0.0 AS similarity
        FROM place_documents WHERE place_id = %s LIMIT 1
    """
    rows = _execute(sql, [place_id])
    if not rows:
        return None
    return PlaceDetails(**_row_to_details(rows[0]))


# --- helpers (kept private to this module) ----------------------------------

def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

def _execute(sql: str, params: list) -> list[dict]:
    """Thin DB shim. Reuses the connection mechanism from app/retriever.py
    (extract `_get_conn()` there as a shared helper as part of this PR)."""
    from app.retriever import get_conn  # extracted in this PR
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

def _row_to_hit(r: dict) -> dict:
    # Identity for now; placeholder for any normalization.
    return r

def _row_to_details(r: dict) -> dict:
    return r
```

### Modify: `app/retriever.py`

Extract two helpers so `app/tools/retrieval.py` can reuse them without duplicating logic:

```python
# Extract these from existing code in app/retriever.py:17-37 and the SQL execution path.

def build_embedding(query: str, settings) -> list[float]:
    """Generate a query embedding via OpenAI. Reused by tools."""
    # body: existing OpenAI embedding call

def get_conn():
    """Yield a psycopg2 connection. Context-manager friendly. Reused by tools."""
    # body: existing connection construction
```

Keep `PgVectorRetriever` exactly as-is so `/predict` and `app/chain.py` keep working.

### Modify: `pyproject.toml`

No new deps in W1. (LangGraph lands in W2.)

## Tests

### New: `tests/unit/test_filters.py`

```python
from datetime import datetime
from app.tools.filters import SearchFilters, compile_filters


def test_empty_filters_no_clauses():
    where, params = compile_filters(SearchFilters(business_status=None))
    assert where == ""
    assert params == []

def test_default_excludes_closed_permanently():
    where, params = compile_filters(SearchFilters())
    assert "business_status = %s" in where
    assert params == ["OPERATIONAL"]

def test_price_and_rating():
    where, params = compile_filters(
        SearchFilters(price_level_max=2, min_rating=4.3, business_status=None)
    )
    assert "price_level <= %s" in where
    assert "rating >= %s" in where
    assert params == [2, 4.3]

def test_neighborhood_uses_ilike_substring():
    where, params = compile_filters(
        SearchFilters(neighborhood="North Beach", business_status=None)
    )
    assert "formatted_address ILIKE %s" in where
    assert params == ["%North Beach%"]

def test_types_uses_array_overlap():
    where, params = compile_filters(
        SearchFilters(types_any=["bar", "wine_bar"], business_status=None)
    )
    assert "types && %s" in where
    assert params == [["bar", "wine_bar"]]

def test_open_at_calls_helper():
    ts = datetime(2026, 4, 26, 19, 30)  # Sunday 7:30pm
    where, params = compile_filters(
        SearchFilters(open_at=ts, business_status=None, min_user_rating_count=0)
    )
    assert "place_is_open(regular_opening_hours, %s)" in where
    assert params == [ts]


def test_default_user_rating_count_floor_present():
    where, params = compile_filters(SearchFilters())
    assert "user_rating_count >= %s" in where
    assert 50 in params  # the floor


def test_user_rating_count_floor_can_be_disabled():
    where, params = compile_filters(SearchFilters(min_user_rating_count=0))
    # 0 still emits a clause so the agent's intent is auditable in the SQL log,
    # but the clause is trivially satisfied and matches the previous default
    # behavior.
    assert "user_rating_count >= %s" in where
    assert 0 in params


def test_neighborhood_uses_structured_column():
    where, params = compile_filters(
        SearchFilters(neighborhood="Mission Bay", business_status=None,
                      min_user_rating_count=0)
    )
    assert "LOWER(neighborhood) = LOWER(%s)" in where
    assert "formatted_address ILIKE %s" in where  # fallback also present
    assert "Mission Bay" in params
    assert "%Mission Bay%" in params


def test_boolean_amenity_filters():
    where, params = compile_filters(
        SearchFilters(serves_cocktails=True, outdoor_seating=True,
                      allows_dogs=False, business_status=None,
                      min_user_rating_count=0)
    )
    assert "serves_cocktails = %s" in where
    assert "outdoor_seating = %s"  in where
    assert "allows_dogs = %s"      in where
    assert params == [True, True, False]


def test_unset_boolean_filters_emit_no_clause():
    where, _ = compile_filters(
        SearchFilters(business_status=None, min_user_rating_count=0)
    )
    for col in ("serves_cocktails", "outdoor_seating", "reservable",
                "allows_dogs", "live_music", "good_for_groups"):
        assert f"{col} =" not in where
```

Add SQL-level tests for the overnight `place_is_open` change in the integration suite:

```python
# tests/integration/test_place_is_open.py — gated on APP_ENV=integration
def test_same_day_window_open():
    # Tuesday 12:30, place open Tue 11–22
    assert _is_open(_period(2, 11, 0, 2, 22, 0), datetime(2026, 4, 28, 12, 30))

def test_same_day_window_closed_after_hours():
    assert not _is_open(_period(2, 11, 0, 2, 22, 0), datetime(2026, 4, 28, 23, 0))

def test_overnight_window_after_open_same_day():
    # Bar open Fri 18:00 → Sat 02:00, query Fri 22:00 → open.
    assert _is_open(_period(5, 18, 0, 6, 2, 0), datetime(2026, 5, 1, 22, 0))

def test_overnight_window_before_close_next_day():
    # Same period, query Sat 01:30 → still open.
    assert _is_open(_period(5, 18, 0, 6, 2, 0), datetime(2026, 5, 2, 1, 30))

def test_overnight_window_closed_in_morning():
    # Same period, query Sat 03:00 → closed.
    assert not _is_open(_period(5, 18, 0, 6, 2, 0), datetime(2026, 5, 2, 3, 0))
```

### New: `tests/unit/test_tools_retrieval.py`

Mirror the fake-cursor pattern from `tests/unit/test_retriever.py:1-98`. Cover:
- `semantic_search` builds the expected SQL when filters are empty.
- `semantic_search` injects filter fragments and orders params correctly.
- `nearby` enforces `place_id <> anchor`, applies the haversine `HAVING`, and orders by distance.
- `get_details` returns `None` when the row is missing.

### Integration (gated on `APP_ENV=integration`)

`tests/integration/test_place_documents_view.py` — confirms the view exists, returns rows after `make ingest` against a seeded fixture, and that `place_is_open` returns the expected booleans across the schedule of one well-known place.

## Manual verification

```bash
make migrate          # apply the new alembic migration
make ingest           # seed via existing pipeline
python -c "
from app.tools.retrieval import semantic_search
from app.tools.filters import SearchFilters
from datetime import datetime
hits = semantic_search(
    'romantic italian',
    SearchFilters(price_level_max=3, min_rating=4.3, neighborhood='North Beach',
                  open_at=datetime(2026, 4, 26, 19, 30)),
    k=5)
for h in hits:
    print(h.name, h.rating, h.price_level, h.similarity)
"
```

Expected: 5 results in/near North Beach, all `rating >= 4.3`, all `price_level <= 3`, all open Sunday 7:30pm.

## Risks / open questions

- **Editorial table schema unknown.** When it lands, the `place_documents` view must be redefined as `UNION ALL`. If editorial rows lack `latitude/longitude`, the `nearby` tool needs a guard. Plan: when the editorial PR is in flight, this view definition gets a follow-up edit; tools don't change.
- **`place_is_open` overnight handling.** Now correct for overnight periods. Edge case still unhandled: a single period spanning more than 24h (essentially "always open"). Google's data uses `openNow=true` with no periods in that case, which falls through to the `IF hours IS NULL OR hours = '{}'::jsonb` short-circuit. Acceptable.
- **HNSW + WHERE filters can degrade recall.** pgvector's HNSW index is great for `ORDER BY <=>`, but heavy `WHERE` filtering on top can bypass the index. With the expanded filter set in this PR, this is more likely than in the original W1 design. Mitigation: retrieve a wider top-N (e.g. 50) and filter in SQL after the vector ORDER BY. The view's structure already supports this — the tool layer just needs `LIMIT 50` then post-filter — but we will not optimize pre-emptively. If eval (W6) shows recall regressions on tightly-filtered queries, this is the first lever to pull.
- **Single-rater quality floor may be too aggressive for niche categories.** A new wine bar with 30 raters but a 4.9 rating gets excluded by `min_user_rating_count = 50`. The agent can opt out per-call; in W2's prompt we tell it to lower the floor for queries that ask for "new" / "recently opened." Watch for false negatives in eval.
- **Boolean amenity columns are computed in the view, not stored.** Each query re-evaluates the COALESCE casts. With ~5,800 rows this is fine. If we ever materialize the view or generate columns on `places_raw`, the cast logic moves with no other change.
- **Two views (v1 + v2) double the migration surface.** Acceptable while we A/B v1 vs v2 in W6 evals. As soon as v2 wins, drop `place_documents` (v1) and rename `place_documents_v2` to `place_documents` in a small follow-up PR.
- **Connection pooling.** `get_conn()` should use whatever pooling the app already does. If the existing retriever opens fresh connections per call, leave that alone in this PR; revisit pool config separately.

---

**Status:** ✅ Merged via [#65](https://github.com/deshmukh-neel/mlops_city_concierge/pull/65) on 2026-05-06. Followed by hotfix [#66](https://github.com/deshmukh-neel/mlops_city_concierge/pull/66) (`price_level` is a Google v1 enum string, not int — adds `price_level_rank()` migration and a `scripts/smoke_w1.py` end-to-end harness).

Deferred / pending:

- Pool integration. `get_conn()` opens fresh connections; PR #56 will swap the body to borrow from the shared pool with no caller change.
- Editorial table `UNION ALL`. View definition gets a follow-up edit when the editorial schema lands; tools don't change.
- HNSW recall over-fetch lever (`_OVERFETCH_FACTOR`) ships at `1`. Bump if W6 evals show recall regressing on tightly-filtered queries.
- v1 retirement. Drop `place_documents` and rename `place_documents_v2 → place_documents` once W6 confirms v2 wins.
- Multi-city. `place_is_open()` hardcodes `America/Los_Angeles`; expand to a `tz TEXT` argument when the app moves beyond SF.
