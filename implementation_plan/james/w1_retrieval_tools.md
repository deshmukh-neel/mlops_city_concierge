# W1 — Unified place view + filterable retrieval tools

**Branch:** `feature/agent-w1-retrieval-tools`
**Depends on:** nothing
**Unblocks:** W2, W3, W6 (and indirectly W4)

## Goal

Replace the single-pass, metadata-blind retriever with a set of **tools** the agent can call with structured filters. Hide the underlying tables (`places_raw` + `place_embeddings`, plus the future editorial table) behind a SQL view so the retrieval tools never touch source tables directly.

After this PR:
- New `place_documents` view exposes every column the agent might filter on, with a `source` column.
- Three tool functions (`semantic_search`, `nearby`, `get_details`) return typed Pydantic objects.
- A `SearchFilters` model compiles to safe parameterized SQL fragments.
- The legacy `PgVectorRetriever` still works for `/predict` so this PR is non-breaking on its own.

## Files

### New: `scripts/db/migrations/001_place_documents_view.sql`

```sql
-- Unified place document view. Backed by places_raw + place_embeddings today;
-- a teammate is adding an editorial source (Eater + Infatuation). When that
-- table lands, this view becomes a UNION ALL — agent code does not change.

CREATE OR REPLACE VIEW place_documents AS
SELECT
    p.place_id,
    p.name,
    p.primary_type,
    p.types,
    p.formatted_address,
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
    'google_places'::text AS source,
    e.embedding,
    e.embedding_model,
    e.embedding_text,
    e.source_updated_at AS embedded_source_updated_at
FROM places_raw p
JOIN place_embeddings e ON e.place_id = p.place_id;

-- Comment so future maintainers know to UNION the editorial table here.
COMMENT ON VIEW place_documents IS
  'Unified retrieval surface. When editorial places table lands, redefine as UNION ALL with source = ''editorial''.';
```

Wire this into `Makefile`'s `migrate` target, or ensure Alembic picks it up if Alembic is initialized (Makefile mentions Alembic but the tree doesn't have a `migrations/` dir yet — if absent, append the SQL to `scripts/db/init.sql` for now and note in PR description).

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

    All fields are optional. Empty SearchFilters() matches everything.
    """

    price_level_max: Optional[int] = Field(
        default=None, ge=0, le=4,
        description="Max Google price_level. 0=free, 4=very expensive.",
    )
    min_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    min_user_rating_count: Optional[int] = Field(default=None, ge=0)
    open_at: Optional[datetime] = Field(
        default=None,
        description="If set, restrict to places open at this local time.",
    )
    neighborhood: Optional[str] = Field(
        default=None,
        description="Substring match against formatted_address. Case-insensitive.",
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
        clauses.append("formatted_address ILIKE %s")
        params.append(f"%{f.neighborhood}%")

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

The `place_is_open(jsonb, timestamptz)` helper is a tiny PL/pgSQL function — add it to the same migration:

```sql
-- Append to 001_place_documents_view.sql
CREATE OR REPLACE FUNCTION place_is_open(hours JSONB, at_ts TIMESTAMPTZ)
RETURNS BOOLEAN AS $$
DECLARE
  dow INT := EXTRACT(DOW FROM at_ts);   -- 0=Sun, matches Google
  hh  INT := EXTRACT(HOUR FROM at_ts);
  mm  INT := EXTRACT(MINUTE FROM at_ts);
  minutes_of_day INT := hh * 60 + mm;
  period JSONB;
  open_minutes INT;
  close_minutes INT;
BEGIN
  IF hours IS NULL OR hours = '{}'::jsonb THEN
    RETURN TRUE;  -- unknown hours: don't exclude
  END IF;
  FOR period IN SELECT * FROM jsonb_array_elements(hours->'periods') LOOP
    IF (period->'open'->>'day')::int = dow THEN
      open_minutes  := (period->'open'->>'hour')::int * 60
                     + COALESCE((period->'open'->>'minute')::int, 0);
      close_minutes := (period->'close'->>'hour')::int * 60
                     + COALESCE((period->'close'->>'minute')::int, 0);
      IF minutes_of_day BETWEEN open_minutes AND close_minutes THEN
        RETURN TRUE;
      END IF;
    END IF;
  END LOOP;
  RETURN FALSE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

Note: this is a v1. It does **not** handle overnight periods (close < open). Document this in the migration comment; we can revisit if real data needs it.

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
        SearchFilters(open_at=ts, business_status=None)
    )
    assert "place_is_open(regular_opening_hours, %s)" in where
    assert params == [ts]
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
make migrate          # apply 001_place_documents_view.sql
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
- **`place_is_open` v1 limitation.** Doesn't handle overnight (e.g. bar open 6pm–2am next day). Acceptable for MVP; document and revisit when bar/late-night data is widely used.
- **HNSW + WHERE filters can degrade recall.** pgvector's HNSW index is great for `ORDER BY <=>`, but heavy `WHERE` filtering on top can bypass the index. If we see this in practice, retrieve a wider top-N then filter in Python, or add a partial index per common filter. Not worth optimizing pre-emptively.
- **Connection pooling.** `get_conn()` should use whatever pooling the app already does. If the existing retriever opens fresh connections per call, leave that alone in this PR; revisit pool config separately.
