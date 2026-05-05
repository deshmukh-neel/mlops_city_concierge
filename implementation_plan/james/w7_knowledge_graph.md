# W7 — Knowledge graph layer (`place_relations`) + `kg_traverse` tool

**Branch:** `feature/agent-w7-knowledge-graph`
**Depends on:** W0a (uses the `neighborhood` extracted by `compose_embedding_text_v2`'s helper), W1 (the `place_documents` view + `SearchFilters`)
**Unblocks:** richer multi-stop planning in W2, KG-aware evals in W6

## Why this exists

W2 reserves a `kg_traverse` tool stub (`app/agent/tools.py`, see W2) that currently returns "not yet available." This PR makes it real.

We discovered on 2026-05-04 that `places_raw.source_json` already contains graph-shaped data Google computed for free:

- **`addressDescriptor.landmarks[]`** — up to ~5 nearby landmarks per place with travel-distance + straight-line-distance + place IDs. This is "what's nearby that I should mention" — pre-computed.
- **`containingPlaces[]`** — places this place is *inside of* (e.g. STEM Kitchen is inside Biohub). This is "X is part of Y."
- **`addressDescriptor.areas[]`** — named containment areas with `WITHIN` / `OUTSKIRTS` / `NEAR` membership types.
- **`addressComponents[neighborhood]`** — the structured neighborhood. W0a extracts this and embeds the name; W7 uses it as a graph edge.

We also have everything we need to compute the rest from columns we already have: `NEAR` from haversine, `SAME_NEIGHBORHOOD` from the W0a-extracted neighborhood, `SIMILAR_VECTOR` from top-k cosine on `place_embeddings_v2`.

Storage decision: **plain edge tables in the same Cloud SQL Postgres instance.** Apache AGE (openCypher in Postgres) is **not available on Cloud SQL for Postgres**, so it's not an option without changing infrastructure. Pure SQL edge tables are simple, indexable, joinable to `places_raw`, and require no new extension. There is no "pgvector knowledge graph extension" — pgvector is vector-only; the GraphRAG-with-pgvector pattern means exactly this: pgvector + an edge table.

## Scope

**In scope (this PR):**

- `place_relations` edge table.
- A builder script that seeds five edge types from existing data, with no LLM calls:
  - `NEAR` — haversine within 800m, both `OPERATIONAL`.
  - `SAME_NEIGHBORHOOD` — same `neighborhood` extracted by W0a's helper.
  - `CONTAINED_IN` — from `source_json.containingPlaces[]`.
  - `NEAR_LANDMARK` — from `source_json.addressDescriptor.landmarks[]`. The "destination" is a landmark place_id; landmarks may not exist in `places_raw`, so this edge stores landmark name + place_id without enforcing FK.
  - `SIMILAR_VECTOR` — top-k cosine over `place_embeddings_v2` per place.
- A real `kg_traverse` implementation that replaces the stub.
- A typed `RelatedPlace` Pydantic model returned from the tool.
- An MLflow-logged param `kg_enabled` on each agent run so eval can A/B with the KG on or off.

**Deferred to a follow-up PR (explicitly out of scope here):**

- LLM-extracted edges (`OPERATED_BY`, `MENTIONED_WITH`, `SAME_CHEF`) — require LLM extraction over editorial / generative summaries, plus a confidence threshold and a review pass. Worth doing once the editorial scrape lands and the cheap edges prove their value.
- Editorial-source edges from a teammate's Eater / Infatuation scrape — unblocked when that table lands; the schema below is forward-compatible.

## Files

### New: `scripts/db/migrations/002_place_relations.sql`

```sql
-- Knowledge graph edges between places. One row per (src, dst, type).
-- Edges are directed so the same pair can have asymmetric relations
-- (e.g. CONTAINED_IN); for symmetric relations (NEAR, SIMILAR_VECTOR,
-- SAME_NEIGHBORHOOD), the builder writes both directions.

CREATE TABLE IF NOT EXISTS place_relations (
    src_place_id    TEXT NOT NULL REFERENCES places_raw(place_id) ON DELETE CASCADE,
    dst_place_id    TEXT NOT NULL,             -- no FK: landmarks may live outside places_raw
    relation_type   TEXT NOT NULL,
    weight          DOUBLE PRECISION,          -- distance_m for NEAR, cosine for SIMILAR_VECTOR, NULL otherwise
    metadata        JSONB DEFAULT '{}',
    source          TEXT NOT NULL,             -- 'haversine' | 'address_components' | 'source_json' | 'vector_topk' | 'editorial_llm' (future)
    built_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (src_place_id, dst_place_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_place_relations_src   ON place_relations(src_place_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_place_relations_dst   ON place_relations(dst_place_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_place_relations_type  ON place_relations(relation_type);

COMMENT ON TABLE place_relations IS
  'Knowledge graph edges. See implementation_plan/james/w7_knowledge_graph.md for the relation_type taxonomy and source values.';
```

### New: `scripts/build_place_relations.py`

Idempotent builder. Reads `places_raw` + `place_embeddings_v2`, writes `place_relations`. Safe to re-run.

```python
"""
Build the knowledge graph edges in place_relations.

Reads:
  - places_raw (lat/lng, source_json, neighborhood from addressComponents)
  - place_embeddings_v2 (for SIMILAR_VECTOR top-k)

Writes (idempotent — UPSERT on PK):
  - NEAR              haversine within 800m
  - SAME_NEIGHBORHOOD same neighborhood string
  - CONTAINED_IN      from source_json.containingPlaces[]
  - NEAR_LANDMARK     from source_json.addressDescriptor.landmarks[]
  - SIMILAR_VECTOR    top-k cosine per place from place_embeddings_v2

Run after: any meaningful change to places_raw (new ingest), or after a
re-embed (so SIMILAR_VECTOR reflects the new vectors).

Usage:
    python scripts/build_place_relations.py
    python scripts/build_place_relations.py --only NEAR,SIMILAR_VECTOR
"""

NEAR_RADIUS_M    = 800
SIMILAR_TOPK     = 10
SIMILAR_MIN_COS  = 0.65   # don't write low-quality similarity edges
```

The four sub-builders are deterministic SQL:

```sql
-- NEAR (haversine ≤ 800m, both OPERATIONAL, exclude self)
INSERT INTO place_relations (src_place_id, dst_place_id, relation_type, weight, source)
SELECT a.place_id, b.place_id, 'NEAR',
       6371000 * 2 * ASIN(SQRT(
         POWER(SIN(RADIANS(b.latitude  - a.latitude ) / 2), 2) +
         COS(RADIANS(a.latitude)) * COS(RADIANS(b.latitude)) *
         POWER(SIN(RADIANS(b.longitude - a.longitude) / 2), 2)
       )) AS dist_m,
       'haversine'
FROM places_raw a JOIN places_raw b
  ON a.place_id <> b.place_id
WHERE a.business_status = 'OPERATIONAL' AND b.business_status = 'OPERATIONAL'
  AND a.latitude IS NOT NULL AND b.latitude IS NOT NULL
  AND 6371000 * 2 * ASIN(SQRT(
        POWER(SIN(RADIANS(b.latitude  - a.latitude ) / 2), 2) +
        COS(RADIANS(a.latitude)) * COS(RADIANS(b.latitude)) *
        POWER(SIN(RADIANS(b.longitude - a.longitude) / 2), 2)
      )) <= 800
ON CONFLICT (src_place_id, dst_place_id, relation_type) DO UPDATE SET
  weight = EXCLUDED.weight, built_at = NOW();
```

```sql
-- SAME_NEIGHBORHOOD (using a SQL helper that mirrors compose_embedding_text_v2's
-- _neighborhood_from_address_components — see helper below)
INSERT INTO place_relations (src_place_id, dst_place_id, relation_type, source)
SELECT a.place_id, b.place_id, 'SAME_NEIGHBORHOOD', 'address_components'
FROM places_raw a JOIN places_raw b
  ON a.place_id <> b.place_id
 AND neighborhood_of(a.source_json) = neighborhood_of(b.source_json)
 AND neighborhood_of(a.source_json) <> ''
ON CONFLICT DO NOTHING;
```

The `neighborhood_of(jsonb)` helper goes in the same migration:

```sql
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
```

```sql
-- CONTAINED_IN (from source_json.containingPlaces[].id)
INSERT INTO place_relations (src_place_id, dst_place_id, relation_type, source)
SELECT p.place_id, container->>'id', 'CONTAINED_IN', 'source_json'
FROM places_raw p,
     jsonb_array_elements(COALESCE(p.source_json->'containingPlaces', '[]'::jsonb)) AS container
WHERE container->>'id' IS NOT NULL
ON CONFLICT DO NOTHING;
```

```sql
-- NEAR_LANDMARK (from source_json.addressDescriptor.landmarks[])
INSERT INTO place_relations (src_place_id, dst_place_id, relation_type, weight, metadata, source)
SELECT p.place_id,
       lm->>'placeId',
       'NEAR_LANDMARK',
       (lm->>'travelDistanceMeters')::float,
       jsonb_build_object(
         'displayName', lm->'displayName'->>'text',
         'types',       lm->'types'
       ),
       'source_json'
FROM places_raw p,
     jsonb_array_elements(COALESCE(p.source_json->'addressDescriptor'->'landmarks', '[]'::jsonb)) AS lm
WHERE lm->>'placeId' IS NOT NULL
ON CONFLICT (src_place_id, dst_place_id, relation_type) DO UPDATE SET
  weight   = EXCLUDED.weight,
  metadata = EXCLUDED.metadata,
  built_at = NOW();
```

`SIMILAR_VECTOR` is the only one that benefits from Python (top-k per row via subquery isn't ergonomic in pure SQL):

```python
def build_similar_vector_edges(conn) -> int:
    """For each place, write its top-k most similar peers (cosine ≥ SIMILAR_MIN_COS)."""
    sql = """
    WITH peers AS (
      SELECT a.place_id AS src,
             b.place_id AS dst,
             1 - (a.embedding <=> b.embedding) AS cos,
             ROW_NUMBER() OVER (PARTITION BY a.place_id
                                ORDER BY a.embedding <=> b.embedding) AS rn
      FROM place_embeddings_v2 a
      JOIN place_embeddings_v2 b ON b.place_id <> a.place_id
    )
    INSERT INTO place_relations (src_place_id, dst_place_id, relation_type, weight, source)
    SELECT src, dst, 'SIMILAR_VECTOR', cos, 'vector_topk'
    FROM peers
    WHERE rn <= %s AND cos >= %s
    ON CONFLICT (src_place_id, dst_place_id, relation_type) DO UPDATE SET
      weight = EXCLUDED.weight, built_at = NOW()
    """
    with conn.cursor() as cur:
        cur.execute(sql, (SIMILAR_TOPK, SIMILAR_MIN_COS))
        return cur.rowcount
```

### Modify: `app/tools/__init__.py` and new `app/tools/graph.py`

```python
"""Graph-traversal tool. Returns related places by relation_type. Joins through
place_documents (W1) so the agent gets the same shape it gets from
semantic_search / nearby — no new shape to learn."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel
from app.tools.retrieval import PlaceHit


class RelatedPlace(PlaceHit):
    relation_type: str
    weight: Optional[float] = None
    relation_metadata: dict = {}


VALID_RELATIONS = {"NEAR", "SAME_NEIGHBORHOOD", "CONTAINED_IN",
                   "NEAR_LANDMARK", "SIMILAR_VECTOR"}


def kg_traverse(
    place_id: str,
    relation_type: str = "SIMILAR_VECTOR",
    k: int = 5,
) -> list[RelatedPlace]:
    """Return places related to `place_id` by `relation_type`, joined with
    place_documents so the agent gets enough metadata to render and filter.

    Edges where dst_place_id is not in places_raw (e.g. some NEAR_LANDMARK
    targets) are silently dropped — the agent only sees results it can use.
    """
    if relation_type not in VALID_RELATIONS:
        raise ValueError(f"Unknown relation_type: {relation_type}")
    sql = """
    SELECT pd.place_id, pd.name, pd.primary_type, pd.formatted_address,
           pd.latitude, pd.longitude, pd.rating, pd.price_level,
           pd.business_status, pd.source,
           0.0 AS similarity,
           LEFT(pd.embedding_text, 400) AS snippet,
           r.relation_type, r.weight, r.metadata
    FROM place_relations r
    JOIN place_documents pd ON pd.place_id = r.dst_place_id
    WHERE r.src_place_id = %s AND r.relation_type = %s
    ORDER BY
      CASE r.relation_type
        WHEN 'NEAR'           THEN  r.weight   -- ascending by distance
        WHEN 'SIMILAR_VECTOR' THEN -r.weight   -- descending by cosine
        ELSE 0
      END
    LIMIT %s
    """
    from app.tools.retrieval import _execute, _row_to_hit  # reuse W1 helpers
    rows = _execute(sql, [place_id, relation_type, k])
    return [RelatedPlace(**_row_to_hit(r)) for r in rows]
```

### Modify: `app/agent/tools.py` (replace the stub)

```python
def kg_traverse(
    ctx: RunContext[None],
    place_id: str,
    relation_type: str = "SIMILAR_VECTOR",
    k: int = 5,
) -> list[RelatedPlace]:
    """Traverse the knowledge graph from `place_id`.

    Use this when:
      - the user wants 'more like this' (SIMILAR_VECTOR);
      - you need a stop in the same neighborhood (SAME_NEIGHBORHOOD);
      - you want anchor points near a landmark (NEAR_LANDMARK);
      - you need a stop physically close to the previous one without re-running
        a full geographic query (NEAR).
    """
    from app.tools.graph import kg_traverse as _kg_traverse
    return _kg_traverse(place_id=place_id, relation_type=relation_type, k=k)
```

`SYSTEM_PROMPT` in W2's `app/agent/prompts.py` gets a small addition: a paragraph telling the agent which `relation_type` to pick when. Add to W2's planning loop docs that `kg_traverse(stop_K, relation_type="NEAR")` is sometimes a cheaper substitute for `nearby(stop_K, ...)` once the graph is dense.

### Modify: `scripts/log_model_to_mlflow.py`

Add `kg_enabled: bool` to the logged params (default True after this PR ships). Documented in W6's param expansion list.

### Modify: `Makefile`

```makefile
build-relations:
	python scripts/build_place_relations.py
```

## Tests

### New: `tests/unit/test_kg_traverse.py`

Tests the SQL template and result shape with a fake cursor (same pattern as `tests/unit/test_retriever.py`). Confirms:

- Unknown `relation_type` raises `ValueError`.
- `NEAR` ordering is ascending by `weight` (distance).
- `SIMILAR_VECTOR` ordering is descending by `weight` (cosine).
- Edges with `dst_place_id` not present in `place_documents` are dropped (the JOIN handles this; verify via fixture).
- `RelatedPlace` includes `relation_type` and `weight` fields.

### New: `tests/integration/test_build_place_relations.py` (gated)

Against a 10-place fixture:

- Runs the full builder.
- Confirms `NEAR` is symmetric (every (a→b) has a (b→a) within tolerance).
- Confirms `CONTAINED_IN` populates from `source_json.containingPlaces[]`.
- Confirms `SIMILAR_VECTOR` weight is in [0,1] and edges are above threshold.
- Confirms re-running is idempotent (no row count growth on second call).

### Manual verification

```bash
psql "$DATABASE_URL" -f scripts/db/migrations/002_place_relations.sql
make embed-v2          # ensure place_embeddings_v2 is current
make build-relations   # ~minutes for ~5,800 places

psql "$DATABASE_URL" -c "
  SELECT relation_type, COUNT(*) AS edges, AVG(weight) AS avg_weight
  FROM place_relations GROUP BY relation_type ORDER BY edges DESC;
"
# Expect: NEAR (largest), SAME_NEIGHBORHOOD, SIMILAR_VECTOR, NEAR_LANDMARK, CONTAINED_IN.

# Spot-check the agent surface:
python -c "
from app.tools.graph import kg_traverse
results = kg_traverse('ChIJExYUW8Z_j4AREJB4F5tJJto', relation_type='NEAR', k=5)
for r in results:
    print(r.name, r.weight, r.formatted_address)
"
```

Expected for the STEM Kitchen example: a list of operational places within 800m, ordered by distance.

## Tracking the change in MLflow

`kg_enabled` becomes a logged param. W6's eval suite gains an `agent_strategy` axis: `vector_only`, `vector_plus_kg`. Comparison plots will show whether the KG actually helps for which query types. Critical because adding KG calls is a token-cost increase on every chat — we need data, not vibes, to justify keeping it on.

## Risks / open questions

- **Edge count blowup.** `NEAR` is `O(N²)` filtered by 800m. For 5,855 places in SF this is bounded — most places don't have 5,854 neighbors within 800m — but worth measuring on first build. If a single dense cluster (e.g. Union Square) generates pathological counts, add a per-source cap (`LIMIT 50` per `src_place_id`).
- **`NEAR_LANDMARK` destinations may not exist in `places_raw`.** Intentional: the `kg_traverse` JOIN through `place_documents` drops these silently. If we later want landmark recommendations, we'd ingest landmarks as first-class places.
- **Rebuild cadence.** `place_relations` becomes stale after ingest changes lat/lng or after a re-embed changes vectors. The builder is idempotent so a nightly cron is fine; for now make it part of a documented "after re-embed, also build-relations" checklist.
- **`SAME_NEIGHBORHOOD` quality depends on Google's `addressComponents` consistency.** Some places will have no `neighborhood` component. Edges only form for places that both have one. Audit edge counts per neighborhood after first build; if a major neighborhood is empty, fall back to address substring match for those rows.
- **`SIMILAR_VECTOR` is computed from v2 embeddings only.** If `EMBEDDING_TABLE=place_embeddings` (v1) is still selected at runtime, the graph reflects v2 similarity which the retriever doesn't use. Document this and tie KG rebuild explicitly to v2.
- **No FK on `dst_place_id`.** Permits landmark / containing-place targets that aren't in `places_raw`. The JOIN through `place_documents` filters at query time. Don't add the FK back.
- **LLM-extracted edges are deferred.** The schema is forward-compatible: a future `OPERATED_BY` builder writes with `source = 'editorial_llm'` and a confidence in `metadata`. Don't bake LLM extraction in now without a clear retrieval-quality wins to justify the cost.
- **Apache AGE was considered and ruled out** because it isn't supported on Cloud SQL for Postgres. Pure SQL edge tables are sufficient at this scale (low tens of thousands of edges).
