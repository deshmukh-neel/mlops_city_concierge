---
phase: 01-knowledge-graph-layer-place-relations-real-kg-traverse
plan: A
slug: db-and-builder
type: execute
wave: 1
depends_on: []
files_modified:
  - alembic/versions/2026_05_14_XXXX-<rev>_add_place_relations.py
  - scripts/build_place_relations.py
  - Makefile
  - tests/integration/test_build_place_relations.py
autonomous: true
requirements: [KG-01, KG-02, BLD-01, BLD-02, BLD-03, BLD-04, BLD-05, BLD-06, BLD-07, TEST-02]
must_haves:
  truths:
    - "Running `make migrate` against a fresh DB creates `place_relations` table with the spec'd columns, PK, and three indexes."
    - "`neighborhood_of(jsonb) → text` exists and returns the SF neighborhood for a sample addressComponents JSON."
    - "Running `make build-relations` against a seeded DB writes rows for all five relation_type values."
    - "Re-running the builder produces zero row growth (PK count stable)."
    - "`--only NEAR` rebuilds only NEAR edges, leaving others untouched."
  artifacts:
    - path: "alembic/versions/2026_05_14_XXXX-<rev>_add_place_relations.py"
      provides: "DDL for place_relations + indexes + CREATE OR REPLACE neighborhood_of"
      contains: "CREATE TABLE IF NOT EXISTS place_relations"
    - path: "scripts/build_place_relations.py"
      provides: "Idempotent builder for five relation types, with --only flag"
      contains: "def build_near, def build_same_neighborhood, def build_contained_in, def build_near_landmark, def build_similar_vector"
    - path: "Makefile"
      provides: "build-relations target"
      contains: "build-relations:"
    - path: "tests/integration/test_build_place_relations.py"
      provides: "10-place fixture integration test for builder + KG-01/KG-02"
      contains: "APP_ENV"
  key_links:
    - from: "scripts/build_place_relations.py"
      to: "place_relations table"
      via: "psycopg2 get_conn() with explicit commit"
      pattern: "ON CONFLICT.*DO (UPDATE|NOTHING)"
    - from: "scripts/build_place_relations.py SIMILAR_VECTOR sub-builder"
      to: "place_embeddings_v2"
      via: "window function ROW_NUMBER() OVER (PARTITION BY ... ORDER BY embedding <=> embedding)"
      pattern: "place_embeddings_v2"
---

<objective>
Ship the database layer of the W7 knowledge graph: the `place_relations` Alembic migration (including the `neighborhood_of` plpgsql helper as `CREATE OR REPLACE`), the idempotent Python builder for five edge types (NEAR, SAME_NEIGHBORHOOD, CONTAINED_IN, NEAR_LANDMARK, SIMILAR_VECTOR), the `make build-relations` target, and the gated integration test on a 10-place fixture.

Purpose: This plan is the bottom of the W7 stack — the tool layer in Plan B depends on the table existing and being populated. Splitting allows the table + builder to land first (testable via integration) and the tool wiring to land second (testable via unit + smoke + functional).

Output: Migration applied, builder script working, Makefile target wired, integration test green under `APP_ENV=integration`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-knowledge-graph-layer-place-relations-real-kg-traverse/01-RESEARCH.md
@.planning/phases/01-knowledge-graph-layer-place-relations-real-kg-traverse/01-CONTEXT.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@CLAUDE.md
@implementation_plan/james/w7_knowledge_graph.md
@alembic/versions/2026_05_06_1356-5187c6b09b25_create_place_embeddings_v2.py
@alembic/versions/2026_05_06_1521-4c4789a14f8f_create_place_documents_view.py
@scripts/embed_places_pgvector_v2.py
@app/db.py
@Makefile

<interfaces>
<!-- The migration head this revision must descend from, verified in
     alembic/versions/2026_05_08_1100-a1b2c3d4e5f6_grant_ci_sa_proposals.py: -->
down_revision = "a1b2c3d4e5f6"

<!-- get_conn signature (from app/db.py): pooled connection, NO autocommit.
     The builder must use `with conn:` (psycopg2 auto-commits on exit) or
     explicit `conn.commit()` after each sub-builder. -->
from app.db import get_conn  # context manager, returns psycopg2 connection

<!-- Spec constants — copy verbatim into builder: -->
NEAR_RADIUS_M = 800
SIMILAR_TOPK = 10
SIMILAR_MIN_COS = 0.65
</interfaces>
</context>

## Requirement Coverage

| Requirement | Task |
|-------------|------|
| KG-01 (place_relations table + indexes) | A1 |
| KG-02 (neighborhood_of helper) | A1 |
| BLD-01 (idempotent + --only flag) | A2, A3 |
| BLD-02 (NEAR haversine ≤ 800m) | A2 |
| BLD-03 (SAME_NEIGHBORHOOD via neighborhood_of) | A2 |
| BLD-04 (CONTAINED_IN from source_json) | A2 |
| BLD-05 (NEAR_LANDMARK from addressDescriptor) | A2 |
| BLD-06 (SIMILAR_VECTOR top-K window) | A2 |
| BLD-07 (Makefile target) | A3 |
| TEST-02 (integration test, 10-place fixture) | A4 |

## Open Questions (locked decisions)

These three deviations from the W7 spec are **locked** per the planning brief; surfacing them here per CONTEXT.md instructions for the verifier:

1. **`_view_name()` over hard-coded `place_documents`** — addressed in Plan B (tool layer), not this plan.
2. **`RelatedPlace(**row)` direct construction (no `_row_to_hit` helper)** — addressed in Plan B.
3. **Four-layer test convention** — this plan covers the integration layer only; Plan B covers unit + smoke + functional for the tool.

Plan-specific deviations from spec:
- Spec's `scripts/db/migrations/002_place_relations.sql` filename is **informational only**. Per CONTEXT.md `<decisions>` and CLAUDE.md, we wrap as an Alembic revision created via `make migration MSG="add place_relations"`.
- `neighborhood_of()` already exists from W1 (`4c4789a14f8f`). New migration re-issues a byte-for-byte identical `CREATE OR REPLACE FUNCTION` body with an inline comment "duplicates W1; keep in sync." `downgrade()` does **not** drop it.

<tasks>

<task type="auto">
  <name>Task A1: Alembic migration for place_relations + neighborhood_of</name>
  <files>alembic/versions/2026_05_14_*-<rev>_add_place_relations.py</files>
  <action>
    Run `make migration MSG="add place_relations"` to scaffold the revision file. Edit the generated file so:

    - `down_revision = "a1b2c3d4e5f6"` (current head — verify with `alembic current` or by reading the latest file in `alembic/versions/`).
    - `upgrade()` issues four `op.execute(...)` blocks (use the skeleton in RESEARCH.md "Code Examples" verbatim):
      1. `CREATE TABLE IF NOT EXISTS place_relations (...)` with columns per KG-01: `src_place_id TEXT NOT NULL REFERENCES places_raw(place_id) ON DELETE CASCADE`, `dst_place_id TEXT NOT NULL` (NO FK), `relation_type TEXT NOT NULL`, `weight DOUBLE PRECISION`, `metadata JSONB DEFAULT '{}'`, `source TEXT NOT NULL`, `built_at TIMESTAMPTZ NOT NULL DEFAULT NOW()`, PRIMARY KEY (src_place_id, dst_place_id, relation_type).
      2. `CREATE INDEX IF NOT EXISTS idx_place_relations_src ON place_relations(src_place_id, relation_type);`
      3. `CREATE INDEX IF NOT EXISTS idx_place_relations_dst ON place_relations(dst_place_id, relation_type);` and `CREATE INDEX IF NOT EXISTS idx_place_relations_type ON place_relations(relation_type);`
      4. `CREATE OR REPLACE FUNCTION neighborhood_of(source_json JSONB) RETURNS TEXT AS $$ ... $$ LANGUAGE plpgsql IMMUTABLE;` — body byte-for-byte identical to W1's migration `2026_05_06_1521-4c4789a14f8f_create_place_documents_view.py:35-50`. Prefix with a Python comment: `# duplicates W1's neighborhood_of body; keep in sync.`

    - `downgrade()` issues only `op.execute("DROP TABLE IF EXISTS place_relations CASCADE")`. Do NOT drop `neighborhood_of` — it belongs to W1.

    Do NOT run `ruff format` manually; pre-commit handles it.
  </action>
  <verify>
    <automated>make migrate && psql "$DATABASE_URL" -c "SELECT 1 FROM information_schema.tables WHERE table_name='place_relations'" | grep -q '1 row' && psql "$DATABASE_URL" -c "SELECT neighborhood_of('{\"addressComponents\":[{\"types\":[\"neighborhood\"],\"longText\":\"Mission\"}]}'::jsonb)" | grep -q Mission</automated>
  </verify>
  <done>
    Migration file exists, `make migrate` applies cleanly with no errors. `\d place_relations` in psql shows seven columns, PK on (src, dst, type), three indexes. `SELECT neighborhood_of(...)` returns the expected SF neighborhood text. `make migrate` is a no-op on the second invocation.
  </done>
</task>

<task type="auto">
  <name>Task A2: Builder script — five sub-builders</name>
  <files>scripts/build_place_relations.py</files>
  <action>
    Create the builder script implementing the spec at `implementation_plan/james/w7_knowledge_graph.md` (sub-builders section). Structure:

    - `argparse` with `--only RELATION_TYPE[,...]` (comma-separated list of relation types; default = all five).
    - Module constants `NEAR_RADIUS_M = 800`, `SIMILAR_TOPK = 10`, `SIMILAR_MIN_COS = 0.65`.
    - `main()` opens a single connection via `from app.db import get_conn` and `with get_conn() as conn:` (this auto-commits on exit per psycopg2 context manager semantics — confirm by reading `app/db.py`). For each selected sub-builder, call it with the same cursor or open a fresh cursor per call; call `conn.commit()` explicitly between sub-builders to scope failure isolation.
    - Five sub-builder functions, each accepting `(conn)` and returning `int` (rows affected). Copy SQL templates verbatim from the W7 spec:
      - `build_near(conn)` — SQL INSERT with self-join on `places_raw`, haversine ≤ 800m, both directions (src→dst AND dst→src), `business_status='OPERATIONAL'`, non-null lat/lng, `src_place_id < dst_place_id` in the JOIN then UNION ALL the reverse; `weight = haversine_meters`, `source = 'haversine'`, `ON CONFLICT (src_place_id, dst_place_id, relation_type) DO UPDATE SET weight = EXCLUDED.weight, built_at = NOW()`.
      - `build_same_neighborhood(conn)` — INSERT from `places_raw a JOIN places_raw b ON neighborhood_of(a.source_json) = neighborhood_of(b.source_json) AND neighborhood_of(a.source_json) <> '' AND a.place_id <> b.place_id`, `source = 'address_components'`, `ON CONFLICT DO NOTHING`.
      - `build_contained_in(conn)` — INSERT from `places_raw, jsonb_array_elements(source_json->'containingPlaces')`, dst = element->>'id', `source = 'source_json'`, `ON CONFLICT DO NOTHING`.
      - `build_near_landmark(conn)` — INSERT from `places_raw, jsonb_array_elements(source_json->'addressDescriptor'->'landmarks')` with `dst_place_id = element->>'placeId'`, `weight = (element->>'travelDistanceMeters')::double precision`, `metadata = jsonb_build_object('displayName', element->'displayName'->>'text', 'types', element->'types')`, `source = 'source_json'`, `ON CONFLICT DO UPDATE SET weight=EXCLUDED.weight, metadata=EXCLUDED.metadata, built_at=NOW()`.
      - `build_similar_vector(conn)` — Use a CTE: `WITH ranked AS (SELECT a.place_id AS src, b.place_id AS dst, 1 - (a.embedding <=> b.embedding) AS cos, ROW_NUMBER() OVER (PARTITION BY a.place_id ORDER BY a.embedding <=> b.embedding) AS rn FROM place_embeddings_v2 a JOIN place_embeddings_v2 b ON a.place_id <> b.place_id) INSERT ... SELECT src, dst, 'SIMILAR_VECTOR', cos, '{}'::jsonb, 'vector_topk' FROM ranked WHERE rn <= 10 AND cos >= 0.65 ON CONFLICT DO UPDATE SET weight = EXCLUDED.weight, built_at = NOW()`. Reads `place_embeddings_v2` ONLY (NOT `place_embeddings` v1). Add a docstring noting this v2-only behavior (RESEARCH.md Pitfall 5).
    - Per RESEARCH.md Pitfall 7: confirm `with get_conn() as conn:` commits; if reading `app/db.py` shows otherwise, add explicit `conn.commit()`.
    - Print a one-line summary per sub-builder: `f"{relation_type}: {n} rows"`.
    - Add a prereq guard: `SELECT COUNT(*) FROM place_embeddings_v2`; if zero and SIMILAR_VECTOR is in the selected set, print a warning ("place_embeddings_v2 is empty; SIMILAR_VECTOR will produce 0 rows. Run `make embed-v2` first.") but do not fail.
    - **Do NOT** pre-emptively cap NEAR per source (CONTEXT.md `<specifics>` — only add if measured to be a problem).
    - Every f-stringed SQL must have `# noqa: S608` with a justification comment, or use parameterized SQL where possible. Constants like `NEAR_RADIUS_M` should be `%s`-parameterized, not f-stringed.
  </action>
  <verify>
    <automated>python -c "from scripts.build_place_relations import build_near, build_same_neighborhood, build_contained_in, build_near_landmark, build_similar_vector, main; import argparse; print('imports ok')" && python scripts/build_place_relations.py --only NEAR --help 2>&1 | grep -q -- "--only"</automated>
  </verify>
  <done>
    Script imports cleanly. `--only` flag parses. All five sub-builder functions exist. Module constants match spec. Pre-commit (ruff) passes.
  </done>
</task>

<task type="auto">
  <name>Task A3: Makefile target + commit hygiene</name>
  <files>Makefile</files>
  <action>
    Append a `build-relations` target to the Makefile that invokes `python scripts/build_place_relations.py "$$@"` (passes through args so `make build-relations -- --only NEAR` works). Follow the existing target style in the file (look at `make ingest` or `make log-mlflow` for the conventional one-line shell pattern). Add a `## description` comment line so `make help` (if present) picks it up.

    Order it near `make ingest` / `make log-mlflow` since those are operator scripts of the same shape.
  </action>
  <verify>
    <automated>grep -q "^build-relations:" Makefile && make -n build-relations | grep -q "python scripts/build_place_relations.py"</automated>
  </verify>
  <done>
    `grep -q '^build-relations:' Makefile` succeeds. `make -n build-relations` prints the python invocation. Target sits next to other operator-script targets.
  </done>
</task>

<task type="auto">
  <name>Task A4: Integration test on 10-place fixture (TEST-02)</name>
  <files>tests/integration/test_build_place_relations.py</files>
  <action>
    Create the integration test, gated by `APP_ENV=integration` per project convention (see `.planning/codebase/TESTING.md:81-90` and existing `tests/integration/` files for the gate pattern). Module header:

      import os, pytest
      pytestmark = pytest.mark.skipif(
          os.environ.get("APP_ENV") != "integration",
          reason="integration test; set APP_ENV=integration",
      )

    Fixtures (use pytest fixtures, function-scoped, that wrap the test in a transaction rolled back at the end, or use a dedicated test schema):
    - `seed_10_places` — inserts 10 rows into `places_raw` with realistic SF coordinates, neighborhoods, and a `source_json` containing `containingPlaces`, `addressDescriptor.landmarks`, and `addressComponents` with a `neighborhood` type. Three places share the same neighborhood, two have `containingPlaces`, three have `addressDescriptor.landmarks`. Five places are within 800m of each other.
    - `seed_embeddings_v2` — inserts 10 rows into `place_embeddings_v2` with deterministic vectors (e.g., random with fixed seed) so SIMILAR_VECTOR has predictable cosine ordering. At least two pairs should have cosine ≥ 0.65.

    Test cases (each calls `main()` or the individual sub-builder functions from `scripts.build_place_relations`):
    - `test_full_build`: runs the builder; assert `SELECT COUNT(*) FROM place_relations WHERE relation_type='NEAR'` > 0, same for the other four types. (BLD-02..BLD-05, BLD-06)
    - `test_near_symmetric_within_tolerance`: for every (src, dst) NEAR row, assert (dst, src) NEAR row exists with weight equal within 1e-6 (haversine is symmetric).
    - `test_contained_in_populates_from_source_json`: assert specific (src, dst) edges exist matching the fixture's `containingPlaces[].id` values.
    - `test_similar_vector_threshold`: assert every SIMILAR_VECTOR row has `weight ∈ [0.65, 1.0]`. (BLD-06)
    - `test_idempotent`: run builder twice; assert `SELECT COUNT(*) FROM place_relations` is identical between runs (assert on row count, NOT on `built_at` — RESEARCH.md Pitfall 6). (BLD-01)
    - `test_only_flag_subset`: run `main(["--only", "NEAR"])` against a DB pre-seeded with other relation types; assert non-NEAR counts are unchanged. (BLD-01)
    - `test_neighborhood_of_helper`: call `SELECT neighborhood_of('{"addressComponents":[{"types":["neighborhood"],"longText":"Mission"}]}'::jsonb)`; assert result is `'Mission'`. (KG-02)
    - `test_place_relations_table_shape`: query `information_schema.columns` for table `place_relations`, assert column set matches KG-01 spec. (KG-01)

    Use `asyncio_mode = "auto"` is already configured; no `@pytest.mark.asyncio` needed. Connection setup via `app.db.get_conn`.
  </action>
  <verify>
    <automated>APP_ENV=integration pytest tests/integration/test_build_place_relations.py -v --no-cov 2>&1 | tail -30</automated>
  </verify>
  <done>
    All 8 test cases pass under `APP_ENV=integration`. Without `APP_ENV=integration`, the module is skipped (no DB connection attempted). Pre-commit ruff is green.
  </done>
</task>

</tasks>

<verification>
- `make migrate` is a no-op on the second invocation (idempotent at the migration level).
- `make build-relations` against the seeded test DB produces non-zero counts for all five relation types.
- Re-running `make build-relations` shows zero PK growth.
- `APP_ENV=integration make test-integration` passes.
- `make lint` passes (pre-commit-equivalent ruff check).
</verification>

<success_criteria>
- Migration applied; `\d place_relations` shows the spec-compliant shape.
- `scripts/build_place_relations.py --only NEAR` rebuilds only NEAR rows.
- `tests/integration/test_build_place_relations.py` passes 8/8 under the integration gate.
- Plan B can begin: the table exists and contains rows, so the tool layer's unit tests (fake-cursor) and functional tests (real DB) have something to verify against.
</success_criteria>

<output>
Each task is committed separately with a one-line message per user preference (small focused commits):
- A1: `feat(kg): add place_relations migration with neighborhood_of helper`
- A2: `feat(kg): add idempotent place_relations builder`
- A3: `feat(kg): add make build-relations target`
- A4: `test(kg): add integration tests for place_relations builder`

After all four tasks complete, create `.planning/phases/01-knowledge-graph-layer-place-relations-real-kg-traverse/01-PLAN-A-SUMMARY.md` per the gsd summary template.
</output>
