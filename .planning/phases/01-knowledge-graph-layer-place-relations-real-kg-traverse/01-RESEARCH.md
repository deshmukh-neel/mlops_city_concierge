# Phase 1: Knowledge Graph Layer - Research

**Researched:** 2026-05-14
**Domain:** Postgres-backed knowledge graph layered over pgvector, exposed as a LangChain tool to a LangGraph agent
**Confidence:** HIGH — the design is already locked in `implementation_plan/james/w7_knowledge_graph.md`; this research grounds it in the current codebase and surfaces three concrete deviations the planner must reconcile.

## Synthesis (recommended approach in 5 sentences)

Implement W7 as a **single Alembic migration** (`alembic revision --autogenerate -m "add place_relations"` then hand-edit) that creates `place_relations` and idempotently `CREATE OR REPLACE`s `neighborhood_of(jsonb)` — the function already exists from W1's `4c4789a14f8f` migration (`alembic/versions/2026_05_06_1521-4c4789a14f8f_create_place_documents_view.py:35-50`), so the spec's helper body must match W1's exactly to avoid drift. Build the relations from `scripts/build_place_relations.py` using **pure SQL for NEAR / SAME_NEIGHBORHOOD / CONTAINED_IN / NEAR_LANDMARK** (inserts copied from the W7 spec, wrapped in `with get_conn() as conn` per the project's psycopg2 convention) and **Python-orchestrated SQL for SIMILAR_VECTOR** (top-K window function over `place_embeddings_v2`). Add `app/tools/graph.py` exposing `kg_traverse(place_id, relation_type, k) → list[RelatedPlace]` that JOINs `place_relations` to **the active embeddings view selected by `_view_name()` from `app/tools/retrieval.py:56-60`** (not a hard-coded `place_documents` — this is deviation #1 from the spec, needed because the project supports v1/v2 view switching). Replace the stub at `app/agent/tools.py:76-85` with a thin wrapper, extend the agent `SYSTEM_PROMPT` at `app/agent/prompts.py:39` with relation_type guidance (the prompt already references `kg_traverse(stop_K, relation_type='NEAR')` at line 85, so this is filling in a forward reference), and add `kg_enabled: bool` to the params dict in `scripts/log_model_to_mlflow.py:194-201`. Ship the four-layer test suite (unit / smoke / functional / integration) the project's testing convention requires — the W7 spec only calls out unit + integration, so the planner needs to add smoke + functional explicitly.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Edge storage (`place_relations` table + indexes) | Database / Storage | — | Plain SQL edge table in Cloud SQL; Apache AGE not available |
| `neighborhood_of()` plpgsql helper | Database / Storage | — | Already exists from W1; reuse via `CREATE OR REPLACE` |
| Relation builder (NEAR, SAME_NEIGHBORHOOD, CONTAINED_IN, NEAR_LANDMARK) | Database / Storage | Scripts | Pure SQL inserts driven by a Python script |
| SIMILAR_VECTOR builder | Database / Storage | Scripts | SQL window function over `place_embeddings_v2`, orchestrated from Python |
| `kg_traverse` tool | API / Backend | Database | Python tool in `app/tools/graph.py` joining `place_relations` to embeddings view |
| Agent tool surface | API / Backend | — | LangChain `StructuredTool` wrapper in `app/agent/tools.py` |
| MLflow `kg_enabled` param | MLOps / Tracking | — | Added to `scripts/log_model_to_mlflow.py` params dict |
| Makefile `build-relations` target | Build / Ops | — | One-line `python scripts/build_place_relations.py` |

## Standard Stack

### Already in the codebase (no new deps)

| Library | Version | Purpose | Verified at |
|---------|---------|---------|-------------|
| psycopg2-binary | as-in-`pyproject.toml` | DB driver for builder + tool [VERIFIED: existing usage in `app/db.py`, `app/tools/retrieval.py:64`] | `.planning/codebase/STACK.md:32-37` |
| pgvector | extension already enabled | Cosine distance operator `<=>` over `place_embeddings_v2.embedding` [VERIFIED: `alembic/versions/2026_05_06_1356-5187c6b09b25_create_place_embeddings_v2.py:38-44`] | existing HNSW index |
| Alembic | `>=1.13.1,<2.0.0` | Migration tooling — current head `a1b2c3d4e5f6` | `alembic/versions/` |
| LangChain `StructuredTool` | `>=0.2.0,<1.0.0` | Tool surface for the agent | `app/agent/tools.py:13` |
| Pydantic | `>=2.7.0,<3.0.0` | `RelatedPlace(PlaceHit)` extension | `app/tools/retrieval.py:32-44` |
| MLflow | (shared GCP server) | `kg_enabled` param logging | `scripts/log_model_to_mlflow.py:194-201` |

**Installation:** none — every dependency is already pinned via Poetry.

### Alternatives Considered (and rejected)

| Instead of | Could Use | Why rejected |
|------------|-----------|--------------|
| Plain SQL edge table | Apache AGE / openCypher | [CITED: `implementation_plan/james/w7_knowledge_graph.md:20`] Cloud SQL doesn't support AGE — non-starter |
| Plain SQL edge table | NetworkX in-memory | Re-loads ~50k+ edges per request; no idempotent persistence; tool calls would be slow |
| Recursive CTE in `kg_traverse` | Single-hop SQL | [CITED: W7 spec — `kg_traverse` is single-hop by design; multi-hop is the agent's job via repeated calls] |
| Hard-coded `place_documents` | Use `_view_name()` allowlist switch | Project supports v1/v2 toggle via `EMBEDDING_TABLE` setting; spec's hard-coded `place_documents` is a deviation that breaks v2 selection [VERIFIED: `app/tools/retrieval.py:20-23, 56-60`] |

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Build-time (run on demand)                       │
│                                                                     │
│  scripts/build_place_relations.py                                   │
│       │                                                             │
│       ├─► reads ─► places_raw  (lat/lng, source_json, neighborhood) │
│       ├─► reads ─► place_embeddings_v2  (for SIMILAR_VECTOR)        │
│       │                                                             │
│       └─► writes (idempotent UPSERT) ─► place_relations             │
│                                                                     │
│  Triggered by: `make build-relations` (manual, after ingest/embed)  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       Request-time (per /chat)                      │
│                                                                     │
│  user → POST /chat → LangGraph agent loop                           │
│              │                                                      │
│              ▼                                                      │
│      app/agent/graph.py  (think → act → critique)                   │
│              │                                                      │
│              ▼  picks tool                                          │
│      app/agent/tools.py: kg_traverse (StructuredTool)               │
│              │                                                      │
│              ▼  thin wrapper                                        │
│      app/tools/graph.py: kg_traverse(place_id, relation_type, k)    │
│              │                                                      │
│              ├─► validate relation_type ∈ VALID_RELATIONS           │
│              │                                                      │
│              ▼  SQL                                                 │
│      place_relations JOIN place_documents{,_v2}                     │
│              │   (the JOIN drops dst rows not in places_raw)        │
│              ▼                                                      │
│      list[RelatedPlace]  (PlaceHit + relation_type + weight)        │
│              │                                                      │
│              ▼                                                      │
│      back to the LangGraph think node as a ToolMessage              │
└─────────────────────────────────────────────────────────────────────┘
```

### Recommended File Layout (delta only)

```
alembic/versions/
└── 2026_05_14_<rev>_add_place_relations.py   # NEW — Alembic-wrapped DDL

scripts/
└── build_place_relations.py                  # NEW — idempotent builder

app/tools/
└── graph.py                                  # NEW — kg_traverse + RelatedPlace

app/agent/
├── tools.py                                  # MODIFY — replace stub at :76-85
└── prompts.py                                # MODIFY — relation_type paragraph in SYSTEM_PROMPT

scripts/
└── log_model_to_mlflow.py                    # MODIFY — kg_enabled in params dict

Makefile                                      # MODIFY — add `build-relations` target

tests/unit/
├── test_kg_traverse.py                       # NEW — fake-cursor pattern
├── test_kg_traverse_smoke.py                 # NEW — import + construct
└── test_kg_traverse_functional.py            # NEW — multi-relation happy path

tests/integration/
└── test_build_place_relations.py             # NEW — gated by APP_ENV=integration
```

### Pattern 1: Alembic-wrapped raw SQL (DDL)

**What:** Wrap the spec's raw SQL inside `op.execute("""...""")` blocks in an Alembic revision. This is exactly how W0a wrapped its pre-existing manual SQL (`scripts/db/migrations/000_place_embeddings_v2.sql`) into `alembic/versions/2026_05_06_1356-5187c6b09b25_create_place_embeddings_v2.py`.

**When to use:** Every DDL change in this codebase. The spec's `scripts/db/migrations/002_place_relations.sql` filename is informational only — CONTEXT.md explicitly locks the Alembic-wrapped path.

**Example (from existing migration):**
```python
# Source: alembic/versions/2026_05_06_1356-5187c6b09b25_create_place_embeddings_v2.py:25-44
def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS place_embeddings_v2 ( ... );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_place_embeddings_v2_vector
            ON place_embeddings_v2
            USING hnsw (embedding vector_cosine_ops);
        """
    )
```

**Apply to W7:** Use `IF NOT EXISTS` on the table and indexes; use `CREATE OR REPLACE FUNCTION` on `neighborhood_of()` so re-applying after W1's migration is safe.

### Pattern 2: `_view_name()` allowlist switch in tools

**What:** Tools that join an embeddings table go through `_VIEW_FOR_TABLE` (`app/tools/retrieval.py:20-23`) and `_view_name()` (`:56-60`) so the active view (`place_documents` vs `place_documents_v2`) is picked from `settings.embedding_table` at call time.

**Apply to W7 (deviation from spec):** `kg_traverse`'s JOIN must use `_view_name()`, not hard-code `place_documents`. The spec's SQL at `implementation_plan/james/w7_knowledge_graph.md:254-271` JOINs `place_documents` literally; this should become `f"FROM place_relations r JOIN {view} pd ON ..."` with `# noqa: S608` per the project convention (`.planning/codebase/CONVENTIONS.md:44`).

### Pattern 3: Fake-cursor unit testing for SQL tools

**What:** Build hand-rolled `FakeCursor` + `FakeConnection` classes implementing `__enter__/__exit__/execute/fetchall`, patch `app.tools.graph.get_conn` via `@contextmanager`-decorated function, assert on `cursor.executed_sql` and `cursor.executed_params`.

**Reference implementations:**
- `tests/unit/test_tools_retrieval.py:47-91` — the closest match (uses `RealDictCursor` rows as dicts, which is what `_execute` returns at `app/tools/retrieval.py:63-66`)
- `tests/unit/test_retriever.py:10-42` — older tuple-row variant

**Apply to W7:** Follow `test_tools_retrieval.py` exactly — `_execute` returns `list[dict]`, so `RelatedPlace(**row)` works directly. The `patch_get_conn` fixture should be lifted into `tests/conftest.py` or duplicated locally; spec says copy from `test_retriever.py` but `test_tools_retrieval.py` is the cleaner parent.

### Pattern 4: LangChain `StructuredTool` from a Python function

**What:** Agent-exposed tools are plain Python functions with full type hints + a docstring; `_to_lc_tool` (`app/agent/tools.py:108-114`) wraps them via `StructuredTool.from_function` with an auto-derived args schema (`_args_schema_for`, `:88-105`).

**Apply to W7:** The replacement `kg_traverse` in `app/agent/tools.py` must keep the same signature shape as the stub (`place_id: str, relation_type: str = "...", k: int = 5`) and must be added to the `_TOOLS` list at `:117-123` (or replace the existing stub entry). **`_args_schema_for` raises on any missing type hint** (`:99-102`) — every param including `k` must be annotated.

### Anti-Patterns to Avoid

- **Hard-coding `place_documents` in the JOIN** — breaks the v1/v2 toggle. Use `_view_name()`.
- **Adding a FK on `dst_place_id`** — explicitly forbidden (`implementation_plan/james/w7_knowledge_graph.md:54, 369`). Landmark targets live outside `places_raw`.
- **Building `RelatedPlace` from `__init__` kwargs that don't match `_row_to_hit`'s shape** — `_row_to_hit` doesn't exist in the current code; the spec references it at `implementation_plan/james/w7_knowledge_graph.md:272-274` but no such helper is in `app/tools/retrieval.py`. **Deviation #2:** the planner must either (a) add `_row_to_hit` to `retrieval.py` first, or (b) construct `RelatedPlace(**row)` directly with the same dict-row pattern used by `semantic_search` (`app/tools/retrieval.py:104`). Recommendation: (b) — simpler, matches existing pattern, no new helper to test.
- **Running the builder via `asyncio` / multiple connections** — `_execute` borrows a pooled connection per call; the builder should open one `psycopg2.connect(...)` (or `get_conn()`) and run all sub-builders in the same transaction or as separate transactions, but NOT spin up parallel writers. The PK + UPSERT semantics rely on serial execution.
- **Forgetting `# noqa: S608`** on any f-stringed SQL — ruff `S` rules are enabled (`.planning/codebase/CONVENTIONS.md:39, 44`).
- **Skipping smoke + functional test layers** — project convention requires all four (`.planning/codebase/TESTING.md:58-63`); spec only mentions unit + integration.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Haversine in Python | Custom math.acos formula | The SQL form in the spec | Already vetted in `app/tools/retrieval.py:136-140`'s `nearby`; identical formula |
| Top-K-per-row similarity | Python loop over N² pairs | `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY a.embedding <=> b.embedding)` | pgvector + window function is one SQL round-trip [CITED: W7 spec :194-216] |
| Neighborhood extraction in builder | Re-implement Python `_neighborhood_from_address_components` | Call `neighborhood_of(source_json)` plpgsql helper | Already exists from W1 (`alembic/versions/2026_05_06_1521-4c4789a14f8f_create_place_documents_view.py:35-50`) and the Python source is at `scripts/embed_places_pgvector_v2.py:156-173` |
| Idempotency in builder | DELETE-then-INSERT | `ON CONFLICT … DO UPDATE / DO NOTHING` | Atomic per row; no race window; spec-required |
| Pydantic args schema for the tool | Hand-write a model | `_args_schema_for(fn)` (`app/agent/tools.py:88-105`) | Already wired into `_to_lc_tool`; derives from type hints |

**Key insight:** This phase is almost entirely **wiring existing patterns** into a new vertical. The only genuinely new code is `app/tools/graph.py` (~50 lines), the migration body (copy from spec), and the builder script (copy SQL from spec + a thin Python `argparse` wrapper). Everything else is a one-line edit (Makefile, MLflow params, agent tool list, prompt).

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `place_relations` is a **new** table — no prior data exists. `place_embeddings_v2` is read but not modified. `places_raw.source_json` is read but not modified. | None — first build is greenfield |
| Live service config | None — Cloud SQL connection params and MLflow tracking URI are unchanged | None |
| OS-registered state | None — no cron, no systemd, no Task Scheduler. `make build-relations` is operator-invoked per CONTEXT.md `## Specific Ideas` and W7 risks section (`implementation_plan/james/w7_knowledge_graph.md:366`) | None |
| Secrets/env vars | None new. Builder uses existing `DATABASE_URL` resolution path (`app.db_url.resolve_database_url`); MLflow uses existing `MLFLOW_TRACKING_URI` | None |
| Build artifacts | None — no compiled binaries, no egg-info, no Docker image baking. Cloud Run image is rebuilt on every push but doesn't include `place_relations` data | None |

**The canonical question — "after every file in the repo is updated, what runtime systems still have the old string cached, stored, or registered?"** Answer: the **`kg_traverse` stub return shape** is the only stale-cache concern. Eval datasets (W6) or any external test fixtures that asserted on `{"available": False, "reason": "..."}` will need updating, but per CONTEXT.md W6 expansion is explicitly deferred — no consumer of the stub return shape exists in this codebase today (`grep -rn 'not yet available' tests/` returns nothing; only the stub itself uses the phrase). [VERIFIED: grep of `tests/`]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Postgres + pgvector | Builder, tool, integration tests | ✓ (Cloud SQL prod; local docker) | PG 18 prod, PG 16 local docker | — |
| `place_embeddings_v2` populated | SIMILAR_VECTOR builder | ✓ assumed — W0a shipped | — | If empty, SIMILAR_VECTOR build writes 0 rows; document as prereq |
| Alembic CLI | Migration | ✓ via Poetry dev deps | `>=1.13.1,<2.0.0` | — |
| MLflow tracking server | `kg_enabled` param logging | ✓ shared GCP server `http://35.223.147.177:5000` | — | Param is also logged locally on disk via MLflow file backend if URI unreachable — `degraded boot` pattern (CONVENTIONS.md:73) |
| Cloud SQL proxy | `test-integration-cloud` only | ✓ (operator-provided) | — | Use local docker DB via `make test-integration` |

**Missing dependencies with no fallback:** none.

**Prereq the planner must call out:** `make embed-v2` must have run at least once against the target DB before `make build-relations`, otherwise SIMILAR_VECTOR yields zero edges. CONTEXT.md `## Specific Ideas` references this implicitly; PLAN.md should add an explicit prereq check or a build-time `SELECT COUNT(*) FROM place_embeddings_v2` and warn-on-zero.

## Common Pitfalls

### Pitfall 1: Drift between `neighborhood_of()` bodies
**What goes wrong:** The W7 spec gives a `CREATE OR REPLACE FUNCTION neighborhood_of(...)` body (`implementation_plan/james/w7_knowledge_graph.md:142-158`). W1 already shipped an identical body (`alembic/versions/2026_05_06_1521-4c4789a14f8f_create_place_documents_view.py:35-50`). If the new migration copies the spec verbatim, that's fine — but if anyone "improves" one and not the other, queries silently use a different definition.
**Why it happens:** Same function defined in two migration files.
**How to avoid:** The new migration should `CREATE OR REPLACE FUNCTION` with a body **byte-for-byte identical** to W1's. Add an inline comment in the new migration noting "duplicates W1; keep in sync." Better: the new migration could omit the function entirely since W1 already created it, but the spec calls for including it. Default: include with the comment.
**Warning signs:** Diff between the two SQL strings at code review time.

### Pitfall 2: Hard-coded `place_documents` breaks v2-mode
**What goes wrong:** The spec's `kg_traverse` SQL JOINs `place_documents` literally. When `settings.embedding_table == 'place_embeddings_v2'`, the retriever uses `place_documents_v2` but `kg_traverse` would use the v1 view — inconsistent metadata between tools.
**Why it happens:** Spec was written assuming a single view.
**How to avoid:** Use `_view_name()` like every other tool in `app/tools/retrieval.py`. Mark deviation #1 in PLAN.md Open Questions for the planner to confirm.
**Warning signs:** Tests that hard-code "place_documents" in expected SQL — assert on the resolved view, not the literal.

### Pitfall 3: `RelatedPlace(**_row_to_hit(r))` references a helper that doesn't exist
**What goes wrong:** Spec calls `_row_to_hit` (`implementation_plan/james/w7_knowledge_graph.md:274`). `grep -n "_row_to_hit" app/` returns nothing — it's not in `app/tools/retrieval.py`. Tasks that copy the spec verbatim will fail import.
**Why it happens:** Spec drift from earlier W1 design.
**How to avoid:** Either (a) add `_row_to_hit(row: dict) -> dict` as a no-op identity helper in `retrieval.py` for symmetry, or (b) construct `RelatedPlace(**row)` directly. Recommendation: (b) — `_execute` already returns dicts (`app/tools/retrieval.py:63-66`).
**Warning signs:** ImportError on first run; planner sees `_row_to_hit` referenced and asks where it lives.

### Pitfall 4: NEAR edge count blow-up
**What goes wrong:** O(N²) filtered by 800m. For ~5,800 SF places, dense clusters (Union Square, Mission, Hayes Valley) could produce hot spots with hundreds of edges per row.
**Why it happens:** Inherent to all-pairs haversine.
**How to avoid:** Build first, measure (`SELECT src_place_id, COUNT(*) FROM place_relations WHERE relation_type='NEAR' GROUP BY 1 ORDER BY 2 DESC LIMIT 20;`). Only add per-source `LIMIT 50` if a row >> 100. CONTEXT.md `## Specific Ideas` explicitly says **don't** pre-emptively cap.
**Warning signs:** Builder takes >>10min, or `pg_stat_statements` shows the NEAR insert dominating.

### Pitfall 5: SIMILAR_VECTOR is computed from v2 only; runtime may use v1
**What goes wrong:** Builder reads `place_embeddings_v2`. If `EMBEDDING_TABLE=place_embeddings` at runtime, retriever uses v1 but KG reflects v2 similarity — they don't agree.
**Why it happens:** Spec decision to tie KG to v2 only.
**How to avoid:** Document explicitly in the builder docstring and in PLAN.md. Don't gate the build on `settings.embedding_table` — the build always uses v2.
**Warning signs:** Eval (W6) shows KG SIMILAR_VECTOR results that don't match the agent's own semantic_search hits.

### Pitfall 6: Idempotency footgun — `ON CONFLICT DO UPDATE` updates `built_at`, mutating timestamps every run
**What goes wrong:** "Idempotent" usually means "no row growth." But `DO UPDATE SET ..., built_at = NOW()` rewrites the timestamp on every re-run, so the integration test's "no row count growth" assertion passes but the table changes. That's the spec's intent (built_at = last touched), but the integration test must assert on **row count**, not on table-level checksums.
**Why it happens:** Subtle semantic difference between idempotent-rows and idempotent-state.
**How to avoid:** Test the explicit assertion: `count(*)` after run 1 == `count(*)` after run 2. Don't test `built_at` equality between runs.
**Warning signs:** Test failure with "built_at changed" — that's a bad test, not a real failure.

### Pitfall 7: `psycopg2` doesn't auto-commit DDL/DML — builder must commit
**What goes wrong:** `get_conn()` in `app/db.py` likely uses pooled connections without autocommit. Builder script forgets `conn.commit()` and nothing persists.
**Why it happens:** Common psycopg2 trap.
**How to avoid:** Either use `with conn:` (psycopg2's context manager auto-commits on exit) or call `conn.commit()` explicitly after each sub-builder. Check `app/db.py` `get_conn` semantics first.
**Warning signs:** Builder reports row counts but `SELECT COUNT(*) FROM place_relations` shows 0.

## Code Examples

### Alembic migration skeleton (matches existing convention)
```python
# Source pattern: alembic/versions/2026_05_06_1356-5187c6b09b25_create_place_embeddings_v2.py
"""add place_relations

Revision ID: <new>
Revises: a1b2c3d4e5f6
Create Date: 2026-05-14 ...
"""
from collections.abc import Sequence
from alembic import op

revision: str = "<new>"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS place_relations (
            src_place_id    TEXT NOT NULL REFERENCES places_raw(place_id) ON DELETE CASCADE,
            dst_place_id    TEXT NOT NULL,
            relation_type   TEXT NOT NULL,
            weight          DOUBLE PRECISION,
            metadata        JSONB DEFAULT '{}',
            source          TEXT NOT NULL,
            built_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (src_place_id, dst_place_id, relation_type)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_place_relations_src ON place_relations(src_place_id, relation_type);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_place_relations_dst ON place_relations(dst_place_id, relation_type);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_place_relations_type ON place_relations(relation_type);")
    # neighborhood_of is already created by W1's migration 4c4789a14f8f.
    # Re-issue CREATE OR REPLACE here with an identical body so the W7 migration
    # is self-contained per spec. Keep these two bodies in sync.
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


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS place_relations CASCADE")
    # Do NOT drop neighborhood_of — it belongs to W1.
```

### `app/tools/graph.py` (recommended shape — view-aware)
```python
"""Graph-traversal tool. Returns related places by relation_type."""
from __future__ import annotations
from typing import Optional
from app.tools.retrieval import PlaceHit, _execute, _view_name


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
    if relation_type not in VALID_RELATIONS:
        raise ValueError(f"Unknown relation_type: {relation_type}")
    view = _view_name()  # validated allowlist member — safe to f-string
    sql = f"""
        SELECT pd.place_id, pd.name, pd.primary_type, pd.formatted_address,
               pd.latitude, pd.longitude, pd.rating, pd.price_level,
               pd.business_status, pd.source,
               0.0 AS similarity,
               LEFT(pd.embedding_text, 400) AS snippet,
               r.relation_type, r.weight,
               r.metadata AS relation_metadata
        FROM place_relations r
        JOIN {view} pd ON pd.place_id = r.dst_place_id
        WHERE r.src_place_id = %s AND r.relation_type = %s
        ORDER BY
          CASE r.relation_type
            WHEN 'NEAR'           THEN  r.weight
            WHEN 'SIMILAR_VECTOR' THEN -r.weight
            ELSE 0
          END
        LIMIT %s
    """  # noqa: S608
    rows = _execute(sql, [place_id, relation_type, k])
    return [RelatedPlace(**row) for row in rows]
```

Note: SELECT aliases `r.metadata AS relation_metadata` so the dict key matches the Pydantic field name.

### Replacement at `app/agent/tools.py:76-85`
```python
def kg_traverse(
    place_id: str,
    relation_type: str = "SIMILAR_VECTOR",
    k: int = 5,
) -> list[RelatedPlace]:
    """Traverse the knowledge graph from `place_id`.

    Use this when:
      - the user wants 'more like this' (SIMILAR_VECTOR);
      - you need a stop in the same neighborhood (SAME_NEIGHBORHOOD);
      - anchor points near a landmark (NEAR_LANDMARK);
      - geographic neighbors without a fresh `nearby` call (NEAR).
    """
    from app.tools.graph import kg_traverse as _kg_traverse
    return _kg_traverse(place_id=place_id, relation_type=relation_type, k=k)
```

Then add `RelatedPlace` to the import block at the top of `app/agent/tools.py:17-29`. The `_TOOLS` entry at `:122` stays the same name; only the underlying function changes.

### MLflow param addition at `scripts/log_model_to_mlflow.py:194-201`
```python
params: dict[str, str | int | float | bool] = {
    "llm_provider": llm_provider,
    "chat_model": resolved_chat_model,
    "k": k,
    "temperature": temperature,
    "embedding_model": settings.openai_embedding_model,
    "vector_store": "pgvector",
    "kg_enabled": True,  # W7: default True after this PR; W6 evals A/B via False
}
```

Note: the type annotation widens to include `bool` so mypy passes. Verify `mlflow.log_params` accepts bools (it does — they're logged as strings).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `kg_traverse` stub returning `{"available": False, ...}` | Real implementation backed by `place_relations` | This phase | Agent gains 5 new useful relation types |
| Hard-coded `place_documents` view in spec | View-aware via `_view_name()` | Mid-W7 (this research surfaces it) | Honors v1/v2 toggle |
| Two separate `neighborhood_of` definitions (planned by spec) | Reuse W1's via `CREATE OR REPLACE` | This research | Single source of truth |

**Deprecated/outdated in spec relative to current code:**
- Spec's `_row_to_hit(r)` helper reference — doesn't exist; use `RelatedPlace(**row)` directly.
- Spec's filename `scripts/db/migrations/002_place_relations.sql` — informational only; Alembic-wrapped per CONTEXT.md.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `_row_to_hit` does not exist in `app/tools/retrieval.py` and the spec's reference to it is stale | Pitfall 3, Code Examples | LOW — verified by grep; planner just needs to not copy spec verbatim |
| A2 | `place_embeddings_v2` is populated in the target environments where `make build-relations` runs | Environment Availability | MEDIUM — if empty, SIMILAR_VECTOR build silently writes 0 rows; manifests as missing edges, not an error |
| A3 | The W7 spec's intent for `kg_traverse` JOIN is to drop dst rows missing from the embeddings view (silent filter), and using v2's view in v2-mode aligns with that intent | Pattern 2 | LOW — the JOIN semantics are the same regardless of which view; v1/v2 differ only on which embeddings table they include |
| A4 | `get_conn()` in `app/db.py` does NOT auto-commit, so the builder must call `conn.commit()` explicitly or use a `with conn:` block | Pitfall 7 | LOW — easy to verify by reading `app/db.py`; default psycopg2 behavior is no autocommit |
| A5 | The four-layer test convention applies — smoke + functional for `kg_traverse` even though the W7 spec only lists unit + integration | Anti-Patterns, Recommended File Layout | LOW — `.planning/codebase/TESTING.md:58-63` is unambiguous; risk is just extra test files |

## Project Constraints (from CLAUDE.md)

- Python 3.10+; ruff `E,F,I,N,UP,B,SIM,S` (line-length 100). New SQL f-strings need `# noqa: S608` with a justification comment.
- Tests use pytest with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio`.
- Integration tests gated by `APP_ENV=integration`; module header `pytestmark = pytest.mark.skipif(...)` per `.planning/codebase/TESTING.md:81-90`.
- Alembic is the migration tool; create with `make migration MSG="..."`; apply with `make migrate`.
- Pre-commit auto-runs ruff; do **not** run `ruff format` manually before committing (memory note).
- Small focused commits with single-line messages (memory note).
- User merges PRs themselves — agents don't run `gh pr merge` (memory note).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest `>=8.2,<9` |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]` lines 61-64) |
| Quick run command | `pytest tests/unit/test_kg_traverse.py -v` |
| Full suite command | `make test` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| KG-01 | `place_relations` table + indexes exist | integration | `APP_ENV=integration pytest tests/integration/test_build_place_relations.py -v` | ❌ Wave 0 |
| KG-02 | `neighborhood_of()` returns SF neighborhood for sample `source_json` | integration | same file, fixture-based | ❌ Wave 0 (covered by W1 today but assert in our test for explicitness) |
| BLD-01 | Re-run produces zero row growth | integration | `test_build_place_relations.py::test_idempotent` | ❌ Wave 0 |
| BLD-01 | `--only NEAR` rebuilds subset | unit | `pytest tests/unit/test_build_place_relations_smoke.py::test_only_flag_parses` | ❌ Wave 0 |
| BLD-02..BLD-05 | Each sub-builder populates expected rows on 10-place fixture | integration | `test_build_place_relations.py::test_full_build` | ❌ Wave 0 |
| BLD-06 | SIMILAR_VECTOR weight ∈ [0,1], threshold respected | integration | `test_build_place_relations.py::test_similar_vector_threshold` | ❌ Wave 0 |
| BLD-07 | `make build-relations` exists | smoke | `grep -q 'build-relations' Makefile` | ❌ Wave 0 |
| TOOL-01 | `kg_traverse` JOINs view, drops missing dst | unit | `pytest tests/unit/test_kg_traverse.py::test_join_drops_missing_dst -x` | ❌ Wave 0 |
| TOOL-02 | `RelatedPlace` has `relation_type`, `weight`, `relation_metadata` | unit | `pytest tests/unit/test_kg_traverse.py::test_related_place_shape` | ❌ Wave 0 |
| TOOL-03 | NEAR ascending, SIMILAR_VECTOR descending | unit | `pytest tests/unit/test_kg_traverse.py::test_ordering_clause` | ❌ Wave 0 |
| TOOL-04 | Unknown relation_type → ValueError | unit | `pytest tests/unit/test_kg_traverse.py::test_invalid_relation_raises` | ❌ Wave 0 |
| TOOL-05 | Agent tool wrapper delegates | unit/smoke | `pytest tests/unit/test_kg_traverse_smoke.py` + existing `test_agent_smoke.py` | ❌ Wave 0 |
| TOOL-06 | `SYSTEM_PROMPT` mentions relation_type guidance | unit | `pytest tests/unit/test_agent_prompts.py` (existing — extend) | partial — file exists, add assertion |
| MLOPS-01 | `kg_enabled` in logged params | unit | `pytest tests/unit/test_mlflow_logging.py` (existing — extend) | partial — file exists, add assertion |
| TEST-01 | All unit tests above pass | unit | `make test-unit` | ❌ Wave 0 |
| TEST-02 | Integration suite passes against fixture | integration | `APP_ENV=integration make test-integration` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/unit/test_kg_traverse.py -x` (≈ <5s)
- **Per wave merge:** `make test-unit && make lint` (≈ <30s)
- **Phase gate:** `make test` green; `APP_ENV=integration make test-integration` green against a seeded fixture DB.

### Wave 0 Gaps
- [ ] `tests/unit/test_kg_traverse.py` — TOOL-01..TOOL-04
- [ ] `tests/unit/test_kg_traverse_smoke.py` — import + `RelatedPlace` constructs + `kg_traverse` is registered in `_TOOLS`
- [ ] `tests/unit/test_kg_traverse_functional.py` — multi-relation end-to-end with fake-cursor (`NEAR` then `SIMILAR_VECTOR` ordering both verified in one flow)
- [ ] `tests/integration/test_build_place_relations.py` — gated by APP_ENV=integration; 10-place fixture
- [ ] Extend `tests/unit/test_agent_prompts.py` — assert SYSTEM_PROMPT contains "relation_type" guidance paragraph
- [ ] Extend `tests/unit/test_mlflow_logging.py` — assert `kg_enabled` is in logged params dict

*(Framework install: not needed — pytest already configured.)*

## Open Questions for the Planner

These are decisions CONTEXT.md leaves to Claude's discretion or that this research has surfaced as needing an explicit call **before** PLAN.md is written. Surface them in PLAN.md `## Open Questions` and either lock them with the user during planning or annotate the chosen default.

1. **Use `_view_name()` or hard-code `place_documents` in `kg_traverse` SQL?** Recommendation: `_view_name()`. Spec says `place_documents` literally; current codebase uses the view-switching pattern everywhere else. This is the most important deviation from the spec; flagging up-front per CONTEXT.md instructions.
2. **Construct `RelatedPlace(**row)` directly, or add a `_row_to_hit` helper?** Recommendation: direct construction. Spec references `_row_to_hit` but the helper doesn't exist in `app/tools/retrieval.py`. Adding a no-op helper just to match the spec is over-engineering.
3. **Single PLAN.md or split into 4?** CONTEXT.md suggests an optional 4-way split (migration, builder, tool+agent+prompt, MLflow+tests). Recommendation: **two plans** — (A) DB layer (migration + builder + Makefile + integration tests) and (B) Tool layer (graph.py + agent wrapper + prompt + MLflow + unit/smoke/functional tests). A and B are independently testable; B depends on A only for the integration tests. Avoids the 4-way coordination tax without losing parallelism.
4. **One Alembic migration, or two (table first, then `CREATE OR REPLACE FUNCTION` second)?** Recommendation: one. The function is idempotent via `CREATE OR REPLACE`; no benefit to splitting.
5. **Should the builder log to MLflow?** (KG edge count per type, build duration.) The W7 spec doesn't ask for this and CONTEXT.md doesn't either. **Recommendation: defer.** MLOPS-01 only requires the `kg_enabled` param on agent runs; builder telemetry is W6/eval territory. If added now, it's scope creep.
6. **Where does `_row_to_hit` reference get fixed in the spec?** Not the planner's concern; this is a doc-drift issue in `implementation_plan/james/w7_knowledge_graph.md`. PLAN.md should note "spec drift acknowledged — built without `_row_to_hit`" in its Open Questions for the verifier.
7. **Add a make target for the STEM Kitchen spot-check?** CONTEXT.md `## Specific Ideas` suggests a small `make` target or PR description snippet. Recommendation: PR description snippet — adding a one-off `make kg-spot-check` target adds Makefile noise for a verification done once.

## Sources

### Primary (HIGH confidence — read in this session)
- `implementation_plan/james/w7_knowledge_graph.md` — Full W7 spec
- `.planning/phases/01-knowledge-graph-layer-place-relations-real-kg-traverse/01-CONTEXT.md` — Locked decisions
- `.planning/REQUIREMENTS.md` — Phase requirements KG-01..TEST-02
- `.planning/ROADMAP.md` — Phase 1 goal + success criteria
- `.planning/PROJECT.md` — Project overview
- `CLAUDE.md` — Project conventions (Alembic, ruff, pytest)
- `app/tools/retrieval.py` — `PlaceHit`, `_execute`, `_view_name`, `_VIEW_FOR_TABLE`
- `app/agent/tools.py` — Tool registration pattern + existing `kg_traverse` stub
- `app/agent/prompts.py` — `SYSTEM_PROMPT` (already references `kg_traverse` at line 85)
- `alembic/versions/2026_05_06_1521-4c4789a14f8f_create_place_documents_view.py` — `neighborhood_of()` already exists here
- `alembic/versions/2026_05_06_1356-5187c6b09b25_create_place_embeddings_v2.py` — migration template
- `scripts/embed_places_pgvector_v2.py:156-173` — Python `_neighborhood_from_address_components`
- `scripts/log_model_to_mlflow.py` — `params` dict at lines 194-201
- `Makefile` — target pattern
- `tests/unit/test_retriever.py` — fake-cursor pattern (older)
- `tests/unit/test_tools_retrieval.py` — `patch_get_conn` fixture pattern (recommended parent)
- `.planning/codebase/CONVENTIONS.md`, `TESTING.md`, `STACK.md` — project conventions

### Secondary (MEDIUM confidence)
- pgvector cosine semantics (`<=>` returns distance, so `1 - (a <=> b)` is similarity) — verified by existing usage in `scripts/embed_places_pgvector_v2.py` and W7 spec window function

### Tertiary (LOW confidence)
- None — no external web sources consulted; the entire design is already specified in-repo and the codebase patterns are verifiable by reading.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every dep already in `pyproject.toml`
- Architecture: HIGH — three deviations from spec surfaced and reasoned through; all anchored in current code references
- Pitfalls: HIGH — six concrete, file-cited
- Test architecture: HIGH — matches the project's four-layer convention

**Research date:** 2026-05-14
**Valid until:** 2026-06-13 (30 days; stable area — no upstream pgvector / LangChain changes expected to affect this scope)
