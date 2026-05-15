# Phase 1: Knowledge Graph Layer - Context

**Gathered:** 2026-05-14
**Status:** Ready for planning
**Source:** Pre-existing W7 spec — `implementation_plan/james/w7_knowledge_graph.md`

<domain>
## Phase Boundary

This phase ships the W7 knowledge graph milestone of City Concierge:

1. A new `place_relations` edge table in Cloud SQL Postgres.
2. An idempotent SQL+Python builder that seeds **five** edge types from data already on hand (no LLM calls): `NEAR`, `SAME_NEIGHBORHOOD`, `CONTAINED_IN`, `NEAR_LANDMARK`, `SIMILAR_VECTOR`.
3. A real `kg_traverse(place_id, relation_type, k)` tool that **replaces** the stub in `app/agent/tools.py` (introduced in W2).
4. A typed `RelatedPlace` Pydantic model returned from the tool, joined through the existing `place_documents` view (W1) so the agent gets the same shape it gets from `semantic_search` / `nearby`.
5. An MLflow-logged param `kg_enabled` on each agent run so W6 evals can A/B with the KG on or off.
6. Unit + gated integration tests; a `make build-relations` target.

Out of scope for this phase: LLM-extracted edges (`OPERATED_BY`, `MENTIONED_WITH`, `SAME_CHEF`); editorial-source edges from the teammate's Eater/Infatuation scrape; Apache AGE; adding an FK on `dst_place_id`.

</domain>

<decisions>
## Implementation Decisions

**ALL DECISIONS BELOW ARE LOCKED.** They come from the W7 spec, which is the design contract for this phase. The planner must not redesign these. If the planner believes a decision is wrong, it should flag in PLAN.md `## Open Questions` rather than deviate.

### Storage / schema
- Edges live in a new table `place_relations` in the existing Cloud SQL Postgres instance — **not** Apache AGE (not supported on Cloud SQL).
- One row per `(src_place_id, dst_place_id, relation_type)` — that triple is the PRIMARY KEY.
- Edges are **directed**. For symmetric relations (`NEAR`, `SAME_NEIGHBORHOOD`, `SIMILAR_VECTOR`), the builder writes both directions explicitly.
- `src_place_id` has FK → `places_raw(place_id) ON DELETE CASCADE`. `dst_place_id` has **no FK** so landmark targets outside `places_raw` can be stored.
- Indexes on `(src_place_id, relation_type)`, `(dst_place_id, relation_type)`, and `(relation_type)` alone.
- A `neighborhood_of(jsonb) → text` immutable plpgsql helper lives in the same migration. It mirrors the Python helper `_neighborhood_from_address_components` used by `compose_embedding_text_v2`.

### Migration tooling
- Add the migration as `scripts/db/migrations/002_place_relations.sql` per the spec's filename. **However**, this codebase uses Alembic (`alembic/versions/`, `make migrate`, `app.db_url.resolve_alembic_database_url()`). The planner should reconcile: either (a) wrap the SQL inside a new Alembic revision created via `make migration MSG="add place_relations"` and apply via `make migrate`, or (b) add the raw SQL file and document a manual `psql -f` step. **Default decision: wrap in Alembic** to stay consistent with project conventions in `CLAUDE.md`. The spec's filename is informational.

### Builder script
- Path: `scripts/build_place_relations.py`.
- Idempotent: re-running produces no PK growth (UPSERT via `ON CONFLICT … DO UPDATE` for relations whose weights can change; `ON CONFLICT … DO NOTHING` otherwise).
- Supports `--only RELATION_TYPE[,…]` to rebuild a subset.
- Constants from spec: `NEAR_RADIUS_M = 800`, `SIMILAR_TOPK = 10`, `SIMILAR_MIN_COS = 0.65`.
- `NEAR`, `SAME_NEIGHBORHOOD`, `CONTAINED_IN`, `NEAR_LANDMARK`: pure SQL inserts (templates in W7 spec).
- `SIMILAR_VECTOR`: top-K-per-row via window function over `place_embeddings_v2`, written from Python (template in W7 spec).
- Reads from `places_raw` and `place_embeddings_v2` only. Does **not** touch `place_embeddings` (v1).

### Tool surface
- New module: `app/tools/graph.py` with `kg_traverse(place_id, relation_type, k=5) → list[RelatedPlace]`.
- `RelatedPlace` extends `app.tools.retrieval.PlaceHit` with `relation_type: str`, `weight: Optional[float] = None`, `relation_metadata: dict = {}`.
- `VALID_RELATIONS = {"NEAR", "SAME_NEIGHBORHOOD", "CONTAINED_IN", "NEAR_LANDMARK", "SIMILAR_VECTOR"}` — unknown values raise `ValueError`.
- The tool JOINs `place_relations` to `place_documents` (W1 view) so destinations missing from `places_raw` are dropped silently. Reuse `_execute` and `_row_to_hit` helpers from `app/tools/retrieval.py`.
- Result ordering: ascending by `weight` for `NEAR`, descending by `weight` for `SIMILAR_VECTOR`, unordered (stable LIMIT) for the rest.
- Replace the W2 stub in `app/agent/tools.py` with a thin wrapper delegating to `app.tools.graph.kg_traverse`. Keep the `RunContext[None]` signature so the LangGraph wiring is unchanged.
- Add a paragraph to W2's `app/agent/prompts.py` `SYSTEM_PROMPT` explaining when the agent should pick which `relation_type`. Don't redesign the surrounding prompt.

### MLflow
- Add `kg_enabled: bool` to the params logged by `scripts/log_model_to_mlflow.py`. Default `True` after this phase ships.
- W6 future-watch only — this phase doesn't expand eval suites.

### Makefile
- Add `build-relations` target invoking `python scripts/build_place_relations.py`.

### Tests
- Unit tests in `tests/unit/test_kg_traverse.py` follow the existing fake-cursor pattern from `tests/unit/test_retriever.py`. Cover: ValueError on unknown relation_type; NEAR ascending by weight; SIMILAR_VECTOR descending by weight; missing-`dst` rows dropped via the JOIN; `RelatedPlace` includes `relation_type` + `weight`.
- Integration tests in `tests/integration/test_build_place_relations.py` are **gated by `APP_ENV=integration`**, consistent with project conventions in CLAUDE.md. Use a 10-place fixture. Confirm: full builder runs; `NEAR` is symmetric within tolerance; `CONTAINED_IN` populates from `source_json`; `SIMILAR_VECTOR` weight ∈ [0,1] above threshold; second run is idempotent (no row growth).
- `make test` and `make lint` must stay green.

### Claude's Discretion

The planner has discretion over:
- How the work is **split into PLAN.md files** (one plan or multiple). Suggested split for parallelization: (1) migration + neighborhood_of helper; (2) builder script; (3) tool + agent wiring + prompt; (4) MLflow param + tests. The planner may consolidate or further split.
- Whether to add a small `app/tools/graph.py`-internal helper for the ORDER BY CASE expression vs. inlining it.
- Test fixture data structure for `tests/integration/test_build_place_relations.py` (only the assertions are locked).
- Whether to rebuild relations as part of CI or document it as a manual operator step (default: manual; CI rebuild is out of scope here).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (planner, executors, plan-checker, verifier) MUST read these before planning or implementing.**

### W7 design contract (THE source of truth)
- `implementation_plan/james/w7_knowledge_graph.md` — Full skeleton: SQL DDL, builder pseudocode, tool implementation, test plan, risks. Treat as the spec; deviate only if you find a contradiction with project conventions, and surface deviations in PLAN.md `## Open Questions`.

### Project-level context
- `implementation_plan/james/README.md` — Workstream index + cross-cutting decisions (esp. "Knowledge graph is real, not deferred" and the v1/v2 embedding policy).
- `implementation_plan/james/FUTURE_WATCH.md` — Known concerns about the LangGraph ↔ Pydantic AI adapter and v1/v2 cleanup.
- `CLAUDE.md` — Project conventions: Python 3.10+, ruff, pytest with `asyncio_mode=auto`, integration tests gated by `APP_ENV=integration`, Alembic for migrations.
- `.planning/codebase/STACK.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — Recently-mapped codebase intel.

### Code touchpoints (the planner should read these to ground tasks in current code)
- `app/agent/tools.py` — Has the `kg_traverse` stub to replace, plus the `RunContext[None]` pattern this code follows.
- `app/agent/prompts.py` — `SYSTEM_PROMPT` to extend with relation_type guidance.
- `app/tools/retrieval.py` — Source for `PlaceHit`, `_execute`, `_row_to_hit`. `RelatedPlace` extends `PlaceHit`.
- `app/embeddings/compose.py` (or wherever `compose_embedding_text_v2` lives) — Source for the Python neighborhood-extraction logic that `neighborhood_of()` plpgsql helper mirrors.
- `tests/unit/test_retriever.py` — The fake-cursor test pattern to follow.
- `scripts/log_model_to_mlflow.py` — Where `kg_enabled` param is added.
- `alembic/versions/` — Where the new migration goes (vs. spec's `scripts/db/migrations/`).
- `Makefile` — Where `build-relations` target is added.

</canonical_refs>

<specifics>
## Specific Ideas

- The W7 spec's "Manual verification" section is the canonical Phase 1 success demo — see ROADMAP.md §Phase 1 success criteria #2 and #3.
- For the spot-check `kg_traverse` against `ChIJExYUW8Z_j4AREJB4F5tJJto` (STEM Kitchen example in the W7 spec), the planner should add a small `make` target or document the snippet in the PR description.
- Per-source cap on `NEAR` (`LIMIT 50` per `src_place_id`) is **deferred** — only add if the first build shows a pathological hot spot. The planner should not pre-emptively add it.

</specifics>

<deferred>
## Deferred Ideas

- LLM-extracted edges (`OPERATED_BY`, `MENTIONED_WITH`, `SAME_CHEF`) — separate future PR.
- Editorial-source edges — blocked on the Eater/Infatuation table landing.
- Per-source NEAR cap — only if measured to be a problem.
- A nightly cron to rebuild relations after re-embeds — for now, documented as a manual step in the PR description ("after re-embed, also `make build-relations`").
- Apache AGE — explicitly ruled out (Cloud SQL doesn't support it).
- W6 eval suite expansion (`agent_strategy: vector_only | vector_plus_kg` axis) — W6 milestone, not W7.

</deferred>

---

*Phase: 01-knowledge-graph-layer-place-relations-real-kg-traverse*
*Context source: pre-existing W7 spec (treated as locked design contract); no /gsd-discuss-phase run was needed because design is already complete.*
