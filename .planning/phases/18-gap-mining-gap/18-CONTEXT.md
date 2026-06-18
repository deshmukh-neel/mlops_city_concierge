# Phase 18: Gap Mining (GAP) - Context

**Gathered:** 2026-06-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a **real demand/supply gap miner** that reads the demand signal logged by
Phase 17 (`user_query_log` — real `/chat` user queries) and crosses it against
supply (`places_raw` coverage) to surface genuinely under-served
`(neighborhood, cuisine)` buckets, then writes those as `pending` seed-query
proposals to `places_ingest_query_proposals` — the canonical seam the existing
ingest already consumes. This **replaces the hardcoded gap constant** that drove
the production loop's gap selection in Phase 16 (`loop_falsifier.py`'s
`GAP = ("Outer Sunset", "vietnamese")`, D-01/D-02).

**Owns:** GAP-01..04 — the real demand/supply gap miner.

**Key existing-code reality:** `scripts/coverage_agent.py` already does
*supply-side* gap detection (per-neighborhood / per-cuisine place counts below
`min_places`) + an LLM proposal step → `places_ingest_query_proposals`, with a
`gather_stats → find_gaps → propose_queries → filter_already_covered →
insert_pending → log_to_mlflow` skeleton and a `coverage_agent` MLflow
experiment. What it is **missing** is a *real demand signal*: today it only has a
weak `recent_query` diversity row from `place_query_hits` (ingestion hits, not
user demand). **Phase 18's distinctive contribution is the demand half** —
mining `user_query_log` for what users actually asked for and using it to gate
and rank gaps.

**Out of scope (belongs to Phase 19):** the productionized Make-targeted
ingest→embed→metric loop and the productionized hit@k/recall@k scorer
(LOOP-01..03 + METRIC). Phase 18 produces the demand-driven gap proposals; Phase
19 runs the full loop and measures it. Phase 18 also does NOT rip the hardcoded
constant out of `loop_falsifier.py` — that constant is the falsifier's *own*
deliberately-independent reproducible mechanism stub (Phase 16 D-01) and stays.
"Replaces the constant" means the **production loop's gap selection** is now
demand-driven, not that the deterministic falsifier becomes coupled to a
non-deterministic miner.

</domain>

<decisions>
## Implementation Decisions

### Extraction strategy (demand → buckets)
- **D-01:** **LLM extraction via the existing `vibe.make_judge()`** is the
  primary strategy for turning raw `user_query_log` rows into
  `(neighborhood, cuisine)` demand tuples. This reuses the exact LLM-proposal
  pattern `coverage_agent.propose_queries` already uses, and handles messy free
  text (`message`) plus the LLM-parsed `requested_primary_types[]` array. Cost:
  one LLM call per mining run; non-deterministic (acceptable — the mined gap
  feeds proposals, not a deterministic gate).

### Gap definition + ranking (demand × supply)
- **D-02:** A bucket is a **gap iff `demand_count > 0` AND
  `place_count < min_places`** (existing default 5). Rank gaps by
  `demand_count` descending. Demand *gates and orders*; supply is the existing
  absolute-floor threshold from `coverage_agent.find_gaps`. Chosen over a tuned
  demand/supply ratio or pure-demand ranking because it is the most honest and
  explainable formulation for the capstone and reuses the existing threshold.

### Loop integration contract (how the miner feeds the loop)
- **D-03:** **The miner writes `pending` seed-query rows to
  `places_ingest_query_proposals`** (exactly `coverage_agent.insert_pending`'s
  existing path). The ingest already consumes pending proposals — this IS the
  productionized loop seam. The miner does NOT import into / replace
  `loop_falsifier.py`'s `GAP` constant (rejected: the falsifier was deliberately
  made independent of GAP-mining in Phase 16 D-01; coupling a deterministic gate
  to a non-deterministic miner is wrong). "Replaces the hardcoded gap constant"
  is satisfied because the *production loop's* gap selection is now demand-driven.
- **D-04:** **Cold start = honest no-op.** When `user_query_log` is empty or
  yields zero mappable demand gaps (likely early — Phase 17 logging just
  shipped), the miner inserts **nothing**, logs `gaps_found = 0` to MLflow, and
  **exits 0** (success, nothing to ingest this run). This matches
  `coverage_agent`'s existing `inserted = 0` behavior. A supply-only fallback was
  rejected — it would blur the demand-driven story Phase 18 exists to prove and
  re-create exactly what `coverage_agent` already does.

### Data source + prod-safety
- **D-05 (RECOMMENDED — researcher to confirm feasibility):** The honest
  productionized loop **reads the demand signal (`user_query_log`) from PROD
  Cloud SQL (read-only `SELECT`) and writes proposals + measures supply against
  the SANDBOX**. Reads of prod are fine — Phase 16's hard rule is only "never
  *write* shared prod `places_raw`." **Constraint / research risk:** Phase 16
  D-10 deliberately rejected threading `db_url` params and instead injects
  `SANDBOX_DATABASE_URL` as the whole process's `DATABASE_URL`. A two-DB miner
  (read prod + write sandbox simultaneously) needs *both* connections live at
  once, which the current single-`DATABASE_URL` `get_conn()` / `resolve_database_url`
  model does not cleanly support. **Acceptable fallback:** sandbox-only (mirror
  Phase 16) — read demand from rows seeded into the sandbox `user_query_log` — if
  prod `user_query_log` is still near-empty OR if two-DB threading proves too
  invasive for this phase. The planner picks after the researcher checks prod row
  counts + two-connection feasibility. Whichever is chosen, the prod-safety guard
  is on the **write target** (must be sandbox), not on a flag.

### Claude's Discretion
- **Miner shape & reuse (user: "you decide"):** RECOMMENDED — **extend
  `scripts/coverage_agent.py`** with a demand CTE/function in `gather_stats`
  (reading `user_query_log`) + a demand-gated scorer, reusing
  `propose_queries` / `filter_already_covered` / `insert_pending` /
  `log_to_mlflow` as-is. Rationale: the propose→filter→insert→MLflow pipeline is
  exactly what GAP needs; adding a demand path is the smallest honest change
  (DRY). **Guardrail:** the existing supply-only path + its W5 tests MUST stay
  intact (don't regress `coverage_agent`'s current contract). A net-new
  `scripts/gap_miner.py` is the acceptable alternative if the researcher finds
  the demand CTE doesn't slot cleanly into `gather_stats` or the supply-only
  contract over-constrains a refactor.
- **Unmappable / noisy demand (user: "you decide"):** RECOMMENDED — **drop
  unmappable demand and constrain mined buckets to catalog membership**
  (`NEIGHBORHOODS × CUISINES` from `ingest_places_sf.py`), logging an
  `unmapped_count` MLflow metric for honesty. **Hard reason:** `loop_falsifier`'s
  `premark_seed_isolation` requires the chosen seed query to be in
  `build_seed_queries()` (exits INFRA otherwise), and the ingest's seed-query
  format assumes catalog buckets — off-catalog gaps would break the loop's seed
  contract. So vague queries ("something fun") and demand for buckets outside the
  catalog are dropped (counted, not silently lost). Allowing off-catalog buckets
  is a Phase-19+ extension, not this phase.
- **CLI / ops conventions (user: no preference — house style):**
  - **Mirror `coverage_agent.py`'s CLI:** `--days` (demand window), `--dry-run`
    (print, don't insert — opt-out, matching the existing convention), 
    `--min-places` (supply threshold), plus `--top-n` to cap inserted gaps.
    `argparse main() → int` exit-code shape like `coverage_agent`.
  - **New `make gap-mine` target** wrapping the script, documented in CLAUDE.md
    commands (matches `make loop-falsifier` / `make sandbox-provision`).
  - **MLflow under the existing `coverage_agent` experiment:** log
    `gaps_found`, `proposals_inserted`, `demand_rows_scanned`, `unmapped_count`,
    and the ranked gap list as an artifact — reusing `coverage_agent`'s
    `log_to_mlflow` pattern so all loop telemetry stays in one experiment.
  - **Keep `--dry-run` opt-out** (writes by default); do NOT invent an opt-in
    `--apply`. Consistency with `coverage_agent` beats a divergent safety default;
    safety lives in the sandbox write-target guard (D-05), not the flag.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope (locked — read first)
- `.planning/ROADMAP.md` § "Phase 18: Gap Mining (GAP)" + the v2.3 milestone
  block (lines ~62-75) — goal ("real demand/supply gap miner replaces Phase 16's
  hardcoded gap constant") + milestone success-gate note.
- `.planning/PROJECT.md` § "Current State" / "v2.3 in flight" (lines ~21, 29) —
  milestone goal, the demand-signal rationale, sandbox-not-prod rule.
- `.planning/phases/16-loop-falsifier/16-CONTEXT.md` § `<deferred>` (Phase 18
  GAP-01..04) + D-01/D-02 (the hardcoded-constant thin-slice this phase replaces)
  + D-10 (sandbox `DATABASE_URL` injection — the two-DB constraint behind D-05).

### The miner to extend / the patterns to reuse (compose, don't reinvent)
- `scripts/coverage_agent.py` — the existing supply-side miner + full
  `gather_stats → find_gaps → propose_queries → filter_already_covered →
  insert_pending → log_to_mlflow` skeleton + the `coverage_agent` MLflow
  experiment. **This is the primary integration target (D-extend recommendation).**
  Note `gather_stats`'s neighborhood/cuisine CTEs over `places_raw`, the
  `min_place_count` threshold in `find_gaps` (D-02), and `insert_pending`'s
  `ON CONFLICT (query_text) DO NOTHING` proposals write (D-03).
- `scripts/ingest_places_sf.py` — `NEIGHBORHOODS` (line ~161), `CUISINES`
  (line ~194), `build_seed_queries()` (line ~319, the `NEIGHBORHOODS × CUISINES`
  catalog that defines valid buckets — the catalog-membership constraint behind
  the unmappable-demand discretion call). The ingest consumes pending proposals.

### Demand signal (Phase 17 — what we mine)
- `alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py` — the
  `user_query_log` schema: `id`, `message` (raw text verbatim, no PII scrub),
  `requested_primary_types[]` (LLM-parsed categories), `num_stops`, `rag_label`,
  `created_at` (indexed for time-window mining — `idx_user_query_log_created_at`),
  `session_id`. Migration note explicitly says "Phase 18 (GAP) mines this table."
- `app/query_log.py` — `log_user_query(...)`, the fire-and-forget writer that
  populates the table (shows exactly which fields are captured).
- `app/main.py` (lines ~737, 781-826) — where `chat()` extracts `extracted_types`
  / `num_stops` and schedules the log write (provenance of the demand fields).

### LLM + DB plumbing
- `scripts/coverage_agent.py` → uses `vibe.make_judge()` for the LLM proposal
  step (D-01 reuses this), and `get_conn()` for DB access.
- `app/config.py` → `resolve_database_url(env)` precedence (`DATABASE_URL` >
  `POSTGRES_*` > None) + `SANDBOX_DATABASE_URL` (new in Phase 16, injected AS
  `DATABASE_URL`) — central to the D-05 two-DB question.
- `scripts/loop_falsifier.py` — `premark_seed_isolation` (requires the seed in
  `build_seed_queries()` — the catalog-membership hard constraint) + the `GAP`
  constant (lines ~64-69) this phase's production path supersedes (but does NOT
  delete — D-03).

### Memory cross-refs (machine-specific gotchas)
- `project_local_postgres_port_collision` — Postgres.app squats 5432; project
  uses 5433 for both prod + sandbox URLs.
- `project_local_backend_prod_db` — prod instance is double-dash, DB inside is
  single-dash (relevant if D-05 reads prod demand).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`scripts/coverage_agent.py` is ~90% of the pipeline already.** Its
  `propose_queries` (LLM), `filter_already_covered` (dedup vs
  `build_seed_queries()` + checkpoints + proposals), `insert_pending`
  (proposals write), and `log_to_mlflow` (coverage_agent experiment) are all
  directly reusable. The net-new work is a **demand CTE/function over
  `user_query_log`** + a **demand-gated gap scorer** (D-01/D-02).
- **`vibe.make_judge()`** is the established LLM accessor (used by
  `coverage_agent.propose_queries`) — reuse for demand extraction (D-01).
- **`user_query_log` is freshly indexed on `created_at`** — the `--days`
  window query is cheap and was anticipated by Phase 17.

### Established Patterns
- **Falsifier/script convention:** `argparse main() → int` exit code + `make`
  wrapper + per-script MLflow logging (no shared helper class). The miner follows
  `coverage_agent`'s `main()` shape (D-CLI discretion).
- **Proposals as the loop seam:** ingest prepends `status='pending'` proposals
  from `places_ingest_query_proposals` to its seed list. Writing proposals IS how
  you drive the loop (D-03).
- **Single-`DATABASE_URL` DB access** (`get_conn()` / `resolve_database_url`):
  inline `DATABASE_URL=... <cmd>` retargets a whole process. This is the friction
  point for D-05's two-DB read-prod / write-sandbox split.

### Integration Points
- **WRITE:** `places_ingest_query_proposals` (`pending` rows) → consumed by
  `scripts/ingest_places_sf.py`. The miner's only side effect on the loop.
- **READ (demand):** `user_query_log` (prod per D-05 recommendation, or sandbox
  fallback).
- **READ (supply):** `places_raw` counts (existing `coverage_agent.gather_stats`
  CTEs).
- **MLflow:** existing `coverage_agent` experiment.

</code_context>

<specifics>
## Specific Ideas

- **Honesty metrics for the capstone:** log `unmapped_count` (demand dropped as
  off-catalog/vague) and `demand_rows_scanned` to MLflow so the demand→gap
  funnel is auditable and the "what we ignored" story is explicit — mirrors Phase
  16's honesty-note discipline.
- **Demand fields available beyond `message`:** `requested_primary_types[]` is
  already LLM-parsed category data — the extractor should exploit it (it's
  cleaner than parsing free text) alongside the raw `message`.
- **The falsifier's `GAP` constant stays put** — it is the deterministic
  mechanism stub, intentionally decoupled from the miner. This phase adds the
  real production-loop driver; it does not touch the falsifier's reproducibility.

</specifics>

<deferred>
## Deferred Ideas

- **Off-catalog gap discovery** (neighborhoods/cuisines users want that the
  static `NEIGHBORHOODS × CUISINES` catalog never seeded) — blocked by the
  loop's catalog-membership seed contract; a Phase-19+ extension once the loop
  can ingest arbitrary seed queries.
- **Productionized ingest→embed→metric loop (LOOP-01..03) + productionized
  hit@k/recall@k scorer (METRIC)** → Phase 19. Phase 18 produces demand-driven
  gap proposals; Phase 19 runs and measures the full loop.
- **Tuned demand/supply ratio scoring** (vs. the simple `demand>0 AND
  supply<floor` gate in D-02) — revisit in Phase 19's metric if the simple gate
  proves too coarse on real populated data.
- **Wiring the miner into the falsifier's gap selection behind a flag** —
  explicitly rejected for this phase (D-03); revisit only if a future phase wants
  an end-to-end demand→falsify run.

None of these are in Phase 18 scope — discussion stayed within the GAP boundary.

</deferred>

---

*Phase: 18-gap-mining-gap*
*Context gathered: 2026-06-17*
