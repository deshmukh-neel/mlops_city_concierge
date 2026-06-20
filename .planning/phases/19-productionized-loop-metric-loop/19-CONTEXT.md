# Phase 19: Productionized Loop + Metric (LOOP-01..03 + METRIC) - Context

**Gathered:** 2026-06-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Stitch the existing adaptive-data-loop pieces (gap-mine → ingest → embed → metric)
into a **single Make-targeted productionized loop** (LOOP-01..03) and build a
**productionized hit@k / recall@k retrieval scorer over a POPULATED dataset**
(METRIC-01..03). This is the **capstone of v2.3 Adaptive Data Loop** — the
milestone that productionizes the `coverage_agent` loop so it learns from real
USER queries.

Every component already exists individually and is reused **as-is**:
- Phase 16: `loop_falsifier.py` (mechanism falsifier, empty-sandbox) +
  `app/loop/falsifier_core.py` (pure `compute_hit_rate`, DB-diff, exit 0/1/2).
- Phase 17: `user_query_log` (real `/chat` demand signal).
- Phase 18: `coverage_agent.gap_mine_main()` (real demand×supply miner →
  `places_ingest_query_proposals`), `sandbox_guard.assert_sandbox_write_target`.
- Reused: `ingest_places_sf.py` (Google Places ingest), `embed_places_pgvector_v2.py`.

**The two missing pieces Phase 19 builds (glue + metric, NOT new pipeline code):**
1. A **loop orchestrator** (`scripts/loop_runner.py` + `make loop`) that chains
   the stages with before/after snapshots — no single Make-level chain exists today.
2. A **populated-dataset scorer**: today's `compute_hit_rate` only works against an
   EMPTY sandbox (target set = DB-diff of newly-ingested IDs; pass = strictly-positive
   `>0`). Phase 19 measures a before→after lift on a NON-empty baseline, adds a
   `compute_recall_at_k` sibling, and a tunable quality floor (the "≥-floor" bar that
   16-CONTEXT explicitly deferred to Phase 19).

**Hard invariant (carried from Phase 16):** NEVER writes shared prod `places_raw`.
All ingest/embed/proposal writes target the isolated `*_sandbox` DB, enforced by
`assert_sandbox_write_target` (live `SELECT current_database()`).

**Does NOT touch** `scripts/loop_falsifier.py` or its hardcoded `GAP` — that
falsifier is the deterministic, miner-independent mechanism stub (Phase 16 D-01,
Phase 18 D-03) and stays reproducible. `loop_runner.py` is a NEW sibling.

</domain>

<decisions>
## Implementation Decisions

> **Cross-AI reviewed.** These decisions were adversarially reviewed by Codex
> across TWO rounds (`/tmp/phase19_decision_brief*.md`). Round 1 found 2 BLOCKERS
> (D-02, D-08) + RISKY items; the decisions below are the AMENDED set. Round 2
> confirmed both BLOCKERS CLOSED and the design **"plannable now"**, surfacing
> three residual execute-time constraints (folded in as LOCKED CONSTRAINTS below).

### Loop target dataset (what reality the loop demonstrates)
- **D-01:** Run the FULL loop against a **POPULATED sandbox with real demand**.
  gap-mine reads demand from prod `user_query_log` via `DEMAND_DATABASE_URL`
  (direct, read-only); writes proposals + ingest + embed to the sandbox. Before→after
  measured on a **non-empty** baseline (the "≥-floor / populated metric" Phase 16
  deferred). **Fixture-mode honesty (Codex):** when prod `user_query_log` is
  near-empty, the seeded-sandbox-demand fallback (`seed_demand_log.py`) hardcodes a
  bucket (`(Outer Sunset, vietnamese)`), so that path is a **FIXTURE, not "real
  demand" — it MUST be documented as such** in the loop output / runbook.

### Baseline provisioning (how the gap-poor "before" is built) — [Round-1 BLOCKER, CLOSED]
- **D-02:** Provision a populated baseline that **EXCLUDES *all* static-catalog seed
  queries that could surface the chosen gap bucket** — not just
  `'{cuisine} restaurants in {neighborhood} SF'`, but also citywide
  `'{cuisine} restaurants in San Francisco'` AND the per-neighborhood
  eatery/generic-food queries that overlap the bucket (the catalog has overlapping
  coverage; a naive "minus one query" leaves the gap already-covered → empty DB-diff →
  false FAIL). **Fully embed the baseline BEFORE the before-snapshot** (no embed
  backlog, or it pollutes the "after" diff). **GUARD:** if the after-ingest
  `places_raw` DB-diff is empty → **EXIT_INFRA(2)** (loop-worked-but-no-new-rows is a
  provisioning/infra error, not a metric FAIL).
  - **LOCKED CONSTRAINT — reset must restore the BASELINE, not just proposals (Codex
    R2):** a prior gap run mutates `places_raw` / `place_query_hits` /
    `place_embeddings_v2`, and embed-v2 skips already-current rows. The idempotent
    reset MUST recreate/restore the exact baseline data **and** embeddings (DROP +
    re-provision), NOT merely clear `places_ingest_query_proposals` + checkpoints.
  - **LOCKED CAVEAT — attribution is observed, not guaranteed (Codex R2):**
    `ingest_places_sf.py` persists every Google Text Search result with no
    neighborhood/cuisine post-filter, so "the new rows are the gap bucket" is an
    **audited/logged observation**, not a guarantee of seed isolation. Log enough to
    audit it; do not claim guaranteed gap-attribution.

### Metric definition (populated-dataset scorer) — [Round-1 RISKY, CLOSED]
- **D-03:** Compute **BOTH hit@k (headline, GATED) AND recall@k (depth, LOGGED).**
  **Target set = the `place_embeddings_v2` DB-diff** (newly-ingested AND embedded
  place_ids — NOT the `places_raw` diff; matches `loop_falsifier`'s `new_v2_ids`, so
  raw rows that fail/skip embedding can't be impossible-to-hit positives). hit@k =
  `#paraphrases retrieving ≥1 new embedded place in top-k / N`. recall@k =
  `#new embedded places found across paraphrases' top-k / #new embedded places`.
  Reuse `falsifier_core.compute_hit_rate` as-is; **add a pure `compute_recall_at_k`
  sibling** in `falsifier_core` (net-new — only `compute_hit_rate` exists today). With
  this target set, `before_hit@k = 0` by construction (the new v2 IDs did not exist at
  before-snapshot), so the populated baseline does NOT inflate the before measurement.

### Held-out paraphrases (non-circularity for a dynamic gap) — [Round-1 RISKY, CLOSED]
- **D-04:** For the miner-chosen gap, **LLM-generate N paraphrases of that intent and
  FREEZE them to a run artifact BEFORE the ingest stage** (so post-ingest data can't
  game them); read-only for scoring. The artifact records **gap + seed_query +
  generation prompt + model + paraphrases + timestamp**. Non-circularity =
  `falsifier_core.check_non_circularity` exact-string check vs the seed query.
  (Adapts Phase 16 D-06/D-07's frozen-before-ingest discipline to a dynamically-chosen
  gap. Residual: exact-string non-circularity won't catch case/whitespace near-dupes —
  acceptable, same posture as Phase 16.)

### Pass bar / gate semantics — [Round-1 RISKY, CLOSED]
- **D-05:** **FLOOR is a runtime-tunable constant** (env/CLI override). The **first
  real run gates on strict-positive-delta only** (`after_hit@k > before_hit@k`) — a
  hard `≥0.5` up front would fail-by-construction on a competitive populated corpus
  (Phase 16's 5/5 was against an EMPTY DB with no retrieval competition). After
  observing the actual populated-corpus `after_hit@k`, **ratchet the floor to the
  highest defensible value and document the floor + the justifying run**. Steady-state
  gate = `positive-delta AND after_hit@k ≥ calibrated FLOOR`. Exit 0/1/2.
- **D-06:** **Gate is OPERATOR-run** (live Google + OpenAI keys + sandbox): prints a
  VERDICT, exits 0/1/2, logs MLflow under the `coverage_agent` experiment. **CI does
  NOT run the live loop** (no sandbox DB / no live keys / costs real API calls);
  instead CI **unit-tests the pure scorers** (hit@k, recall@k, floor logic) **AND the
  runner decision logic** (floor handling, no-gap/cold-start, one-gap handoff,
  stale-proposal rejection, exit-code mapping) at zero API cost. Mirrors Phase 16's
  gate-operator-run / core-unit-tested split exactly.

### Loop orchestration shape — [Round-1 RISKY, CLOSED w/ locked ordering]
- **D-07:** A **NEW `scripts/loop_runner.py`** (sibling to `loop_falsifier.py`) owns
  stage ordering; one `make loop` target wraps it. Stages: clear-stale-pending →
  gap-mine → parse gap → generate+freeze paraphrases → before-snapshot → ingest
  subprocess (`DATABASE_URL=sandbox`) → embed-v2 subprocess → DB-diff (v2) →
  after-snapshot → hit@k/recall@k + floor → MLflow → exit 0/1/2. Cold-start (no gap)
  short-circuits to exit 0. `loop_falsifier.py` is **untouched**.
  - **LOCKED CONSTRAINT — coercion ordering (Codex R2; the one execute-time watch):**
    `gap_mine_main` does NOT fully self-manage its DB target — its pair-supply reads
    and proposal writes use the shared `app.db` pool via `get_conn()`, which lazily
    resolves CACHED settings. So `loop_runner` MUST coerce `os.environ["DATABASE_URL"]
    = sandbox` → `get_settings.cache_clear()` → `close_db_pool()` → assert the
    resolved target == sandbox **BEFORE any in-process `coverage_agent` import,
    `gap_mine_main` call, OR proposal mutation.** Running gap-mine (or clearing stale
    proposals) before the coercion/cache-clear is unsafe (could hit prod). This is the
    same lru_cache+pool footgun `loop_falsifier.py` already documents and handles.
  - **LOCKED CONSTRAINT — embedding-table assertion:** assert
    `settings.embedding_table == 'place_embeddings_v2'` after the cache-clear —
    `semantic_search` picks its view from settings while the subprocess embeds v2; a
    mismatch silently scores against the wrong view.

### Gap handoff (miner → metric) — [Round-1 BLOCKER, CLOSED]
- **D-08:** **Deterministic one-gap contract** (the proposals table is the seam,
  Phase 18 D-03; NO miner signature change): (1) reject/clear stale `'pending'` rows in
  the sandbox proposals table; (2) snapshot the set of pending `query_text` values;
  (3) run `gap_mine_main(['--top-n','1'])`; (4) `new = pending_after − pending_before`;
  (5) **if `len(new)==1`** → that row IS the gap (parse `(neighborhood, cuisine)` from
  its `query_text` via the known `gap_to_seed_query` format); **if `len(new)==0`** →
  honest cold-start, exit 0 no-op (covers both "no demand gap" and "seed already
  deduped away" by `ON CONFLICT DO NOTHING`); **if `len(new)>1`** → EXIT_INFRA
  (shouldn't happen with `--top-n 1`). No reliance on `created_at` ordering (ties on
  `NOW()`).
  - **LOCKED CAVEAT — key on `query_text` (Codex R2):** the proposals table PK IS
    `query_text` (no separate ID column), so the set-diff keys on `query_text`. The
    single new row stays `'pending'` until ingest flips it to `'applied'` after
    processing — so it is correctly consumed by the gap ingest.

### Claude's Discretion
- **MLflow naming** under the existing `coverage_agent` experiment — follow the
  `loop_falsifier.log_to_mlflow` pattern (params → artifacts-before-metrics → metrics;
  IN-05). Run name like `loop-runner-{neighborhood}-{cuisine}`. Log: before/after
  snapshots, frozen paraphrases, v2 DB-diff IDs, before/after hit@k, recall@k, delta,
  new_place_count, embed_added_count, the floor value, and the fixture-vs-real-demand
  flag (D-01).
- **k / N constant values** for the populated run — default to the Phase 16 module
  constants (`K=5, N=5` in `falsifier_core`); revisit only if the smoke run shows the
  metric hinges on a lucky/unlucky single paraphrase.
- **Exact `make loop` env-guard** for `SANDBOX_DATABASE_URL` (+ `GOOGLE_PLACES_API_KEY`
  / `OPENAI_API_KEY`) — mirror the `make loop-falsifier` / `make sandbox-provision`
  guard style.
- **Internal `loop_runner` decomposition + unit-test seams** — keep all gate/diff/floor
  logic in pure `falsifier_core` functions so they unit-test at zero API cost (D-06).
- **Concrete provisioning recipe + the calibrated floor value** — chosen at
  plan/execute time after a place-count + after_hit@k smoke run (mirrors Phase 16 D-02
  deferring the concrete gap value).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope (locked — read first)
- `.planning/ROADMAP.md` § "Phase 19: Productionized Loop + Metric (LOOP-01..03 +
  METRIC)" + the v2.3 "Adaptive Data Loop" milestone block — the FIXED phase boundary
  ("full Make-targeted ingest→embed→metric loop + productionized hit@k/recall@k
  scorer", never writes prod `places_raw`).
- `.planning/phases/16-loop-falsifier/16-CONTEXT.md` § `<deferred>` (Phase 19 owns
  LOOP-01..03 + METRIC-01..03 + the "≥-floor quality bar" + the dedicated
  sandbox-URL-threading productionization) + D-03/D-04/D-05 (hit@k metric semantics) +
  D-06/D-07 (frozen-paraphrase + non-circularity discipline) + D-10 (DATABASE_URL
  injection / lru_cache footgun).
- `.planning/phases/18-gap-mining-gap/18-CONTEXT.md` § D-03 (proposals table is the
  loop seam; falsifier `GAP` stays put) + D-05 (two-DB demand-read / sandbox-write
  path) + the `gap_mine_main` CLI/cold-start contract.
- `.planning/PROJECT.md` § "Current State" / "v2.3 in flight" — milestone goal,
  sandbox-not-prod rule, demand-signal rationale.

### Cross-AI review record (this phase)
- `/tmp/phase19_decision_brief.md` (Round 1) + `/tmp/phase19_decision_brief_v2.md`
  (Round 2) — the adversarial Codex reviews that produced the amended decisions and the
  three locked execute-time constraints. (Transient temp files; the decisions + caveats
  they produced are captured above and in 19-DISCUSSION-LOG.md.)

### Reused pipeline (compose, do NOT reinvent or modify)
- `scripts/coverage_agent.py` → `gap_mine_main(argv) -> int` — the Phase 18 demand
  miner. CLI `--days --dry-run --min-places --top-n`; cold-start no-op exit 0; dedups
  vs ALL proposals with `ON CONFLICT DO NOTHING`; pair-supply + proposal-write use the
  shared `get_conn()` pool (the D-07 coercion-ordering constraint). `gap_to_seed_query`
  format = `"{cuisine} restaurants in {neighborhood} San Francisco"`.
- `scripts/ingest_places_sf.py` — bare `python scripts/ingest_places_sf.py`; needs
  `GOOGLE_PLACES_API_KEY` + `DATABASE_URL`; consumes ALL `'pending'`
  `places_ingest_query_proposals` ordered by `created_at`, marks them `'applied'`
  post-run; `build_seed_queries()` is the overlapping static catalog
  (`NEIGHBORHOODS` ~L161, `CUISINES` ~L194); persists every Google result with NO
  bucket post-filter (the D-02 attribution caveat).
- `scripts/embed_places_pgvector_v2.py` — `python -m scripts.embed_places_pgvector_v2`;
  needs `OPENAI_API_KEY` + `DATABASE_URL`; embeds only missing/stale rows; prints on
  success, RAISES (non-zero) on error (subprocess `check=True` catches it — the
  D-02 "fully embed before snapshot" + reset-restores-embeddings constraint).
- `scripts/sandbox_guard.py` → `assert_sandbox_write_target(conn)` — live
  `SELECT current_database()`; refuses non-`*sandbox*` writes. Call on the SAME conn as
  the write, immediately before it.
- `scripts/seed_demand_log.py` — the hardcoded `(Outer Sunset, vietnamese)` demand
  FIXTURE used when prod `user_query_log` is empty (the D-01 fixture-honesty note).

### Metric primitives + orchestrator pattern (the templates to mirror)
- `app/loop/falsifier_core.py` — pure, stdlib-only: `compute_hit_rate`,
  `is_pass`/`is_strictly_positive_delta`, `db_diff`, `check_non_circularity`,
  `check_prod_safety`, `K=5`, `N=5`, `EXIT_PASS/FAIL/INFRA = 0/1/2`. **ADD
  `compute_recall_at_k` here** (D-03; net-new, pure, unit-tested).
- `scripts/loop_falsifier.py` — the orchestrator template: subprocess pattern
  (`run_subprocess_or_infra`), before/after snapshot via in-process `semantic_search`,
  `_snapshot_ids_from_url` (direct psycopg2, pool-independent), DATABASE_URL coercion +
  `get_settings.cache_clear()` + `close_db_pool()` + resolved-target assertion
  (~L424-466), `log_to_mlflow` (artifacts-before-metrics). **Do NOT modify it.**
- `scripts/eval_falsifier.py` — the report-grading + exit-code (0/1/2) convention for
  an operator-run gate.
- `configs/falsifier_paraphrases.json` — the frozen-paraphrase file shape
  (`seed_query`, `generation_prompt`, `non_circularity_note`, `paraphrases[]`) that
  D-04's per-run artifact mirrors.
- `app/tools/retrieval.py` → `semantic_search(query, filters=None, k=10) ->
  list[PlaceHit]` (sync; returns `place_id` + `similarity`; reads the view chosen by
  `settings.embedding_table` — the D-07 embedding-table assertion).

### DB plumbing + provisioning
- `app/config.py` → `resolve_database_url(env)` (precedence `DATABASE_URL` >
  `POSTGRES_*`); `settings.embedding_table` default `place_embeddings_v2`.
- `app/db_pool.py` → `get_conn()` / `close_db_pool()` — the lazily-cached pool behind
  the D-07 coercion-ordering constraint.
- `scripts/provision_sandbox.sh` + `make sandbox-provision` / `make sandbox-migrate` —
  the existing idempotent sandbox provisioning to extend for the populated baseline
  (D-02), including a DROP+re-provision reset.
- `Makefile` — `gap-mine` (L110), `ingest-places` (L76), `embed-v2` (L84),
  `loop-falsifier` (L244), `sandbox-provision` (L53) — the targets `make loop`
  composes; NO existing chain wires them together.
- `alembic/versions/2026_05_08_1000-34679f77f726_add_places_ingest_query_proposals.py`
  — proposals schema: PK = `query_text` (no ID column → D-08 keys on `query_text`);
  `created_at` defaults `NOW()` (ties → D-08 uses set-diff, not ordering).

### Memory cross-refs (machine-specific gotchas)
- `project_local_postgres_port_collision` — Postgres.app squats 5432; project uses 5433
  for both prod + sandbox URLs.
- `project_falsify01_gate_passed` — the runtime-DATABASE_URL-retarget MUST
  `cache_clear()` + `close_db_pool()` (the D-07 footgun, proven in Phase 16).
- `project_local_backend_prod_db` — prod instance is double-dash, DB inside single-dash
  (relevant if D-01 reads prod demand).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **The entire pipeline already exists** — Phase 19 is glue + one new pure metric.
  `gap_mine_main`, `ingest_places_sf`, `embed_places_pgvector_v2`,
  `assert_sandbox_write_target`, `semantic_search`, and `falsifier_core`'s
  hit@k/diff/guard primitives are all reused unchanged. The ONLY net-new code:
  `scripts/loop_runner.py`, `compute_recall_at_k` in `falsifier_core`, a `make loop`
  target, and the populated-baseline provisioning/reset target.
- **`loop_falsifier.py` is the proven orchestrator template** — its subprocess driving,
  before/after snapshot, pool-independent `_snapshot_ids_from_url`, the
  coerce→cache_clear→close_pool→assert sequence, and MLflow logging are directly
  borrowable patterns (copy the patterns; do not import-and-mutate the falsifier).

### Established Patterns
- **Operator-run gate / pure-core-unit-tested split** (Phase 16): the live loop is a
  `make` target; the gate logic is pure functions in `falsifier_core` with zero-cost
  unit tests. Phase 19 mirrors this exactly (D-06).
- **Proposals table as the loop seam** (Phase 18 D-03): writing a `'pending'` proposal
  IS how you drive the loop. D-08 reads the seam back deterministically.
- **Single-`DATABASE_URL` coercion + lru_cache/pool footgun**: retarget a whole process
  at the sandbox via `os.environ["DATABASE_URL"]`, but `get_settings.cache_clear()` +
  `close_db_pool()` are mandatory before any settings-touching code runs (the D-07
  locked ordering constraint).
- **Exit-code convention 0/1/2 (PASS/FAIL/INFRA)** locked across
  `eval_falsifier`/`loop_falsifier`; `loop_runner` follows it.

### Integration Points
- **WRITE (loop):** `places_ingest_query_proposals` (`'pending'`, guarded), then
  `places_raw` + `place_query_hits` + `place_embeddings_v2` — ALL sandbox.
- **READ (demand):** prod `user_query_log` via `DEMAND_DATABASE_URL` (direct,
  read-only) or seeded sandbox fixture (D-01).
- **READ (metric):** in-process `semantic_search` over the sandbox `place_documents_v2`
  view (must match `place_embeddings_v2` — D-07 assertion).
- **MLflow:** existing `coverage_agent` experiment.
- **MISSING (build):** `make loop` chain, `compute_recall_at_k`, the populated-baseline
  provisioning + DROP-and-restore reset, the tunable floor constant.

</code_context>

<specifics>
## Specific Ideas

- **Honest capstone framing (essay/interview):** Phase 16 was a *mechanism* falsifier
  (empty sandbox, artificial gap, before=0 by construction). Phase 19 is the
  *productionized* loop on a **populated** baseline — a real before→after lift where the
  new gap places compete against the existing corpus. The fixture-vs-real-demand flag
  (D-01) and the "gap-attribution is audited, not guaranteed" caveat (D-02) keep the
  story honest about what is fixture and what is real.
- **Floor as a calibrated outcome, not a guess (D-05):** the first run reports the
  actual populated `after_hit@k`; the committed floor is then set to the highest
  defensible value, documented alongside the run that justified it — same honesty
  discipline as the v2.x baselines.
- **Reuse Phase 16's frozen-before-ingest paraphrase discipline** for a *dynamic* gap
  (D-04) — the per-run artifact is the moving-target-resistant analog of the committed
  `falsifier_paraphrases.json`.

</specifics>

<deferred>
## Deferred Ideas

- **Off-catalog gap discovery** (buckets the static `NEIGHBORHOODS × CUISINES` catalog
  never seeds) — blocked by the loop's catalog-membership seed contract; a future-phase
  extension once ingest can take arbitrary seed queries. (Carried from Phase 18.)
- **Tuned demand/supply ratio scoring** (vs. the simple `demand>0 AND supply<floor`
  gate) — revisit if the simple gate proves too coarse on real populated data.
  (Carried from Phase 18.)
- **A neighborhood/cuisine post-filter on ingested Google results** (to *guarantee*
  gap-attribution rather than audit it — the D-02 caveat) — would change
  `ingest_places_sf` semantics (new pipeline code), so out of scope for this
  glue-only phase; note as a future hardening if attribution noise proves material.
- **Wiring the miner into the deterministic falsifier's gap selection** — explicitly
  rejected (Phase 18 D-03); the falsifier stays miner-independent.

None of these are in Phase 19 scope — discussion stayed within the
LOOP-01..03 + METRIC boundary.

</deferred>

---

*Phase: 19-productionized-loop-metric-loop*
*Context gathered: 2026-06-20*
