# Phase 19: Productionized Loop + Metric (LOOP-01..03 + METRIC) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-20
**Phase:** 19-productionized-loop-metric-loop
**Areas discussed:** Loop target dataset, Metric definition, Pass bar / gate semantics, Loop orchestration shape (+ two adversarial Codex review rounds)

---

## Loop target dataset

| Option | Description | Selected |
|--------|-------------|----------|
| Populated sandbox, real demand | Seed a real SF subset as a non-empty before-baseline; gap-mine reads prod user_query_log via DEMAND_DATABASE_URL (seeded fallback); ingest→embed→metric on a non-empty baseline | ✓ |
| Empty sandbox (mechanism, like Phase 16) | Reuse the empty-sandbox baseline but drive gap selection from the real miner; trivial gap | |
| Prod-read → sandbox-write only | Two-DB demand path but no clean populated before→after baseline | |

**User's choice:** Populated sandbox, real demand (D-01)
**Notes:** Most honest capstone story; realizes the populated-metric Phase 16 deferred.

| Option | Description | Selected |
|--------|-------------|----------|
| Seed a real SF subset, then loop adds to it | Ingest catalog MINUS the gap bucket, embed = 'before'; loop ingests the gap; measure the lift; idempotent provision target | ✓ |
| Snapshot/restore a prod dump | Realistic but ~6k-row embed pass, muddier DB-diff, export/import dependency (16-CONTEXT rejected this for the falsifier) | |
| You decide at plan time | Lock the shape; defer the recipe | |

**User's choice:** Seed a real SF subset, then loop adds to it (D-02)
**Notes:** Real baseline, non-trivial gap. (Codex Round-1 BLOCKER → amended to exclude ALL overlapping seed sources + empty-diff→INFRA guard + full pre-embed; Round-2 added the reset-restores-baseline constraint + attribution caveat.)

---

## Metric definition (populated)

| Option | Description | Selected |
|--------|-------------|----------|
| Both hit@k AND recall@k vs the gap's new places | Target = DB-diff (new gap place_ids); hit@k headline (gated), recall@k depth (logged); reuse compute_hit_rate + add recall@k sibling | ✓ |
| hit@k only (extend existing scorer) | Generalize compute_hit_rate to a populated baseline; leaves recall@k unbuilt | |
| recall@k vs a committed ground-truth set | Hand-curated labeled relevance set; most rigorous but heavy / scope-creep | |

**User's choice:** Both hit@k AND recall@k (D-03)
**Notes:** Matches ROADMAP's "hit@k/recall@k" wording. (Codex refined target set from `places_raw` diff → `place_embeddings_v2` diff so raw rows that fail embedding aren't impossible-to-hit positives.)

| Option | Description | Selected |
|--------|-------------|----------|
| Generate-then-freeze per run, before ingest | LLM-generate N paraphrases of the miner-chosen gap, freeze to a run artifact before ingest, non-circularity check, read-only scoring | ✓ |
| Keep the committed frozen file, pin the gap to it | Reuse falsifier_paraphrases.json; couples the productionized loop to one hardcoded bucket | |
| You decide at plan time | Lock the requirement; defer per-run vs committed | |

**User's choice:** Generate-then-freeze per run, before ingest (D-04)
**Notes:** Adapts Phase 16 D-06/D-07 to a dynamic gap. (Codex: artifact should record gap+seed+prompt+model+paraphrases+timestamp.)

---

## Pass bar / gate semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Both: strictly-positive delta AND after-hit@k ≥ floor | PASS iff after > before AND after_hit@k ≥ floor (documented tunable const); realizes the deferred quality bar | ✓ |
| Strictly-positive delta only (same as Phase 16) | PASS iff after > before; no quality floor | |
| ≥ floor only (absolute, no delta) | PASS iff after ≥ floor; loses the causal delta | |

**User's choice:** Both: strictly-positive delta AND after-hit@k ≥ floor (D-05)
**Notes:** (Codex Round-1 RISKY → de-risked: floor is runtime-tunable; first run gates on strict-positive-delta only, then ratchet the floor after observing the real populated after_hit@k, documenting the justifying run. A hard ≥0.5 up front would fail-by-construction on a competitive corpus.)

| Option | Description | Selected |
|--------|-------------|----------|
| Operator-run gate + CI unit-tests the scorer | Full loop is an operator-run make target (live keys + sandbox, exit 0/1/2, MLflow); CI unit-tests the pure scorer at zero API cost | ✓ |
| Hard CI gate (block the build) | Infeasible — no sandbox/keys in CI; would require mocking the whole pipeline | |
| Logged-not-gated (tracked metric only) | Observational; loses the executable pass/fail gate | |

**User's choice:** Operator-run gate + CI unit-tests the scorer (D-06)
**Notes:** Mirrors Phase 16's gate/core split. (Codex: also unit-test the runner decision logic — floor, no-gap, one-gap handoff, stale-rejection, exit mapping.)

---

## Loop orchestration shape

| Option | Description | Selected |
|--------|-------------|----------|
| Python orchestrator script + one make target | New scripts/loop_runner.py (sibling to loop_falsifier.py) owns stage ordering, snapshots, subprocesses, DB-diff, metric, MLflow, exit 0/1/2; `make loop` wraps it; cold-start exits 0; core stays unit-testable | ✓ |
| Pure Make chain (sequential targets) | `make loop` = gap-mine && ingest && embed && metric; Make can't do snapshots/DB-diff/cold-start/cache-clear | |
| Extend loop_falsifier.py to a 'production mode' | DRY-reuse plumbing but risks coupling the deterministic falsifier to the non-deterministic miner (Phase 16 D-01 / Phase 18 D-03 reject this) | |

**User's choice:** Python orchestrator script + one make target (D-07)
**Notes:** (Codex Round-2 locked constraint: coerce DATABASE_URL=sandbox + cache_clear + close_db_pool + assert-target BEFORE any in-process gap-mine call or proposal mutation — gap-mine's supply/write use the cached pool. Also assert EMBEDDING_TABLE=place_embeddings_v2.)

| Option | Description | Selected |
|--------|-------------|----------|
| Read the top pending proposal back from the DB | After gap-mine, SELECT the top pending row, parse (neighborhood, cuisine) from the known gap_to_seed_query format; proposals table is the seam; no miner signature change | ✓ |
| Add a return value / artifact to gap_mine_main | Cleaner contract but mutates Phase 18's tested signature | |
| Single-gap-per-run: ingest only the top gap | top-n=1; DB-diff = whole ingest | |

**User's choice:** Read the top pending proposal back from the DB (D-08)
**Notes:** (Codex Round-1 BLOCKER → amended to a deterministic one-gap contract: clear stale pending → snapshot query_text set → gap_mine_main(['--top-n','1']) → assert exactly one new pending row [0 = honest cold-start exit 0]; no created_at tie-break; set-diff keys on query_text since the PK is query_text with no ID column.)

---

## Cross-AI review (Codex, two rounds)

- **Round 1** (`/tmp/phase19_decision_brief.md`): found 2 BLOCKERS — D-02 (catalog-minus-gap not clean; overlapping seeds → possible empty DB-diff → false FAIL) and D-08 (proposals "top pending" nondeterministic; ON CONFLICT DO NOTHING can silently produce zero new rows). RISKY on D-01/D-03/D-04/D-05/D-07. Decisions amended accordingly.
- **Round 2** (`/tmp/phase19_decision_brief_v2.md`): confirmed both BLOCKERS **CLOSED**; verdict **"plannable now"**. Three residual execute-time constraints folded into CONTEXT.md as LOCKED constraints/caveats: (1) D-07 coercion ordering before any in-process gap-mine; (2) D-02 reset must restore baseline data+embeddings (not just proposals/checkpoints); (3) D-02/D-08 caveats — gap-attribution is audited not guaranteed (ingest doesn't post-filter Google results), recall@k is net-new, set-diff keys on query_text. Codex ran 156 existing loop/miner unit tests = passing.

## Claude's Discretion

- MLflow run/experiment naming under `coverage_agent` (follow loop_falsifier.log_to_mlflow pattern).
- k / N constant values for the populated run (default to falsifier_core K=5/N=5).
- Exact `make loop` env-guard style for SANDBOX_DATABASE_URL + live keys.
- Internal loop_runner decomposition + unit-test seams (pure logic in falsifier_core).
- Concrete provisioning recipe + the calibrated floor value (deferred to plan/execute after a smoke run — mirrors Phase 16 D-02).

## Deferred Ideas

- Off-catalog gap discovery (blocked by the catalog-membership seed contract) — future phase. (Carried from Phase 18.)
- Tuned demand/supply ratio scoring vs the simple gate — revisit if too coarse on real data. (Carried from Phase 18.)
- A neighborhood/cuisine post-filter on ingested Google results (to *guarantee* gap-attribution rather than audit it) — would change ingest semantics (new pipeline code), out of scope for this glue-only phase.
- Wiring the miner into the deterministic falsifier's gap selection — explicitly rejected (Phase 18 D-03).
