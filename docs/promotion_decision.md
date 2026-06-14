# v2.2 Milestone Promotion Decision

**Created:** 2026-06-14
**Phase:** 15 — Gate Promotion + Baseline Regen
**Milestone:** v2.2 Reasoning-Model Decisiveness
**Status:** In progress — Plans 02 (A2 retest) and 03 (baseline regen) still to execute

## Cross-References (Immutable Input Documents)

- `docs/decisiveness_arm_verdicts.md` — Phase-13 DEC record: INST-05 falsifier, A1/A2/A3
  results, CR-01 forced-synthesizer-broken annotation. **Closed record — do not append.**
- `docs/replay_arm_verdicts.md` — Phase-14 REPLAY record: R1/R2/R3/valve results,
  ARCH-FUT-01 evaluation and deferral, D-14-08 user checkpoint resolving Phase-15 scope.
  **Closed record — do not append.**

---

## Root-Cause: refinement_cheaper = 0.000

**Diagnostic question (D-15-08):** WHY does gpt-5-mini commit at median 1.000 on
omakase_mission_open_ended but at median 0.000 on refinement_cheaper across every arm?

### Run Dirs Inspected

- **Inspected:** `eval_reports/2026-06-12T07-27-03Z` — the Phase-13 A2 run dir
  (`FORCED_COMMIT_STEP=6`) used as the primary evidence base. All 5 gpt-5-mini
  omakase runs (run-0..4) and all 5 gpt-5-mini refinement_cheaper runs (run-0..4)
  were individually parsed. gpt-4o-mini omakase and refinement runs in the same dir
  were also inspected for anchor contrast.
- **Not inspected for this diagnostic:** Phase-14 REPLAY run dirs (not needed —
  refinement_cheaper = 0.000 was already confirmed via the A2 verdict doc's closing
  numbers; Phase-14 R1/R2 did not change the refinement_cheaper outcome).

### Actual Telemetry Values — gpt-5-mini, refinement_cheaper (A2 run dir)

| run | viable_candidates_per_step | rule8_met_per_step | first_commit_call_step | forced_commit_step | commit_forced | tool_names (no commit_itinerary unless noted) | committed_itinerary_rate |
|-----|----------------------------|--------------------|------------------------|-------------------|---------------|-----------------------------------------------|--------------------------|
| 0 | [1, 0, 0, 1] | [F, F, F, F] | None | None | False | semantic_search, nearby, get_details | 0.0 |
| 1 | [1] | [F] | None | None | False | semantic_search, get_details, nearby | 0.0 |
| 2 | [0, 1, 0, 0, 0, 0, 0, 0] | [F,F,F,F,F,F,F,F] | None | None | False | semantic_search, nearby, get_details | 0.0 |
| 3 | [1, 0, 0, 1] | [F, F, F, F] | **4** | None | False | semantic_search, nearby, **commit_itinerary** | **1.0** |
| 4 | [1, 0, 0, 1, 0, 0, 0] | [F,F,F,F,F,F,F] | None | None | False | semantic_search, nearby | 0.0 |

**arm_flags (from deterministic.arm_flags in each JSON):**
`{'forced_commit_step': 6, 'parallel_tool': False, 'viability_contract': False, 'viability_threshold_override': None}`

**viability_threshold recorded in all runs:** 0.55

**Median committed_itinerary_rate across 5 refinement runs:** 0.000 (4/5 runs = 0.0; run-3 = 1.0;
median of [0,0,0,1,0] = 0). This is the "0.000" in the plan title and in the
`docs/decisiveness_arm_verdicts.md` A2 verdict table.

**Notes on runs 1 and 2:** run-1 shows viable_candidates_per_step=[1] (only step 0 had a semantic
search viable hit; steps 1-7 were nearby/get_details which hardcode similarity=0.0 in SQL and
cannot clear the 0.55 threshold). run-1 and run-2 both reached the step limit and returned
"I hit the planning step limit." as the final reply.

### Actual Telemetry Values — gpt-5-mini, omakase_mission_open_ended (A2 run dir, for contrast)

| run | viable_candidates_per_step | rule8_met_per_step | first_commit_call_step | forced_commit_step | commit_forced | committed_itinerary_rate |
|-----|----------------------------|--------------------|------------------------|-------------------|---------------|--------------------------|
| 0 | [2, 2, 0, 2, 0, 1, 0, 2] | [F,F,F,F,F,**T**,**T**,**T**] | None | None | False | 0.0 |
| 1 | [2, 0, 0, 0, 0, 0, 0, 0] | [F,F,F,F,F,F,F,F] | None | None | False | 0.0 |
| 2 | [2, 2, 0, 2, 0, 0, 4] | [F,F,F,F,F,F,F] | **7** | None | False | **1.0** |
| 3 | [2, 0, 0, 0, 0, 2, 2, 2] | [F,F,F,F,F,F,F,F] | **4** | None | False | **1.0** |
| 4 | [2, 0, 1, 0, 0] | [F,F,F,F,F] | **7** | None | False | **1.0** |

**Median committed_itinerary_rate across 5 omakase runs:** 1.000 (runs 2,3,4 = 1.0; runs 0,1 = 0.0;
median of [0,0,1,1,1] = 1). This yields the "1.000 omakase / 0.000 refinement / 0.500 pooled"
split recorded in the A2 verdict doc.

**Pooled median:** median([0,0,1,1,1]) = 1.0 (omakase) averaged with median([0,0,0,1,0]) = 0.0
(refinement) → 0.500 pooled. Matches `docs/decisiveness_arm_verdicts.md` exactly.

### Diagnostic Question 1: Does the refinement prompt structure prevent viable candidates at step 6?

**YES — confirmed by the actual viable_candidates_per_step counts.**

The `refinement_cheaper` scenario has `requested_primary_types: ["Restaurant", "Cocktail Bar",
"Dessert Shop"]` (from `configs/eval_queries.yaml`). The `all_slots_viable` gate in
`app/agent/viability.py` uses the TYPED path when `requested_primary_types` is non-empty:
it requires at least one distinct viable place_id (cosine >= 0.55) **per requested type**.
For refinement_cheaper: needs 1+ viable Restaurant AND 1+ viable Cocktail Bar AND 1+ viable
Dessert Shop from the cumulative `semantic_search` scratch.

The maximum viable_candidates_per_step seen across ALL 5 refinement runs is **1** (at steps 0
and 3 in runs 0, 3, 4; step 1 in run 2; step 0 in run 1). With only 1 total viable hit at any
step, `all_slots_viable` can cover at most 1 of the 3 required typed slots. The other 2 types
(Cocktail Bar and/or Dessert Shop) have zero viable candidates in the semantic_search scratch
across all 5 runs.

**Contrast with omakase:** The `omakase_mission_open_ended` scenario sets no
`requested_primary_types` (the query "Plan an omakase night in the Mission" carries no explicit
type constraints). `all_slots_viable` therefore uses the UNTYPED path: needs N=3 distinct viable
place_ids cumulatively. The omakase viable_candidates_per_step reaches 2 per step (runs 0,3)
and 4 in one step (run 2), readily satisfying the 3-place target. Run 0 shows
`rule8_met_per_step` flipping to True at step 5: `[F,F,F,F,F,T,T,T]` — confirming that the
untyped coverage gate CAN be satisfied on omakase, unlike the typed gate on refinement.

### Diagnostic Question 2: Are all slots truly viable per the all_slots_viable condition at step 6 on refinement?

**NO — confirmed by rule8_met_per_step: ALL False, for ALL 5 refinement_cheaper runs.**

| run | rule8_met_per_step (all steps) | Verdict at step 6 |
|-----|-------------------------------|-------------------|
| 0 | [F, F, F, F] (only 4 steps had semantic_search) | N/A — only 4 semantic_search calls |
| 1 | [F] (only 1 semantic_search call) | N/A — never reached step 6 semantically |
| 2 | [F,F,F,F,F,F,F,F] | False at step 6 |
| 3 | Model committed at step 4, never reached 6 | N/A |
| 4 | [F,F,F,F,F,F,F] | False at step 6 |

`all_slots_viable` returns False at every step for all 5 refinement_cheaper runs because the
typed-path coverage for Cocktail Bar and Dessert Shop is never satisfied. Hayes Valley is a
well-defined neighborhood in the pgvector corpus, but the model searches it with
`semantic_search` in the first 1-4 steps and then pivots to `nearby` and `get_details` for
remaining steps. The `nearby` tool hardcodes `similarity=0.0` in SQL (documented in
`app/agent/viability.py` WR-01), so it can never contribute to the viability gate.

### Forced-Path Telemetry: forced_commit_step and commit_forced vs arm_flags.forced_commit_step

| field | Value in ALL 5 refinement_cheaper runs |
|-------|---------------------------------------|
| `deterministic.arm_flags.forced_commit_step` | **6** (flag was set) |
| `deterministic.forced_commit_step` (actual firing) | **None** (path never fired) |
| `deterministic.commit_forced` | **False** |

This confirms the forced-commit path DID NOT fire on refinement_cheaper in any run.

**CR-01 over-determination note:** Per `docs/decisiveness_arm_verdicts.md` (CR-01 annotation),
the forced path was also inoperative in this run dir due to the synthesizer bug (fixed in Plan
13-08): `best_viable_candidate_per_slot` returned empty dicts for real PlaceHit Pydantic
objects, and `Stop.rationale` was REQUIRED with no default. The fix ships in the current
codebase. However, for refinement_cheaper, even the fixed synthesizer would not have fired:
`all_slots_viable` was False at step 6 in every run (rule8_met_per_step all-False), so the
gate condition in `app/agent/graph.py` line 650 would have skipped the branch regardless of
the synthesizer health.

**Two independent blockers prevented forced-commit from firing on refinement_cheaper:**
1. CR-01 synthesis bug (now fixed in `app/agent/graph.py` / `app/agent/viability.py`)
2. `all_slots_viable` never True for the typed 3-slot refinement_cheaper request (structural)

The Plan 02 A2 retest (with the fixed synthesizer) will confirm whether blocker 1 being removed
changes the outcome, but blocker 2 (structural viability failure) predicts it will not.

### Why the run-3 Commit Does Not Help refinement_minimal_edit

Run-3 is the only run where gpt-5-mini committed on the refinement_cheaper scenario
(`first_commit_call_step=4`, `committed_itinerary_rate=1.0`). Yet `refinement_minimal_edit`
scores 0.0 even in run-3. This is correct by design:

The `refinement_cheaper` scenario is a 2-TURN evaluation (turn 0: initial plan; turn 1:
"make stop 2 cheaper"). The `refinement_minimal_edit` scorer measures whether the agent's
turn-1 response swaps exactly stop 2 without disturbing stops 1 and 3. In run-3, gpt-5-mini
commits a valid 3-stop plan on turn 0 — but the refinement scorer still returns 0.0,
indicating that the turn-1 refinement response failed the minimal-edit contract. The exact
cause (too many edits, wrong stop swapped, or failure to commit on turn 1) is recorded in the
violation list (`violations: ['refinement_minimal_edit']`).

For the 4 runs where gpt-5-mini does NOT commit on turn 0, the eval harness sets
`refinement_context=True` with empty `prior_committed_stops`, and the scorer returns 0.0
via the Branch-2 fail-loud path (stderr: "threading_mode=prod turn 0 produced no
committed_stops; refinement_minimal_edit will score 0.0 via Branch 2 fail-loud").

### Root-Cause Summary

The refinement_cheaper = 0.000 (median) pattern has two independent structural causes:

1. **Decisiveness gap (primary):** gpt-5-mini explores via `semantic_search` for 1-4 steps
   and then pivots to `nearby`/`get_details` for the remaining steps, burning the step budget
   without calling `commit_itinerary`. Only run-3 commits (step 4); the other 4 runs hit the
   step limit or stop exploring without committing. This is the same decisiveness gap the
   entire v2.2 milestone was designed to diagnose; it is more pronounced on refinement than
   omakase because the typed-constraint search space is narrower.

2. **Typed viability gate never satisfied (structural):** The `all_slots_viable` gate requires
   distinct viable candidates for each of Restaurant, Cocktail Bar, and Dessert Shop (cosine
   >= 0.55 from semantic_search only). In all 5 runs, the maximum cumulative viable count is
   1, covering only one type. The forced-commit safety net (FORCED_COMMIT_STEP=6) cannot
   fire because its precondition (`all_slots_viable`) is False at step 6 in every run.
   This is a retrieval/scenario-coverage property, not a code bug.

3. **Refinement scoring mechanics (secondary):** Even the one run where gpt-5-mini commits
   on turn 0 (run-3) fails `refinement_minimal_edit` on the turn-1 refinement response.
   The `refinement_minimal_edit` = 0.000 metric is therefore additionally capped by
   refinement-turn behavior, even beyond the decisiveness gap on turn 0.

**The omakase vs refinement asymmetry** stems from (a) the untyped vs typed viability path
(omakase has no `requested_primary_types`, so the gate needs only 3 distinct viable place_ids
cumulatively — achievable because the Mission has many untyped candidates), and (b) the
2-turn scoring requirement on refinement_cheaper (omakase scores `refinement_minimal_edit=1.0`
trivially because it has no refinement turn to fail).

---

## D-15-08 Fix Disposition

**Disposition: DEFERRED**

### Analysis Against the Trivial-Fix Bar

The D-15-08 trivial-fix boundary (from `15-CONTEXT.md`) requires: "a one-flag /
one-line / low-risk change — e.g. the gate's viability condition is computed wrong, or
`FORCED_COMMIT_STEP=6` fires after the model has already given up."

The root-cause finding above shows the problem is a **scenario/retrieval-availability
property, not a one-line gate bug**:

- `all_slots_viable` in `app/agent/viability.py` is correctly implemented for the typed
  3-slot case. It correctly requires 3 distinct typed viable candidates. The function is
  not wrong; the candidates don't exist in the semantic_search scratch for this scenario.
- There is no off-by-one in the step count (step 6 is correctly evaluated at
  `state.step_count >= 6`).
- The gate condition is not accidentally inverting True/False.
- The CR-01 synthesis bug IS fixed (Plan 13-08, current codebase) — but that fix does not
  help here because `all_slots_viable` is False, so the synthesizer is never reached.

A potential "fix" would be to loosen the `all_slots_viable` gate — e.g. lower the
per-type threshold, allow partial-type coverage, or count `nearby` hits. However:
- Loosening the gate changes the forced-commit semantics for ALL scenarios (not just
  refinement_cheaper), affecting the anchor's behavior.
- Allowing `nearby` hits into viability would require changing the WR-01 constraint
  documented in `app/agent/viability.py` — a multi-file semantic change, not one-line.
- These changes would exceed the one-flag/one-line/low-risk bar and constitute
  architectural modifications to the experiment contract.

No change has been made to `app/agent/graph.py` or `tests/unit/test_graph_forced_commit.py`.

**Verification:** `git diff --stat app/agent/graph.py tests/unit/test_graph_forced_commit.py`
returns empty (no changes).

### Tracked Debt Note

This finding is recorded as tracked technical debt with the Phases 13-14 evidence chain as
its trigger criteria. Specifically:

- The refinement_cheaper viability gap (max 1 viable candidate per step, all 3 typed slots
  needed) points to a retrieval/candidate-availability problem: either the pgvector corpus
  lacks sufficient Cocktail Bar and Dessert Shop entries near Hayes Valley with cosine >= 0.55
  for these semantic queries, or the model's search strategy does not exercise the right
  query terms to surface them.
- Addressing this would require: (a) tuning semantic_search query generation for typed
  slots, (b) lowering the viability threshold per-scenario, (c) expanding the pgvector
  corpus with more typed entries, or (d) redesigning the refinement_cheaper scenario to use
  less restrictive type constraints. None of these is a one-line change.

**This DOES NOT re-open ARCH-FUT-01.** ARCH-FUT-01 (custom imperative agent loop) was
evaluated and deferred at the D-14-08 user checkpoint (recorded in `docs/replay_arm_verdicts.md`)
as an architectural contingency, not a Phase-15 action item. The refinement_cheaper viability
gap is a separate, narrower finding about retrieval coverage for typed-slot scenarios.

**Future milestone trigger:** If a future milestone addresses reasoning-model decisiveness on
typed-slot scenarios, the evidence package is:
- These run JSONs (`eval_reports/2026-06-12T07-27-03Z` gpt-5-mini refinement_cheaper run-0..4)
- The `viable_candidates_per_step` all-1 pattern confirming typed-slot coverage failure
- The `rule8_met_per_step` all-False pattern confirming the forced-commit gate is blocked

### Prod Default

The prod default (`FORCED_COMMIT_STEP=0`, flag off) is unchanged. ARCH-FUT-01 is not
executed. No gate values have been flipped.

### Baseline Regen Sequencing Note

Per D-15-05, baseline regen (Plan 03) is sequenced LAST against final merged code. Since no
`app/agent/graph.py` fix ships in this plan, the only code changes that precede baseline regen
are whatever Plan 02 (A2 retest) reveals. The `check_baselines_fresh.py` staleness watch-set
is not tripped by this plan (no app code changes).

---

## Zero-Spend Verification

Before task execution: `eval_reports/` contained 43 run directories.
After task execution: `eval_reports/` still contains 43 run directories (verified via
before/after `ls -1 eval_reports/ | sort` diff — output empty, no new directories).

No `eval_matrix`, `eval_agent`, or live-provider command was run. All analysis performed
by reading existing committed run JSONs in `eval_reports/2026-06-12T07-27-03Z/`.

---

## Sections Pending (Plans 02 and 03)

The following sections will be populated by subsequent plans:

- **A2 Retest Results** — Plan 02 will run the A2 experiment (FORCED_COMMIT_STEP=6) on
  the fixed synthesizer and record the gpt-5-mini refinement_cheaper and omakase results
  against the INST-05 falsifier bar.
- **Gate Promotion Decisions (PROMO-02)** — Whether gpt-5-mini is promoted from
  `aspirational` to `enforced` in `configs/eval_gates.yaml`, and the provenance record
  per D-15-06/07.
- **Baseline Regen Record (PROMO-01)** — Provenance of the regenerated committed baselines
  from `scripts/write_baselines.py` run against the flag-off prod-config.
- **PROMO-03 Latency Report** — Per-turn latency decomposition (LLM-call vs tool-exec
  seconds) from INST-04 telemetry vs the ~30s/turn prod budget.
