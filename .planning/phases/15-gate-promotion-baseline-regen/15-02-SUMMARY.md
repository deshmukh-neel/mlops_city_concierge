---
phase: 15-gate-promotion-baseline-regen
plan: "02"
subsystem: eval-harness
tags: [eval, live-run, baseline, promotion, measurement]
dependency_graph:
  requires: [15-01]
  provides: [A2-retest-delta, flag-off-run-dir]
  affects: [docs/promotion_decision.md, Plan-03]
tech_stack:
  added: []
  patterns: [eval-matrix-arm, eval-falsifier-arm, clean-env-invocation, arm-flags-inspection]
key_files:
  created: []
  modified:
    - docs/promotion_decision.md
decisions:
  - "Run #2 anchor regression (gpt-4o-mini/refinement_cheaper 0.000 vs baseline 1.000): RECORD-PARTIAL-AND-STOP-REGEN per D-11-14 branch C"
  - "Forced path DID NOT fire on refinement_cheaper (confirms Plan 01 root cause: all_slots_viable never True for typed 3-slot constraint)"
  - "write_baselines.py NOT run in this plan (D-15-05 regen-last + anchor regression blocks regen)"
metrics:
  duration: "~116 minutes (Run #1: 44:44 to 17:34 PDT; Run #2: 17:46 to 18:40 PDT)"
  completed: "2026-06-14"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 1
---

# Phase 15 Plan 02: Live Measurement (A2 Retest + Flag-Off Baseline) Summary

Two full n=5 live matrix runs against the final post-fix code: Run #1 (A2 retest, FORCED_COMMIT_STEP=6) reproduced the Phase-13 gpt-5-mini 0.500 pooled rate with the fixed synthesizer confirming the structural viability root cause; Run #2 (flag-off prod-config) revealed an anchor regression on gpt-4o-mini/refinement_cheaper (measured 0.000 vs committed baseline 1.000), blocking Plan 03 baseline regen.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Run #1 — A2 retest (FORCED_COMMIT_STEP=6) smoke + n=5 | 72514e4 | docs/promotion_decision.md (A2 Retest Delta section) |
| 2 | Run #2 — flag-off prod-config smoke + n=5 | 72514e4 | docs/promotion_decision.md (Run #2 provenance section) |

## Run Summary

### Run #1 (A2 Retest: FORCED_COMMIT_STEP=6)

- **Run dir:** `eval_reports/2026-06-14T23-44-15Z`
- **summary.json:** `eval_reports/2026-06-14T23-44-15Z/summary.json`
- **Smoke dir:** `eval_reports/2026-06-14T23-33-58Z` (n=1 guard)
- **Smoke arm_flags confirmed:** `forced_commit_step=6, viability_contract=false, parallel_tool=false, replay_multi_message=false, replay_content_blocks=false, viability_threshold_override=null`

| Model | omakase median | refinement median | pooled |
|-------|---------------|------------------|--------|
| openai/gpt-5-mini | 1.000 | 0.000 | **0.500** |
| openai/gpt-4o-mini | 1.000 | 1.000 | **1.000** (anchor NON-REGRESSION) |
| deepseek/deepseek-reasoner | 0.000 | 0.000 | **0.000** |

**Forced-path finding:** The forced commit path DID NOT FIRE on refinement_cheaper in any of the 5 gpt-5-mini runs (`forced_commit_step=None`, `commit_forced=False` in all). This confirms the Plan 01 root cause: `all_slots_viable` requires distinct viable candidates for each of Restaurant, Cocktail Bar, and Dessert Shop, but the maximum viable count per step across all runs was 1 (only one typed slot covered). The forced path DID fire once on omakase (run-1: `rule8_met_per_step` True at step 5 with 5 viable candidates). The CR-01 synthesizer fix is irrelevant to refinement_cheaper because the gate precondition is never satisfied.

**Falsifier verdict:** FAIL (exit 2) — gpt-5-mini 0.500 < 0.600 aspirational bar. gpt-4o-mini PASS (1.000 >= baseline 1.000). Run #1 is the EXPERIMENT run; `write_baselines.py` NOT run on it.

### Run #2 (Flag-Off Prod-Config: All Six Flags Unset)

- **Run dir:** `eval_reports/2026-06-15T00-46-43Z`
- **summary.json:** `eval_reports/2026-06-15T00-46-43Z/summary.json`
- **Smoke dir:** `eval_reports/2026-06-15T00-35-56Z` (n=1 guard)
- **Smoke arm_flags confirmed:** `forced_commit_step=0, viability_contract=false, parallel_tool=false, replay_multi_message=false, replay_content_blocks=false, viability_threshold_override=null`

| Model | omakase median | refinement median | pooled |
|-------|---------------|------------------|--------|
| openai/gpt-5-mini | 1.000 | 0.000 | **0.500** |
| openai/gpt-4o-mini | 1.000 | 0.000 | **0.500** |
| deepseek/deepseek-reasoner | 0.000 | 0.000 | **0.000** |

**ANCHOR REGRESSION:** gpt-4o-mini refinement_cheaper measured 0.000 vs committed baseline 1.000. Failing runs (0, 2, 4): hit step limit without committing (`first_commit_call_step=None`). Passing runs (1, 3): committed at steps 2 and 4. The regression pattern matches the root-cause finding: without `FORCED_COMMIT_STEP=6`, the agent hits the step limit in 3/5 runs when the viability gate is narrow. This suggests the Phase-11 committed baseline of 1.000 for gpt-4o-mini/refinement_cheaper may have been recorded under arm-flag conditions, not flag-off conditions.

**Decision path taken:** RECORD-PARTIAL-AND-STOP-REGEN (D-11-14 branch C). Run #2 is recorded as the honest measurement but CANNOT be used as a baseline source until the anchor regression is investigated. Plan 03 (baseline regen) is blocked.

**Falsifier verdict:** FAIL (exit 1) — gpt-4o-mini anchor regression. `write_baselines.py` NOT run.

## Pre-Matrix Probes

- `scripts/probe_provider_capture.py --provider openai`: PASS (gpt-5-mini responded)
- `scripts/probe_provider_capture.py --provider deepseek`: PASS (deepseek-reasoner responded)
- OpenAI embeddings sanity probe: PASS ("The Story of Ramen" returned, not a 429)
- anthropic/gemini probes: INTENTIONALLY SKIPPED per D-12-09 (no keys/top-up)

## Run-Cap Accounting

- Prior Phase-15 full n=5 runs at plan start: 0
- Smokes consumed: 2 (not counted against cap)
- Full n=5 runs consumed: 2 (Run #1 + Run #2)
- Total Phase-15 full runs after this plan: 2 of <=4 cap
- No billing top-ups occurred

## Deviations from Plan

### Auto-fixed Issues

None.

### Anchor Regression (Run #2) — BLOCKING

**Found during:** Task 2

**Issue:** gpt-4o-mini/refinement_cheaper committed_itinerary_rate = 0.000 (median) in the flag-off run, vs committed baseline 1.000. Per docs/baseline_regen.md and plan rules, this halts baseline regen.

**Finding:** The committed baseline of 1.000 for gpt-4o-mini/refinement_cheaper appears to have been set under conditions where FORCED_COMMIT_STEP or similar arm flags were active. In the flag-off prod-config, gpt-4o-mini fails to commit in 3/5 refinement runs (hits step limit). This is structurally identical to gpt-5-mini's behavior, but only gpt-4o-mini was falsely expected to reach 1.000 in flag-off mode.

**Decision path taken:** RECORD-PARTIAL-AND-STOP-REGEN per D-11-14 branch C. Run #2 is recorded for provenance but cannot be written as baselines.

**Plan 03 impact:** Blocked pending baseline provenance investigation. Before running `write_baselines.py`, the origin of the Phase-11 gpt-4o-mini/refinement_cheaper baseline must be confirmed. If it was written from an arm run, the baseline value must be corrected to the honest flag-off measurement (~0.000 or ~0.400 based on this run's [0,1,0,1,0] pattern).

**Files modified:** `docs/promotion_decision.md` (anchor regression documented under Run #2 section)

**Commit:** 72514e4

## Self-Check

### Files exist:
- `docs/promotion_decision.md` — exists with A2 Retest Delta + Run #2 Provenance sections
- `eval_reports/2026-06-14T23-44-15Z/summary.json` — exists (Run #1, 31 files)
- `eval_reports/2026-06-15T00-46-43Z/summary.json` — exists (Run #2, 31 files)

### Commits:
- `72514e4` — feat(15-02): run A2 retest + flag-off baseline measurement

## Self-Check: PASSED

All files exist and commits confirmed.

## Known Stubs

None — this plan produces measurement artifacts (eval run dirs + docs), no UI/API stubs.

## Threat Flags

None — no new network endpoints, auth paths, or schema changes introduced.
