---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: "08"
subsystem: eval-harness
tags: [baselines, regen, ratification, runbook, live-infra]
dependency_graph:
  requires: ["11-01", "11-02", "11-03", "11-04", "11-05", "11-06", "11-07"]
  provides: [BASE-01, BASE-02, BASE-03]
  affects: [configs/eval_baselines, configs/eval_gates.yaml, docs/baseline_regen.md, tests/unit/test_eval_matrix.py]
tech_stack:
  added: []
  patterns: [snapshot-then-regen, runbook-ordered-steps, baselines-mode-gate-check, aggregate-cell-jsons-path]
key_files:
  created:
    - docs/baseline_regen.md
  modified:
    - configs/eval_baselines/omakase_mission_open_ended.json
    - configs/eval_baselines/refinement_cheaper.json
    - configs/eval_baselines/_snapshots/omakase_mission_open_ended.pre-phase11.json
    - configs/eval_baselines/_snapshots/refinement_cheaper.pre-phase11.json
    - configs/eval_baselines/_snapshots/late_night_closure_cascade.pre-phase11.json
    - configs/eval_gates.yaml
decisions:
  - "D-11-08: baseline_regen.md runbook authored with preconditions (embeddings probe, DB reachability, 4 API keys), ordered 7-step procedure, failure branches (gemini deferral D-11-11, gated-provider error block, gpt-4o-mini <0.8 STOP)"
  - "D-11-10 confirmed: late_night_closure_cascade explicitly documented as NOT regenerated"
  - "D-11-09 completed: three pre-phase11 snapshots written (c0a4d34)"
  - "D-11-20 completed: anthropic demoted to logged-not-gated (billing exhaustion, 2941807); gpt-5-mini refinement committed_itinerary_rate median=0.0 measured (n=5)"
  - "Refinement matrix aggregated via aggregate_cell_jsons on partial run dir; write_baselines wrote gpt-4o-mini/gpt-5-mini/deepseek-reasoner; REFUSED anthropic (0/5 billing), deepseek-chat (4/5 tool_calls error), gemini (1/5 partial crash)"
  - "gemini run-0 in refinement SCORED (committed_itinerary_rate=1.0) — first-ever gemini refinement observation; full n=5 deferred per D-11-11 (only 1/5 complete due to prior crash)"
metrics:
  duration: "~90m total (two executor sessions)"
  completed_date: "2026-06-11"
  tasks_completed: 4
  tasks_total: 4
  files_changed: 7
---

# Phase 11 Plan 08: Live Regen and Ratification Summary

**One-liner:** Honest n=5 baselines written for omakase and refinement scenarios; gpt-4o-mini anchor holds at median 1.0; anthropic deferred (billing), gemini deferred (1/5 partial); gates re-ratified with D-11-20 rationale; parity test atomically consistent.

## What Was Built

### Task 1: docs/baseline_regen.md runbook (COMPLETE — commits 916c19d, af74ec7)

Created `docs/baseline_regen.md` — the BASE-01 / D-11-08 ordered runbook. Contents:

- **Preconditions section:** embeddings sanity probe (the exact 21-14-30Z poison condition guard), DB reachability via cloud-sql-proxy on port 5433 for instance `mlops-491820:us-central1:mlops--city-concierge`, all four API keys
- **Ordered 7-step procedure:** `make probe-providers` → `make snapshot-baselines` → `APP_ENV=eval make eval-matrix RUNS=5` → `APP_ENV=eval make eval-matrix-refinement RUNS=5` → `make write-baselines SUMMARY=... RUNS=5` (×2) → `make eval-gates-check-baselines` → commit
- **Failure branches:** gemini cells erroring (D-11-11 deferral), gated provider errors (rerun), gpt-4o-mini commit_rate < 0.8 (STOP — real anchor regression), anthropic billing deferral
- **Explicit note:** `late_night_closure_cascade` is NOT regenerated (D-10-09/10 standing)
- af74ec7 fixed GEMINI_API_KEY env var name (was GOOGLE_API_KEY) and semantic_search import path

### Task 2: Snapshot, run matrices, write baselines, re-ratify gates (COMPLETE — commits c0a4d34, 2941807, fcb9fda, e0573e4, 93e3a4b)

**Step 1 — Snapshots (c0a4d34):**
Three `configs/eval_baselines/_snapshots/*.pre-phase11.json` files written, preserving fail-open-saturated v2.0 numbers as audit trail.

**Step 2 — Anthropic demotion (2941807):**
`anthropic/claude-sonnet-4-6` demoted from `provisional-n1` to `logged` in `configs/eval_gates.yaml` + `docs/eval_gates.md` § Anthropic deferral added. All 5 omakase and refinement cells returned HTTP 400 "credit balance too low" — deterministic billing blocker, not a code defect.

**Step 3 — Omakase matrix baselines written (fcb9fda):**
`configs/eval_baselines/omakase_mission_open_ended.json` regenerated from run `2026-06-11T19-09-10Z` at n=5. Results:

| Provider | committed_itinerary_rate median | Status |
|----------|--------------------------------|--------|
| openai/gpt-4o-mini | 1.0 | WRITTEN — ANCHOR HOLDS |
| openai/gpt-5-mini | 1.0 | WRITTEN (stdev high; aspirational gate) |
| deepseek/deepseek-chat | 0.0 | WRITTEN (logged; 1/5 converged) |
| deepseek/deepseek-reasoner | 0.0 | WRITTEN (logged; decisiveness gap) |
| anthropic/claude-sonnet-4-6 | n/a | REFUSED (0/5 billing errors; D-11-20) |
| gemini | — | Not in eval_matrix.yaml (PROV-04/D-09-08 exclusion) |

**Step 4 — Refinement matrix baselines written (e0573e4):**
Run `2026-06-11T19-38-23Z` had 5/5 complete for gpt-4o-mini, gpt-5-mini, deepseek-reasoner; partial for others. Aggregated via `aggregate_cell_jsons` (harness-own aggregation path). `write_baselines.py` results:

| Provider | n_scored | committed_itinerary_rate median | Status |
|----------|---------|--------------------------------|--------|
| openai/gpt-4o-mini | 5/5 | 1.0 | WRITTEN — ANCHOR HOLDS |
| openai/gpt-5-mini | 5/5 | 0.0 | WRITTEN (aspirational; refinement harder) |
| deepseek/deepseek-reasoner | 5/5 | 0.0 | WRITTEN (logged) |
| deepseek/deepseek-chat | 4/5 | 0.0 | REFUSED (run-1 tool_calls 400; D-10-03) |
| anthropic/claude-sonnet-4-6 | 0/5 | n/a | REFUSED (billing; D-11-20) |
| gemini/gemini-3.1-pro-preview | 1/5 | 1.0 | REFUSED (1/5 partial; D-11-11) — gemini run-0 SCORED first-ever refinement measurement |

**Step 5 — Gate re-ratification (93e3a4b):**
`configs/eval_gates.yaml` gpt-5-mini rationale updated with actual refinement n=5 data:
- omakase committed_itinerary_rate median=1.0, refinement median=0.0
- Remains aspirational; v2.2 decisiveness work pending

**Step 6 — Anchor STOP check:**
gpt-4o-mini `committed_itinerary_rate` median = 1.0 (omakase) and 1.0 (refinement). Both exceed the 0.8 floor (D-10-07). No regression — STOP condition not triggered.

**Step 7 — Parity test and gates check:**
- `make eval-gates-check-baselines` exits 0; only aspirational miss on gpt-5-mini (non-blocking)
- `test_baseline_provider_cells_match_matrix_entries[eval_matrix.yaml]` PASSED
- `test_baseline_provider_cells_match_matrix_entries[eval_matrix_refinement.yaml]` PASSED
- Full suite: 1192 passed, 53 skipped, 9 deselected

## Deferred Cells (Documented)

| Provider | Matrix | Reason | Decision |
|----------|--------|--------|----------|
| anthropic/claude-sonnet-4-6 | eval_matrix.yaml | HTTP 400 "credit balance too low" (all 5 cells) | D-11-20: demoted to logged; promotion path in docs/eval_gates.md |
| anthropic/claude-sonnet-4-6 | eval_matrix_refinement.yaml | Same billing blocker | D-11-20 deferral |
| gemini/gemini-3.1-pro-preview | eval_matrix_refinement.yaml | Only 1/5 complete (crash mid-run); run-0 SCORED (committed_itinerary_rate=1.0) | D-11-11: deferred; retry when GEMINI_API_KEY quota permits |
| deepseek/deepseek-chat | eval_matrix_refinement.yaml | run-1 HTTP 400 tool_calls error (4/5) | Old data carried forward from prior baseline; logged-not-gated |

Note: deepseek/deepseek-chat is NOT in `_DEFERRED_BASELINE_CELLS` because the parity test allows its old cell to remain (the test only checks `missing = matrix_keys - baseline_keys == deferred`; deepseek-chat IS in baseline from prior run).

## Deviations from Plan

### Auto-handled: Partial refinement run (crash at gemini run-0)

**Found during:** Task 2 execution
**Issue:** The refinement matrix run `2026-06-11T19-38-23Z` was incomplete at crash time: gemini had only run-0, no summary.json existed, deepseek-chat run-1 errored.
**Resolution:** Used the harness's own `aggregate_cell_jsons` function to generate summary.json from existing cell files — no manual editing. write_baselines mechanically refused partial/errored cells (D-10-03). This is the documented D-11-11 failure branch, not a deviation from plan design.
**Files:** eval_reports/2026-06-11T19-38-23Z/summary.json (gitignored), configs/eval_baselines/refinement_cheaper.json

### Gemini run-0 first-ever measurement

**Observation:** gemini run-0 in refinement SCORED (n_scored=1, committed_itinerary_rate=1.0) — this is the first-ever empirical gemini refinement observation. The Phase-9 adapters appear to work for gemini in refinement_cheaper. Only 1/5 complete, so REFUSED by write_baselines per D-10-03. Noted for future n=5 measurement.

## Security Scan

No API key strings in any committed artifact:
```
grep -rIl "sk-\|AIzaSy" configs/eval_baselines docs/baseline_regen.md → NO KEYS - SAFE
```

## Self-Check

- [x] `docs/baseline_regen.md` exists: FOUND (commits 916c19d + af74ec7)
- [x] Three `_snapshots/*.pre-phase11.json` files exist: FOUND (commit c0a4d34)
- [x] `omakase_mission_open_ended.json` has `committed_itinerary_rate`: FOUND (commit fcb9fda)
- [x] `refinement_cheaper.json` has `committed_itinerary_rate`: FOUND (commit e0573e4)
- [x] `anthropic/claude-sonnet-4-6` status = `logged` in eval_gates.yaml: CONFIRMED (commit 2941807)
- [x] `openai/gpt-5-mini` status = `aspirational` in eval_gates.yaml: CONFIRMED
- [x] `make eval-gates-check-baselines` exits 0: VERIFIED
- [x] Parity tests pass: VERIFIED (2/2 passed)
- [x] Full test suite: 1192 passed

## Self-Check: PASSED
