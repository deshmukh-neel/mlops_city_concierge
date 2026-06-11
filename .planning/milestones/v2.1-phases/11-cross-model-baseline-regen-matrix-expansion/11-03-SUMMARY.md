---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: "03"
subsystem: eval-matrix
tags: [eval, matrix, aggregation, committed_itinerary_rate, exit-codes, structural-check]
dependency_graph:
  requires: [11-01]
  provides: [D-11-02-committed-rate-threading, D-11-16-exit-classification, WR-11-structural-check-fix]
  affects: [scripts/eval_matrix.py, tests/unit/test_eval_matrix.py]
tech_stack:
  added: []
  patterns:
    - committed_itinerary_rate supplemental scalar threading in aggregate_cell_jsons
    - 3-tuple run_matrix return (rc, violation_cells, error_cells)
    - real make_error_record call in structural-check Check 6
key_files:
  created: []
  modified:
    - scripts/eval_matrix.py
    - tests/unit/test_eval_matrix.py
decisions:
  - "D-11-02: committed_itinerary_rate threaded into summary.json scorers block as supplemental scalar bypassing CRITIQUE_THRESHOLDS whitelist"
  - "D-11-16: run_matrix returns 3-tuple (rc, violation_cells, error_cells); rc==2 on errors, rc==1 on violations-only, rc==0 clean"
  - "WR-11: Check 6 replaced tautological synthetic-dict with real make_error_record(EvalQuery, 'turn0', RuntimeError) call"
metrics:
  duration: 6m
  completed: "2026-06-11"
  tasks: 2
  files: 2
---

# Phase 11 Plan 03: Eval Matrix Commit Rate and Exit — Summary

**One-liner:** Thread `committed_itinerary_rate` into `summary.json` scorers block (D-11-02), split `run_matrix` failures into distinct `violation_cells`/`error_cells` lists (D-11-16/WR-07), and replace the structural-check Check 6 tautology with a real `make_error_record` schema validation (WR-11).

## What Was Built

### Task 1: committed_itinerary_rate Threading (D-11-02)

Added a supplemental scalar accumulation loop immediately after the existing `_scorer_means_from_cell` accumulation in `aggregate_cell_jsons`. The metric is already computed at `eval_agent.py:1092` but was excluded from the gate-checker-readable `scorers` block because it is not in `CRITIQUE_THRESHOLDS`. The fix explicitly threads it from `cell_aggregate.get("committed_itinerary_rate")` into the provider block's values list, bypassing the whitelist. `_stats_for_values` then produces the `{median, min, max, stdev, n}` block automatically. Hard gates that ride on `committed_itinerary_rate` now flip from NOT-EVALUABLE to enforced.

**Files:** `scripts/eval_matrix.py` (12 lines added), `tests/unit/test_eval_matrix.py` (110 lines — 3 tests + helper)

### Task 2: run_matrix Exit Classification + WR-11 Structural-Check Fix (D-11-16, WR-11)

Updated `run_matrix` to replace the single `failures` list with two distinct lists:
- `violation_cells` (returncode == 1): model-behavior violations, expected in exploratory runs, non-blocking
- `error_cells` (returncode >= 2): infra failures, rerun needed

Return signature changed from `tuple[int, list]` to `tuple[int, list, list]`. Matrix exit code precedence: 2 if any error-cell, 1 if any violation-cell and no error-cells, 0 otherwise. `main()` updated to unpack 3-tuple and report violation/error cell counts with distinct stderr lines.

WR-11 fix: replaced the tautological Check 6 in `--structural-check` (which only validated a hand-crafted dict and could never fail even if the real schema regressed) with a call to the real `make_error_record(EvalQuery(...), "turn0", RuntimeError("quota exceeded"))`, validating `status == "error"` and `error["stage"] in {"setup", "turn0", "turnN"}`.

Updated all existing callers of `run_matrix` in the test file (2 direct calls + 3 mock patches) to use the new 3-tuple interface.

**Files:** `scripts/eval_matrix.py` (83 lines changed), `tests/unit/test_eval_matrix.py` (173 lines added + 42 lines updated)

## Verification Results

- `poetry run pytest tests/unit/test_eval_matrix.py -v` — 68 passed, 0 failed
- `make eval-matrix-refinement-structural-check` — exits 0: "structural-check: OK — matrix has 6 cell(s)..."
- `poetry run ruff check scripts/eval_matrix.py` — All checks passed
- `make test` — 1178 passed, 53 skipped, 0 failed (full suite)

## Deviations from Plan

None — plan executed exactly as written. The WR-11 fix required updating 5 existing test call sites (2 direct `run_matrix` unpacks + 3 `mocker.patch` return values) when the signature changed from 2-tuple to 3-tuple; this was an expected consequence of the signature change noted in the plan and handled inline without requiring Rule 4 escalation.

## Known Stubs

None — all functionality is fully wired.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries introduced.

## Self-Check: PASSED

- `scripts/eval_matrix.py` — FOUND (modified)
- `tests/unit/test_eval_matrix.py` — FOUND (modified)
- Commit 6faaf3e — FOUND (test RED Task 1)
- Commit 44284cb — FOUND (feat GREEN Task 1)
- Commit 5e09b4d — FOUND (test RED Task 2)
- Commit c92136f — FOUND (feat GREEN Task 2)
- All 68 eval_matrix tests pass
- Full suite 1178 passed
