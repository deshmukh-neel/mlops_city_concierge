---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 01
subsystem: eval-harness
tags: [eval, measurement, exit-code, tdd, wave-0]
dependency_graph:
  requires: []
  provides:
    - WR-06: single-turn error capture in evaluate_case (stage='turn0')
    - WR-07: 0/1/2 exit-code contract in main()
    - WR-08: phantom scratch key exclusion from tool-call counting
    - WR-09: zero-n derived-rate guards (None on all-errored cells)
  affects:
    - scripts/eval_agent.py
    - tests/unit/test_eval_agent.py
tech_stack:
  added: []
  patterns:
    - "TDD red-green: failing tests committed before implementation"
    - "_NON_TOOL_SCRATCH_KEYS frozenset membership exclusion pattern"
    - "try/except Exception + finally for latency measurement"
    - "None-guard pattern: `expr if n_scored > 0 else None`"
    - "Three-way exit code contract (0=clean, 1=violations, 2=infra)"
key_files:
  created: []
  modified:
    - scripts/eval_agent.py
    - tests/unit/test_eval_agent.py
decisions:
  - "D-11-05: _NON_TOOL_SCRATCH_KEYS = frozenset({'prior_committed_stops', 'prior_stops_obj'}) placed module-level near tool helpers"
  - "D-11-06: single-turn error capture uses nested try/except inside finally block to preserve latency measurement"
  - "D-11-04: all five derived rates guarded with `if n_scored > 0 else None` — None not 0.0 to distinguish from legitimate 0% rates"
  - "D-11-16: infra failures (build_report exception + n_errored > 0) both map to exit 2; violations map to exit 1"
metrics:
  duration: 20m
  completed: "2026-06-11T06:48:00Z"
  tasks_completed: 2
  files_modified: 2
---

# Phase 11 Plan 01: Eval Agent Measurement Fixes Summary

**One-liner:** Wave-0 eval-harness measurement bugs fixed — phantom scratch keys excluded from tool counts, single-turn exceptions caught and returned as error records, all-errored cells publish None derived rates, and exit codes now distinguish infra failures (2) from model violations (1).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Failing tests for WR-06/07/08/09 | 8c7ee71 | tests/unit/test_eval_agent.py |
| GREEN | Implementation of all four fixes | f8fa3c2 | scripts/eval_agent.py |

## What Was Built

### WR-08 / D-11-05: Phantom Scratch Key Exclusion

Added `_NON_TOOL_SCRATCH_KEYS = frozenset({"prior_committed_stops", "prior_stops_obj"})` near the tool-counting helpers. Both `count_tool_calls` and `tool_names_from_state` now filter these two keys from their iteration. The prod-threading branch injects these keys as conversation-state serialization helpers, not tool invocations. Without the fix they inflated `tool_calls_mean` in baselines.

### WR-06 / D-11-06: Single-Turn Error Capture

Wrapped the single-turn `graph.ainvoke(...)` call inside `evaluate_case` in a nested `try/except Exception` block. Exceptions now return `make_error_record(case, "turn0", exc)` instead of propagating to the `evaluate_cases` loop and aborting the entire run. The `finally` block preserves latency measurement. Uses `# noqa: BLE001` per repo convention.

### WR-09 / D-11-04: Zero-N Derived-Rate Guards

All five derived rate fields in `aggregate_results` are now guarded with `if n_scored > 0 else None`:
- `deterministic_pass_rate`
- `deterministic_violation_rate`
- `expected_results_mismatch_rate`
- `tool_error_rate`
- `tool_success_rate`

Before the fix: `rate(0, 0)` returned 0.0, so `deterministic_pass_rate` = `1.0 - 0.0 = 1.0` on an all-errored cell — a measurement error that would be permanently baked into regenerated baselines.

### WR-07 / D-11-16: 0/1/2 Exit-Code Contract

`main()` now implements the three-way contract:
- **2** = infra failure: `build_report` raised an exception, OR `report_has_errors` is True (n_errored > 0)
- **1** = model-behavior violations: `report_has_violations` is True (non-blocking; rerun not needed)
- **0** = clean

Before the fix: both infra failures and violations returned 1, making CI unable to distinguish a rerun-needed failure from an expected model-quality miss.

## Test Coverage

Added 4 new test classes to `tests/unit/test_eval_agent.py`:

| Class | Tests | Covers |
|-------|-------|--------|
| `TestPhantomKeyExclusion` | 6 | WR-08 / D-11-05 |
| `TestSingleTurnErrorCapture` | 3 | WR-06 / D-11-06 |
| `TestZeroNDerivedRateGuards` | 7 | WR-09 / D-11-04 |
| `TestExitCodeContract` | 4 | WR-07 / D-11-16 |

Full eval_agent unit suite: **122 passed** (was 102 before this plan). Full unit suite: **1156 passed, 11 skipped**.

## Verification

- `poetry run pytest tests/unit/test_eval_agent.py -v` — 122 passed
- `poetry run ruff check scripts/eval_agent.py` — All checks passed
- `poetry run pytest tests/unit/ -v` — 1156 passed, 11 skipped
- `grep -c "if n_scored > 0 else None" scripts/eval_agent.py` — 5 (all five rates guarded)
- `grep -n "_NON_TOOL_SCRATCH_KEYS" scripts/eval_agent.py` — constant referenced in both helpers

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all implemented functionality is fully wired.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. All changes are internal to `scripts/eval_agent.py` (measurement logic and exit-code semantics only).

## Self-Check: PASSED

- [x] `scripts/eval_agent.py` modified — exists at path
- [x] `tests/unit/test_eval_agent.py` modified — exists at path
- [x] Commit 8c7ee71 exists (RED phase tests)
- [x] Commit f8fa3c2 exists (GREEN phase implementation)
- [x] `_NON_TOOL_SCRATCH_KEYS` constant present and referenced in both helpers
- [x] `return make_error_record(case, "turn0", exc)` present in evaluate_case
- [x] 5 derived-rate guards present (confirmed by grep -c)
- [x] `return 2` on both infra-failure branches in main()
- [x] All 122 eval_agent tests pass
