---
phase: 10
plan: "07"
subsystem: eval-harness
tags: [gap-closure, cr-01, gate-checker, tdd]
dependency_graph:
  requires: []
  provides: [EVAL-03-verified]
  affects: [scripts/check_eval_gates.py, tests/unit/test_check_eval_gates.py]
tech_stack:
  added: []
  patterns: [tdd-red-green, nested-shape-walk, integration-test]
key_files:
  created: []
  modified:
    - scripts/check_eval_gates.py
    - tests/unit/test_check_eval_gates.py
decisions:
  - "_check_gate walks summary['scenarios'][*]['providers'][family]; skips baseline_eligible=False scenarios (D-10-09 quarantine alignment)"
  - "Integration test uses category_compliance (whitelisted in CRITIQUE_THRESHOLDS) instead of committed_itinerary_rate which is not yet wired (Phase 11 BASE-01); unit tests retain committed_itinerary_rate as a direct scorer-block metric"
metrics:
  duration: "10m"
  completed: "2026-06-11"
  tasks_completed: 2
  files_changed: 2
---

# Phase 10 Plan 07: Gate Checker Schema Fix Summary

Closes CR-01: `_check_gate` read `summary.get('providers', {})` — a flat top-level key that `aggregate_cell_jsons` never writes. Every hard-gated family reported NOT-EVALUABLE and the script exited 0 against real output. The fix walks the real nested `scenarios -> providers` shape, skipping quarantined scenarios (`baseline_eligible=False`). Tests rewritten to the real shape; integration test wires `aggregate_cell_jsons()` output straight into the checker.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 RED | Failing tests for nested shape | e2e9edf | tests/unit/test_check_eval_gates.py |
| 1 GREEN | Fix _check_gate to walk nested scenarios->providers | 2a58a43 | scripts/check_eval_gates.py |
| 2 GREEN | Rewrite _make_summary + add aggregate_cell_jsons integration test | 1c3aaba | tests/unit/test_check_eval_gates.py |

## What Changed

### scripts/check_eval_gates.py

Replaced the broken flat lookup (lines 155-156):
```python
providers: dict = summary.get("providers", {})
cell = providers.get(family)
```
with a walk over `summary.get("scenarios", {})`:
- Iterates `(scenario_id, scenario_block)` pairs
- Skips blocks where `scenario_block.get("baseline_eligible", True) is False` (D-10-09 quarantine)
- Takes the first non-None `candidate = scenario_block.get("providers", {}).get(family)`
- Falls through to the existing NOT-EVALUABLE branch if no eligible cell found

Updated module docstring to document the nested `scenarios -> providers` lookup path.

### tests/unit/test_check_eval_gates.py

- `_make_summary` rewritten to produce `{"scenarios": {"refinement_cheaper": {"providers": provider_cells}}, "errors": []}` — the real `aggregate_cell_jsons` shape. All eight existing exit-code tests exercise the real nested path without any assertion changes.
- Added `_SYNTHETIC_SCENARIO_ID`, `_INTEGRATION_GATES_YAML` helpers.
- Added four TDD RED tests verifying `_check_gate` behaviors on the nested shape (violation, pass, not_evaluable, quarantine skip).
- Added `test_integration_real_aggregate_output_fires_hard_gate`: writes per-cell JSONs, calls `aggregate_cell_jsons()`, writes result to `summary.json`, asserts `script.main(...) == 1` — the test that would have caught CR-01.

## Verification

- `poetry run pytest tests/unit/test_check_eval_gates.py -q` exits 0 (16 tests pass).
- `grep -n "scenarios" scripts/check_eval_gates.py` shows the nested walk.
- `grep -n 'summary.get("providers"' scripts/check_eval_gates.py` returns nothing.
- `make test` exits 0 (1122 passed, 53 skipped).

## Deviations from Plan

### Auto-adapted: integration test uses category_compliance scorer

**Found during:** Task 2
**Issue:** `committed_itinerary_rate` is not registered in `CRITIQUE_THRESHOLDS`, so `_scorer_means_from_cell` filters it out via the whitelist. An integration test writing `committed_itinerary_rate_mean` to a cell aggregate would produce an empty scorers block and trigger NOT-EVALUABLE rather than a gate violation — the test would fail in GREEN even with a correct implementation.
**Fix:** The integration test uses `category_compliance` (registered in `CRITIQUE_THRESHOLDS`) as the gated scorer in `_INTEGRATION_GATES_YAML`. This proves the nested-shape end-to-end path without requiring changes to the scorer registry. Unit tests retain `committed_itinerary_rate` as a direct scorer-block key (bypassing the aggregator whitelist).
**Plan intent preserved:** The integration test still proves that `aggregate_cell_jsons()` output fed to `script.main()` produces exit 1 on a hard-gate violation — which is the CR-01 regression proof.
**Files modified:** tests/unit/test_check_eval_gates.py
**Commit:** 1c3aaba

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. The fix is entirely within the offline gate-checker script and its tests.

## TDD Gate Compliance

- RED commit: `e2e9edf` — `test(10-07): add failing RED tests...` (2 tests failed against old implementation)
- GREEN commit: `2a58a43` — `feat(10-07): fix _check_gate...` (all 4 new tests pass)
- GREEN commit (Task 2): `1c3aaba` — `feat(10-07): rewrite _make_summary...` (integration test + _make_summary rewrite)
- No REFACTOR phase needed — implementation is already clean.

## Self-Check

### Created files exist:
- `.planning/phases/10-eval-harness-honesty/10-07-SUMMARY.md` — present (this file)

### Modified files committed:
- `scripts/check_eval_gates.py` — commit 2a58a43 FOUND
- `tests/unit/test_check_eval_gates.py` — commits e2e9edf + 1c3aaba FOUND
