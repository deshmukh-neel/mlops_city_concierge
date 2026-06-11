---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: "04"
subsystem: eval-matrix-config
tags: [eval, matrix, cross-model, BASE-02, D-11-12, D-11-13]
dependency_graph:
  requires: ["11-03"]
  provides: ["BASE-02 matrix config: 5 flag-OFF entries, 1 scenario"]
  affects:
    - configs/eval_matrix.yaml
    - tests/unit/test_eval_matrix.py
    - tests/unit/test_eval_config.py
tech_stack:
  added: []
  patterns:
    - "Wave-1 deferral pattern: _DEFERRED_BASELINE_CELLS documents missing cells until Wave-2 regen lands"
    - "Atomic co-update: YAML + parity test + count test changed in same commit"
key_files:
  created: []
  modified:
    - configs/eval_matrix.yaml
    - tests/unit/test_eval_matrix.py
    - tests/unit/test_eval_config.py
decisions:
  - "D-11-12: gpt-5-mini, claude-sonnet-4-6, deepseek-reasoner added flag-OFF (no env block)"
  - "D-11-12: gemini intentionally excluded — lives only in eval_matrix_refinement.yaml"
  - "D-11-13: late_night_closure_cascade removed from default scenarios (stays runnable via SCENARIOS=)"
  - "Wave-1 deferral: _DEFERRED_BASELINE_CELLS[eval_matrix.yaml] set to the 3 new entries"
metrics:
  duration: "10m"
  completed: "2026-06-11"
  tasks_completed: 2
  files_modified: 3
---

# Phase 11 Plan 04: Matrix Expansion (D-11-12/13) Summary

**One-liner:** Expanded `configs/eval_matrix.yaml` to 5 flag-OFF cross-model entries and removed `late_night_closure_cascade` from the default scenario list, atomically co-updating the parity test and count assertions.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add 3 cross-model entries + remove late_night scenario from eval_matrix.yaml | 08a3a59 | configs/eval_matrix.yaml, tests/unit/test_eval_matrix.py |
| 2 | Atomically update parity test counts + deferred-cells map | 08a3a59 | tests/unit/test_eval_matrix.py |

Note: Tasks 1 and 2 were committed together in a single atomic commit (08a3a59) to satisfy the T-11-09 threat mitigation — the parity test must never see the YAML change without the matching test update.

## What Was Built

### Task 1: `configs/eval_matrix.yaml` changes (D-11-12 + D-11-13)

Three flag-OFF entries added under a Phase-11 comment:
- `openai/gpt-5-mini`
- `anthropic/claude-sonnet-4-6`
- `deepseek/deepseek-reasoner`

Gemini is intentionally excluded per D-11-12 / PROV-04 — its coverage lives only in `eval_matrix_refinement.yaml`.

`late_night_closure_cascade` removed from the default `scenarios` list per D-11-13. A comment block documents: the D-10-09 quarantine, that it stays runnable via `make eval-matrix SCENARIOS=late_night_closure_cascade`, and the cost rationale (5 providers × 5 runs of a baseline-ineligible scenario is pure burn). The baseline JSON file is NOT deleted (D-10-10 annotate-not-delete standing).

### Task 2: `tests/unit/test_eval_matrix.py` atomic updates

- `test_repo_eval_matrix_yaml_loads_via_load_eval_matrix`: entry count 2→5, scenario count 2→1, new provider assertions, D-11-13 `late_night not in scenarios` invariant
- `_DEFERRED_BASELINE_CELLS["eval_matrix.yaml"]`: set to `{"openai/gpt-5-mini", "anthropic/claude-sonnet-4-6", "deepseek/deepseek-reasoner"}` with Wave-1/Wave-2 removal comment
- `_MATRIX_TO_BASELINES["eval_matrix.yaml"]`: `late_night_closure_cascade.json` removed from parity check (file preserved on disk)
- `test_dry_run_prints_default_matrix_cells`: cell count comment updated to reflect 5×1×3=15 cells

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `test_eval_config.py` duplicate entry-count assertion**

- **Found during:** Task 2 verification (full suite run)
- **Issue:** `tests/unit/test_eval_config.py::TestPhase6EvalConfigAdditions::test_existing_eval_matrix_yaml_still_loads` independently asserted `len(config.entries) == 2`; the plan only mentioned `test_eval_matrix.py` as the file to update.
- **Fix:** Updated the assertion to `== 5` with a D-11-12 comment explaining the expansion; the `all(env is None)` invariant is preserved because all new entries are flag-OFF.
- **Files modified:** `tests/unit/test_eval_config.py`
- **Commit:** 8b5c3ef

## Verification

```
poetry run pytest tests/unit/test_eval_matrix.py -v -k "yaml_loads or baseline_provider_cells_match"
# 4 passed

poetry run pytest tests/unit/test_eval_matrix.py -v
# 68 passed

make eval-matrix-refinement-structural-check
# structural-check: OK

poetry run pytest tests/unit/ -q
# 1178 passed, 11 skipped
```

## Known Stubs

The three new entries (`gpt-5-mini`, `claude-sonnet-4-6`, `deepseek-reasoner`) in `eval_matrix.yaml` have no baseline cells yet. This is intentional and tracked in `_DEFERRED_BASELINE_CELLS["eval_matrix.yaml"]`. Wave-2 live regen (BASE-01, Plan 11-05/06) removes them from the deferral set.

## Self-Check: PASSED

- `configs/eval_matrix.yaml` exists and parses with 5 entries, 1 scenario
- `tests/unit/test_eval_matrix.py` updated (parity test passes)
- `tests/unit/test_eval_config.py` updated (Rule 1 fix)
- Commits exist: 08a3a59 (feat), 8b5c3ef (fix)
- `configs/eval_baselines/late_night_closure_cascade.json` still exists (not deleted)
