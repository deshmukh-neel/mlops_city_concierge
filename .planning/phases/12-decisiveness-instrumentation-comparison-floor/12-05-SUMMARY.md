---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: "05"
subsystem: eval-falsifier
tags: [falsifier, exit-codes, wrong-matrix-guard, unit-tests]
dependency_graph:
  requires: []
  provides: [zero-overlap-exit-2-guard]
  affects: [scripts/eval_falsifier.py, tests/unit/test_eval_falsifier.py]
tech_stack:
  added: []
  patterns: [escalate-warn-to-exit-2, monkeypatch-best-effort-guard]
key_files:
  modified:
    - scripts/eval_falsifier.py
    - tests/unit/test_eval_falsifier.py
decisions:
  - "Escalate zero-overlap from warn-and-continue to refuse-with-exit-2: wrong-matrix run now returns exit 2 with no VERDICT, scoped to run-dir mode only when expected_scenarios is truthy"
  - "Fixed four pre-existing tests that used synthetic scenario IDs (scenario_a, scenario_b, s, scenario_not_in_baselines) that would now trigger the zero-overlap guard; updated to use omakase_mission_open_ended or tailored synthetic baselines"
metrics:
  duration: "5m"
  completed: "2026-06-12"
  tasks_completed: 2
  files_changed: 2
requirements: [INST-05]
---

# Phase 12 Plan 05: Falsifier Zero-Overlap Exit-2 Guard Summary

**One-liner:** Run-dir mode now refuses to grade a wrong-matrix summary with exit 2 (no PASS/FAIL verdict), preventing spurious milestone PASSes from refinement-matrix runs.

## What Was Built

### Task 1: Escalate zero-overlap from warn-and-continue to refuse-with-exit-2 (cb09ab3)

In `scripts/eval_falsifier.py`, inside the run-dir-mode `else` branch, the existing zero-overlap predicate `if expected_scenarios and not (found_scenarios & expected_scenarios):` was escalated from print-warning-and-continue to:

1. Print the existing WARNING diagnosis verbatim (unchanged)
2. Write a one-line refusal to stderr: `"eval_falsifier: refusing to grade — resolved run dir shares zero scenarios with configs/eval_matrix.yaml (wrong-matrix run); exit 2, no verdict."`
3. `return 2` immediately — before the gpt-5-mini check, before the anchor check, before any VERDICT line

Updated both the module docstring and `main()` docstring to name the zero-overlap case as an exit-2 cause alongside "missing run dir / malformed JSON".

Guard scoping is preserved: fires only when `expected_scenarios` is truthy (`_expected_matrix_scenarios()` returned non-empty). An unreadable/empty matrix config yields empty expected set; the guard does not fire (best-effort design).

Untouched: `--baselines-mode` branch; defensive anchor "no scenario overlap → warn and pass" branch (~line 381).

### Task 2: Unit tests pinning exit 2 / no VERDICT contract (4a896be)

Added `TestZeroOverlapRefusesWithExit2` class to `tests/unit/test_eval_falsifier.py`:

- **(A) test_zero_overlap_run_dir_returns_exit_2:** `refinement_cheaper` summary (otherwise-passing gpt-5-mini 0.8 + anchor 1.0) → asserts `rc == 2`. Proves the guard prevents a spurious milestone PASS.
- **(B) test_zero_overlap_emits_no_verdict_line:** `scenario_not_in_matrix` summary → asserts `"VERDICT" not in captured.out`, plus confirms diagnosis text (`"may belong to a different matrix"`, `"eval_matrix_refinement"`) still prints.
- **(C) test_in_matrix_run_still_grades:** `omakase_mission_open_ended` (in-matrix) with gpt-5-mini 0.8 + anchor 1.0 → asserts `rc == 0` and `"VERDICT" in captured.out`. Negative control proving the guard is scoped to wrong-matrix runs.
- **(D) test_empty_expected_set_does_not_refuse:** monkeypatches `_expected_matrix_scenarios` to `set()` → asserts `rc in {0, 1}` and `rc != 2`. Proves the best-effort empty-set path still grades.

Also fixed four pre-existing tests that broke because they used synthetic scenario IDs not in `eval_matrix.yaml`:

- `TestBarLogic.test_pooled_05_fails_06_bar`: changed `scenario_a` + `scenario_b` to single `omakase_mission_open_ended` at median 0.5
- `TestBarLogic.test_pooled_08_passes_06_bar`: changed `scenario_a` to `omakase_mission_open_ended`
- `TestMalformedSummaryShapes.test_main_with_malformed_n_returns_verdict_not_traceback`: changed `s` to `omakase_mission_open_ended`
- `TestAnchorCommonScenarioPooling.test_no_common_scenarios_warns_and_passes`: changed `scenario_not_in_baselines` to `omakase_mission_open_ended` and replaced the shared synthetic baselines dir with a refinement-only dir (no omakase entry), so the anchor comparison still hits the "no scenario overlap" branch

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Four pre-existing tests used synthetic scenario IDs that triggered the new zero-overlap guard**
- **Found during:** Task 2 (first pytest run)
- **Issue:** `TestBarLogic.test_pooled_05_fails_06_bar`, `test_pooled_08_passes_06_bar`, `TestMalformedSummaryShapes.test_main_with_malformed_n_returns_verdict_not_traceback`, and `TestAnchorCommonScenarioPooling.test_no_common_scenarios_warns_and_passes` all used synthetic scenario IDs (`scenario_a`, `scenario_b`, `s`, `scenario_not_in_baselines`) absent from `eval_matrix.yaml`. After Task 1, the zero-overlap guard fires for these IDs and returns 2 before the paths they were testing.
- **Fix:** Updated each test to use `omakase_mission_open_ended` (the sole in-matrix scenario). For `test_no_common_scenarios_warns_and_passes`, created a refinement-only synthetic baselines dir (no omakase entry) so the test still exercises the anchor "no scenario overlap" branch.
- **Files modified:** `tests/unit/test_eval_falsifier.py`
- **Commit:** 4a896be (included in Task 2 commit)

## Verification Results

- `poetry run pytest tests/unit/test_eval_falsifier.py -q`: 39 passed (35 prior + 4 new)
- `make lint`: passed (ruff E,F,I,N,UP,B,SIM)
- `make test`: 1275 passed, 53 skipped, 9 deselected
- Manual contract check: zero-overlap run-dir exits 2 with no VERDICT line, stderr refusal message present
- `--baselines-mode` regression: exits 1 (real VERDICT = FAIL), unaffected by guard
- `grep -c "import openai|from langchain|build_chat_model" scripts/eval_falsifier.py`: 0

## Self-Check: PASSED

- `scripts/eval_falsifier.py` exists and contains `return 2` at the zero-overlap guard
- `tests/unit/test_eval_falsifier.py` exists and contains `TestZeroOverlapRefusesWithExit2`
- Task 1 commit cb09ab3 exists in git log
- Task 2 commit 4a896be exists in git log
