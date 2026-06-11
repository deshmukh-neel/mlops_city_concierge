---
phase: 10
plan: "08"
subsystem: eval-harness
tags: [eval, harness-honesty, gap-closure, tdd, cr-02, cr-03]
dependency_graph:
  requires: []
  provides: [baseline_eligible in summary.json, crash-free full eval suite]
  affects: [scripts/eval_matrix.py, scripts/eval_agent.py]
tech_stack:
  added: []
  patterns: [TDD red-green, try/except fallback guard, None-guard defensive coding]
key_files:
  created: []
  modified:
    - scripts/eval_matrix.py
    - scripts/eval_agent.py
    - tests/unit/test_eval_matrix.py
    - tests/unit/test_eval_agent.py
decisions:
  - "CR-03: main() wires load_eval_queries(args.eval_queries) into aggregate_cell_jsons via a try/except fallback to None on OSError/ValueError — missing config degrades summary (no baseline_eligible keys) without crashing"
  - "CR-02: _constraints_for_case guards dereference with `case.expected_results is not None` so clarification cases (expected_results=None) return UserConstraints(num_stops=None) cleanly"
metrics:
  duration: "15m"
  completed: "2026-06-10"
  tasks_completed: 2
  files_changed: 4
---

# Phase 10 Plan 08: Quarantine Wiring and Crash Guard Summary

Closed two BLOCKER gaps (CR-03 and CR-02) that share the eval runner/matrix surface. One-line fix wires the quarantine flag into real summary.json output; one-line guard makes the default full eval suite survive every checked-in case.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 RED | Failing tests for CR-03 baseline_eligible wiring | a8a6467 | tests/unit/test_eval_matrix.py |
| 1 GREEN | Wire load_eval_queries into main() aggregation (CR-03) | a52cd03 | scripts/eval_matrix.py, tests/unit/test_eval_matrix.py |
| 2 RED | Failing tests for CR-02 None-guard | 1a5c6a2 | tests/unit/test_eval_agent.py |
| 2 GREEN | None-guard expected_results in _constraints_for_case (CR-02) | 795bad1 | scripts/eval_agent.py |

## What Was Built

### CR-03 Fix (EVAL-02) — scripts/eval_matrix.py

`main()` previously called `aggregate_cell_jsons(output_dir, llm_provider_override=...)` without
passing `eval_queries_config`, so every real `summary.json` omitted `baseline_eligible` entirely.
Phase 11 baseline tooling would default the quarantined `late_night_closure_cascade` scenario to
eligible — silently negating the D-10-09 quarantine.

Fix: added `load_eval_queries` to the existing `from app.eval.config import (...)` block, then
wrapped `load_eval_queries(args.eval_queries)` in a try/except (OSError, ValueError) that logs a
warning and falls back to `None` on failure. The loaded config is passed as
`eval_queries_config=_eval_queries_cfg` to `aggregate_cell_jsons`. A missing or malformed
eval-queries file degrades gracefully (no `baseline_eligible` keys in output) rather than aborting
summary writing.

### CR-02 Fix (EVAL-01) — scripts/eval_agent.py

`_constraints_for_case` unconditionally dereferenced `case.expected_results.min_stops` /
`.max_stops` when `explicit_num_stops_from_text` returned None. Five of 30 hand_written cases
(`impossible_four_am_five_star`, `impossible_cheap_michelin`, `impossible_north_beach_sushi_4am`,
`overconstrained_walkable_three_neighborhoods`, `closed_monday_brunch`) have
`expected_results=None` by design and crashed the default full-suite run with `AttributeError`.

Fix: changed `if num_stops is None:` to `if num_stops is None and case.expected_results is not None:`
so the dereferences only run when `expected_results` is present. The inner `min_s == max_s` fallback
logic is unchanged.

### Tests Added

**tests/unit/test_eval_matrix.py** — two new tests:
- `test_main_aggregation_surfaces_baseline_eligible`: seeds cell JSONs for `late_night_closure_cascade`
  and `refinement_cheaper`, mocks `run_matrix`, drives `main()`, asserts
  `baseline_eligible=False` for late_night and `=True` for refinement_cheaper in written summary.json.
- `test_main_aggregation_survives_missing_eval_queries_file`: confirms fallback path — nonexistent
  `--eval-queries` file still writes summary.json (without `baseline_eligible` keys, rc=0).

**tests/unit/test_eval_agent.py** — two new tests in `TestConstraintsForCaseClarificationGuard`:
- `test_no_crash_on_known_clarification_case`: focused regression on `impossible_four_am_five_star`;
  asserts `num_stops is None` and no exception.
- `test_no_crash_over_all_hand_written_cases`: loops over every `hand_written` case in
  `configs/eval_queries.yaml` and asserts `_constraints_for_case` returns a `UserConstraints` instance.

## Verification Results

```
poetry run pytest tests/unit/test_eval_matrix.py tests/unit/test_eval_agent.py -q
160 passed in 1.43s

make test
1126 passed, 53 skipped, 9 deselected, 9 warnings in 16.76s
```

Manual checks:
```
grep -c "load_eval_queries" scripts/eval_matrix.py  → 2 (import + main callsite)
grep -c "expected_results is not None" scripts/eval_agent.py  → 1
poetry run python -c "from scripts.eval_agent import _constraints_for_case; ..."  → OK
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Renamed helper `_write_cell_with_aggregate` to avoid collision**
- **Found during:** Task 1 GREEN (full suite run)
- **Issue:** An existing `_write_cell_with_aggregate(directory, provider, model, scenario_id, run_n, aggregate: dict)` at line 411 of `test_eval_matrix.py` had the same name but different signature. The new test helper I added would shadow it and break two existing tests (`test_scorer_means_excludes_non_scorer_keys`, `test_scorer_means_rejects_bool_values_disguised_as_numeric`).
- **Fix:** Renamed new helper to `_write_cell_scored(...)` with a `score_value: float` parameter instead of `aggregate: dict`.
- **Files modified:** tests/unit/test_eval_matrix.py
- **Commit:** a52cd03

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED (Task 1) | a8a6467 | test(10-08): failing tests for CR-03 |
| GREEN (Task 1) | a52cd03 | feat(10-08): wire load_eval_queries |
| RED (Task 2) | 1a5c6a2 | test(10-08): failing tests for CR-02 |
| GREEN (Task 2) | 795bad1 | feat(10-08): None-guard |

Both RED/GREEN gate pairs present. No REFACTOR commits needed (minimal targeted fixes).

## Known Stubs

None.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Fixes are internal to eval tooling only.

## Self-Check: PASSED

- [x] `scripts/eval_matrix.py` modified (load_eval_queries import + main() wiring)
- [x] `scripts/eval_agent.py` modified (None-guard in _constraints_for_case)
- [x] `tests/unit/test_eval_matrix.py` modified (2 new tests for CR-03)
- [x] `tests/unit/test_eval_agent.py` modified (2 new tests for CR-02)
- [x] Commits a8a6467, a52cd03, 1a5c6a2, 795bad1 all exist in git log
- [x] Full suite (make test) passes: 1126 passed
