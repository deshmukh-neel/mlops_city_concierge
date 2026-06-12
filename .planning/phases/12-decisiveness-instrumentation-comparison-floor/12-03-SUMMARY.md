---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: "03"
subsystem: eval-harness
tags: [eval, falsifier, instrumentation, INST-05, artifact-reading]
dependency_graph:
  requires:
    - "12-01: step_telemetry field + graph timing hooks"
    - "12-02: eval_agent.py INST metrics (for live run mode)"
    - "configs/eval_baselines: committed baseline JSONs"
  provides:
    - "INST-05: make eval-falsifier milestone falsifier report"
    - "exit-code 0/1/2 contract for Phase 13 mechanical consumption"
  affects:
    - "Makefile: eval-falsifier target added"
    - "scripts/eval_falsifier.py: new artifact-reading script"
    - "tests/unit/test_eval_falsifier.py: 17 tests including smoke"
tech_stack:
  added: []
  patterns:
    - "artifact-reading falsifier (no live API calls, D-12-06)"
    - "pooled committed_itinerary_rate across eligible scenarios (D-12-08)"
    - "reuse check_eval_gates machinery (D-12-07)"
    - "importlib.util module-loading test pattern (mirrors test_check_eval_gates)"
key_files:
  created:
    - scripts/eval_falsifier.py
    - tests/unit/test_eval_falsifier.py
  modified:
    - Makefile
decisions:
  - "Anchor regression check skipped in --baselines-mode (comparing baselines to themselves is vacuous); run-dir mode compares live run against committed baseline floor"
  - "_get_metric_value removed from imports (unused — anchor check reuses _pooled_commit_rate on baseline summary, not scalar metric extraction)"
  - "Lazy import of check_eval_gates inside main() to keep module top-level free of app imports (D-12-06 artifact-reading only)"
metrics:
  duration: "5 minutes"
  completed: "2026-06-12"
  tasks: 3
  files_created: 2
  files_modified: 1
---

# Phase 12 Plan 03: Falsifier Report Summary

**One-liner:** Artifact-reading falsifier (`make eval-falsifier`) that pools gpt-5-mini committed_itinerary_rate across all scored scenario cells and checks gpt-4o-mini anchor non-regression, with exit codes 0/1/2 for Phase 13 mechanical consumption.

## What Was Built

Three deliverables implement INST-05 end-to-end:

**`scripts/eval_falsifier.py`** — the falsifier report script:
- `_latest_run_dir(base)` — resolves the most recent ISO8601 run directory
- `_pooled_commit_rate(summary, provider_key)` — pools committed_itinerary_rate medians weighted by n across all baseline_eligible scenarios; returns `(pooled_or_None, per_scenario_dict)`
- `main(argv)` — resolves summary (run-dir or --baselines-mode), checks gpt-5-mini pooled rate >= 0.6, checks gpt-4o-mini anchor non-regression, prints per-model numbers + verdict, returns exit code 0/1/2
- Zero live API calls; reuses `_build_summary_from_baselines` and `_load_baseline_eligibility` from `check_eval_gates.py`

**`tests/unit/test_eval_falsifier.py`** — 17 tests:
- `TestPooledCommitRate`: 7 unit tests covering two-scenario pooling (result: 0.5), weighted pool with unequal n, baseline_eligible exclusion, absent provider cells, empty scenarios
- `TestBarLogic`: 4 tests covering exit 1 (pooled < 0.6), exit 0 (pooled >= 0.6 + anchor holds), exit 2 (missing summary.json), exit 2 (nonexistent run dir)
- `TestBaselinesModeUnit`: 5 tests covering the `--baselines-mode` code path via synthetic baselines dir — real verdict (0 or 1), above-bar → 0, below-bar → 1, empty dir → 2, missing dir → 2
- `test_smoke_runs_against_real_baselines`: committed smoke test against real `configs/eval_baselines`, asserts exit in {0, 1} (not 2)

**`Makefile`** — `eval-falsifier` target:
- `RUN_DIR ?=` variable in variables block
- `.PHONY: eval-falsifier` with `## INST-05: ...` help comment after `snapshot-baselines`
- Recipe passes `$(if $(RUN_DIR),--run-dir $(RUN_DIR),)` and `--baselines-dir configs/eval_baselines`
- `make help` lists the target; `make -n eval-falsifier` prints the command

## Observed Behavior Against Real Baselines

```
[openai/gpt-5-mini] committed_itinerary_rate per scenario:
  omakase_mission_open_ended: 1.000
  refinement_cheaper: 0.000

openai/gpt-5-mini: pooled committed_itinerary_rate = 0.500 < 0.6  FAIL

openai/gpt-4o-mini: baselines-mode — pooled baseline = 1.000

eval_falsifier: VERDICT = FAIL  (exit 1)
```

The gpt-5-mini splits sharply: omakase 1.0 vs refinement 0.0 → pooled ~0.5. This is the expected pre-Phase-13 state confirming the falsifier is correctly measuring the decisiveness gap that Phase 13 must close.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | f478589 | feat(12-03): eval_falsifier.py artifact-reading falsifier (INST-05) |
| 2 | 308ea96 | test(12-03): unit + smoke tests for eval_falsifier.py (INST-05) |
| 3 | b2a42c8 | chore(12-03): eval-falsifier Makefile target + RUN_DIR var (INST-05) |

## Verification Results

- `poetry run pytest tests/unit/test_eval_falsifier.py -q`: 17 passed
- `poetry run ruff check scripts/eval_falsifier.py tests/unit/test_eval_falsifier.py`: LINT OK
- `make -n eval-falsifier`: prints expected command
- `make help | grep eval-falsifier`: listed with INST-05 description
- `poetry run python scripts/eval_falsifier.py --baselines-mode --baselines-dir configs/eval_baselines`: exit 1 (FAIL, real verdict — not infra error)
- `poetry run python scripts/eval_falsifier.py --run-dir /nonexistent`: exit 2
- `grep -c "import openai|from langchain|build_chat_model" scripts/eval_falsifier.py`: 0

## Deviations from Plan

**1. [Rule 1 - Bug fix] Removed unused `_get_metric_value` import**
- **Found during:** Task 1 lint check
- **Issue:** Plan's `key_links` listed `_get_metric_value` as an import to include, but the implementation computes anchor non-regression by running `_pooled_commit_rate` against the baseline summary dict (not scalar metric extraction). The import was unused.
- **Fix:** Removed `_get_metric_value` from the `check_eval_gates` import to satisfy ruff F401.
- **Rationale:** The anchor non-regression is better implemented as a pooled comparison (consistent with how gpt-5-mini is checked) rather than a per-metric scalar lookup.

**2. [Rule 2 - Design clarity] Baselines-mode anchor check is informational, not a gate**
- **Found during:** Task 1 implementation
- **Issue:** In `--baselines-mode`, comparing the anchor against itself (baselines = source AND reference) is vacuous and would always pass. The report clearly notes "anchor regression check skipped in baselines-mode".
- **Fix:** The anchor regression check only enforces in run-dir mode (comparing live run results against committed baseline floor). In baselines-mode, the anchor pool value is reported for reference.

## Self-Check

### Created Files
- `scripts/eval_falsifier.py`: EXISTS
- `tests/unit/test_eval_falsifier.py`: EXISTS

### Commits
- f478589: EXISTS (feat(12-03): eval_falsifier.py)
- 308ea96: EXISTS (test(12-03): unit + smoke tests)
- b2a42c8: EXISTS (chore(12-03): Makefile target)

## Self-Check: PASSED
