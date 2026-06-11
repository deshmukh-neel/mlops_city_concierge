---
phase: 10-eval-harness-honesty
plan: "03"
subsystem: eval-harness
tags: [eval, quarantine, baseline-eligible, tdd, parity, annotation]
dependency_graph:
  requires: [EVAL-01-error-status-runner, EVAL-01-error-status-aggregator]
  provides: [EVAL-02-quarantine-flag, EVAL-04-parity-verified]
  affects:
    - app/eval/config.py
    - configs/eval_queries.yaml
    - configs/eval_matrix.yaml
    - configs/eval_baselines/late_night_closure_cascade.json
    - scripts/eval_matrix.py
    - tests/unit/test_eval_matrix.py
tech_stack:
  added: []
  patterns: [baseline_eligible-field, annotation-not-regen, quarantine-distinct-from-deferral]
key_files:
  created: []
  modified:
    - app/eval/config.py
    - configs/eval_queries.yaml
    - configs/eval_matrix.yaml
    - configs/eval_baselines/late_night_closure_cascade.json
    - scripts/eval_matrix.py
    - tests/unit/test_eval_matrix.py
decisions:
  - "D-10-09: late_night_closure_cascade quarantined via baseline_eligible=False field on EvalQuery; recorded in three places (eval_queries.yaml, eval_matrix.yaml, baseline JSON _observations)"
  - "D-10-10: late_night baseline JSON annotated with _observations citing legacy-threading shape; NOT regenerated, NOT deleted; 1 insertion, 0 deletions"
  - "T-10-03-03: baseline_eligible defaults to True (fail toward enforcement; quarantine is opt-in and explicit)"
  - "EVAL-04: parity test test_baseline_provider_cells_match_matrix_entries verified to cover all current matrix files; late_night quarantine is NOT a deferral (not added to _DEFERRED_BASELINE_CELLS)"
metrics:
  duration: "~25 minutes"
  completed: "2026-06-10"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 6
---

# Phase 10 Plan 03: Quarantine and Parity Summary

**One-liner:** `late_night_closure_cascade` quarantined from baselines and gates via a parsed `baseline_eligible: false` field on `EvalQuery`, recorded in three places (scenario config, matrix YAML comment, baseline JSON `_observations`); PR #104 parity test verified to cover all matrix files with quarantine distinct from deferral.

## What Was Built

Two TDD tasks implemented EVAL-02 (quarantine flag) and verified EVAL-04 (parity test coverage) for the `late_night_closure_cascade` scenario.

### Task 1: baseline_eligible field + quarantine wiring (RED + GREEN)

**`EvalQuery.baseline_eligible` field (app/eval/config.py):**

Added `baseline_eligible: bool = True` as a declared optional field with a default of True. The default preserves all 30 legacy cases and `omakase_mission_open_ended`; the field being declared means `extra="forbid"` accepts it when present in YAML. A docstring block cites D-10-09 and explains the legacy threading shape rationale.

**eval_queries.yaml quarantine annotation:**

The `late_night_closure_cascade` case gains `baseline_eligible: false` with a multi-line inline comment citing D-10-09 and linking to `eval_gates.yaml`.

**eval_matrix.yaml breadcrumb comment:**

A D-10-09/10 comment block is added adjacent to the `late_night_closure_cascade` scenario entry explaining the quarantine rationale, the deferred migration, and the three-place recording rule.

**`aggregate_cell_jsons` extension (scripts/eval_matrix.py):**

The function gains an optional `eval_queries_config: EvalQueriesConfig | None = None` parameter. When provided, it builds a `scenario_id -> baseline_eligible` lookup from the eval-queries config and surfaces a `baseline_eligible` boolean in each scenario block of `summary.json`. Unknown scenario IDs default to `True` (fail toward enforcement, not silent exclusion — T-10-03-03). The scenario still RUNS as a diagnostic; only the flag distinguishes it from an eligible scenario.

### Task 2: Baseline JSON annotation + EVAL-04 parity test (Task 2)

**`_observations` annotation (configs/eval_baselines/late_night_closure_cascade.json):**

A top-level `_observations` key was added with a D-10-10 citation. The `providers` block and all score values are byte-unchanged: `git diff --stat` shows 1 insertion, 0 deletions.

**`test_late_night_scenario_is_baseline_ineligible` test:**

Added to `tests/unit/test_eval_matrix.py`. Asserts:
1. `load_eval_queries(...)` yields `baseline_eligible=False` for `late_night_closure_cascade`
2. The baseline JSON has a `_observations` key citing D-10-10
3. The `providers` block is still present (not deleted)

**EVAL-04 parity test verified:**

`test_baseline_provider_cells_match_matrix_entries` is parametrized over `_MATRIX_TO_BASELINES` (both `eval_matrix.yaml` and `eval_matrix_refinement.yaml`). The test passes for both matrix files. `late_night_closure_cascade` is NOT added to `_DEFERRED_BASELINE_CELLS` — the quarantine is not a deferral (the baseline JSON still has the providers block matching the matrix entries).

## Commits

| Hash    | Description |
|---------|-------------|
| 7b42bf2 | test(10-03): add failing tests for baseline_eligible quarantine flag (RED) |
| 80218ea | feat(10-03): add baseline_eligible to EvalQuery; quarantine late_night in configs; honor flag in aggregator |
| a2c8792 | feat(10-03): annotate late_night baseline JSON with _observations; add EVAL-04 parity + ineligibility tests |

## TDD Gate Compliance

RED gate: `7b42bf2` (test commit — 6 failing tests for `baseline_eligible` field and aggregator behavior)
GREEN gate: `80218ea` (implementation commit — all 6 tests pass)
REFACTOR gate: Not needed — implementation was clean on first pass.

## Deviations from Plan

None — plan executed exactly as written.

The TDD RED/GREEN cycle followed the plan's `tdd="true"` directive:
- RED: 6 tests added covering `EvalQuery.baseline_eligible` field (default True), `late_night` YAML parse (False), `omakase` guard (True), aggregator `eval_queries_config` param, and scenario block marker
- GREEN: field added to `EvalQuery`, YAML updated, `aggregate_cell_jsons` extended

## Verification

All plan verification criteria passed:

```
poetry run pytest tests/unit/test_eval_matrix.py -q
→ 56 passed in 0.84s

python -c "import json; json.load(open('configs/eval_baselines/late_night_closure_cascade.json'))"
→ OK (valid JSON after annotation)

poetry run ruff check app/eval/config.py scripts/eval_matrix.py tests/unit/test_eval_matrix.py
→ All checks passed!

poetry run python -c "from app.eval.config import load_eval_queries; cfg=load_eval_queries('configs/eval_queries.yaml'); ln=[c for c in cfg.hand_written if c.id=='late_night_closure_cascade'][0]; om=[c for c in cfg.hand_written if c.id=='omakase_mission_open_ended'][0]; assert ln.baseline_eligible is False and om.baseline_eligible is True"
→ OK

git diff --stat configs/eval_baselines/late_night_closure_cascade.json
→ 1 file changed, 1 insertion(+) — annotation-only, no score line deletions
```

## Known Stubs

None. The `baseline_eligible` flag is a first-class parsed field that wires directly from YAML to `EvalQuery` to `aggregate_cell_jsons` output.

## Threat Flags

No new threat surface introduced. Mitigations applied per the plan's STRIDE register:

- T-10-03-01 (Tampering): annotation-only change; git diff confirms 1 insertion, 0 deletions on the baseline JSON — no score values altered.
- T-10-03-02 (Repudiation): quarantine decision recorded in three places (eval_queries.yaml inline comment, eval_matrix.yaml comment block, baseline JSON `_observations`) all citing D-10-09/10.
- T-10-03-03 (Elevation of privilege): `baseline_eligible` defaults to True — new scenarios are parity-checked by default; quarantine is opt-in and explicit.
- T-10-03-SC (package installs): no package installs in this plan.

## Self-Check

### Files exist:
- [x] app/eval/config.py (modified — baseline_eligible field)
- [x] configs/eval_queries.yaml (modified — baseline_eligible: false on late_night)
- [x] configs/eval_matrix.yaml (modified — D-10-09/10 comment block)
- [x] configs/eval_baselines/late_night_closure_cascade.json (modified — _observations added)
- [x] scripts/eval_matrix.py (modified — eval_queries_config param + baseline_eligible in scenario block)
- [x] tests/unit/test_eval_matrix.py (modified — 7 new tests)

### Commits exist:
- [x] 7b42bf2 (RED tests)
- [x] 80218ea (GREEN implementation)
- [x] a2c8792 (Task 2 annotation + parity tests)

## Self-Check: PASSED
