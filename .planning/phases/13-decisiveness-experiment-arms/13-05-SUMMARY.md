---
phase: 13-decisiveness-experiment-arms
plan: "05"
subsystem: eval/configs
tags: [arm-matrix-config, falsifier, eval-plumbing, tdd, D-13-02, D-13-04]
dependency_graph:
  requires: [13-01-viability-predicate-and-telemetry]
  provides: [arm-matrix-config, falsifier-arm-mode, eval-matrix-arm-target]
  affects: [configs/eval_matrix_arm.yaml, scripts/eval_falsifier.py, Makefile]
tech_stack:
  added: []
  patterns: [tdd-red-green, argparse-extension, glob-based-run-dir-reading]
key_files:
  created:
    - configs/eval_matrix_arm.yaml
  modified:
    - scripts/eval_falsifier.py
    - Makefile
    - tests/unit/test_eval_falsifier.py
decisions:
  - "scenario IDs in arm config exactly match committed baseline file stems (omakase_mission_open_ended, refinement_cheaper) so falsifier zero-overlap exit-2 guard is satisfied without special-casing"
  - "_commit_split_from_run_dir uses provider_slug prefix stripping (not split('--')) to correctly handle multi-segment provider slugs like openai--gpt-5-mini"
  - "split annotation added to both gpt-5-mini AND anchor verdict lines in run-dir mode — symmetric with D-13-04 intent"
  - "arm flags omitted from per-cell env in eval_matrix_arm.yaml (set via export before invocation) per D-13-01 design"
metrics:
  duration: "~7 minutes"
  completed: "2026-06-12T06:05:36Z"
  tasks: 3
  files: 4
---

# Phase 13 Plan 05: Arm Matrix Config and Falsifier Summary

3-provider x 2-scenario arm matrix config + falsifier --matrix-config flag with model-initiated vs forced commit split reader + Makefile eval-matrix-arm/eval-falsifier-arm targets.

## What Was Built

**Task 1: Arm matrix config — `configs/eval_matrix_arm.yaml`**

New YAML config declaring exactly 3 providers (openai/gpt-4o-mini anchor,
openai/gpt-5-mini, deepseek/deepseek-reasoner) and 2 scenarios
(omakase_mission_open_ended, refinement_cheaper). anthropic/gemini cells
deferred (D-12-09). late_night_closure_cascade excluded (D-10-09 quarantine).
Scenario IDs match committed baseline file stems in configs/eval_baselines/
so the falsifier zero-overlap exit-2 guard passes when grading arm run dirs.
Header comment documents the arm flag design (set via env export before
invocation, not per-cell env in the YAML).

**Task 2 (TDD): Falsifier extensions — `scripts/eval_falsifier.py`**

Two extensions:
1. `--matrix-config` argument added to `_parse_args`. The zero-overlap exit-2
   guard now reads whichever matrix config is passed (default: eval_matrix.yaml)
   via `_expected_matrix_scenarios(matrix_config_path)`. Arm run dirs containing
   refinement_cheaper (not in default eval_matrix.yaml) pass the guard when
   `--matrix-config configs/eval_matrix_arm.yaml` is given.
2. `_commit_split_from_run_dir(run_dir, provider_key, scenario_ids=None)` helper
   reads individual *.json run files (not summary.json), aggregates
   `commit_forced` (True → forced += 1) and `first_commit_call_step` (not None
   → model_initiated += 1). Malformed JSON / OSError silently skipped
   (T-13-05-01). The split is printed in the gpt-5-mini AND anchor verdict lines
   in run-dir mode: `(model-initiated {mi}/{total}, forced {fc}/{total})`.

TDD: 16 new tests across `TestMatrixConfigFlag` and `TestCommitSplitFromRunDir`.
All 51 falsifier tests pass (35 pre-existing + 16 new).

**Task 3: Makefile targets — `Makefile`**

Two new targets added after `eval-falsifier`:
- `eval-matrix-arm`: runs eval_matrix.py with `--matrix-config configs/eval_matrix_arm.yaml`
  and `--runs $(RUNS)`. Help text documents per-arm env flag export patterns
  (A1=VIABILITY_CONTRACT_ENABLED, A2=FORCED_COMMIT_STEP, A3=PARALLEL_TOOL_EXECUTION_ENABLED).
- `eval-falsifier-arm`: runs eval_falsifier.py with `--run-dir`, `--matrix-config`
  (defaults to arm config), and `--baselines-dir`. Graders the arm run dir
  against the two-scenario universe.
- Added `MATRIX_CONFIG ?=` parameter variable near `RUN_DIR ?=`.

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- `poetry run pytest tests/unit/test_eval_falsifier.py -v` — 51 passed (35 pre-existing + 16 new)
- `poetry run python scripts/eval_matrix.py --matrix-config configs/eval_matrix_arm.yaml --runs 1 --llm-provider-override scripted` — parses config and writes summary.json (exits 1 on violation cells per scripted mode, no infra error)
- `make -n eval-matrix-arm` — prints correct eval_matrix.py invocation with --matrix-config arm.yaml
- `make -n eval-falsifier-arm RUN_DIR=eval_reports/x` — prints correct eval_falsifier.py invocation with --matrix-config arm.yaml
- `make lint` — All checks passed

## Commits

| Hash | Message |
|------|---------|
| 12e46a2 | feat(13-05): add arm matrix config (3 providers x 2 scenarios) |
| b31850a | test(13-05): add failing tests for --matrix-config flag and forced-commit split reader |
| 9233c10 | feat(13-05): extend falsifier with --matrix-config flag and forced-commit split reader |
| 2f049dd | feat(13-05): add eval-matrix-arm and eval-falsifier-arm Makefile targets |

## Self-Check: PASSED

- [x] `configs/eval_matrix_arm.yaml` exists with 3 entries and 2 scenarios
- [x] `scripts/eval_falsifier.py` contains `--matrix-config` arg and `_commit_split_from_run_dir`
- [x] `Makefile` contains `eval-matrix-arm` and `eval-falsifier-arm` targets
- [x] All 51 falsifier tests pass
- [x] Lint clean
- [x] No new network endpoints, auth paths, or trust-boundary crossings

## Known Stubs

None - all data flows are wired through real config files and script arguments.

## Threat Flags

None - no new network endpoints, auth paths, file access patterns beyond the local
eval_reports filesystem, or schema changes at trust boundaries.
T-13-05-01 (malformed JSON crashing split reader) mitigated via try/except in
`_commit_split_from_run_dir`.
T-13-05-03 (forced commits silently inflating commit rate) mitigated via explicit
split annotation in verdict lines.
