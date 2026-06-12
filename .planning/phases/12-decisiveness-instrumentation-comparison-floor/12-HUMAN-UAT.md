---
status: partial
phase: 12-decisiveness-instrumentation-comparison-floor
source: [12-VERIFICATION.md]
started: 2026-06-11T00:00:00Z
updated: 2026-06-11T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Falsifier grades the correct matrix run in run-dir mode

expected: `make eval-falsifier RUN_DIR=<a real eval_matrix run dir>` (or letting `_latest_run_dir` auto-resolve when the most recent run is a main-matrix run) reports gpt-5-mini pooled commit rate and the gpt-4o-mini anchor from THAT run — not from a refinement-matrix run. `_latest_run_dir` picks by ISO8601 directory name with no validation that the summary came from `configs/eval_matrix.yaml`; if a refinement run is more recent, the falsifier silently grades the wrong artifact (also flagged as WR-06 in 12-REVIEW.md).
result: [pending]

### 2. Anchor non-regression fails correctly when anchor score drops below baseline

expected: With a temporarily lowered gpt-4o-mini baseline (or a live run where the anchor regresses), `eval_falsifier.py` exits 1 and reports the anchor FAIL with per-model numbers. CR-01 in 12-REVIEW.md flags that the anchor comparison pools mismatched scenario sets (baselines include `refinement_cheaper`; an eval-matrix run summary contains only `omakase_mission_open_ended`) — dormant today because both committed anchor medians are 1.0, but any honest regen below 1.0 could produce a wrong verdict.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
