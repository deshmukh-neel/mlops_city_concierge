---
status: complete
phase: 12-decisiveness-instrumentation-comparison-floor
source: [12-VERIFICATION.md]
started: 2026-06-11T00:00:00Z
updated: 2026-06-11T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Falsifier grades the correct matrix run in run-dir mode

expected: `make eval-falsifier RUN_DIR=<a real eval_matrix run dir>` (or letting `_latest_run_dir` auto-resolve when the most recent run is a main-matrix run) reports gpt-5-mini pooled commit rate and the gpt-4o-mini anchor from THAT run — not from a refinement-matrix run. `_latest_run_dir` picks by ISO8601 directory name with no validation that the summary came from `configs/eval_matrix.yaml`; if a refinement run is more recent, the falsifier silently grades the wrong artifact (also flagged as WR-06 in 12-REVIEW.md).
result: issue
reported: "Live run confirmed: source line + wrong-matrix WARNING print correctly (WR-06 fix works), but the falsifier still emits a verdict and exits 1 after the warning. In scripted/CI use the exit code IS the contract — a wrong-matrix run can produce a spurious FAIL (or worse, a spurious PASS on the milestone bar). Zero scenario overlap with eval_matrix.yaml should be exit 2 (infrastructure/usage error, no verdict), not warn-and-continue. User delegated judgment to Claude; verdict given project history of fail-open gate bugs."
severity: major

### 2. Anchor non-regression fails correctly when anchor score drops below baseline

expected: With a temporarily lowered gpt-4o-mini baseline (or a live run where the anchor regresses), `eval_falsifier.py` exits 1 and reports the anchor FAIL with per-model numbers. CR-01 in 12-REVIEW.md flags that the anchor comparison pools mismatched scenario sets (baselines include `refinement_cheaper`; an eval-matrix run summary contains only `omakase_mission_open_ended`) — dormant today because both committed anchor medians are 1.0, but any honest regen below 1.0 could produce a wrong verdict.
result: pass
evidence: "Live run printed intersection-exclusion note (CR-01 fix working against real artifacts); committed tests cover regression exit-1 path and per-model FAIL message (TestAnchorCommonScenarioPooling, 6/6 green)." 

## Summary

total: 2
passed: 1
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "When the resolved run dir's summary shares zero scenarios with configs/eval_matrix.yaml, eval_falsifier refuses to grade: prints the wrong-matrix diagnosis and exits 2 (no PASS/FAIL verdict line)"
  status: failed
  reason: "User reported (via delegated judgment): warning prints but verdict + exit 1 still emitted; scripted consumers read exit code only — spurious FAIL today, spurious milestone PASS possible if a refinement run clears 0.6"
  severity: major
  test: 1
  artifacts: ["scripts/eval_falsifier.py", "tests/unit/test_eval_falsifier.py"]
  missing: ["zero-overlap guard before verdict emission: exit 2 instead of warn-and-continue", "unit test: zero-overlap summary -> exit 2, no VERDICT line printed"]
