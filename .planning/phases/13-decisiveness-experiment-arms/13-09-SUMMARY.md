---
phase: 13-decisiveness-experiment-arms
plan: "09"
subsystem: scripts/eval + docs
tags: [cr-02, falsifier, split-reader, tdd, docs, regression-test]
dependency_graph:
  requires: [13-08]
  provides: [CR-02 split reader fix, fixture pinned to real EvalRunReport shape, verdict doc CR-02 annotation]
  affects: [scripts/eval_falsifier.py, tests/unit/test_eval_falsifier.py, docs/decisiveness_arm_verdicts.md]
tech_stack:
  added: []
  patterns: [TDD red/green, asdict(EvalRunReport) fixture shape derivation]
key_files:
  created: []
  modified:
    - scripts/eval_falsifier.py
    - tests/unit/test_eval_falsifier.py
    - docs/decisiveness_arm_verdicts.md
decisions:
  - "CR-02 fix: _commit_split_from_run_dir iterates data.get('queries') and reads query.get('deterministic') per entry; top-level data.get('deterministic') is gone"
  - "Fixture _write_run_file rebuilt to use real EvalRunReport/QueryEvalResult/DeterministicEvalResult dataclasses from scripts/eval_agent.py via report_to_dict(asdict); checks dict populated from DETERMINISTIC_CHECKS with score=None entries (make_error_record pattern); status='error' avoids KeyError from aggregate_results scorer means"
  - "_write_run_file_old_shape added to pin the pre-fix top-level shape; CR-02 regression test asserts (0,0) on old shape under fixed reader"
  - "Scenario filtering: fixed reader prefers query.get('scenario_id') from the query record, falls back to filename parsing only when absent"
  - "A1 and A3 verdict sections get brief cross-reference notes pointing to the A2 CR-02 annotation block"
metrics:
  duration: "5m"
  completed: "2026-06-12"
  tasks: 2
  files: 3
---

# Phase 13 Plan 09: CR-02 Falsifier Split Reader Fix Summary

**One-liner:** Fix `_commit_split_from_run_dir` to read `queries[i].deterministic` (not top-level), pin fixture to real EvalRunReport shape, add CR-02 regression tests, and annotate the verdict doc that pasted 0/0 falsifier output was a tool bug.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 RED | Failing CR-02 regression tests for queries[i].deterministic reader | 55e89b6 | tests/unit/test_eval_falsifier.py |
| 1 GREEN | Fix _commit_split_from_run_dir to iterate queries[i] | 5748c0e | scripts/eval_falsifier.py, tests/unit/test_eval_falsifier.py |
| 2 | Annotate verdict doc with CR-02 tool bug note | 7214316 | docs/decisiveness_arm_verdicts.md |

## What Was Built

### Task 1: Split reader fix (CR-02) + regression tests

**Root cause:** `_commit_split_from_run_dir` called `data.get("deterministic") or {}` at the top level of each per-run JSON. The real `EvalRunReport` serialized by `eval_agent.py` (via `report_to_dict = asdict(report)`) puts `deterministic` inside each `queries[i]`, not at the top level. Every call returned `{}`, so forced and model_initiated were always 0/0.

**Fix:** Replaced the single top-level read with iteration over `data.get("queries") or []`. For each `query` dict, reads `det = query.get("deterministic")`, skips if not a dict, then applies the existing forced/model_initiated classification logic. Scenario filtering prefers `query.get("scenario_id")` from the query record, falls back to filename parsing for backward compat. The existing `summary.json` skip and `OSError/ValueError` swallow-and-skip guard (T-13-05-01) are preserved.

**Fixture fix:** `_write_run_file` in `TestCommitSplitFromRunDir` was rewritten to use the real dataclasses — `EvalRunReport`, `QueryEvalResult`, `DeterministicEvalResult`, `CheckResult` — imported from `scripts.eval_agent`. The `checks` dict is populated from `DETERMINISTIC_CHECKS` with `score=None` entries (same pattern as `make_error_record`). `status="error"` avoids a `KeyError` from `aggregate_results` iterating scorer means on empty checks. The fixture now cannot pass while the reader reads the wrong JSON level.

**Added tests (4 CR-02 regression tests):**
- `test_cr02_real_shape_returns_nonzero_counts` — 2 forced + 2 model-initiated + 1 never-committed on real EvalRunReport shape returns (2, 2); FAILS on pre-fix reader (returns (0,0))
- `test_cr02_old_top_level_shape_returns_zeros` — old top-level-only shape returns (0, 0) under fixed reader; pins that old fixture format no longer produces false positives
- `test_cr02_commit_forced_and_model_initiated_classification` — per-query forced vs model-initiated vs neither classification
- `test_cr02_scenario_filtering_reads_from_query_scenario_id` — scenario filtering via `query.get("scenario_id")` correctly excludes out-of-scope scenarios

**Acceptance criteria verified:**
- `grep -n "queries" scripts/eval_falsifier.py` — line 207: `for query in data.get("queries") or []:`
- `grep -n 'data.get("deterministic")' scripts/eval_falsifier.py` — NOT FOUND (top-level read gone)
- Fixture top-level keys include `queries`; `deterministic` nested under `queries[i]`
- Fixture field set derived from `EvalRunReport`/`QueryEvalResult`/`DeterministicEvalResult` dataclasses
- `poetry run pytest tests/unit/test_eval_falsifier.py -q` — 55 passed

### Task 2: Verdict doc CR-02 annotation

Added a main CR-02 annotation block in the A2 section, immediately before the "Key finding — FORCED_COMMIT_STEP=6" paragraph. The annotation:
- (a) States the pasted `(model-initiated 0/0, forced 0/0)` lines in all pasted verbatim falsifier outputs were produced by a tool bug (CR-02: reader read `deterministic` at wrong JSON level)
- (b) States the hand-computed split numbers in the per-model tables (e.g. "model-initiated 4/10, forced 0/10") are CORRECT and were verified directly from `queries[i].deterministic` data
- (c) States the bug is fixed in `scripts/eval_falsifier.py` (Plan 13-09) and names the regression test
- Explicitly notes the pasted 0/0 lines are preserved as historical record, not deleted

Added brief cross-reference notes in the A1 and A3 pasted verbatim output sections, pointing to the main CR-02 annotation in A2.

**Pasted verbatim 0/0 output preserved:** `grep -c "model-initiated 0/0" docs/decisiveness_arm_verdicts.md` = 9 (historical record intact). Honest null result ("No arm cleared the INST-05 falsifier bar") preserved and unaltered.

## Deviations from Plan

None — plan executed exactly as written.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or trust boundaries introduced. The falsifier remains a local read-only analysis tool.

## Known Stubs

None.

## Verification Results

- `poetry run pytest tests/unit/test_eval_falsifier.py -q` — 55 passed
- `make test` — 1354 passed, 53 skipped, 17 warnings (full suite clean)
- `grep -n "queries" scripts/eval_falsifier.py` — line 207 shows iteration
- `grep -n 'data.get("deterministic")' scripts/eval_falsifier.py` — NOT FOUND
- Verdict doc: "tool bug", "CR-02", "4/10", "model-initiated 0/0" (preserved), "No arm cleared" all present
- `git status configs/eval_baselines/` — clean (no baselines written)

## Self-Check: PASSED

Files verified:
- scripts/eval_falsifier.py — exists; `for query in data.get("queries") or []` at line 207; no `data.get("deterministic")`
- tests/unit/test_eval_falsifier.py — 4 new CR-02 tests added; fixture uses real EvalRunReport dataclasses; all 55 tests pass
- docs/decisiveness_arm_verdicts.md — CR-02 annotation block present with "tool bug", "queries[i]", "hand-computed", "fixed"; cross-references in A1 and A3; pasted 0/0 lines preserved

Commits verified: 55e89b6, 5748c0e, 7214316 — all present in git log.
