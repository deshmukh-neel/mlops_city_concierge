---
phase: 10-eval-harness-honesty
plan: "02"
subsystem: eval-harness
tags: [eval, error-threading, aggregation, tdd, summary-json]
dependency_graph:
  requires: [EVAL-01-error-status-runner]
  provides: [EVAL-01-error-status-aggregator]
  affects: [scripts/eval_matrix.py, tests/unit/test_eval_matrix.py]
tech_stack:
  added: []
  patterns: [error-threading, cell-validity, exit-code-distinct-counts, structural-check-check6]
key_files:
  created: []
  modified:
    - scripts/eval_matrix.py
    - tests/unit/test_eval_matrix.py
decisions:
  - "D-10-03 (second half): aggregate_cell_jsons reads n_scored/n_errored/errors from each cell JSON and surfaces per-provider n_scored/n_errored/cell_valid in summary.json"
  - "T-10-02-02: total_errored > 0 forces non-zero exit with a distinct stderr line (INVALID_FOR_BASELINE), separate from the subprocess failures line"
  - "Check 6 added to structural-check block: synthetic error cell with stage in {'setup','turn0','turnN'} validates error-schema contract without live calls"
metrics:
  duration: "~30 minutes"
  completed: "2026-06-10"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 10 Plan 02: Summary Error Threading Summary

**One-liner:** `aggregate_cell_jsons` now surfaces per-cell `n_scored`/`n_errored`/`cell_valid` and a top-level `errors` array in `summary.json`; the matrix exits non-zero with a distinct INVALID_FOR_BASELINE stderr line when any cell has errored runs, and structural-check Check 6 validates the error-schema shape in CI without live calls.

## What Was Built

Two TDD tasks extended the matrix aggregation layer (EVAL-01, second half) to make errored cells visible in `summary.json` and enforce a non-zero exit code when any cell is INVALID_FOR_BASELINE.

### Task 1: Thread n_scored / n_errored / errors through aggregate_cell_jsons + error-aware exit (RED + GREEN)

**`aggregate_cell_jsons` extension:**

The aggregation loop now reads `aggregate.n_scored`, `aggregate.n_errored`, and `aggregate.errors` from each cell JSON (written by 10-01). Per-(scenario, provider_key) block in `summary.json` gains three new fields:

```python
providers_out[provider_key] = {
    "scorers": {...},        # existing
    "n_scored": <int>,       # NEW: runs that produced a scored record
    "n_errored": <int>,      # NEW: runs that produced an error record
    "cell_valid": <bool>,    # NEW: n_errored == 0  (D-10-03)
}
```

**Top-level `errors` array:** Every errored cell's error entries are collected into a top-level `summary["errors"]` list shaped `{"cell": <filename>, "stage": <stage>, "type": <type>, "message": <msg>}`. The key is only emitted when non-empty (clean runs produce no `errors` key).

**Legacy backward-compat:** Cell JSONs missing the new fields (pre-10-01 shape) default to `n_errored=0`, keeping existing baselines and CI outputs unchanged.

**Provider-only-errored cells:** If a provider had only errored runs (no scored rows in `grouped`), the aggregation loop still emits the block via a post-pass over `error_counts`, with `"scorers": {}`.

**Error-aware exit code (main):**

`total_errored` is computed by summing `n_errored` across all provider-key blocks. When `total_errored > 0`, `main()` prints a DISTINCT stderr line:

```
eval_matrix: N run(s) had errors (INVALID_FOR_BASELINE); see .../summary.json#/errors
```

And forces `rc = max(rc, 1)`. This is separate from the existing subprocess failures line, satisfying T-10-02-02: an errored matrix cannot exit 0 and be confused for a clean baseline-eligible run.

### Task 2: Extend --structural-check with error-schema Check 6 + aggregation tests (RED + GREEN)

**Check 6 in structural-check block:**

Added after Check 5 (the shared-helper callable check), following the established 5-check pattern exactly:

```python
synthetic_error_cell = {
    "status": "error",
    "error": {"stage": "turn0", "type": "RateLimitError", "message": "quota"},
}
assert "status" in synthetic_error_cell and "error" in synthetic_error_cell
assert synthetic_error_cell["error"].get("stage") in {"setup", "turn0", "turnN"}
```

Failure exits 1 with a descriptive stderr line. Success folds into the existing OK print (which now appends `, error-schema valid` to the output).

**Unit tests added** (5 new tests + 1 new test class):

| Test | Asserts |
|------|---------|
| `test_aggregate_cell_jsons_threads_error_counts` | OK cell cell_valid=True, errored cell cell_valid=False, top-level errors non-empty |
| `test_aggregate_cell_jsons_cell_valid_true_when_no_errors` | Clean run: n_errored=0, cell_valid=True, errors=[] |
| `test_aggregate_cell_jsons_legacy_cell_json_defaults_to_zero_errored` | Legacy cell without n_errored field → defaults to n_errored=0 |
| `test_aggregate_cell_jsons_error_count_in_exit_code` | Errored cell visible in aggregated summary |
| `TestStructuralCheckErrorSchema.test_structural_check_validates_error_schema` | structural-check exits 0 with Check 6 in place |
| `TestStructuralCheckErrorSchema.test_structural_check_error_schema_check_is_present_in_source` | Stage membership guard present in main() source |

## Commits

| Hash | Description |
|------|-------------|
| 28c9fcc | test(10-02): add failing tests for error-threading in aggregate_cell_jsons (RED) |
| 44eec46 | feat(10-02): thread n_scored/n_errored/cell_valid through aggregate_cell_jsons + error-aware exit + structural-check Check 6 |

## Deviations from Plan

None — plan executed exactly as written. The implementation aligned precisely with the PATTERNS.md structural-check extension pattern and the D-10-03 cell validity spec from 10-01-SUMMARY.md.

## Verification

All plan verification criteria passed:

```
poetry run pytest tests/unit/test_eval_matrix.py -q
→ 49 passed in 0.74s

make eval-matrix-refinement-structural-check
→ structural-check: OK — matrix has 6 cell(s), env-override preserved through
   _apply_override, scorer registered, shared helper functional, error-schema valid

poetry run ruff check scripts/eval_matrix.py tests/unit/test_eval_matrix.py
→ All checks passed!

grep -n "n_errored" scripts/eval_matrix.py | head -5
→ shows n_errored read inside aggregate_cell_jsons AND used in main() exit computation

grep -n "cell_valid" scripts/eval_matrix.py
→ 3 matches in aggregate_cell_jsons output

poetry run python -c "from scripts.eval_matrix import aggregate_cell_jsons; import inspect; assert 'n_errored' in inspect.getsource(aggregate_cell_jsons)"
→ OK
```

## Known Stubs

None. All error-threading fields, cell_valid flags, and top-level errors array are wired to real behavior derived from per-cell JSON aggregate fields.

## Threat Flags

No new threat surface introduced. Mitigations applied per the plan's STRIDE register:

- T-10-02-01 (Tampering): `cell_valid` derived directly from `n_errored` read from each cell JSON; structural-check Check 6 validates the error-schema shape so a malformed error block is caught in CI before any live run.
- T-10-02-02 (Repudiation): `n_errored > 0` forces non-zero exit with a distinct `INVALID_FOR_BASELINE` stderr line; an errored matrix cannot exit 0 and be mistaken for a clean baseline.
- T-10-02-03 (DoS): Check 6 uses only in-memory synthetic dicts — no subprocess, no live calls, zero external attack surface.

## Self-Check

### Files exist:
- [x] scripts/eval_matrix.py (modified)
- [x] tests/unit/test_eval_matrix.py (modified)

### Commits exist:
- [x] 28c9fcc
- [x] 44eec46

## Self-Check: PASSED
