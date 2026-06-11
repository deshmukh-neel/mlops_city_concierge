---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 05
subsystem: eval-tooling
tags: [baseline-writer, eval-matrix, D-11-07, D-10-03, D-10-09, BASE-01, TDD]
dependency_graph:
  requires: ["11-03"]
  provides: ["scripts/write_baselines.py", "make write-baselines", "make snapshot-baselines"]
  affects: ["11-06", "11-08"]
tech_stack:
  added: []
  patterns:
    - stdlib-only CLI script (argparse, json, datetime, pathlib) mirroring check_eval_gates.py shape
    - 0/1/2 exit-code contract (0=success, 1=refusal/content, 2=infra)
    - TDD RED/GREEN commit sequence
key_files:
  created:
    - scripts/write_baselines.py
    - tests/unit/test_write_baselines.py
  modified:
    - Makefile
decisions:
  - "D-11-07: write_baselines.py is stdlib-only; refuses n_scored < n_requested (D-10-03) and baseline_eligible=False (D-10-09); exits 1 on refusal, 2 on infra failure"
metrics:
  duration: 12m
  completed: 2026-06-11
  tasks: 2
  files: 3
---

# Phase 11 Plan 05: write-baselines tool Summary

**One-liner:** Stdlib-only baseline writer tool with D-10-03 partial-cell refusal, D-10-09 quarantine enforcement, provenance stamping, and `_observations` carry-forward — baselines are never hand-rolled again.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for write_baselines.py | 0e9605e | tests/unit/test_write_baselines.py |
| 1 (GREEN) | Implement scripts/write_baselines.py | 9f495ed | scripts/write_baselines.py |
| 2 | Makefile write-baselines + snapshot-baselines targets | 8424fe9 | Makefile |

## What Was Built

### `scripts/write_baselines.py`

Machine-enforced baseline writer for eval-matrix output. Key properties:

- **D-10-03 refusal:** any provider cell with `n_scored < n_requested` is refused; no partial baseline is written; exit 1
- **D-10-09 quarantine:** scenarios with `baseline_eligible=False` have all their provider cells refused; exit 1
- **Provenance stamping:** every written cell carries `generated_at` (UTC compact timestamp) and `generated_by = "scripts/write_baselines.py"`
- **`_observations` carry-forward:** prior `_observations` annotations on existing baseline cells are preserved through rewrites
- **Cell merge:** provider cells not in the current summary.json are preserved from the prior baseline file (additive update, not full replace)
- **Exit-code contract:** 0 = all eligible cells written, 1 = any refusal, 2 = infra failure (missing or malformed summary.json)
- **stdlib-only:** argparse, json, datetime, pathlib — no LLM SDK imports; importable with all provider keys unset

### `tests/unit/test_write_baselines.py`

Seven behavior tests following the TDD RED→GREEN sequence:

1. Eligible cell (n_scored == n_requested) writes baseline JSON with correct shape and exits 0
2. Partial cell (n_scored < n_requested) is refused, D-10-03 in stderr, no file written, exits 1
3. Quarantined scenario (baseline_eligible=False) cells refused, D-10-09 in stderr, no file written, exits 1
4. Prior `_observations` carried forward on rewrite
5. Missing summary.json exits 2 (distinct from refusal)
6. Malformed (non-JSON) summary.json exits 2
7. Module imports with all four provider API keys unset

### Makefile targets

- `write-baselines`: `$(POETRY_RUN) python scripts/write_baselines.py $(SUMMARY) --n-requested $(RUNS) --baselines-dir configs/eval_baselines`
- `snapshot-baselines`: copies three canonical baselines to `_snapshots/*.pre-phase11.json` for pre-Wave-2 audit trail

## Verification Results

- `poetry run pytest tests/unit/test_write_baselines.py` — 7/7 PASSED
- `poetry run ruff check scripts/write_baselines.py` — clean
- `poetry run python scripts/write_baselines.py --help` — exit 0
- `make test` (full unit suite) — 1185 passed, 11 skipped
- `make -n write-baselines SUMMARY=/tmp/x.json RUNS=5` — shows `--n-requested 5` invocation
- `make -n snapshot-baselines` — shows three cp commands targeting `_snapshots/*.pre-phase11.json`

## TDD Gate Compliance

- RED gate: `test(11-05)` commit 0e9605e (failing tests — all 7 failed on ModuleNotFoundError)
- GREEN gate: `feat(11-05)` commit 9f495ed (implementation — all 7 pass)
- REFACTOR gate: not needed (implementation was clean on first pass)

## Deviations from Plan

None — plan executed exactly as written. The `datetime.utcnow()` deprecation warning is intentional (the DTZ003 noqa suppresses it) per the codebase's existing `YYYY-MM-DDTHH-MM-SSZ` compact format convention used across all baseline/eval timestamp fields.

## Threat Flags

None. The writer copies only numeric scorer stats and provenance stamps into baseline JSON files. No env vars, prompts, raw model output, or provider secrets are written. T-11-11, T-11-12, T-11-13, T-11-05-SC all mitigated as designed.

## Self-Check: PASSED

- `scripts/write_baselines.py` exists: FOUND
- `tests/unit/test_write_baselines.py` exists: FOUND
- Commits 0e9605e, 9f495ed, 8424fe9 all exist in git log: FOUND
