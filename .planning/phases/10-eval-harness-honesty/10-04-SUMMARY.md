---
phase: 10-eval-harness-honesty
plan: "04"
subsystem: eval-gates
tags: [eval, gates, yaml, machine-readable, tdd]
dependency_graph:
  requires: []
  provides:
    - configs/eval_gates.yaml
    - scripts/check_eval_gates.py
    - docs/eval_gates.md
    - Makefile eval-gates-check target
  affects:
    - configs/eval_matrix_refinement.yaml (gate comments migrated)
tech_stack:
  added: []
  patterns:
    - machine-readable config in configs/ consumed by scripts/ with Make target wrapper
    - check_baselines_fresh.py exit-code convention (0/1/2) replicated exactly
    - TDD: RED commit (failing tests) → GREEN commit (implementation) → verified clean
key_files:
  created:
    - configs/eval_gates.yaml
    - scripts/check_eval_gates.py
    - docs/eval_gates.md
    - tests/unit/test_check_eval_gates.py
  modified:
    - configs/eval_matrix_refinement.yaml
    - Makefile
decisions:
  - "D-10-05: configs/eval_gates.yaml is the single source of truth for merge gates"
  - "D-10-06: strict refinement_minimal_edit == 1.0 gate formally retired"
  - "D-10-07: active gpt-4o-mini committed_itinerary_rate >= 0.8; gpt-5-mini aspirational >= 0.6; anthropic provisional-n1 >= 0.8; deepseek/gemini logged"
  - "D-10-08: YAML per-entry fields: family, status, rationale, hard, advisory"
  - "T-10-04-01: metric absent from summary is NOT-EVALUABLE, not a silent pass"
metrics:
  duration: 45m
  completed_date: "2026-06-10"
  tasks_completed: 2
  files_created: 4
  files_modified: 2
---

# Phase 10 Plan 04: Eval Gates — Machine-Readable Gate Source of Truth Summary

Machine-readable per-family merge gates in `configs/eval_gates.yaml` with an executable gate-check script (`scripts/check_eval_gates.py`) and `make eval-gates-check` target; formally retires the unsatisfiable strict `refinement_minimal_edit == 1.0` gate.

## What Was Built

### Task 1: configs/eval_gates.yaml + docs/eval_gates.md + eval_matrix_refinement.yaml migration

`configs/eval_gates.yaml` is the single source of truth for all per-family merge gates (EVAL-03 / D-10-05). It contains seven entries covering the full provider matrix:

| Family | Status | Hard Gate |
|--------|--------|-----------|
| openai/gpt-4o-mini | active | committed_itinerary_rate >= 0.8 |
| openai/gpt-5-mini | aspirational | committed_itinerary_rate >= 0.6 |
| anthropic/claude-sonnet-4-6 | provisional-n1 | committed_itinerary_rate >= 0.8 |
| deepseek/deepseek-reasoner | logged | none |
| deepseek/deepseek-chat | logged | none |
| gemini/gemini-3.1-pro-preview | logged | none |
| late_night_closure_cascade | quarantined-legacy-threading | none |

The strict Phase-6-era `refinement_minimal_edit == 1.0` gate is formally retired (D-10-06). It was authored against fail-open baselines; the honest anchor sits at median 0.0/max 0.5 post-D-07-05. `refinement_minimal_edit` medians are now advisory everywhere until v2.2.

`docs/eval_gates.md` explains each status value and how to run the gate check. It links to the YAML and never duplicates numeric gate values.

`configs/eval_matrix_refinement.yaml` comments were updated to point at `configs/eval_gates.yaml` — numeric gate values removed from comments per D-10-05 (the fossilized-Makefile-number failure mode that produced the strict-1.0 gate is eliminated).

**Commit:** d5d8615

### Task 2: scripts/check_eval_gates.py + Makefile eval-gates-check + unit tests (TDD)

**RED commit:** f6e23d9 — 11 failing tests for all four exit-code outcomes, not-evaluable reporting, and logged/quarantined skip behavior.

**GREEN commit:** 2059b5c — implementation passes all 11 tests.

`scripts/check_eval_gates.py` follows the `check_baselines_fresh.py` pattern exactly:
- `main(argv=None) -> int` with `raise SystemExit(main())`
- Exit 0: all active/provisional-n1 hard gates pass; aspirational misses printed non-blocking
- Exit 1: one or more hard-gate violations; family names in stderr with HARD GATE VIOLATION
- Exit 2: infrastructure failure (missing YAML, missing/malformed summary.json)

Key security property (T-10-04-01): when `committed_itinerary_rate` is absent from a cell's scorers (Phase 10 — metric wired in Phase 11 BASE-01), the gate is reported as NOT-EVALUABLE rather than silently passing. Similarly for a family with no cell in the summary.

Makefile `eval-gates-check` target:
```
make eval-gates-check SUMMARY=eval_reports/{ts}/summary.json
```

## Verification

- `poetry run pytest tests/unit/test_check_eval_gates.py -q` → 11 passed
- `poetry run ruff check scripts/check_eval_gates.py tests/unit/test_check_eval_gates.py` → clean
- `make eval-gates-check SUMMARY=<passing synthetic summary>` → exits 0
- `poetry run python -c "import yaml; yaml.safe_load(open('configs/eval_gates.yaml'))"` → parses
- Full unit suite: 1091 passed, 7 skipped (no regressions)

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. The gate infrastructure is complete. The `committed_itinerary_rate` metric is intentionally not yet wired into the eval matrix scorers — this is documented as Phase 11 BASE-01 scope. The not-evaluable reporting is the correct behavior for Phase 10.

## Threat Flags

None. The T-10-04-01 threat (summary.json spoofing silent pass) is mitigated via the NOT-EVALUABLE reporting path. T-10-04-02 (gate value tampering) is mitigated by the single-source-of-truth YAML with per-gate D-ID rationale.

## Self-Check: PASSED

Files created:
- [x] configs/eval_gates.yaml exists
- [x] scripts/check_eval_gates.py exists
- [x] docs/eval_gates.md exists
- [x] tests/unit/test_check_eval_gates.py exists

Commits verified:
- [x] d5d8615 (Task 1: YAML + docs + matrix migration)
- [x] f6e23d9 (Task 2 RED: failing tests)
- [x] 2059b5c (Task 2 GREEN: implementation + Makefile)

TDD Gate Compliance:
- [x] RED gate: test(10-04) commit f6e23d9 exists with failing tests
- [x] GREEN gate: feat(10-04) commit 2059b5c exists after RED
