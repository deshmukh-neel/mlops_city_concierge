---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: "02"
subsystem: eval-scorer
tags: [scorer, critique, category-compliance, measurement-fix, tdd]
dependency_graph:
  requires: []
  provides: [category_compliance-zero-stop-abstain]
  affects: [scripts/eval_agent.py aggregate_results, configs/eval_baselines]
tech_stack:
  added: []
  patterns: [None-abstain scorer pattern, float | None return type annotation]
key_files:
  modified:
    - app/agent/critique/checks.py
    - tests/unit/test_critique_checks.py
decisions:
  - "D-11-03: category_compliance zero-stop guard moved before D-03 empty-requested guard; returns None not 1.0"
  - "itinerary_violations _try helper explicitly guards against None scores to avoid TypeError and document abstain semantics"
metrics:
  duration: "3m"
  completed: "2026-06-11"
  tasks_completed: 1
  files_modified: 2
---

# Phase 11 Plan 02: category_compliance Zero-Stop Abstain Summary

**One-liner:** `category_compliance` now returns `None` on zero-stop runs (WR-12 / D-11-03), preventing decisiveness-failing providers from inflating category-compliance medians.

## What Was Built

Fixed the `category_compliance` scorer in `app/agent/critique/checks.py` so that zero-stop runs (where the agent failed to commit any itinerary) abstain with `None` instead of returning a spurious `1.0`. This is a measurement-semantics-only fix — no agent behavior changes.

**Key changes:**

1. `category_compliance` return type: `-> float` changed to `-> float | None`
2. Zero-stop guard moved BEFORE the D-03 empty-requested guard and changed from `return 1.0` to `return None` with comment `# WR-12 / D-11-03: abstain`
3. Docstring updated to document: (a) zero-stop abstain contract, (b) guard ordering (zero-stop fires first), (c) why None not 1.0
4. `itinerary_violations._try` now explicitly checks `if score is None: return` before the threshold comparison, making abstain semantics explicit rather than relying on the exception-catch path

## TDD Gate Compliance

| Gate | Status |
|------|--------|
| RED: failing tests committed | cca854b |
| GREEN: implementation committed | 82241ad |

RED committed first: 3 new tests failing (zero-stop None, zero-stop before empty-requested, aggregation propagation). GREEN committed after: all 92 tests in the file pass.

## Acceptance Criteria Verification

- `grep -n "return None" app/agent/critique/checks.py` → line 259 inside `category_compliance` zero-stop guard
- `def category_compliance(` signature → `-> float | None:` at line 229
- `category_compliance_strict` still returns `1.0` on zero stops (unchanged)
- `poetry run pytest tests/unit/test_critique_checks.py -k "category_compliance"` → 28 passed
- `make test` → 1159 passed, 0 failed
- `poetry run mypy app/agent/critique/checks.py` → Success: no issues found

## Commits

| Hash | Message |
|------|---------|
| cca854b | test(11-02): add failing tests for category_compliance zero-stop abstain (D-11-03 / WR-12) |
| 82241ad | feat(11-02): category_compliance abstains (None) on zero stops (WR-12 / D-11-03) |

## Deviations from Plan

**One auto-fix beyond the plan spec:**

**[Rule 2 - Missing critical handling] Explicit None guard in `itinerary_violations._try`**
- **Found during:** Implementing the fix
- **Issue:** The `_try` helper did `if score < CRITIQUE_THRESHOLDS[name]` — with a `None` return from `category_compliance`, this would raise a `TypeError` that the outer `except Exception` would catch, silently swallowing the abstain (treating it as a skipped check due to "error"). While functionally safe, it was semantically ambiguous — a real exception and an abstain were handled identically.
- **Fix:** Added explicit `if score is None: return` before the threshold comparison with a docstring comment explaining abstain semantics. The exception-catch path remains for genuine DB/logic failures.
- **Files modified:** `app/agent/critique/checks.py`
- **Commit:** 82241ad

## Known Stubs

None. This is a pure measurement-semantics fix with no stubbed data paths.

## Threat Flags

None. This fix reduces measurement error (T-11-04 from the plan's threat register: Tampering via category_compliance zero-stop return). No new network endpoints, auth paths, or schema changes introduced.

## Self-Check: PASSED

- `app/agent/critique/checks.py` exists and contains `return None  # WR-12 / D-11-03`
- `tests/unit/test_critique_checks.py` contains `test_category_compliance_returns_none_for_empty_stops_with_requested_types`
- Commits cca854b and 82241ad exist in git log
