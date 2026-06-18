---
phase: 18-gap-mining-gap
plan: "01"
subsystem: data-loop
tags: [sandbox, seed, guard, demand-log, tdd]
dependency_graph:
  requires: [17-02]
  provides: [scripts/sandbox_guard.py, scripts/seed_demand_log.py]
  affects: [scripts/coverage_agent.py, 18-02, 18-03]
tech_stack:
  added: []
  patterns: [current_database()-based write guard, capturing-stub TDD, parameterised INSERT]
key_files:
  created:
    - scripts/sandbox_guard.py
    - tests/unit/test_sandbox_guard.py
    - scripts/seed_demand_log.py
    - tests/unit/test_seed_demand_log.py
  modified:
    - Makefile
    - .env.example
key_decisions:
  - "Shared guard lives in scripts/sandbox_guard.py — both seed_demand_log.py (Plan 01) and coverage_agent.py (Plan 03) import from there; no forward-reference, DRY (REVIEW ROUND-2 MEDIUM-2)"
  - "Pass condition is live SELECT current_database() result, NOT SANDBOX_DATABASE_URL env-var equality — a mis-set env var pointing at prod cannot whitelist a prod write (REVIEW ROUND-2 H3 refinement)"
  - "Fixture uses Nepalese not Burmese (burmese absent from CUISINES); fix discovered during RED→GREEN cycle"
metrics:
  duration: "~20 minutes"
  completed: "2026-06-18"
  tasks_completed: 4
  files_count: 6
---

# Phase 18 Plan 01: Sandbox Prerequisites Summary

One-liner: Shared `current_database()`-based sandbox-write guard in `scripts/sandbox_guard.py`, sandbox-migrate make target, and TDD-green catalog-valid demand seed helper with H3-hardened write enforcement.

## Tasks Completed

| Task | Name | Type | Commit |
|------|------|------|--------|
| 1 | make sandbox-migrate + DEMAND_DATABASE_URL in .env.example | auto | c592589 |
| 2 | Apply user_query_log migration to sandbox (human-action) | checkpoint | operator-verified |
| 3 | Shared sandbox-write guard (scripts/sandbox_guard.py) + unit test | auto/TDD | RED: 101a1b9, GREEN: 07d5b27 |
| 4 | Seed demand log helper (scripts/seed_demand_log.py) + unit test | auto/TDD | RED: b25be52, GREEN: 54fa074 |

## Key Artifacts

### scripts/sandbox_guard.py (shared guard — Plan 03 imports from here)

`assert_sandbox_write_target(conn=None)` runs `SELECT current_database()` against the active write connection and raises `RuntimeError` unless the live dbname is `city_concierge_sandbox` or a name containing `sandbox` and not in the known-prod set. The pass condition is the live `current_database()` result only — `SANDBOX_DATABASE_URL` env-var equality is never the deciding factor (H3 refinement).

### scripts/seed_demand_log.py

`seed_demand_rows()` returns a deterministic 5-row fixture — all rows within `NEIGHBORHOODS × CUISINES`; includes the `("Outer Sunset", "vietnamese")` overlap row for the thin-bucket gap test. `insert_demand_rows(rows, conn=None)` calls `assert_sandbox_write_target()` before any INSERT, uses parameterised `%s` placeholders (T-18-SQLi), commits once after the loop.

## Test Coverage

- `tests/unit/test_sandbox_guard.py` — 6 tests: sandbox passes, prod raises, H3 mis-set-env-var raises, sandbox-pattern name passes, non-sandbox-pattern raises, connection reuse uses passed conn
- `tests/unit/test_seed_demand_log.py` — 11 tests: non-empty list, required keys, catalog-valid cuisine types, neighborhood in message, Outer-Sunset/vietnamese row present, parameterised placeholders, exactly-one-commit, insert returns count, guard called before insert, raising guard prevents all inserts, guard imported from sandbox_guard

All 17 tests pass. `poetry run ruff check` passes on all 4 files.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixture used "Burmese Restaurant" which maps to "burmese" — absent from CUISINES**
- **Found during:** Task 4 GREEN phase (import-time assertion fires)
- **Issue:** CUISINES list has no "burmese" entry; "Burmese Restaurant" → "burmese" fails the catalog-validity assertion baked into the module
- **Fix:** Replaced the Burmese row with "Nepalese Restaurant" / "Nepalese restaurants in Inner Sunset" — "nepalese" is in CUISINES
- **Files modified:** scripts/seed_demand_log.py
- **Commit:** 54fa074

## Known Stubs

None — all seed rows are fully wired and all functions are implemented.

## Shared Guard Module Path (for Plan 03)

Plan 18-03 (`coverage_agent.py`) must import from:
```python
from scripts.sandbox_guard import assert_sandbox_write_target
```
The module was created in this wave-0 plan (Task 3). No lazy import, no forward-reference, no re-definition.

## Self-Check

Files exist:
- scripts/sandbox_guard.py: FOUND
- scripts/seed_demand_log.py: FOUND
- tests/unit/test_sandbox_guard.py: FOUND
- tests/unit/test_seed_demand_log.py: FOUND

Commits exist:
- 101a1b9: test(18-01): add failing tests for sandbox write guard (RED) — FOUND
- 07d5b27: feat(18-01): implement sandbox write guard (current_database-based, GREEN) — FOUND
- b25be52: test(18-01): add failing tests for seed_demand_log helper (RED) — FOUND
- 54fa074: feat(18-01): implement seed_demand_log with sandbox guard (GREEN) — FOUND

## Self-Check: PASSED
