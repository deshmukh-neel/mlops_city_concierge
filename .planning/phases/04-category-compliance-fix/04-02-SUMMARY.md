---
phase: 04-category-compliance-fix
plan: 02
subsystem: eval
tags: [eval-harness, category-compliance, pydantic, yaml, runner]

requires:
  - phase: 03-eval-harness-extension
    provides: EvalQuery, ExpectedConstraints, multi-turn eval runner, category_compliance scorer
provides:
  - ExpectedConstraints.requested_primary_types typed config field
  - Per-slot category expectations for the two Phase 4 target eval cases
  - Eval runner bypass wiring from YAML expectations into ItineraryState.constraints
affects: [phase-04-category-compliance-fix, eval-baselines, category-compliance-gate]

tech-stack:
  added: []
  patterns:
    - Pydantic v2 multi-field field_validator reuse
    - Eval bypasses production intake by constructing UserConstraints from checked-in YAML

key-files:
  created: []
  modified:
    - app/eval/config.py
    - configs/eval_queries.yaml
    - scripts/eval_agent.py
    - tests/unit/test_eval_agent.py

key-decisions:
  - "Eval slot expectations are authoritative YAML data and do not call the production intake LLM."
  - "requested_primary_types uses the same ExpectedConstraints list validator as types_any."

patterns-established:
  - "ExpectedConstraints list-like string fields can share one Pydantic v2 multi-field validator."
  - "Eval runner constraints are rebuilt for each single-turn and multi-turn graph invocation."

requirements-completed: [CAT-01, CAT-04]

duration: 15 min
completed: 2026-05-22
---

# Phase 04 Plan 02: Eval Config and Runner Bypass Summary

**Eval cases now carry per-slot category expectations from YAML into graph state without invoking the production intake LLM.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-22T21:45:14Z
- **Completed:** 2026-05-22T21:59:48Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added `ExpectedConstraints.requested_primary_types` with default `[]` and shared blank-stripping validation.
- Added target category slots to `omakase_mission_open_ended` and `refinement_cheaper`, while leaving `late_night_closure_cascade` ungated.
- Wired `evaluate_case` and `evaluate_multi_turn_case` to pass `UserConstraints.requested_primary_types` into every graph invocation.

## Task Commits

1. **Task 1 RED: ExpectedConstraints category field tests** - `843ac38` (test)
2. **Task 1 GREEN: ExpectedConstraints requested_primary_types field** - `35aa8b0` (feat)
3. **Task 2 RED: Eval YAML category slot tests** - `426b4cc` (test)
4. **Task 2 GREEN: Eval query requested_primary_types YAML** - `07aef13` (feat)
5. **Task 3 RED: Eval runner bypass wiring tests** - `4615d2a` (test)
6. **Task 3 GREEN: Eval runner UserConstraints wiring** - `77d6461` (feat)

**Plan metadata:** committed separately with this summary.

## Files Created/Modified

- `app/eval/config.py` - Added `requested_primary_types` to `ExpectedConstraints` and reused one list validator for both type lists.
- `configs/eval_queries.yaml` - Added requested primary types to the two Phase 4 target cases.
- `scripts/eval_agent.py` - Added `_constraints_for_case` and passed constraints into single-turn and multi-turn `ItineraryState` construction.
- `tests/unit/test_eval_agent.py` - Added TDD coverage for config parsing, live YAML expectations, and graph-invoke state wiring.

## Decisions Made

- Eval bypasses production slot extraction: checked-in YAML is deterministic and authoritative for offline scoring.
- The late-night closure cascade case remains without `requested_primary_types` per D-04-12.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created an isolated verification environment outside the repo**
- **Found during:** Task 1
- **Issue:** `poetry` was not installed and the system Python had no project dependencies (`pytest`, `pydantic`, `langchain_core`).
- **Fix:** Created `/tmp/city-concierge-venv` and installed the project plus test/type tools there. For Makefile verification, used `POETRY_RUN="/tmp/city-concierge-venv/bin/python -m"` instead of editing project config.
- **Files modified:** None
- **Verification:** Plan-specific pytest and mypy commands ran successfully through the isolated venv.
- **Committed in:** N/A, environment-only

---

**Total deviations:** 1 auto-fixed (Rule 3)
**Impact on plan:** No code scope change. The repo remains unchanged by the temporary verification environment.

## Issues Encountered

- Full-unit verification was briefly blocked while parallel Plan 04-01 had landed RED tests but not the matching state/schema implementation. After 04-01's dependent commits landed, the same `make test-unit` target passed.
- `gsd-sdk` was not available on PATH in this shell. Per the parallel Wave 1 write boundary, no STATE/ROADMAP/REQUIREMENTS files were edited by this executor.

## Verification

- `/tmp/city-concierge-venv/bin/python -m pytest tests/unit/test_eval_agent.py -v` - passed, 61 tests.
- `/tmp/city-concierge-venv/bin/python -m mypy app/eval/config.py` - passed.
- `/tmp/city-concierge-venv/bin/python -m mypy scripts/eval_agent.py` - passed.
- `/tmp/city-concierge-venv/bin/python -c "from app.eval.config import load_eval_queries; ..."` - printed the expected requested primary type lists for both target cases.
- Acceptance greps for `requested_primary_types`, shared validator count, and no eval intake calls passed. The `late_night_closure_cascade` grep correctly exited 1 because that case remains ungated.
- `make test-unit POETRY_RUN="/tmp/city-concierge-venv/bin/python -m"` - passed, 764 passed, 7 skipped.

## Known Stubs

None - no stubs introduced. Empty defaults added in this plan are intentional abstain behavior for existing eval cases without slot expectations.

## Threat Flags

None - no new network endpoints, auth paths, file access patterns, or trust-boundary schema changes beyond typed eval YAML parsing already covered by the plan threat model.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

04-02 is ready for downstream Phase 4 plans. The eval harness can now score category-slot behavior once 04-01's strict scorer and later graph/prompt changes land.

## Self-Check: PASSED

- Found modified files: `app/eval/config.py`, `configs/eval_queries.yaml`, `scripts/eval_agent.py`, `tests/unit/test_eval_agent.py`.
- Found commits: `843ac38`, `35aa8b0`, `426b4cc`, `07aef13`, `4615d2a`, `77d6461`.
- Summary created at `.planning/phases/04-category-compliance-fix/04-02-SUMMARY.md`.

---
*Phase: 04-category-compliance-fix*
*Completed: 2026-05-22*
