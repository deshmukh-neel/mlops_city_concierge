---
phase: 04-category-compliance-fix
plan: 01
subsystem: agent
tags: [langchain-tools, critique-scorers, pydantic-state, tdd]

requires:
  - phase: 03-eval-harness-extension
    provides: UserConstraints.requested_primary_types and family-level category_compliance scorer
provides:
  - Optional slot_index tool arg on semantic_search and nearby
  - category_compliance_strict scorer with keyword table and itinerary violation registration
  - rationale_misaligned RevisionReason literal for downstream revision dispatch
affects: [phase-04-category-compliance-fix, graph-injection, prompts, revision-hints, eval-gates]

tech-stack:
  added: []
  patterns:
    - LangChain StructuredTool args_schema generated from wrapper annotations
    - Pure state-only scorer with static keyword table and no DB dependency
    - TDD RED/GREEN commits per task

key-files:
  created:
    - .planning/phases/04-category-compliance-fix/04-01-SUMMARY.md
  modified:
    - app/agent/tools.py
    - app/agent/critique/checks.py
    - app/agent/state.py
    - tests/unit/test_agent_smoke.py
    - tests/unit/test_agent_tools.py
    - tests/unit/test_critique_checks.py
    - tests/unit/test_agent_state.py

key-decisions:
  - "slot_index is accepted by the LLM-facing wrappers and exposed in args_schema, but never forwarded to app.tools.retrieval."
  - "category_compliance_strict uses exact keyword-table matches for mapped user terms and falls back to family_of for unmapped primary-type values."
  - "RevisionReason changed only by adding rationale_misaligned; dispatch behavior remains for downstream plans."

patterns-established:
  - "Tool wrapper schemas are also attached to exported functions for direct validation probes."
  - "Strict category tests distinguish free-text keywords from Title Case primary_type values used by family_of."

requirements-completed: [CAT-01, CAT-02]

duration: 16 min
completed: 2026-05-22
---

# Phase 04 Plan 01: Tool Slot Index and Strict Scorer Summary

**Retrieval tools now expose per-slot indices, strict category scoring catches within-family drift, and revision state can type rationale misalignment hints.**

## Performance

- **Duration:** 16 min
- **Started:** 2026-05-22T21:45:10Z
- **Completed:** 2026-05-22T22:00:54Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added optional `slot_index: int | None = None` to `semantic_search` and `nearby`, including tool docstrings and schema exposure.
- Added `_STRICT_TYPE_KEYWORDS` and `category_compliance_strict`, then registered it in `CRITIQUE_THRESHOLDS` and `itinerary_violations()`.
- Added `rationale_misaligned` to `RevisionReason` with Pydantic construction coverage.

## Task Commits

1. **Task 1 RED: slot_index tool schema tests** - `152db0d` (test)
2. **Task 1 GREEN: slot_index retrieval tool wrappers** - `5e444d9` (feat)
3. **Task 2 RED: strict category scorer tests** - `3b56fda` (test)
4. **Task 2 GREEN: strict category scorer implementation** - `637d54e` (feat)
5. **Task 3 RED: rationale revision reason tests** - `09d9436` (test)
6. **Task 3 GREEN: rationale_misaligned RevisionReason** - `4a851fa` (feat)

**Plan metadata:** committed separately with this summary.

## Files Created/Modified

- `app/agent/tools.py` - Added `slot_index` to retrieval tool wrappers and attached generated args schemas to exported functions.
- `app/agent/critique/checks.py` - Added strict keyword table, strict scorer, threshold, and violation registration.
- `app/agent/state.py` - Added the `rationale_misaligned` `RevisionReason` literal.
- `tests/unit/test_agent_smoke.py` - Added slot_index schema and wrapper pass-through coverage.
- `tests/unit/test_agent_tools.py` - Updated direct `_args_schema_for` expectation for the new slot_index field.
- `tests/unit/test_critique_checks.py` - Added strict scorer branch, registration, and no-DB source-token tests.
- `tests/unit/test_agent_state.py` - Added RevisionReason and RevisionHint validation tests.

## Decisions Made

- Attached the generated `args_schema` to each exported tool function because the plan acceptance probes import `semantic_search.args_schema` and `nearby.args_schema` directly.
- Used `"Italian Restaurant"` rather than an arbitrary free-text word for the unmapped fallback test, because existing `family_of()` intentionally maps only known Title Case primary types.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Used local venv verification because Poetry was unavailable**
- **Found during:** Task 1 RED
- **Issue:** `poetry` was not installed and system Python had no `pytest`.
- **Fix:** Created ignored `.venv/` and installed the project plus test/type tools there.
- **Files modified:** None tracked
- **Verification:** `.venv/bin/python -m pytest tests/unit -v` passed with 764 passed, 7 skipped; `.venv/bin/python -m mypy app/` passed.
- **Committed in:** N/A, environment-only

**2. [Rule 3 - Blocking] Updated stale direct args-schema test used by Task 1 verification**
- **Found during:** Task 1
- **Issue:** `tests/unit/test_agent_tools.py::test_args_schema_includes_all_params` asserted the old `semantic_search` parameter set, while Task 1 verification explicitly runs this file.
- **Fix:** Updated the assertion to include `slot_index` and its `None` default.
- **Files modified:** `tests/unit/test_agent_tools.py`
- **Verification:** `.venv/bin/python -m pytest tests/unit/test_agent_smoke.py tests/unit/test_agent_tools.py -v` passed.
- **Committed in:** `152db0d`

**3. [Rule 2 - Missing Critical] Exposed generated schemas on tool functions for acceptance probes**
- **Found during:** Task 1 acceptance
- **Issue:** The LangChain `StructuredTool` instances had `args_schema`, but exported wrapper functions did not, causing the plan's direct Python probes to fail.
- **Fix:** `_to_lc_tool()` now attaches the generated schema to the wrapper function before building the `StructuredTool`.
- **Files modified:** `app/agent/tools.py`
- **Verification:** Direct probes for `semantic_search.args_schema.model_fields["slot_index"]` and `nearby.args_schema` passed.
- **Committed in:** `5e444d9`

**4. [Rule 1 - Test Bug] Corrected strict scorer fallback test assumptions**
- **Found during:** Task 2 GREEN
- **Issue:** Two RED expectations treated free-text `"omakase"` and `"unusual_word"` as family-level mappable values, but the existing `family_of()` contract maps only known Title Case primary_type strings.
- **Fix:** Kept the within-family drift assertion by comparing family scoring on `"Sushi Restaurant"` versus strict scoring on `"omakase"`, and used `"Italian Restaurant"` for unmapped-by-strict family fallback.
- **Files modified:** `tests/unit/test_critique_checks.py`
- **Verification:** `.venv/bin/python -m pytest tests/unit/test_critique_checks.py -v -k category_compliance_strict` passed.
- **Committed in:** `637d54e`

---

**Total deviations:** 4 auto-fixed (Rule 1: 1, Rule 2: 1, Rule 3: 2)
**Impact on plan:** All fixes were required to satisfy the plan's acceptance probes or align tests with existing contracts. No 04-02-owned source files were edited.

## Issues Encountered

- `gsd-sdk` was not on `PATH`; the compatibility CLI at `/Users/pnhek/.codex/get-shit-done/bin/gsd-tools.cjs` was used for GSD metadata reads.
- A parallel 04-02 executor committed its summary while this plan was running. Shared STATE/ROADMAP/REQUIREMENTS files were not edited here to avoid cross-plan write conflicts.

## Verification

- `.venv/bin/python -m pytest tests/unit/test_agent_smoke.py tests/unit/test_agent_tools.py -v` - passed, 21 tests.
- `.venv/bin/python -m pytest tests/unit/test_critique_checks.py -v` - passed, 56 tests.
- `.venv/bin/python -m pytest tests/unit/test_agent_state.py -v` - passed, 23 tests.
- `.venv/bin/python -m pytest tests/unit -v` - passed, 764 passed and 7 skipped.
- `.venv/bin/python -m mypy app/` - passed.
- Combined smoke import for slot_index, `category_compliance_strict`, `_STRICT_TYPE_KEYWORDS`, and `RevisionReason` - passed.
- Exact `poetry run ...` commands were not runnable because `poetry` is not installed in this shell; equivalent commands ran through the local venv.

## Known Stubs

None. Stub-pattern scan found only existing rationale placeholder regression-test text and intentional empty/default values.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plans 04-03 through 04-05 can now depend on the `slot_index` tool schema, strict scorer metric, and `rationale_misaligned` state literal.

## Self-Check: PASSED

- Found summary file at `.planning/phases/04-category-compliance-fix/04-01-SUMMARY.md`.
- Found key modified files: `app/agent/tools.py`, `app/agent/critique/checks.py`, `app/agent/state.py`.
- Found commits: `152db0d`, `5e444d9`, `3b56fda`, `637d54e`, `09d9436`, `4a851fa`.

---
*Phase: 04-category-compliance-fix*
*Completed: 2026-05-22*
