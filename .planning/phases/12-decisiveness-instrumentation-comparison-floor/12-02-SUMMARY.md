---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: "02"
subsystem: eval
tags: [eval_agent, instrumentation, decisiveness, viability, rule8, INST-01, INST-02, INST-03]

requires:
  - phase: 12-decisiveness-instrumentation-comparison-floor
    plan: "01"
    provides: step_telemetry on ItineraryState

provides:
  - first_commit_call_step_from_state helper in scripts/eval_agent.py
  - viable_candidates_per_step_from_state helper in scripts/eval_agent.py
  - rule8_met_per_step_from_state helper in scripts/eval_agent.py
  - DeterministicEvalResult extended with 7 INST fields
  - query_result_from_state wired with all INST fields
  - make_error_record backward-compatible with safe defaults
  - 12 unit tests covering all three helpers

affects:
  - Phase 13 DEC arms (read first_commit_call_step, viable_candidates_per_step, rule8_met fields)
  - scripts/eval_agent.py write path (step_telemetry forwarded verbatim via asdict)

tech-stack:
  added: []
  patterns:
    - Pure helper function pattern (takes ItineraryState, returns simple value, guards with isinstance)
    - D-12-04: import LOW_SIMILARITY_THRESHOLD, never hardcode 0.55
    - D-12-01: eval semantics stay harness-side (no prod graph code touched)
    - Per-step vs cumulative contract: viable_per_step is non-cumulative, rule8 accumulates internally

key-files:
  created: []
  modified:
    - scripts/eval_agent.py
    - tests/unit/test_eval_agent.py

key-decisions:
  - "viable_candidates_per_step_from_state is PER-STEP (non-cumulative): element i counts only hits from scratch entries whose step==i; cumulative accumulation lives entirely in rule8"
  - "rule8_met_per_step_from_state re-reads scratch entries internally (not from the flat int list) because per-type coverage cannot be reconstructed from the flat per-step count"
  - "first_commit_mention_step defaults to None (opaque-reasoning-safe) per D-12-03; never gated"
  - "rule8_kept_searching excludes commit steps — a step where rule8 was met AND a commit was made is not a gap"
  - "Existing DeterministicEvalResult test fixtures updated with the 7 new required fields (Rule 1 fix)"

metrics:
  duration: 6min
  completed: "2026-06-12"
  tasks: 3
  files_modified: 2
---

# Phase 12 Plan 02: Harness-Derived Decisiveness Fields Summary

**Three pure derived-field helpers + extended DeterministicEvalResult with 7 INST fields wired end-to-end into the eval run JSON via asdict write path, with LOW_SIMILARITY_THRESHOLD imported never hardcoded**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-12T01:15:00Z
- **Completed:** 2026-06-12T01:22:01Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Added `from app.agent.revision import LOW_SIMILARITY_THRESHOLD` to `scripts/eval_agent.py` (D-12-04; no hardcoded 0.55 in viability logic)
- Extended `DeterministicEvalResult` with 7 new typed fields: `first_commit_call_step`, `first_commit_mention_step`, `viable_candidates_per_step`, `rule8_met_per_step`, `rule8_met_but_kept_searching_steps`, `step_telemetry`, `viability_threshold`
- Added `first_commit_call_step_from_state(state) -> int | None`: reads `scratch["commit_itinerary"]` entries, returns min step index across all commit entries (INST-01)
- Added `viable_candidates_per_step_from_state(state, threshold, requested_types) -> list[int]`: PER-STEP non-cumulative counts grouped by `"step"` key across semantic_search/nearby scratch entries (INST-02)
- Added `rule8_met_per_step_from_state(state, viable_per_step, requested_types) -> list[bool]`: cumulative set-coverage check — element i is True iff every requested type has had >=1 viable hit across steps 0..i; empty-types fallback uses cumulative count vs num_stops (INST-03)
- Wired all 7 fields into `query_result_from_state` with correct computation chain
- Added safe defaults for all 7 fields to `make_error_record` (backward-compat)
- Verified `report_to_dict` is a simple `asdict(report)` with no filtering — step_telemetry reaches the run JSON verbatim (INST-04 / D-12-02)
- Added 12 unit tests covering all three helpers including per-step-vs-cumulative contract, type filter, cosine-only fallback, and JSON safety

## Task Commits

1. **Task 1: Add three pure derived-field helpers + extend DeterministicEvalResult** - `eebe3a9` (feat)
2. **Task 2: Wire derived fields into query_result_from_state + make_error_record** - `1cd3864` (feat)
3. **Task 3: Unit tests for the three derived-field helpers** - `3882e27` (test)

## Files Created/Modified

- `scripts/eval_agent.py` - Import LOW_SIMILARITY_THRESHOLD; 7 new DeterministicEvalResult fields; 3 helper functions; wired in query_result_from_state and make_error_record
- `tests/unit/test_eval_agent.py` - Added 3 new helpers to import block; 12 new tests in TestFirstCommitCallStepFromState, TestViableCandidatesPerStepFromState, TestRule8MetPerStepFromState; updated 2 existing DeterministicEvalResult fixture calls to include new required fields

## Decisions Made

- `viable_candidates_per_step_from_state` is PER-STEP (non-cumulative): element i counts only hits from scratch entries whose `"step"` key equals i. Cumulative logic belongs exclusively in `rule8_met_per_step_from_state`. Both docstrings make this contract explicit.
- `rule8_met_per_step_from_state` re-reads scratch entries internally rather than reconstructing coverage from the flat int list, because per-type coverage cannot be inferred from the flat per-step count (two different types each with 1 hit is identical to one type with 2 hits in a flat count).
- `rule8_kept_searching` excludes steps where a commit also occurred in the same step — those are successful decisiveness, not gaps.
- `first_commit_mention_step` defaults to None per D-12-03 (opaque-reasoning-safe default; heuristic, never gated).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed two existing DeterministicEvalResult test fixture calls missing new required fields**
- **Found during:** Task 2 verification (`poetry run pytest tests/unit/test_eval_agent.py -x -q`)
- **Issue:** `DeterministicEvalResult` added 7 new required positional fields; two `query_result()` fixture calls in test file failed with `TypeError: missing 7 required positional arguments`
- **Fix:** Added `first_commit_call_step=None, first_commit_mention_step=None, viable_candidates_per_step=[], rule8_met_per_step=[], rule8_met_but_kept_searching_steps=[], step_telemetry=[], viability_threshold=0.55` to both fixture constructors
- **Files modified:** `tests/unit/test_eval_agent.py`
- **Committed in:** Task 2 commit (`1cd3864`)

## Known Stubs

None — all INST fields are computed from real state data (scratch, constraints, step_telemetry). No placeholder values flow to consumers.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries. T-12-04 (threshold drift) mitigated: LOW_SIMILARITY_THRESHOLD imported, no new literal 0.55 in viability logic. T-12-11 (step_telemetry silently dropped) mitigated: `report_to_dict` is a plain `asdict(report)` with no allow-list/pop/filter; round-trip test confirms presence in dict.

## Self-Check

- `scripts/eval_agent.py` has LOW_SIMILARITY_THRESHOLD import at line 38: confirmed
- `scripts/eval_agent.py` has all 7 new DeterministicEvalResult fields: confirmed (dataclasses.fields assertion passed)
- `scripts/eval_agent.py` has all 3 helpers: confirmed (import verification passed)
- `query_result_from_state` passes all 7 new fields: confirmed (grep shows first_commit_call_step=, viability_threshold=, step_telemetry= in function body)
- `make_error_record` passes safe defaults for all 7 new fields: confirmed
- `report_to_dict` at line 1392 is `return asdict(report)` with no filtering: confirmed
- Commits exist: eebe3a9, 1cd3864, 3882e27 — all present in git log
- 145 tests pass (127 original + 12 new + 6 from fixture fixes)
- `make lint` passes

## Self-Check: PASSED

---
*Phase: 12-decisiveness-instrumentation-comparison-floor*
*Completed: 2026-06-12*
