---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: "01"
subsystem: agent
tags: [langgraph, telemetry, timing, instrumentation, step_telemetry]

requires:
  - phase: 11-baseline-regeneration
    provides: stable ItineraryState model and graph.py plan()/act() structure

provides:
  - step_telemetry field on ItineraryState (JSON-safe list[dict[str, Any]])
  - in-graph LLM-call and tool-execution wall-time hooks in plan() and act()
  - unit tests proving per-step telemetry production, key set, type safety, JSON-safety

affects:
  - 12-02-scratch-viability-signals (reads step_telemetry from ItineraryState)
  - 12-03-eval-result-instrumentation (forwards step_telemetry via QueryEvalResult)
  - phase 13 DEC arms (latency decomposition data)

tech-stack:
  added: [stdlib time (no new dependency)]
  patterns:
    - JSON-safe state field pattern (plain int/float only, never Pydantic models — aimessage_tool_call_args_json_safe precedent)
    - patch-current-step telemetry pattern (plan() writes entry, act() patches it)
    - D-12-01 always-on-in-prod telemetry (cheap enough: 2 monotonic reads + dict append per step)

key-files:
  created: []
  modified:
    - app/agent/state.py
    - app/agent/graph.py
    - tests/unit/test_agent_graph.py

key-decisions:
  - "D-12-01 reaffirmed: step_telemetry stays always-on in prod (two time.monotonic() reads + dict append is negligible latency)"
  - "act() patches the plan()-written entry rather than appending a separate one — keeps one entry per logical step, not two"
  - "step index uses state.step_count (pre-increment) to match scratch entry convention from Plan 12-02 step-index contract"
  - "tool_calls_this_step incremented in both the commit_itinerary and the regular tool branch to correctly count all tool calls per step"

patterns-established:
  - "JSON-safe-only values in ItineraryState: all step_telemetry dict values must be plain int/float/str — never Pydantic model instances"
  - "plan()-writes-act()-patches telemetry split: plan() appends entry with llm_call_seconds; act() replaces tool_exec_seconds/tool_calls_this_step"

requirements-completed: [INST-04]

duration: 3min
completed: "2026-06-12"
---

# Phase 12 Plan 01: In-Graph Step Telemetry Summary

**Always-on per-step LLM-call and tool-execution wall-time telemetry added to ItineraryState via plan()/act() timing hooks, JSON-safe and keyed by step_count index**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-06-12T00:56:19Z
- **Completed:** 2026-06-12T00:59:35Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added `step_telemetry: list[dict[str, Any]]` field to `ItineraryState` with `Field(default_factory=list)`, JSON-safe default, and D-12-01 description naming all four keys
- Added `import time` and timing hooks to `plan()` (wraps `ainvoke` with monotonic start/stop) and `act()` (wraps the entire tool loop, patches the current-step entry)
- 3 new unit tests cover: key set + types + JSON-safety after tool call, JSON-safety on commit_itinerary path, multi-step accumulation with monotone step indices

## Task Commits

1. **Task 1: Add step_telemetry field to ItineraryState** - `4c30dff` (feat)
2. **Task 2: Add timing hooks to plan() and act() in graph.py** - `d0dfdd0` (feat)
3. **Task 3: Unit-test per-step telemetry production through the graph** - `b78e2bc` (test)

## Files Created/Modified

- `app/agent/state.py` - Added `step_telemetry` field after `closure_context`, before `model_config`
- `app/agent/graph.py` - Added `import time`; timing hooks in `plan()` returning `step_telemetry` key; timing + patching logic in `act()` returning `step_telemetry` key
- `tests/unit/test_agent_graph.py` - 3 new async tests under "Phase 12 Plan 01: INST-04 step_telemetry tests" section

## Decisions Made

- `act()` patches the entry written by `plan()` (matching on `telemetry[-1]["step"] == state.step_count`) rather than appending a separate second entry per step. This keeps one telemetry entry per logical plan step with both LLM and tool timing in the same dict, which is the shape Phase 13 latency analysis will diff.
- `tool_calls_this_step` is incremented in both the `commit_itinerary` branch (before `continue`) and the regular tool branch, so all tool calls in a step are counted regardless of type.
- Ruff format modified the telemetry dict literal in `act()` (multi-line formatting) — picked up and re-staged before the final commit.

## Deviations from Plan

None — plan executed exactly as written. The two ruff-format pre-commit hook modifications (Tasks 2 and 3) are expected behavior, not deviations.

## Issues Encountered

None. Ruff format ran automatically via pre-commit hook twice (Tasks 2 and 3) and reformatted code; re-staged and committed cleanly on the second attempt.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries introduced. Only adds timing reads (two `time.monotonic()` calls per step) and a plain-dict field to existing ItineraryState. T-12-01 (Tampering via non-primitive values) is mitigated by Task 3's `isinstance(v, (int, float)) and not isinstance(v, bool)` assertions. T-12-03 (Information disclosure) is accept: only wall-clock floats and counts, no PII.

## Next Phase Readiness

- Phase 12 Plan 02 (`12-02-scratch-viability-signals`) can proceed — `step_telemetry` is on `ItineraryState` and the step-index contract (`state.step_count` as key) is established
- Phase 13 latency decomposition has the per-step `llm_call_seconds` vs `tool_exec_seconds` data it needs
- All 51 tests in `test_agent_graph.py` pass; `make lint` passes

## Self-Check

- `app/agent/state.py` has `step_telemetry` field: confirmed (`grep -n step_telemetry` returns line 278)
- `app/agent/graph.py` has timing hooks: confirmed (5 references: 3 in plan, 2 in act; 4 monotonic calls)
- `tests/unit/test_agent_graph.py` has 3 new telemetry tests: confirmed (51 total vs 48 before)
- Commits exist: 4c30dff, d0dfdd0, b78e2bc — all present in git log

## Self-Check: PASSED

---
*Phase: 12-decisiveness-instrumentation-comparison-floor*
*Completed: 2026-06-12*
