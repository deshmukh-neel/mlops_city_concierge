---
phase: 13-decisiveness-experiment-arms
plan: "01"
subsystem: agent/eval
tags: [viability-predicate, forced-commit-telemetry, arm-flags, tdd]
dependency_graph:
  requires: [12-02-harness-derived-decisiveness-fields]
  provides: [shared-viability-predicate, forced-commit-state-fields, arm-flags-run-json]
  affects: [app/agent/viability.py, app/agent/state.py, scripts/eval_agent.py]
tech_stack:
  added: []
  patterns: [tdd-red-green, json-safe-state-fields, truthy-set-env-parsing]
key_files:
  created:
    - app/agent/viability.py
    - tests/unit/test_viability.py
  modified:
    - app/agent/state.py
    - scripts/eval_agent.py
    - tests/unit/test_eval_agent.py
decisions:
  - "Viability predicate lives in app/agent/ (not scripts/) to avoid circular import with eval_agent.py which imports app.agent.*"
  - "New DeterministicEvalResult fields lack Python default values to surface any new callers that forget them; existing test fixtures were fixed as a Rule-1 auto-fix"
  - "arm_flags assembled from env at query_result_from_state call time (not at graph-build time) so the eval harness snapshot self-describes the run without needing to re-read the graph config"
metrics:
  duration: "~10 minutes"
  completed: "2026-06-12T05:22:02Z"
  tasks: 3
  files: 5
---

# Phase 13 Plan 01: Viability Predicate and Telemetry Summary

Shared viability predicate + forced-commit state fields + eval run self-description via arm_flags — the foundation all four DEC arms consume.

## What Was Built

**Task 1 (TDD): Shared viability predicate — `app/agent/viability.py`**

New module exposing `all_slots_viable(state, threshold) -> bool` and
`best_viable_candidate_per_slot(state, threshold) -> list[dict | None]`.
Matches `rule8_met_per_step_from_state` semantics exactly (WR-01 semantic_search
only; WR-02 multiset distinct place_id coverage). Importable from both
`app/agent/graph.py` and `app/agent/revision.py` without circular import.
23 tests, all passing.

**Task 2 (TDD): Forced-commit state fields — `app/agent/state.py`**

Two new JSON-safe plain-primitive fields appended after `step_telemetry`:
- `commit_forced: bool = False` — True only when DEC-02 triggers
- `forced_commit_step: int | None = None` — step index when forced

Both default to the no-force state so default-path behavior is byte-identical.
`mypy app/` still reports `Success: no issues found in 40 source files`.

**Task 3 (TDD): Arm telemetry in eval run JSON — `scripts/eval_agent.py`**

`DeterministicEvalResult` gains three new fields:
- `commit_forced: bool` — forwarded from `state.commit_forced` via getattr
- `forced_commit_step: int | None` — forwarded from `state.forced_commit_step`
- `arm_flags: dict[str, Any]` — `{"viability_contract": bool, "forced_commit_step": int, "parallel_tool": bool, "viability_threshold_override": str|None}`

`make_error_record` carries safe defaults (`commit_forced=False`, `forced_commit_step=None`, `arm_flags={}`). No new `DETERMINISTIC_CHECKS` entries — these are telemetry-only (D-13-04 anti-scope).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pre-existing `DeterministicEvalResult` test fixtures missing new required fields**

- **Found during:** Task 3 implementation — making the three new dataclass fields positional (no defaults) to catch callers that forget them
- **Issue:** Two helper constructions of `DeterministicEvalResult` in `test_eval_agent.py` (the `query_result()` fixture and the `test_aggregate_results_flattens_mean_metrics` inline construction) lacked the new fields, causing 14 pre-existing tests to fail
- **Fix:** Added `commit_forced=False, forced_commit_step=None, arm_flags={}` to both fixture constructors
- **Files modified:** `tests/unit/test_eval_agent.py`
- **Commit:** c68a2c8

## Verification

- `poetry run pytest tests/unit/test_viability.py tests/unit/test_eval_agent.py -v` — 188 passed, 2 warnings (pre-existing coroutine warnings unrelated to this plan)
- `python -c "import app.agent.viability; import app.agent.graph; import app.agent.revision"` — OK, no circular import
- `make typecheck` — Success: no issues found in 40 source files
- `make lint` — All checks passed
- `grep -c "DETERMINISTIC_CHECKS\[" scripts/eval_agent.py` — 0 (no new scorers)
- Default-path behavior with all arm flags unset: `arm_flags == {"viability_contract": False, "forced_commit_step": 0, "parallel_tool": False, "viability_threshold_override": None}`

## Commits

| Hash | Message |
|------|---------|
| 5c9c060 | test(13-01): add failing tests for shared viability predicate and state fields |
| 6c97e01 | feat(13-01): add shared viability predicate and forced-commit state fields |
| 4ae024e | test(13-01): add failing tests for arm_flags and forced-commit telemetry fields |
| c68a2c8 | feat(13-01): thread arm_flags + forced-commit telemetry into eval run JSON |

## Self-Check: PASSED

- [x] `app/agent/viability.py` exists and exports `all_slots_viable` + `best_viable_candidate_per_slot`
- [x] `app/agent/state.py` contains `commit_forced` field
- [x] `scripts/eval_agent.py` contains `arm_flags` field in `DeterministicEvalResult`
- [x] All 188 tests pass
- [x] No circular imports
- [x] mypy clean
- [x] ruff clean
- [x] No new scorers registered

## Known Stubs

None — all data flows are wired through real state fields.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary crossings. The `arm_flags` dict carries only flag booleans/ints/strings (no secrets, no PII); run JSONs are local eval artifacts (T-13-01-02 accepted in plan threat model).
