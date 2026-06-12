---
phase: 13-decisiveness-experiment-arms
plan: "03"
subsystem: agent/critique
tags: [dec-03, critique-recalibration, threshold-override, low-similarity-scoping, tdd]
dependency_graph:
  requires: [13-01-viability-predicate]
  provides: [dec-03-threshold-env-override, dec-03-low-similarity-scoping, dec-03-decision-doc]
  affects: [app/agent/revision.py, docs/decisiveness_dec03_decision.md]
tech_stack:
  added: []
  patterns: [tdd-red-green, truthy-set-env-parsing, lazy-import-circular-guard]
key_files:
  created:
    - docs/decisiveness_dec03_decision.md
    - tests/unit/test_agent_revision.py
  modified:
    - app/agent/revision.py
decisions:
  - "LOW_SIMILARITY_THRESHOLD is now env-overridable via LOW_SIMILARITY_THRESHOLD_OVERRIDE; code default stays 0.55 (seed finding 2 — it already was 0.55, so setting it there is a no-op)"
  - "Suppression gate lives in _diagnose_last_tool_result (has access to state) not _diagnose_one (no state); only the low_similarity reason is suppressed — all other reasons fire unconditionally"
  - "Lazy import of all_slots_viable inside the flag gate avoids top-level circular import risk while keeping flag-off path zero-overhead"
  - "First A1 run keeps LOW_SIMILARITY_THRESHOLD_OVERRIDE unset to isolate the scoping effect; 0.45 tested only if A1 shows positive-but-short signal"
metrics:
  duration: "~4 minutes"
  completed: "2026-06-12"
  tasks: 2
  files: 3
---

# Phase 13 Plan 03: DEC-03 Doc and Critique Scoping Summary

DEC-03 critique recalibration — decision doc written before code (roadmap criterion 4), threshold made env-overridable (code default stays 0.55), and low_similarity scoped to pre-candidate steps only under the shared A1 arm flag.

## What Was Built

**Task 1: DEC-03 decision doc (doc-first, roadmap criterion 4)**

Created `docs/decisiveness_dec03_decision.md` documenting both DEC-03 changes before any code landed:

- **Threshold direction:** Code default stays 0.55 (seed finding 2 — it already is 0.55). The experiment knob is `LOW_SIMILARITY_THRESHOLD_OVERRIDE`, used in A1 to test values below 0.55. First A1 run keeps it unset (scoping measured alone); 0.45 tried only if signal is positive-but-short.
- **Low_similarity scoping decision:** Suppress the hint once every requested stop has a viable candidate (rule8-met). This resolves the `critique-loop-and-commit-tool-conflict` tension: critique stops pulling the model back to "rephrase" when it should commit.
- **Co-tuning enforcement:** Both DEC-03 changes ride the same `VIABILITY_CONTRACT_ENABLED` flag as DEC-01 (D-13-05, D-13-07). DEC-03 can never fire in isolation.

**Task 2 (TDD RED): Failing tests for threshold override and low_similarity scoping**

New `tests/unit/test_agent_revision.py` with 6 tests:
1. `test_threshold_default_when_override_unset` — resolves to 0.55 when unset
2. `test_threshold_override_applied_when_set` — resolves to 0.45 when set
3. `test_threshold_falls_back_on_empty_override` — empty string falls back to 0.55 (T-13-03-01)
4. `test_flag_off_low_similarity_fires_as_before` — flag-off: hint fires as before
5. `test_flag_on_all_viable_low_similarity_suppressed` — flag-on + all viable: returns None
6. `test_flag_on_not_all_viable_low_similarity_fires` — flag-on + not all viable: hint fires

2 tests failing at RED commit (threshold override + suppression unimplemented).

**Task 2 (TDD GREEN): Implementation in `app/agent/revision.py`**

Three changes:
1. `import os` added at top
2. `LOW_SIMILARITY_THRESHOLD` made env-overridable: `float(os.environ.get("LOW_SIMILARITY_THRESHOLD_OVERRIDE", "") or "0.55")`
3. `_VIABILITY_CONTRACT_ENABLED` flag parsed at module level via truthy-set
4. `_diagnose_last_tool_result` extended with DEC-03 gate: after `_diagnose_one` returns a `low_similarity` hint, if flag ON and `all_slots_viable(state, LOW_SIMILARITY_THRESHOLD)` returns True, return None instead. All other reasons pass through unconditionally. Flag-off path never calls `all_slots_viable` (byte-identical).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test helper used plain dicts for PlaceHit but revision.py uses `getattr`**

- **Found during:** RED test run — `all_closed` hint fired instead of `low_similarity`
- **Issue:** `_diagnose_one` uses `getattr(h, "business_status", None)` which returns `None` on dicts (dict keys are not attributes). Plain dict hits always triggered `all_closed` before reaching the `low_similarity` check.
- **Fix:** Changed test helper `_hit()` to return `PlaceHit` Pydantic objects (matching real tool output) instead of dicts. Updated type annotation in `_state_with_search_hits`.
- **Files modified:** `tests/unit/test_agent_revision.py`
- **Commit:** 89c3a55 (corrected at RED time, before GREEN)

## Verification

- `poetry run pytest tests/unit/test_agent_revision.py tests/unit/test_critique_checks.py -v` — 98 passed, 0 failed
- `make typecheck` — Success: no issues found in 40 source files
- `make lint` — All checks passed
- `poetry run python -c "import app.agent.viability; import app.agent.graph; import app.agent.revision; print('No circular import')"` — OK
- `test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts` — PASSED (revision.py changes do not add behavioral phrases to prompts.py or io.py)
- Flag-off path is byte-identical: `all_slots_viable` is only called when `_VIABILITY_CONTRACT_ENABLED` is True
- Decision doc committed before code (Task 1 commit `4d3c0ec` precedes Task 2 commits `89c3a55`, `18b918f`)

## Commits

| Hash | Message |
|------|---------|
| 4d3c0ec | docs(13-03): record DEC-03 threshold direction and low_similarity scoping decision before code |
| 89c3a55 | test(13-03): add failing tests for threshold env-override and low_similarity scoping |
| 18b918f | feat(13-03): make threshold env-overridable and scope low_similarity behind arm flag |

## TDD Gate Compliance

- [x] RED gate: `test(13-03)` commit `89c3a55` exists with 2 failing tests
- [x] GREEN gate: `feat(13-03)` commit `18b918f` exists with all 6 tests passing
- [x] REFACTOR gate: not needed (implementation was clean)

## Self-Check: PASSED

- [x] `docs/decisiveness_dec03_decision.md` exists and is 135 lines (>= 20 required)
- [x] `docs/decisiveness_dec03_decision.md` contains `LOW_SIMILARITY_THRESHOLD_OVERRIDE`
- [x] `docs/decisiveness_dec03_decision.md` contains `VIABILITY_CONTRACT_ENABLED`
- [x] `docs/decisiveness_dec03_decision.md` references D-13-07 and roadmap criterion 4
- [x] `app/agent/revision.py` contains `LOW_SIMILARITY_THRESHOLD_OVERRIDE`
- [x] `app/agent/revision.py` contains `_VIABILITY_CONTRACT_ENABLED` flag
- [x] `app/agent/revision.py` imports `all_slots_viable` inside the gate (lazy, circular-safe)
- [x] `tests/unit/test_agent_revision.py` exists with 6 tests, all passing
- [x] Task 1 commit (`4d3c0ec`) precedes Task 2 commits (doc-first ordering)
- [x] `make typecheck` clean (40 source files)
- [x] `make lint` clean
- [x] No circular imports

## Known Stubs

None — all logic is wired to real module-level flag reads and the `all_slots_viable` predicate from plan 13-01.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary crossings. The env-override is read at module import time and raises `ValueError` visibly on bad input (T-13-03-01); the suppression gate is off by default (T-13-03-02).
