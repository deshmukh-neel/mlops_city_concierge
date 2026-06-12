---
phase: 13-decisiveness-experiment-arms
plan: "04"
subsystem: agent/graph
tags: [decisiveness, arm-flags, forced-commit, parallel-tools, tdd, A2-forced-commit, A3-parallel]
dependency_graph:
  requires: [13-01-viability-predicate, 13-02-viability-prompt]
  provides: [A1-prompt-wiring, A2-forced-commit-branch, A3-parallel-act]
  affects: [app/agent/graph.py, tests/unit/test_graph_forced_commit.py, tests/unit/test_graph_parallel_tools.py]
tech_stack:
  added: []
  patterns: [arm-flag-closure, asyncio.gather-order-stable, tdd-red-green, forced-commit-through-normal-path]
key_files:
  created:
    - tests/unit/test_graph_forced_commit.py
    - tests/unit/test_graph_parallel_tools.py
  modified:
    - app/agent/graph.py
decisions:
  - "A2 placed in critique() BEFORE max_steps check so it fires even when max_steps would be the next stop"
  - "critique_final_with_stops called on state.model_copy(stops=committed_stops) to thread committed state into hard checks (empty original state would make violations vacuously empty)"
  - "_exec_one inner function typed as Any (not ToolCall TypedDict) to satisfy mypy — ToolCall is a TypedDict that mypy does not coerce to dict[str,Any]"
  - "Test approach: full-graph tests use mocked commit_stops + critique_final_with_stops to ensure single plan->act->critique cycle (avoids LangGraph recursion limit from DB-less loops)"
  - "Sequential path preserved verbatim in if/else branch (not routed through _exec_one) to guarantee byte-identical behavior when flag is off"
metrics:
  duration: "~25 minutes"
  completed: "2026-06-12T05:55:40Z"
  tasks: 3
  files: 3
requirements: [DEC-01, DEC-02, DEC-04]
---

# Phase 13 Plan 04: Graph Arms — Forced-Commit and Parallel Tools Summary

Three DEC arm flags wired into `app/agent/graph.py` at graph-build time: A1 prompt addendum, A2 forced-commit branch in critique(), A3 parallel tool execution in act(). With all flags off, the graph is byte-identical to pre-Phase-13 behavior.

## What Was Built

**Task 1 (TDD): Arm-flag reads at graph-build time + A1 prompt addendum wiring**

Three arm flags resolved once in `build_agent_graph()` before any inner function definitions:

- `_forced_commit_step: int` — `int(os.environ.get("FORCED_COMMIT_STEP", "0") or "0")`
- `_viability_contract_enabled: bool` — truthy-set parse of `VIABILITY_CONTRACT_ENABLED`
- `_parallel_tool_execution_enabled: bool` — truthy-set parse of `PARALLEL_TOOL_EXECUTION_ENABLED`

A1 wiring: `_viability_prompt_addendum = rule8_viability_addendum(_viability_contract_enabled)` is pre-computed at build time and appended to the `SystemMessage` in `plan()` after `SYSTEM_PROMPT.format(...)`. When flag off, addendum is `""` → byte-identical to baseline.

New imports added: `os`, `rule8_viability_addendum` (from `app.agent.prompts`), `LOW_SIMILARITY_THRESHOLD`, `all_slots_viable`, `best_viable_candidate_per_slot`.

**Task 2 (TDD): A2 forced-commit branch in critique() — D-13-04 required test**

A2 branch added in `critique()` BEFORE the `max_steps` check:

```
if _forced_commit_step > 0
   AND state.step_count >= _forced_commit_step
   AND not state.stops          # no prior commit
   AND all_slots_viable(state, LOW_SIMILARITY_THRESHOLD)
```

When firing:
1. `best_viable_candidate_per_slot(state, ...)` → list of grounded candidate dicts
2. `commit_stops(state, raw_stops)` → validated `committed_stops` (place_id grounding enforced)
3. `critique_final_with_stops(state.model_copy(update={"stops": committed_stops}), ...)` — CRITICAL: state copy with committed stops threads the plan into the hard-check gauntlet (calling with the original empty-stops state would make `itinerary_violations` vacuously empty)
4. Merge result dict with `commit_forced=True`, `forced_commit_step=state.step_count`, `stops=committed_stops`

When NOT firing (any slot lacks viable candidate): falls through to `short_circuit_max_steps`. `commit_forced` stays `False`, `forced_commit_step` stays `None`.

**D-13-04 required test** (`test_forced_commit_triggers_at_step_n`): a mock model that always emits `semantic_search` (never commits) triggers the A2 branch at the configured step. The test mocks `all_slots_viable=True` and `commit_stops` to return a valid Stop, and verifies `commit_forced is True` and `forced_commit_step >= 2`.

**Task 3 (TDD): A3 parallel tool execution in act() — D-13-08**

`act()` refactored with an inner `async def _exec_one(tc)` helper that covers all per-call logic (commit branch, unknown-tool branch, closure injection, `asyncio.to_thread`).

When `_parallel_tool_execution_enabled`:
```python
par_results = await asyncio.gather(*[_exec_one(tc) for tc in ai.tool_calls])
for msg, scratch_name, scratch_entry, stops, was_commit in par_results:
    new_messages.append(msg)   # ORIGINAL ORDER preserved by gather
    ...
```
`asyncio.gather` preserves input order — results[i] corresponds to ai.tool_calls[i], so original tool_call order is maintained regardless of completion order (D-13-08).

When flag off: existing sequential `for tc in ai.tool_calls:` loop is preserved verbatim in the `else` branch — no routing through `_exec_one`, guaranteeing byte-identical sequential behavior.

INST-04 timing: `_tool_start`/`_tool_elapsed` wrap the entire parallel/sequential block unchanged.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] mypy type error: ToolCall TypedDict incompatible with dict[str,Any]**

- **Found during:** Task 3 implementation — mypy reported `Argument 1 to "_exec_one" has incompatible type "ToolCall"; expected "dict[str, Any]"`
- **Issue:** `ai.tool_calls` yields `ToolCall` TypedDict items; mypy cannot coerce these to `dict[str, Any]`
- **Fix:** Changed `_exec_one` parameter type annotation from `dict[str, Any]` to `Any`
- **Files modified:** `app/agent/graph.py`
- **Commit:** c017e72 (part of feat commit)

**2. [Rule 2 - Missing Critical Safety] State-threading in A2 forced-commit**

- **Found during:** Task 2 implementation — analysis of `critique_final_with_stops` revealed it calls `itinerary_violations(state)` which uses `state.stops`
- **Issue:** Calling `critique_final_with_stops(state, ...)` with the original uncommitted state (stops=[]) would make `itinerary_violations` return no violations → forced commits would skip all hard checks
- **Fix:** Pass `state.model_copy(update={"stops": committed_stops})` so the gauntlet evaluates the synthesized itinerary
- **Files modified:** `app/agent/graph.py`
- **Commit:** 3fb6134

**3. [Rule 1 - Bug] Test design: LangGraph recursion limit hit in graph-level tests**

- **Found during:** Task 2 and Task 3 test implementation — full-graph tests with mock LLMs that loop (emit tool calls indefinitely) hit LangGraph's recursion limit before `max_steps` could fire
- **Fix:** Redesigned tests to mock `commit_stops` and `critique_final_with_stops` so the graph terminates in one plan->act->critique cycle; used `step_count` pre-seeding for the forced-commit "skip" tests
- **Files modified:** `tests/unit/test_graph_forced_commit.py`, `tests/unit/test_graph_parallel_tools.py`
- **Commits:** bef43ea → 3fb6134, 297862a → c017e72

## Verification

- `poetry run pytest tests/unit/test_graph_forced_commit.py tests/unit/test_graph_parallel_tools.py -v` — **11 passed**
- `make test` (full suite) — **1333 passed, 53 skipped, 9 deselected**
- `make typecheck` — **Success: no issues found in 40 source files**
- `make lint` — **All checks passed**
- PROMPT-02 grep gate — **PASSED** (no forbidden phrases in prompts.py / io.py)
- `grep -n "FORCED_COMMIT_STEP" app/agent/graph.py` — flag present (greppable)
- `grep -n "VIABILITY_CONTRACT_ENABLED" app/agent/graph.py` — flag present (greppable)
- `grep -n "PARALLEL_TOOL_EXECUTION_ENABLED" app/agent/graph.py` — flag present (greppable)
- `grep -n "asyncio.gather" app/agent/graph.py` — A3 implementation present
- No hardcoded `6` in A2 firing condition (env-driven `_forced_commit_step` variable)

## Commits

| Hash | Message |
|------|---------|
| bef43ea | test(13-04): add failing tests for arm flags, A1 prompt addendum, and A2 forced-commit (RED) |
| 4df4a25 | feat(13-04): read arm flags at graph-build time and wire A1 viability prompt addendum |
| 3fb6134 | feat(13-04): add A2 forced-commit branch in critique() and arm-flag imports |
| 297862a | test(13-04): add failing tests for A3 parallel tool execution (RED) |
| c017e72 | feat(13-04): implement A3 parallel tool execution in act() with order-stable results |

## TDD Gate Compliance

- RED gate (Task 1+2): `bef43ea` — `test(13-04): add failing tests...` (RED)
- GREEN gate (Task 1): `4df4a25` — `feat(13-04): read arm flags...` (GREEN)
- GREEN gate (Task 2): `3fb6134` — `feat(13-04): add A2 forced-commit...` (GREEN)
- RED gate (Task 3): `297862a` — `test(13-04): add failing tests for A3...` (RED)
- GREEN gate (Task 3): `c017e72` — `feat(13-04): implement A3 parallel...` (GREEN)

## Self-Check

- [x] `app/agent/graph.py` — contains `FORCED_COMMIT_STEP`, `VIABILITY_CONTRACT_ENABLED`, `PARALLEL_TOOL_EXECUTION_ENABLED`, `asyncio.gather`
- [x] `tests/unit/test_graph_forced_commit.py` — exists, 6 tests passing, D-13-04 required test present
- [x] `tests/unit/test_graph_parallel_tools.py` — exists, 5 tests passing
- [x] All 1333 tests pass (full suite)
- [x] mypy clean (40 files)
- [x] ruff clean
- [x] PROMPT-02 grep gate green
- [x] No hardcoded `6` in A2 firing condition

## Known Stubs

None — all arm flags are wired and operational; flag-off paths are byte-identical to baseline.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes. The arm flags are environment-controlled (operator trust boundary, T-13-04-01/02/03 mitigated as specified in plan threat model). The forced-commit synthesizer routes through `commit_stops` for place_id grounding validation (T-13-04-01 mitigated). The `asyncio.gather` parallel execution is bounded by the model's single AIMessage tool_calls (T-13-04-04 accepted).
