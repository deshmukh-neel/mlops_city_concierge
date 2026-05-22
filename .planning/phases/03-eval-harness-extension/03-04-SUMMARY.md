---
phase: 03-eval-harness-extension
plan: 04
subsystem: eval-runner
tags: [eval, multi-turn, runner, tdd, evaluator, json-safety]
requires:
  - "app/eval/config.py:EvalQuery.turns (plan 03-02)"
  - "app/agent/graph.py:build_agent_graph"
  - "scripts/eval_agent.py:evaluate_case (single-turn body)"
  - "scripts/eval_agent.py:query_result_from_state"
  - "scripts/eval_agent.py:state_from_graph_output"
  - "scripts/eval_agent.py:EVAL_CONTEXT_TEMPLATE"
provides:
  - "scripts/eval_agent.py:evaluate_multi_turn_case (async helper; EVAL-06)"
  - "scripts/eval_agent.py:_eval_context_for (shared SystemMessage renderer)"
  - "scripts/eval_agent.py:evaluate_case (case.turns branch)"
affects:
  - "scripts/eval_agent.py:evaluate_cases (no signature change; new branch is internal to evaluate_case)"
  - "Plan 03-05 (matrix runner): subprocess-fanned-out matrix invokes evaluate_case which now routes multi-turn cases automatically"
  - "Plan 03-07 (baselines): refinement-cheaper and closure-cascade YAML cases can finally express their turn-2 regressions"
tech-stack:
  added: []
  patterns:
    - "Async multi-turn helper threading prior state.messages via add_messages reducer"
    - "Fail-open synthetic tool-error entry (multi_turn_runner) on intermediate-turn exceptions — mirrors evaluate_cases pattern"
    - "_RecordingScriptedLLM (BaseChatModel subclass) capturing per-invocation message lists for threading assertions"
    - "Surgical scripts.eval_agent.time module replacement (types.SimpleNamespace) to isolate helper-clock mocking from asyncio's time use"
    - "json.dumps(asdict(QueryEvalResult)) wire-contract assertion at the multi-turn boundary (EVAL-08 / PR #94 be541a3 regression class)"
key-files:
  created: []
  modified:
    - "scripts/eval_agent.py"
    - "tests/unit/test_eval_agent.py"
decisions:
  - "Multi-turn branch lives INSIDE evaluate_case (single-line `if case.turns: return await evaluate_multi_turn_case(graph, case)`) rather than at evaluate_cases — keeps the per-case branch invisible to all current callers (build_report, eval_matrix runner, future MLflow-backed callers) and preserves the byte-equivalent single-turn JSON output for the 29 existing hand-written cases"
  - "Extracted _eval_context_for() to DRY the EVAL_CONTEXT_TEMPLATE rendering between single-turn and multi-turn paths — no duplicated template"
  - "Final reported state = LAST turn's state (not first or merged). v2.0's confirmed regression bugs (category compliance, rationale alignment, refinement explosion) all surface on turn 2+; scoring the last state targets the actual failure surface"
  - "latency_seconds = SUM of per-turn latencies (not max, not last). Aggregate p50/p95 then reflect the total user-visible scenario cost, which is what the merge-gate is comparing across baselines"
  - "Fail-open on intermediate-turn exception: synthetic multi_turn_runner entry recorded in state.scratch (matches tool_errors_from_state's `result.error` contract) — the eval run continues for other cases. Re-raising would crash the whole report (32 cases lost for one turn-2 failure)"
  - "First-turn failure still gets a JSON row: when turn 0 raises, the helper synthesizes a minimal ItineraryState(messages=messages_in) rather than letting the exception bubble — graceful degradation across the failure modes"
  - "Used _RecordingScriptedLLM (duplicate of _ScriptedLLM from test_chat_functional.py + a seen-list field) per plan guidance: duplication is acceptable for small per-file helpers; promoting to conftest deferred to a future test-infrastructure plan"
  - "Latency test patches scripts.eval_agent.time as a SimpleNamespace (not just time.monotonic) so asyncio's separate time-module reference is unaffected — without this, side_effect=[...] gets drained by asyncio.base_events.time() before the helper's call sites consume them"
  - "Test trajectory uses single AIMessage(content=..., tool_calls=[]) per turn so the graph runs plan -> critique -> finalize_as_is in ONE plan() call per turn; stops=[] keeps all scorers in fail-open mode, no DB pool leak even under full-suite collection (project_full_suite_db_pool_contamination)"
  - "Did NOT touch the legacy sys.path.insert block in scripts/eval_agent.py:21-23 (out of scope per plan + project memory project_app_editable_install)"
metrics:
  duration_minutes: 30
  tasks_completed: 2
  files_modified: 2
  tests_added: 6
  red_commits: 1
  green_commits: 1
  test_commits: 1
  completed: "2026-05-21"
requirements:
  - EVAL-06
  - EVAL-08
---

# Phase 3 Plan 04: Multi-Turn Runner Summary

One-liner: Extended `scripts/eval_agent.py` with `evaluate_multi_turn_case` (EVAL-06) + a `case.turns` branch in `evaluate_case` so the refinement-cheaper and closure-cascade baseline scenarios (turn 2+ regression surface) can finally be expressed in YAML and scored — with five `_RecordingScriptedLLM`-driven unit tests including the EVAL-08 / PR #94 / be541a3 `json.dumps(asdict(QueryEvalResult))` regression guard at the multi-turn message-threading boundary.

## What Shipped

### Task 1 — `evaluate_multi_turn_case` helper + branch in `evaluate_case` (EVAL-06)

Three additions to `scripts/eval_agent.py`:

1. **`_eval_context_for(case) -> str`** — a thin helper extracted from the existing `evaluate_case` body so the offline-eval SystemMessage template renders identically for single-turn and multi-turn paths. No behavior change for single-turn callers; pure refactor.
2. **`evaluate_multi_turn_case(graph, case) -> QueryEvalResult`** — a new async helper that runs `len(case.turns) + 1` sequential `graph.ainvoke` calls against a shared graph instance. Turn 1 builds `[SystemMessage(eval_context), HumanMessage(case.query)]`; each subsequent turn re-uses the prior turn's resulting `state.messages` and appends `HumanMessage(turn_text)`. A fresh `ItineraryState` per turn ensures `step_count`, `scratch`, and `revision_counts` reset between turns — only `messages` threads through, matching the frontend's `conversation_state` round-trip semantics.
3. **`evaluate_case` branch** — top-of-function `if case.turns: return await evaluate_multi_turn_case(graph, case)`. When `case.turns is None or []`, the original single-turn body runs unchanged, producing byte-equivalent JSON for the 29 existing hand-written cases.

The fail-open contract on intermediate-turn exceptions: any turn raising during `graph.ainvoke` is caught, the elapsed time is still added to `total_latency`, a synthetic `{"args": {"turn_index": N, "turn": turn_text}, "result": {"error": "turn N raised: <repr>"}, "step": N, "id": "multi_turn_runner_N"}` entry is appended to `state.scratch["multi_turn_runner"]`, and the partial `QueryEvalResult` is returned. `tool_errors_from_state` surfaces it as `"multi_turn_runner: turn N raised: ..."`. The whole eval run does NOT crash — same fail-open shape as `evaluate_cases`.

**Commits:**
- `e21fd0d` — **RED:** failing test that imports `evaluate_multi_turn_case`; fails at collection with `ImportError`
- `733681c` — **GREEN:** helper + branch land; all 28 tests pass; mypy clean

### Task 2 — Multi-turn behavior tests with EVAL-08 / PR #94 regression guard (EVAL-06 + EVAL-08)

Five new unit tests in `tests/unit/test_eval_agent.py`, all using a duplicated `_RecordingScriptedLLM` (variant of `_ScriptedLLM` from `tests/unit/test_chat_functional.py` with an extra `seen: list[list[BaseMessage]]` field that captures the messages each `_generate` invocation saw — duplication is acceptable per the plan guidance; the helper is six lines of trivial pydantic-field setup):

1. **`test_evaluate_case_single_turn_unchanged`** — `turns=None` runs through `evaluate_case` via the pre-03-04 single-turn body. Asserts exactly one plan() invocation, no `multi_turn_runner` entry in `deterministic.tool_errors`, and the scripted reply propagates to `result.final_reply`. Regression guard against accidental coupling of the multi-turn branch to the single-turn path.

2. **`test_evaluate_multi_turn_threads_messages`** — `turns=["make stop 2 cheaper"]`. Asserts that turn 2's plan() invocation (`llm.seen[1]`) contains:
   - The original `HumanMessage("coffee in soma")`
   - The new `HumanMessage("make stop 2 cheaper")`
   - The original eval-context `SystemMessage`
   - Two `seen` entries total (one per turn)

   The recorder-based assertion is the strongest threading proof: any future regression that nukes prior messages between turns fails this test directly, rather than via a downstream symptom.

3. **`test_multi_turn_latency_sums`** — Patches `scripts.eval_agent.time` with a `types.SimpleNamespace` whose `monotonic()` pops from `[0.0, 1.0, 1.0, 3.0]`. Asserts `result.latency_seconds == pytest.approx(3.0)` — exact arithmetic on a 2-turn case (4 clock reads). The surgical module-replacement isolates the helper from asyncio's own `time.monotonic` calls; patching `time.monotonic` directly drains the side_effect iterator and crashes the loop with `StopIteration`.

4. **`test_multi_turn_intermediate_failure_captured`** — Scripts only one AIMessage so turn 2's `_generate` raises `IndexError` on the empty pop. Asserts:
   - `result` is still a `QueryEvalResult` (the run did not crash)
   - `result.final_reply == "turn1 reply"` (turn 1's reply survives in the partial state)
   - `deterministic.tool_errors` contains an entry matching `"multi_turn_runner"` AND `"turn 1"` (the failing turn's index)

5. **`test_multi_turn_tool_calls_are_json_safe`** (EVAL-08 / PR #94 / `be541a3`) — Runs a 2-turn case to completion and:
   - Calls `json.dumps(asdict(result))` — proves the dataclass wire shape from the multi-turn helper is json-safe (the canonical EVAL-08 contract)
   - Walks every `AIMessage` across `llm.seen` and `json.dumps(tc["args"])` for each tool_call — vacuous in this finalize-only trajectory but locks the contract shape so a future tool-calling test inherits the guarantee for free

   The test docstring explicitly names `PR #94 commit be541a3` and inlines the regression rationale: any future change that smuggles a Pydantic model or datetime into `tool_calls[i]["args"]` at the multi-turn boundary fails here at one of the two json.dumps walks.

The five tests collectively exercise every behavior bullet from the plan's `<behavior>` block.

**Commits:**
- `7995e6c` — **TEST:** 5 multi-turn behavior tests + `_RecordingScriptedLLM` helper; all pass against the GREEN implementation from Task 1; full unit suite 651 → 651 (no regressions)

## Verification Runs

```text
poetry run pytest tests/unit/test_eval_agent.py -v
  33 passed in 0.53s    (27 baseline + 1 import-RED + 5 multi-turn behavior)

poetry run pytest tests/unit/test_eval_agent.py -v -k multi_turn
  5 passed, 28 deselected in 0.55s

poetry run pytest tests/unit/    # full unit suite (project_full_suite_db_pool_contamination test)
  651 passed, 9 warnings in 11.17s    (645 baseline -> +6 new tests, zero regressions)

poetry run mypy scripts/eval_agent.py
  Success: no issues found in 1 source file

poetry run ruff check tests/unit/test_eval_agent.py scripts/eval_agent.py
  All checks passed!
```

**Plan-level inspect verification:**

```python
poetry run python -c "
from scripts.eval_agent import evaluate_multi_turn_case, evaluate_case
import inspect
assert inspect.iscoroutinefunction(evaluate_multi_turn_case)
assert inspect.iscoroutinefunction(evaluate_case)
print('OK')
"
  OK
```

## Acceptance Criteria Status

All acceptance criteria from both tasks verified:

**Task 1:**
- ✅ `grep -n "def evaluate_multi_turn_case\|async def evaluate_multi_turn_case" scripts/eval_agent.py` → 1 line (line 494)
- ✅ `grep -n "case\.turns" scripts/eval_agent.py` → 4 lines (the branch on line 475 + 3 docstring references)
- ✅ `inspect.iscoroutinefunction(evaluate_multi_turn_case)` → True
- ✅ `poetry run pytest tests/unit/test_eval_agent.py -v` → exit 0; 33/33 pass
- ✅ `poetry run mypy scripts/eval_agent.py` → Success
- ✅ Existing 27 tests still pass without modification — single-turn path is byte-equivalent

**Task 2:**
- ✅ `grep -c "multi_turn"` returns ≥ 5 multi-turn-named test functions (6 in total, including Task 1's import test)
- ✅ `grep -n "json.dumps" tests/unit/test_eval_agent.py` → 12 lines (asdict, tc["args"], and docstring references)
- ✅ `grep -n "be541a3\|PR #94\|EVAL-08" tests/unit/test_eval_agent.py` → 8 lines (provenance comments in docstring + inline comments)
- ✅ `poetry run pytest tests/unit/test_eval_agent.py -v -k multi_turn` → 5 pass, 0 fail
- ✅ `poetry run make test-unit` (full unit suite via direct invocation) → 651 pass
- ✅ `poetry run ruff check tests/unit/test_eval_agent.py` → clean

**Plan-level:**
- ✅ `poetry run make test-unit` passes (651 tests)
- ✅ Both helpers are coroutine functions
- ✅ `grep -c "EVAL-08\|json.dumps" tests/unit/test_eval_agent.py` → 16

## Deviations from Plan

**None — plan executed exactly as written.**

The plan called out three design choices in advance that I followed without amplification:

1. The plan suggests `if case.turns: return await evaluate_multi_turn_case(graph, case)` at the top of `evaluate_case` — exact pattern used. `case.turns` is `None` (default) or a non-empty list (the field validator in `app/eval/config.py:turns_non_empty_when_present` rejects `[]`), so the `if case.turns:` truthiness check is correct and explicit.
2. The plan suggests building a synthetic tool-error entry under `state.scratch["multi_turn_runner"]` shaped so `tool_errors_from_state` surfaces it — implemented with the exact `{"result": {"error": "..."}}` shape that the existing extractor reads.
3. The plan says first-turn failures should still produce a JSON row rather than bubble — handled via `partial_state = state if state is not None else ItineraryState(messages=messages_in)` in the exception branch, so even a turn-0 crash yields a graceful `QueryEvalResult`.

No Rule 1/2/3 auto-fixes were needed:

- The `EvalQuery.turns` validator from plan 03-02 already rejects `[]`, so `if case.turns:` is sound (no need for `if case.turns is not None and len(case.turns) > 0`).
- The `add_messages` reducer behavior is the documented LangGraph contract; threading `[*state.messages, HumanMessage(...)]` into a fresh `ItineraryState` works without special accommodations.
- mypy and ruff both flagged zero issues on the first GREEN run.

The latency test required ONE in-test design adjustment (replace `scripts.eval_agent.time` as a SimpleNamespace rather than patching `time.monotonic` directly) — this is documented in the test's docstring with the rationale. Not a deviation from the plan, but a deviation from my first implementation pass which failed under pytest-asyncio's event-loop time queries.

## Authentication Gates

None — pure async helper + scripted-LLM tests; no external services touched.

## Known Stubs

None. The multi-turn runner is immediately consumable by plan 03-05 (matrix runner: subprocess fanout invokes `evaluate_case` which now auto-routes multi-turn cases) and plan 03-07 (baselines: refinement-cheaper and closure-cascade YAML scenarios can express their turn-2 regressions). No placeholder `pass` statements, no `TODO`/`FIXME` markers, no "coming soon" UI text — every shipped line is the final shape.

The plan deliberately did NOT add multi-turn cases to `configs/eval_queries.yaml` (those land in plan 03-07's baseline commit), and DID NOT introduce a `--llm scripted` CLI flag (that's plan 03-06's CI work). Both are intentional out-of-scope deferrals per the plan, not stubs.

## Threat Flags

None. No new network endpoints, auth paths, file-access patterns, schema changes, or trust boundaries introduced. The multi-turn helper is a pure-async loop over `graph.ainvoke` calls — same I/O posture as the pre-03-04 single-turn body. The synthetic `multi_turn_runner` scratch entries contain only the failing turn's text and exception repr; no place_ids, no secrets, no PII.

## TDD Gate Compliance

Task 1 followed RED → GREEN strictly:

| Task | RED       | GREEN     | Notes |
| ---- | --------- | --------- | ----- |
| 1    | `e21fd0d` | `733681c` | RED fails at collection with `ImportError: cannot import name 'evaluate_multi_turn_case'` — strongest form of RED |
| 2    | n/a       | `7995e6c` | Task 2 is a pure test-addition task — written against the GREEN helper from Task 1; commit type is `test(...)` per the executor TDD reference |

Task 2's "GREEN against an existing GREEN implementation" framing is the appropriate TDD shape for a test-only task: the new tests pin behavior that already exists, with the contracts (threading, latency-sum, fail-open, json-safety) becoming regression guards going forward. If any future change breaks one of these behaviors, the test goes red — which is the entire point of writing the tests in the first place.

## Self-Check: PASSED

- FOUND: `scripts/eval_agent.py` — modified, contains `_eval_context_for` (line 456), `if case.turns:` branch (line 475), `async def evaluate_multi_turn_case` (line 494)
- FOUND: `tests/unit/test_eval_agent.py` — modified, contains `test_evaluate_multi_turn_case_is_async_helper`, `_RecordingScriptedLLM`, and five `*multi_turn*` / `*single_turn*` behavior tests
- FOUND: commit `e21fd0d` (Task 1 RED — test imports evaluate_multi_turn_case)
- FOUND: commit `733681c` (Task 1 GREEN — helper + branch)
- FOUND: commit `7995e6c` (Task 2 — 5 behavior tests)
- VERIFIED: 33/33 tests pass in `tests/unit/test_eval_agent.py`
- VERIFIED: 651/651 tests pass in `tests/unit/` (full suite, no regressions)
- VERIFIED: mypy `scripts/eval_agent.py` clean; ruff both files clean
