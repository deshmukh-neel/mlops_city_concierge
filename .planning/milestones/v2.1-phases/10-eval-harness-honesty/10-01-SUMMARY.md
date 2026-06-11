---
phase: 10-eval-harness-honesty
plan: "01"
subsystem: eval-harness
tags: [eval, error-handling, scoring, tdd]
dependency_graph:
  requires: []
  provides: [EVAL-01-error-status-runner]
  affects: [scripts/eval_agent.py, app/agent/critique/checks.py, tests/unit/test_eval_agent.py]
tech_stack:
  added: []
  patterns: [status-discriminator, error-record-builder, scored-only-aggregation]
key_files:
  created: []
  modified:
    - scripts/eval_agent.py
    - app/agent/critique/checks.py
    - tests/unit/test_eval_agent.py
decisions:
  - "D-10-01: QueryEvalResult gains status discriminator (default 'ok') + error dict field"
  - "D-10-02: Exceptions in both threading branches now return make_error_record(), not query_result_from_state() on partial state"
  - "D-10-03: aggregate_results filters on status=='ok'; gains n_scored/n_errored/cell_valid/errors[] fields"
  - "D-10-04: report_has_errors extended to detect n_errored > 0; Branch-1 abstain in checks.py documented but byte-unchanged"
metrics:
  duration: "~60 minutes"
  completed: "2026-06-10"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 3
---

# Phase 10 Plan 01: Error-Status Runner Summary

**One-liner:** Eval runner now emits `status=error` records on any turn exception, excludes errored runs from scorer aggregation, and proves the 21-14-30Z fail-open is gone via stub-driven replay tests.

## What Was Built

Three TDD tasks closed the fail-open scorer paths (EVAL-01) that caused the 2026-06-05T21-14-30Z matrix to show 1.0 medians from a quota-429-all-errors run.

### Task 1: RunErrorRecord schema + make_error_record builder

**Schema change:** `QueryEvalResult` dataclass gains two fields with defaults:
- `status: str = "ok"` — discriminator for aggregate filter
- `error: dict[str, str] | None = None` — D-10-01 error block: `{stage, type, message}`

**New function:** `make_error_record(case, stage, exc) -> QueryEvalResult`
- Returns `status="error"` record with `error={"stage": ..., "type": type(exc).__name__, "message": str(exc)[:500]}`
- All deterministic check scores are `None` (scorers never invoked)
- Stage values: `"setup"`, `"turn0"`, `"turnN"` per D-10-01
- Serializes cleanly via `asdict() -> json.dumps()`
- All pre-existing scored rows keep `status="ok"` via the default

### Task 2: Replace partial-state scoring in both threading branches (D-10-02)

**`_run_legacy_threading`:** The `except Exception` block that previously built a `partial_state`, stamped `multi_turn_runner` into scratch, and called `query_result_from_state()` on it is replaced with:
```python
stage = "turn0" if index == 0 else "turnN"
return make_error_record(case, stage, exc)
```

**`_run_prod_threading`:** Same replacement; returns the error record plus a sentinel `ItineraryState()` (the function signature requires a tuple):
```python
stage = "turn0" if index == 0 else "turnN"
return (make_error_record(case, stage, exc), ItineraryState())
```

**`app/agent/critique/checks.py`:** Branch-1 abstain code (`checks.py:488-491`) is byte-unchanged. Added a clarifying D-10-04 comment documenting the invariant that `score_checks` is only called on completed `status="ok"` runs — Branch-1 never fires from exception-corrupted partial state anymore.

### Task 3: Scored-only aggregation + 21-14-30Z replay acceptance tests (D-10-03)

**`aggregate_results` rewrite:**
- Splits results into `scored_results = [r for r in results if r.status == "ok"]` and `errored_results`
- All scorer means, rates, and counts computed over `scored_results` only
- New aggregate fields: `n_scored`, `n_errored`, `cell_valid` (n_errored==0), `errors` list
- An all-error run yields `refinement_minimal_edit_mean = 0.0` (not 1.0)

**`report_has_errors` extended:** Returns True when `n_errored > 0` in addition to `check_error_count > 0`. This distinguishes infra failures (whole-run errors) from individual scorer exceptions on completed runs.

**21-14-30Z replay acceptance tests:**
- `RaisingChatModel` test stub: raises configurable exception on every `_generate()` call
- `test_21_14_30z_replay_turn0_exception_produces_error_record`: turn-0 LLM exception → status="error", stage="turn0", all scores=None
- `test_21_14_30z_replay_all_error_report_has_zero_scored`: all-error run → n_scored==0, refinement_minimal_edit_mean==0.0 (not 1.0)
- `test_21_14_30z_replay_turn_n_exception_produces_error_record`: ScriptedLLM exhaustion on turn 1 → status="error", stage="turnN"
- Updated existing tests `test_multi_turn_intermediate_failure_captured` and `test_prod_mode_preserves_fail_open_on_exception` to reflect the new error-status contract

## Commits

| Hash | Description |
|------|-------------|
| 134a696 | test(10-01): add failing tests for Task 1 ERROR-status schema (RED) |
| 4e9415b | feat(10-01): add status discriminator + make_error_record to eval_agent.py |
| 67888cd | test(10-01): add failing tests for Task 2 error-record threading (RED) |
| 0f64ac7 | feat(10-01): replace partial-state scoring with error records in both threading branches |
| a2cfafb | test(10-01): add failing tests for Task 3 aggregate + 21-14-30Z replay (RED) |
| c419664 | feat(10-01): aggregate scored-only + emit error counts; 21-14-30Z replay tests |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated two existing tests that expected old fail-open behavior**
- **Found during:** Task 2 GREEN implementation
- **Issue:** `test_multi_turn_intermediate_failure_captured` and `test_prod_mode_preserves_fail_open_on_exception` asserted the old `multi_turn_runner` tool error behavior from the partial-state scoring path. After D-10-02 removal, these tests failed.
- **Fix:** Updated both tests to assert the new error-status contract (`status="error"`, `error.stage="turnN"`, all check scores=None)
- **Files modified:** `tests/unit/test_eval_agent.py`

**2. [Rule 2 - Missing functionality] test_no_partial_state_scoring_in_eval_agent uses AST-based check**
- **Found during:** Task 2 RED test writing
- **Issue:** Initial test checked `"partial_state" not in source` but the new code still mentions `partial_state` in comments (documenting what was removed)
- **Fix:** Changed test to use AST analysis — walks `ExceptHandler` nodes and asserts none call `query_result_from_state` (the actual scoring path), which is a more precise and future-proof check

**3. [Rule 1 - Bug] RaisingChatModel placed before BaseChatModel import**
- **Found during:** Task 2 RED test writing
- **Issue:** Placed `RaisingChatModel(BaseChatModel)` before the `BaseChatModel` import at line ~682 (mid-file import pattern). Caused `NameError: name 'BaseChatModel' is not defined`.
- **Fix:** Moved `RaisingChatModel` class definition to after the `from langchain_core.language_models import BaseChatModel` import block; added `BaseChatModel` to the late imports section.

## Verification

All plan verification criteria passed:

```
poetry run pytest tests/unit/test_eval_agent.py tests/unit/test_critique_checks.py -q
→ 189 passed in 0.93s

grep -c "partial_state" scripts/eval_agent.py
→ 2 (comments only; no scoring code)

grep -n 'status == "ok"' scripts/eval_agent.py
→ 1011: scored_results = [r for r in results if r.status == "ok"]

poetry run ruff check scripts/eval_agent.py tests/unit/test_eval_agent.py
→ All checks passed!
```

## Known Stubs

None. All error records, aggregate fields, and replay tests are wired to real behavior.

## Threat Flags

No new threat surface introduced. The threat mitigations from the plan's STRIDE register were applied:

- T-10-01-01 (Repudiation): Every turn exception now writes an auditable status=error record with stage+type+message. No silent swallow.
- T-10-01-02 (Tampering): Errored runs excluded from means via status=="ok" filter. Quota-429 matrix can no longer masquerade as 1.0 baseline.
- T-10-01-03 (Information disclosure): error.message truncated to 500 chars. Exception messages from OpenAI/Anthropic SDKs do not echo raw API keys.

## Self-Check

### Files exist:
- [x] scripts/eval_agent.py (modified)
- [x] app/agent/critique/checks.py (modified)
- [x] tests/unit/test_eval_agent.py (modified)

### Commits exist:
- [x] 134a696
- [x] 4e9415b
- [x] 67888cd
- [x] 0f64ac7
- [x] a2cfafb
- [x] c419664

## Self-Check: PASSED
