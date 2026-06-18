---
phase: 17-query-logging-log
plan: "02"
subsystem: query-logging
tags: [fastapi, background-tasks, psycopg2, log-user-query, adaptive-data-loop, v2.3]
dependency_graph:
  requires: [user_query_log table (17-01, revision d1be72aea7d4)]
  provides: [log_user_query helper, BackgroundTasks wiring in chat(), test isolation fixture]
  affects: [Phase 18 GAP miner (downstream consumer of user_query_log), /chat endpoint latency (zero — fire-and-forget)]
tech_stack:
  added: [FastAPI BackgroundTasks (first use in app/)]
  patterns: [sync psycopg2 INSERT via get_conn(), except Exception fail-open, autouse conftest isolation fixture, TDD RED/GREEN cycle]
key_files:
  created:
    - app/query_log.py
    - tests/unit/test_query_log.py
    - tests/integration/test_query_log.py
  modified:
    - app/main.py
    - tests/conftest.py
    - tests/unit/test_chat_endpoint.py
decisions:
  - D-01: BackgroundTasks fire-and-forget after ChatResponse built; only main graph path logged; closure early-returns excluded (comment in main.py)
  - D-02: Captured slots passed as args — message, requested_primary_types (extracted_types), num_stops (hoisted local), rag_label; never re-derived
  - D-04: except Exception catch-all + logger.warning(exc_info=True); never re-raises; runs in BackgroundTasks threadpool, cannot block reply
  - test-isolation: autouse _neutralize_query_log in conftest patches app.main.log_user_query to no-op lambda; override contract documented with grep-able keyword
metrics:
  duration: "390s (~6.5min)"
  completed: "2026-06-16"
  tasks_completed: 3
  files_changed: 6
---

# Phase 17 Plan 02: Query Log Write Path Summary

**One-liner:** Fire-and-forget `log_user_query` sync INSERT via FastAPI BackgroundTasks, wired into `chat()` with fail-open (catch Exception + logger.warning), isolated from existing /chat tests via autouse conftest fixture, unit-tested with FakeCursor seam (5 tests), and integration-verified with real-DB round-trip including the empty-array case.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing unit tests for log_user_query | d2bae40 | tests/unit/test_query_log.py |
| 1 (GREEN) | Implement log_user_query fail-open sync INSERT helper | f2ac698 | app/query_log.py |
| 2 | Wire BackgroundTasks + isolate existing /chat tests | 10f2254 | app/main.py, tests/conftest.py |
| 3 | Scheduling-assertion spy + integration round-trip | 14d4756 | tests/unit/test_chat_endpoint.py, tests/integration/test_query_log.py |

## Verification Results

- `pytest tests/unit/test_query_log.py`: 5 passed (populated, empty-array, param-safety, fail-open x2)
- `app/query_log.py`: `grep 'INSERT INTO user_query_log'`, `grep 'except Exception'`, `grep 'exc_info=True'` all match
- AST check: `background_tasks` in `chat()` params; `BackgroundTasks` and `log_user_query` in `app/main.py`
- `tests/conftest.py`: autouse `_neutralize_query_log` contains `log_user_query` and `override` keyword
- `pytest tests/unit/test_chat_endpoint.py tests/unit/test_chat_functional.py`: 37 passed (pre-wiring baseline), still 37 passed post-wiring
- `test_chat_endpoint_schedules_query_log_with_captured_slots`: spy `assert_called_once_with` exact D-02 kwargs passes
- `APP_ENV=integration` integration tests: 2 passed (populated round-trip + empty-array round-trip, rows cleaned up)
- `make test` (FULL suite): **1516 passed, 59 skipped, 0 failures**; `app/query_log.py` coverage 100%

## Deviations from Plan

None — plan executed exactly as written.

- TDD RED/GREEN cycle followed: test committed first (ImportError failure confirmed), then implementation.
- `num_stops` hoisted to local before `ItineraryState` construction with a comment documenting D-02 rationale.
- `background_tasks.add_task(...)` placed after `ChatResponse` built, before `return response` — exactly as specified.
- Empty-array integration test asserts `row[0] == []` (not None) — proves psycopg2 adapts empty Python list as empty Postgres text[].

## Known Stubs

None — all data flows are live. The INSERT is a real parameterised SQL call; no hardcoded values.

## Threat Flags

None — no new network endpoints or auth paths beyond what the plan's threat model covers. T-17-04 (SQL injection) mitigated via `%s` bound parameters (param-safety unit test asserts message literal absent from executed SQL string). T-17-05 (DB failure blocking /chat) mitigated via BackgroundTasks fire-and-forget + except Exception swallow.

## Self-Check: PASSED

- [x] `app/query_log.py` exists and exports `log_user_query`
- [x] `tests/unit/test_query_log.py` exists with 5 tests
- [x] `tests/integration/test_query_log.py` exists with skipif gate
- [x] `tests/conftest.py` has `_neutralize_query_log` autouse fixture
- [x] Commits d2bae40, f2ac698, 10f2254, 14d4756 all confirmed in git log
- [x] `make test` (full suite): 1516 passed, 0 failures
- [x] Integration tests: 2 passed under APP_ENV=integration
