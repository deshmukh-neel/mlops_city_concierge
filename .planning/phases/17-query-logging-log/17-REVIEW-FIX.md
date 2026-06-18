---
phase: 17-query-logging-log
fixed_at: 2026-06-16T00:00:00Z
review_path: .planning/phases/17-query-logging-log/17-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 17: Code Review Fix Report

**Fixed at:** 2026-06-16
**Source review:** .planning/phases/17-query-logging-log/17-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3 (WR-01, WR-02, WR-03 — `critical_warning` scope; the 4 Info items IN-01..04 were out of scope and intentionally not touched)
- Fixed: 3
- Skipped: 0

All three Warning findings interact around the `add_task(log_user_query, ...)`
scheduling site in `app/main.py`. Verified ground truth before fixing: the
`with trace_request("chat", ...) as trace_id:` block opens at `app/main.py:701`
and closes before `final_state = ...`; in Python a `with` block does not create
a new scope, so `trace_id` remains a live, always-bound local at the `add_task`
call. This made WR-03 a zero-risk in-place edit and confirmed WR-02's premise
(the exception path propagates out of the `with` block before `add_task`).

Verification: `tests/unit/test_chat_endpoint.py` + `tests/unit/test_query_log.py`
ran GREEN — **26 passed, 0 failed**. (The integration suite
`tests/integration/test_query_log.py` is skipped without `APP_ENV=integration`;
that is expected, not a failure.)

## Fixed Issues

### WR-03: `session_id` column is always NULL in production despite `trace_id` being trivially in scope

**Files modified:** `app/main.py`, `tests/unit/test_chat_endpoint.py`
**Commit:** 50beb02
**Applied fix:** Threaded the already-in-scope `trace_id` into the existing
`add_task(log_user_query, ...)` call as `session_id=trace_id` (with an inline
`# cheap per-turn correlation marker (17-CONTEXT D-02)` comment). Chose the
conservative in-place option — `add_task` was NOT moved and scheduling was NOT
reordered, because `trace_id` is already live after the `with` block. This turns
an always-NULL column into a per-turn correlation key at zero cost.

The pre-existing `test_chat_endpoint_schedules_query_log_with_captured_slots`
asserted the exact kwargs of the scheduled call, so adding the `session_id=`
kwarg would have broken it. Inspected `app/observability/__init__.py`:
`trace_request` yields `trace.id` (non-deterministic) when a Langfuse client
exists, else `None`. In the unit-test env `LANGFUSE_SECRET_KEY` is unset, so the
value is `None`, but to be robust against a real client the assertion was updated
to `session_id=mocker.ANY` (pins the kwarg's presence, not a brittle value).
This production+test change is committed atomically because the production edit
breaks that test by construction.

### WR-02: Failed-graph requests are never logged; behavior is undocumented and untested

**Files modified:** `app/main.py`
**Commit:** 69e8afb
**Applied fix:** Applied the reviewer's RECOMMENDED option (a) — DOCUMENT, do not
change scheduling. Added a sentence to the existing comment block above
`add_task` explaining that requests where `graph.ainvoke(...)` raises are
intentionally NOT logged: the exception propagates out of the
`with trace_request(...)` block before `add_task` is reached, and the demand
signal deliberately favors successfully-planned queries (D-01/D-02 intent).
Scheduling order was NOT changed (option (b) was explicitly not chosen, as it
would alter D-01/D-02 intent). Comment-only change.

### WR-01: Early-return accept/decline paths skip logging with no test guarding it

**Files modified:** `tests/unit/test_chat_endpoint.py`
**Commit:** 73d230f
**Applied fix:** Purely additive test work — no production logic changed. Added
two endpoint tests, `test_chat_endpoint_accept_path_does_not_log` and
`test_chat_endpoint_decline_path_does_not_log`, that install a real
`mocker.patch("app.main.log_user_query")` spy (overriding the autouse
`_neutralize_query_log` no-op, the same pattern as
`test_chat_endpoint_schedules_query_log_with_captured_slots`) and assert
`spy.assert_not_called()` on both early-return paths. The accept-path test reuses
the full `get_details` / `_place_is_open_now` / `_per_stop_closure_status` /
`_bounded_retime_after_swap` / `enrich_stops_with_booking` mock setup and the
shared `_pending_state()` fixture so the accept branch genuinely early-returns
rather than escalating to the graph. These tests now pin the documented contract:
a future refactor that hoists `add_task` above the early-return branches would
turn the suite RED instead of silently logging closure replies.

---

_Fixed: 2026-06-16_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
