---
phase: 17-query-logging-log
reviewed: 2026-06-16T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py
  - app/main.py
  - app/query_log.py
  - tests/conftest.py
  - tests/integration/test_query_log.py
  - tests/unit/test_chat_endpoint.py
  - tests/unit/test_query_log.py
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---

# Phase 17: Code Review Report

**Reviewed:** 2026-06-16
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 17 adds a fire-and-forget demand-query write path: a new `user_query_log`
Alembic migration, a fail-open `app/query_log.log_user_query` helper, and a
`BackgroundTasks.add_task(...)` call wired into the `/chat` endpoint after the
response is built. The implementation is small, focused, and well-tested for the
slice it claims (unit + integration + endpoint scheduling + fail-open coverage),
and it honors the documented design decisions (D-01 fire-and-forget, D-02
pass-captured-args, D-04 raw-verbatim/fail-open).

I found **no Critical (BLOCKER) issues** — the SQL is correctly parameterized,
the `uuid_generate_v4()` default is backed by the `uuid-ossp` extension that
`init.sql` provisions in every environment (consistent with existing tables at
`init.sql:17,66`), the migration chain (`e0cd7069bc8f` -> `d1be72aea7d4`) is
intact, and the fail-open posture is genuinely fail-open and fail-quiet.

The findings below are **3 Warnings** and **4 Info** items. The two highest-value
Warnings are (1) the early-return accept/decline paths bypass logging with *no
test guarding that intended behavior*, so a future refactor that hoists
`add_task` earlier would silently start logging closure replies undetected, and
(2) failed-graph requests are silently never logged, which is plausibly fine but
is undocumented and untested. Neither blocks ship.

## Warnings

### WR-01: Early-return accept/decline paths skip logging with no test guarding it

**File:** `app/main.py:691-698`, `app/main.py:819-825`
**Issue:** The design comment at `app/main.py:815-818` asserts that the
`_try_accept_path` and `_decline_path` early-returns are intentionally NOT
logged. This is a real behavioral contract, but there is no test that pins it.
The accept/decline endpoint tests
(`test_chat_endpoint_accept_path_inserts_proposed_alternative`,
`test_chat_endpoint_decline_path_drops_closed_stop`) run under the autouse
`_neutralize_query_log` no-op (`tests/conftest.py:45`), so they cannot detect a
regression: if someone later moves `background_tasks.add_task(...)` above the
early-return branches — a natural refactor — closure replies would silently
start being logged with `extracted_types`/`num_stops` undefined (NameError) or
stale, and the suite would stay green. The fail-open swallow in
`log_user_query` would even hide the resulting NameError-equivalent at runtime.
**Fix:** Add an endpoint test that installs a real spy (override the autouse
no-op, same pattern as `test_chat_endpoint_schedules_query_log_with_captured_slots`)
and asserts `spy.assert_not_called()` for both the accept and decline early-return
paths:
```python
def test_chat_decline_path_does_not_log(mocker):
    spy = mocker.patch("app.main.log_user_query")
    # ... post a decline message with a pending conversation_state ...
    assert response.status_code == 200
    spy.assert_not_called()
```

### WR-02: Failed-graph requests are never logged; behavior is undocumented and untested

**File:** `app/main.py:797-825`
**Issue:** `background_tasks.add_task(...)` (line 819) sits *after* the
`with trace_request(...)` block. If `graph.ainvoke(...)` (line 797) raises, the
exception propagates out of `chat()` before `add_task` is reached, so the demand
query is silently dropped. For a demand-signal table this may be acceptable (you
generally want successful plans), but it is a non-obvious gap: the docstring at
lines 815-818 carefully explains why the *early-return* paths aren't logged but
says nothing about the *exception* path, and no test exercises it. Phase 18's
miner will therefore systematically under-count queries that triggered planner
failures — exactly the kind of under-served/hard demand the loop is meant to
surface.
**Fix:** Either (a) document the exception-path drop explicitly in the comment
block alongside the early-return rationale, or (b) if failed queries are
considered valuable demand signal, schedule the log before `graph.ainvoke` (the
captured args `req.message`, `extracted_types`, `num_stops`, `rag_label` are all
available by line 796) — the fail-open helper already guarantees it can't break
the request. Recommend (a) for this slice, with a one-line decision note, since
changing scheduling order touches D-01/D-02 intent.

### WR-03: `session_id` column is always NULL in production despite `trace_id` being trivially in scope

**File:** `app/main.py:819-825`, `app/query_log.py:27`, `alembic/...d1be72aea7d4.py:49`
**Issue:** The `session_id` column exists in the schema, in the helper signature,
and in both integration tests, but the production `/chat` path never populates it
— `add_task` omits it and the helper defaults to `None`, so every prod row will
have `session_id IS NULL`. D-02 explicitly sanctions leaving it null, so this is
not a defect, but the same note (`17-CONTEXT.md:86-88`) calls out that the
existing `trace_id` is an available cheap per-turn marker — and `trace_id` is
literally in scope at line 819 (bound by `with trace_request(...) as trace_id`).
Threading a column through migration + helper + two integration tests while
leaving it permanently empty in prod is wasted surface: it reads as "wired" but
carries zero signal, which can mislead the Phase 18 miner author into trusting a
column that is always NULL.
**Fix:** Pass the marker that's already in hand, turning an always-NULL column
into a per-turn correlation key at zero cost:
```python
background_tasks.add_task(
    log_user_query,
    message=req.message,
    requested_primary_types=extracted_types,
    num_stops=num_stops,
    rag_label=rag_label,
    session_id=trace_id,  # cheap per-turn marker (17-CONTEXT D-02)
)
```
(Requires moving `add_task` inside the `with` block, or hoisting `trace_id` to an
outer local — note this interacts with WR-02's scheduling decision.) If the
column is intentionally deferred to Phase 18, add a code comment saying so, so
the next reader doesn't assume it's populated.

## Info

### IN-01: `requested_primary_types` is declared nullable but the only writer never writes NULL

**File:** `alembic/...d1be72aea7d4.py:40`, `app/query_log.py:24`
**Issue:** The migration declares `requested_primary_types` as `nullable=True`,
but the sole writer (`log_user_query`, and its only caller in `chat()`) always
passes a `list[str]` — `[]` for free-text, never `None`. The integration test
`test_empty_array_round_trip` confirms `[]` reads back as `[]` (not NULL). So the
NULL state is unreachable in practice, which means Phase 18 queries must defend
against a `NULL` that the app never produces (or, worse, won't defend against an
empty-array case they didn't expect). The helper's type hint
`requested_primary_types: list[str]` (non-optional) actually disagrees with the
nullable column.
**Fix:** Either tighten the column to `nullable=False, server_default='{}'` to
match the contract the code actually enforces, or, if a future non-`/chat` writer
might pass `None`, relax the helper hint to `list[str] | None`. Low priority —
purely a contract-clarity issue.

### IN-02: `_neutralize_query_log` fixture signature can't catch a positional call

**File:** `tests/conftest.py:45`
**Issue:** The autouse no-op is `lambda **kwargs: None`. It only accepts keyword
arguments. The production call site uses kwargs exclusively (`app/main.py:820-824`),
so this works today, but it silently couples the test harness to a call
convention. If `log_user_query` is ever invoked positionally anywhere, the no-op
raises `TypeError` and every test in the suite fails confusingly rather than
flagging the real change.
**Fix:** Use `lambda *args, **kwargs: None` for robustness against call-convention
changes. Trivial.

### IN-03: Integration cleanup `DELETE` is not wrapped to survive a failed connection

**File:** `tests/integration/test_query_log.py:62-69`, `103-109`
**Issue:** The `finally` cleanup blocks open a fresh `get_conn()` and `DELETE` by
`session_marker`. If the *body* failed because the DB/pool was unhealthy, the
cleanup `get_conn()` will also raise, masking the original assertion failure with
a connection error in the `finally`. Minor (integration-only, gated behind
`APP_ENV=integration`), but it can make a real failure harder to diagnose.
**Fix:** Wrap the cleanup in a best-effort `try/except` that logs and swallows,
so the original assertion error surfaces:
```python
finally:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM user_query_log WHERE session_id = %s", [session_marker])
            conn.commit()
    except Exception:
        pass  # don't mask the body's failure
```

### IN-04: Module docstring claims "0ms to user-perceived latency" — slightly overstated

**File:** `app/query_log.py:4-5`
**Issue:** The docstring states the background INSERT "adds 0ms to user-perceived
latency." Under Starlette/FastAPI, `BackgroundTasks` registered on a normal
(non-streaming) response run *after the response is sent*, so this is effectively
true for the client. The phrasing is fine as intent documentation; flagging only
because "0ms" is an absolute claim that could mislead a future reader into
thinking the work is free of all server-side cost (it still consumes a threadpool
slot and a pooled DB connection). No code change needed; consider softening to
"off the synchronous response path" to match the more accurate comment at
`app/main.py:813-814`.

---

_Reviewed: 2026-06-16_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
