---
phase: 17
reviewers: [codex]
reviewed_at: 2026-06-16T07:48:53Z
plans_reviewed: [17-01-PLAN.md, 17-02-PLAN.md]
model: codex-cli 0.135.0 (default model)
---

# Cross-AI Plan Review — Phase 17 (Query Logging / LOG)

## Codex Review

**Summary**

The plans are generally solid and well scoped: 17-01 cleanly creates the demand-side table, and 17-02 composes with the existing sync DB layer without changing `/chat` contracts. No HIGH blocker, but there are MEDIUM risks around FastAPI `BackgroundTasks` semantics, unit-test DB side effects, and the "log every query" wording versus the planned main-path-only logging.

**Strengths**

- Migration scope is tight: new `user_query_log` only, no `place_query_hits`, `coverage_agent.py`, or ingest changes.
- Schema matches D-02, and local head is indeed `e0cd7069bc8f`.
- Captured-slot plan is viable: `rag_label` is available at `app/main.py:668`, `extracted_types` at `app/main.py:724`, and a hoisted `num_stops` will remain in function scope after the `with trace_request(...)` block.
- SQL injection posture is correct: fixed SQL plus psycopg2 `%s` parameters for raw `message`.
- Test layering is appropriate: mocked unit tests plus an `APP_ENV=integration` real-DB round trip, then full `make test`.

**Concerns**

- **MEDIUM:** The "0ms latency" claim is overstated. FastAPI/Starlette background tasks run after the response body is sent, but the ASGI request coroutine still waits for them, and `TestClient` usually waits for background tasks before `client.post(...)` returns. This is off the response construction path, but not cost-free for worker/threadpool occupancy.
- **MEDIUM:** Existing `/chat` unit/functional tests will execute the new background task unless patched. The test env sets `DATABASE_URL` to localhost:5432 in `tests/conftest.py:34`, and `get_conn()` lazily initializes a real pool via `app/db_pool.py:93`. Even fail-open logging can create warning noise, slow tests, or trigger the known DB-pool contamination class.
- **MEDIUM:** 17-02 intentionally logs only the main graph path, but the phase objective says "log every real `/chat` USER query." Accept/decline returns at `app/main.py:689` and `app/main.py:695` are excluded. That may be the right product decision, but the plan should reconcile the wording.
- **LOW:** Add coverage for `requested_primary_types=[]`. Most free-text requests leave `extracted_types` empty, so the write helper should be proven against the common empty-array case, not only `["restaurant"]`.
- **LOW:** `make migration` uses `alembic revision --autogenerate` per `Makefile:51`, which requires a reachable, up-to-date DB. The plan should explicitly run `make db-up && make migrate` before generation, or use plain `alembic revision -m` if no autogenerate diff is needed.

**Suggestions**

- In 17-02 Task 2, add a targeted `/chat` test that patches `app.main.log_user_query` and asserts the exact kwargs: `message`, `requested_primary_types`, `num_stops`, `rag_label`.
- Patch `app.main.log_user_query` in existing `/chat` unit tests to avoid real DB pool activation.
- Change wording from "0ms added latency" to "scheduled after response construction/body send; not on the synchronous response contract."
- Either log accept/decline with `requested_primary_types=[]` and `num_stops=None`, or explicitly state that Phase 17 logs "main-path demand queries," not every `/chat` turn.
- Add downgrade/upgrade round-trip to the automated 17-01 verification command, not only acceptance text.

**Risk Assessment**

Overall risk: **MEDIUM**. The table and write helper are straightforward and low-risk, and the boundary guard is good. The main risk is operational/test behavior from introducing `BackgroundTasks` into a test suite that is already sensitive to accidental DB pool activation. Fixing that in the plan should make execution low risk.

---

## Consensus Summary

Single reviewer (codex) — no cross-reviewer consensus to compute. Codex independently verified the plans' factual claims against the live codebase (`e0cd7069bc8f` head, `app/main.py:668/724`, `tests/conftest.py:34`, `app/db_pool.py:93`, `Makefile:51`) and confirmed them accurate.

### Agreed Strengths

- Tight migration scope; boundary guard holds (no `place_query_hits` / `coverage_agent.py` / ingest changes).
- Schema matches D-02; `down_revision` correct; parameterized INSERT (no SQL-injection surface).
- Captured-slot hoisting is sound (variables survive the `with trace_request(...)` block).

### Agreed Concerns (priority order)

1. **[MEDIUM] Existing `/chat` tests will trigger the new background task → real DB pool activation.** This is the highest-value finding: it intersects the project's known full-suite DB-pool-contamination gotcha. The fix is cheap and concrete — patch `app.main.log_user_query` in `tests/conftest.py` / the existing `/chat` unit+functional tests so the background task never activates a real pool. This should likely become a task in 17-02 rather than the plan's current "loose, self-healing" stance.
2. **[MEDIUM] "0ms latency" wording is technically overstated.** Background tasks run after body send but still occupy the worker/threadpool; reword to "off the synchronous response contract" rather than literally 0ms. Documentation-only fix.
3. **[MEDIUM] "log every query" goal vs. main-path-only implementation.** The planner's discretion call (main-path-only, to avoid re-deriving slots for low-signal accept/decline replies) is defensible, but the plan + must_haves wording should explicitly say "main-path demand queries," not "every turn," so verify-phase doesn't flag a false gap. Reconcile the wording.
4. **[LOW] Empty-array test case.** Add a unit test with `requested_primary_types=[]` (the common free-text case), not only the populated `["restaurant"]` case.
5. **[LOW] Migration generation prerequisite.** `make migration` autogenerates against a live DB — the plan should run `make db-up && make migrate` first, or use plain `alembic revision -m` since this is a hand-written new-table migration with no autogenerate diff needed.

### Divergent Views

None (single reviewer).

### Disposition Note

All findings are MEDIUM/LOW — none block execution. Concerns 1–3 are the meaningful ones and are all cheaply addressable by replanning. Recommended: incorporate via `/gsd-plan-phase 17 --reviews` (folds the test-isolation task + wording reconciliation + empty-array test + migration-prereq into the plans), then execute.
