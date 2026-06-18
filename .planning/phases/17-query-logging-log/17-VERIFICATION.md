---
phase: 17-query-logging-log
verified: 2026-06-16T00:00:00Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
re_verification: null
gaps: []
deferred: []
human_verification: []
---

# Phase 17: Query Logging (LOG) Verification Report

**Phase Goal:** Query Logging (LOG) — log `/chat` user queries to Cloud SQL (Postgres) as the v2.3 adaptive loop's demand-side learning signal. Deliver (1) a net-new `user_query_log` table via a single Alembic migration chained onto head e0cd7069bc8f, and (2) a fire-and-forget `log_user_query` write path wired into the `chat()` endpoint via BackgroundTasks, with unit + integration tests and test isolation. Zero behavior change to the existing retrieval/agent loop.
**Verified:** 2026-06-16
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | D-03: A brand-new `user_query_log` table is created via a single Alembic migration whose `down_revision` is `e0cd7069bc8f`; no existing table is touched. | VERIFIED | `alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py` line 25: `down_revision: str \| Sequence[str] \| None = "e0cd7069bc8f"`. Migration only creates `user_query_log`; no ops on `place_query_hits` or any existing table. |
| 2 | D-02: The table has exactly 7 columns — id (uuid pk), message (text not null), requested_primary_types (text[]), num_stops (int null), rag_label (text), created_at (timestamptz not null default now()), session_id (text null). | VERIFIED | Migration lines 33-50: 7 `sa.Column` declarations matching the exact spec. Types confirmed: `sa.UUID`, `sa.Text(nullable=False)`, `sa.ARRAY(sa.Text)(nullable=True)`, `sa.Integer(nullable=True)`, `sa.Text(nullable=True)`, `sa.TIMESTAMP(timezone=True, nullable=False, server_default=NOW())`, `sa.Text(nullable=True)`. |
| 3 | D-04: The migration carries a comment documenting that raw message text is stored verbatim with no PII scrubbing because this is a private capstone DB. | VERIFIED | Migration docstring lines 7-10: "NOTE: raw message text is stored verbatim — no PII scrubbing — because the entire value of this table is mining real demand text. This is a private capstone database, not a public service (see Phase 17 CONTEXT D-04)." All three grep tokens match: `no PII`, `verbatim`, `private capstone`. |
| 4 | D-01: `/chat` schedules the user-query INSERT via FastAPI BackgroundTasks AFTER the ChatResponse is built, on the main graph path, off the synchronous response path. | VERIFIED | `app/main.py:806` builds `response = ChatResponse(...)`. `app/main.py:819` calls `background_tasks.add_task(log_user_query, ...)`. `app/main.py:826` executes `return response`. Order confirmed: build → schedule → return. `BackgroundTasks` imported at line 13; `log_user_query` imported at line 48. Only the main graph path logs; closure early-returns are explicitly excluded at lines 815-818 with a documented rationale. |
| 5 | D-02: `log_user_query` receives message, requested_primary_types, num_stops, and rag_label as captured handler arguments and INSERTs them into `user_query_log`. | VERIFIED | `app/main.py:783`: `num_stops = explicit_num_stops_from_conversation(...)` hoisted to local. `app/main.py:819-824`: `add_task(log_user_query, message=req.message, requested_primary_types=extracted_types, num_stops=num_stops, rag_label=rag_label)`. `app/query_log.py:42-46`: INSERT with all 5 %s params bound in order `[message, requested_primary_types, num_stops, rag_label, session_id]`. No re-derivation. |
| 6 | D-04: `log_user_query` swallows ALL exceptions (`except Exception`) and emits `logger.warning(..., exc_info=True)` so a logging failure can never affect the reply. | VERIFIED | `app/query_log.py:49`: `except Exception:`. `app/query_log.py:50`: `logger.warning("log_user_query failed; skipping", exc_info=True)`. Two fail-open unit tests confirm no propagation for both `get_conn()` raising and `cur.execute()` raising. |
| 7 | The INSERT uses psycopg2 `%s` parameters (never f-string/%-formatted SQL), and is unit-tested with FakeCursor/FakeConnection seam for both populated and empty `requested_primary_types=[]` cases. | VERIFIED | `app/query_log.py:42-46`: fixed SQL template with 5 `%s` placeholders. Param-safety test asserts message literal is absent from executed SQL. 5 tests in `tests/unit/test_query_log.py`: `test_happy_path_populated`, `test_happy_path_empty_array`, `test_param_safety_message_not_interpolated`, `test_fail_open_get_conn_raises`, `test_fail_open_cursor_execute_raises`. Empty-array test explicitly asserts `executed_params[1] == []` (not None) and `executed_params[2] is None`. |
| 8 | The new BackgroundTask never activates a real DB pool in existing `/chat` unit/functional tests via an autouse conftest fixture that suppresses `app.main.log_user_query`, with override contract documented. | VERIFIED | `tests/conftest.py:25-45`: `_neutralize_query_log` autouse fixture. `monkeypatch.setattr("app.main.log_user_query", lambda **kwargs: None)`. Comment contains "override" (grep-able contract). Documents: "MUST override this suppression" for future real-write tests. |
| 9 | A scheduling-assertion test proves the BackgroundTask is scheduled with exact D-02 kwargs; integration test provides real-DB round-trip (skipped by default under `APP_ENV=integration`). | VERIFIED | `tests/unit/test_chat_endpoint.py:94-132`: `test_chat_endpoint_schedules_query_log_with_captured_slots` installs local spy via `mocker.patch("app.main.log_user_query")` and calls `spy.assert_called_once_with(message=..., requested_primary_types=[], num_stops=None, rag_label="openai:gpt-4o-mini")`. `tests/integration/test_query_log.py`: 2 tests, gated by `pytestmark = pytest.mark.skipif(os.getenv("APP_ENV", "test") != "integration", ...)`. Both round-trip the real DB including empty-array case. Full suite: 1516 passed, 0 failures. |

**Score:** 9/9 truths verified

---

## Requirement IDs Coverage

The PLAN frontmatter declares requirements: D-01, D-02, D-03, D-04. These requirement IDs are defined in the Phase 17 CONTEXT document (not in a separate REQUIREMENTS.md — the CONTEXT explicitly notes "there is no v2.3-REQUIREMENTS.md and no formal LOG-01..05 requirement text exists anywhere; this CONTEXT's decisions are the authoritative spec for Phase 17").

| Requirement ID | Declared In | Scope | Status | Evidence |
|---------------|-------------|-------|--------|----------|
| D-01 | 17-02-PLAN.md | Fire-and-forget BackgroundTasks on main graph path; after response, off synchronous path | SATISFIED | `app/main.py:819` `background_tasks.add_task(log_user_query, ...)` placed between `ChatResponse` build (line 806) and `return response` (line 826). Early-return closure paths excluded by design with documented rationale (lines 815-818). |
| D-02 | 17-01-PLAN.md, 17-02-PLAN.md | 7-column schema; pass captured args, never re-derive | SATISFIED | Migration has exactly 7 columns matching spec. `num_stops` hoisted to local (line 783), `extracted_types` from existing computation (line 727). All 4 slots passed as kwargs to `add_task`. |
| D-03 | 17-01-PLAN.md | Brand-new table; `down_revision = e0cd7069bc8f`; no existing table modified | SATISFIED | Migration `down_revision = "e0cd7069bc8f"`. No ops on `place_query_hits`, `places_raw`, or any existing table. |
| D-04 | 17-01-PLAN.md, 17-02-PLAN.md | Raw verbatim store documented; catch-all exception; fail-open, fail-quiet | SATISFIED | Migration docstring documents raw-store decision. `except Exception` + `logger.warning(exc_info=True)` in `app/query_log.py`. |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py` | `user_query_log` DDL, 7 columns, `down_revision e0cd7069bc8f`, D-04 comment, `idx_user_query_log_created_at` index | VERIFIED | 63 lines. All acceptance criteria met. `upgrade()` creates table + index; `downgrade()` drops index then table in correct order. |
| `app/query_log.py` | `log_user_query(*, message, requested_primary_types, num_stops, rag_label, session_id=None) -> None` — sync fire-and-forget INSERT via `get_conn()`, fail-open | VERIFIED | 51 lines. Keyword-only signature, parameterised INSERT, `except Exception` catch-all, `conn.commit()`, `logger.warning(exc_info=True)`. |
| `tests/conftest.py` | Autouse fixture neutralizing `app.main.log_user_query`; override contract documented | VERIFIED | `_neutralize_query_log` autouse fixture at lines 25-45. `monkeypatch.setattr("app.main.log_user_query", lambda **kwargs: None)`. "override" keyword present in comment. |
| `tests/unit/test_query_log.py` | 5 tests: happy-path populated, empty-array, param-safety, fail-open x2 | VERIFIED | 5 `def test_*` functions. FakeCursor/FakeConnection seam with `commit_called` flag. All 5 acceptance criteria tests present and structured as specified. |
| `tests/integration/test_query_log.py` | Real-DB INSERT round-trip; `APP_ENV=integration` gate; skipped by default | VERIFIED | `pytestmark = pytest.mark.skipif(...)` at line 20. 2 test methods: `test_populated_round_trip` and `test_empty_array_round_trip`. Both include cleanup `finally` blocks. |
| `app/main.py` (modified) | `BackgroundTasks` import, `log_user_query` import, `background_tasks` param in `chat()`, `add_task` call on main path | VERIFIED | Line 13: `from fastapi import BackgroundTasks, ...`. Line 48: `from .query_log import log_user_query`. Line 665: `background_tasks: BackgroundTasks` in signature. Line 819: `background_tasks.add_task(log_user_query, ...)`. |
| `tests/unit/test_chat_endpoint.py` (modified) | Scheduling-assertion test with `spy.assert_called_once_with` for exact D-02 kwargs | VERIFIED | `test_chat_endpoint_schedules_query_log_with_captured_slots` at line 94. `mocker.patch("app.main.log_user_query")` spy. `assert_called_once_with(message=..., requested_primary_types=[], num_stops=None, rag_label="openai:gpt-4o-mini")`. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `alembic/versions/*_add_user_query_log.py` | alembic head `e0cd7069bc8f` | `down_revision` linkage | WIRED | `down_revision: str \| Sequence[str] \| None = "e0cd7069bc8f"` at line 25. Confirmed against the head migration `2026_05_14_1200-e0cd7069bc8f_add_place_relations.py`. |
| `app/main.py chat()` | `app.query_log.log_user_query` | `background_tasks.add_task(...)` | WIRED | `app/main.py:48`: import present. `app/main.py:819-825`: `background_tasks.add_task(log_user_query, message=req.message, requested_primary_types=extracted_types, num_stops=num_stops, rag_label=rag_label)`. |
| `app/query_log.py log_user_query` | `user_query_log` table | `INSERT INTO user_query_log` via `get_conn() + conn.commit()` | WIRED | `app/query_log.py:39-48`: `with get_conn() as conn, conn.cursor() as cur: cur.execute("INSERT INTO user_query_log ...", [...])` then `conn.commit()`. |

---

## Data-Flow Trace (Level 4)

`app/query_log.py` is a write-only helper (no render); Level 4 data-flow trace is applied to the write path.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `app/query_log.py` | `message`, `requested_primary_types`, `num_stops`, `rag_label`, `session_id` | Passed as captured kwargs from `chat()` — `req.message` (user input), `extracted_types` (computed by slot-intake LLM or empty list), `num_stops` (hoisted local from `explicit_num_stops_from_conversation`), `rag_label` (from `request.app.state`). | Yes — real user inputs, no hardcoded values, no static fallback that substitutes for a fetch. | FLOWING |
| `INSERT INTO user_query_log` | Params `[message, requested_primary_types, num_stops, rag_label, session_id]` | All 5 params from caller-supplied arguments, none derived inside `log_user_query`. `%s` placeholders prove values flow through as bound params. | Yes — parameterised SQL binds live runtime values. | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `log_user_query` exports correct symbol | `python -c "from app.query_log import log_user_query; print(type(log_user_query))"` | Importable (conftest + unit tests rely on this; 5 unit tests pass) | PASS |
| Migration down_revision linkage | `grep "down_revision.*e0cd7069bc8f" alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py` | Match confirmed | PASS |
| Migration has 7 columns | `grep -c "sa.Column" ...d1be72aea7d4_add_user_query_log.py` | 7 | PASS |
| 5 unit tests exist and pass | `pytest tests/unit/test_query_log.py` | 5 passed (per SUMMARY.md; full suite 1516 passed 0 failures) | PASS |
| BackgroundTasks wiring order | Lines 806 (response built) → 819 (add_task) → 826 (return) | Confirmed via static grep | PASS |
| No TBD/FIXME/XXX debt markers in phase-modified files | grep scan of all 5 new/modified source files | 0 matches — "TODO" false-positives in comments about `%s` placeholders are documentation, not debt markers | PASS |

---

## Probe Execution

No probe scripts declared in PLAN.md for this phase. Phase 17 is not a migration/tooling phase in the probe-script sense; the SUMMARY.md documents live migration verification that requires a running Docker DB (not re-runnable statically). Step 7c: SKIPPED (no declared probes; migration round-trip verified dynamically by executor against local Docker DB and confirmed in 17-01-SUMMARY.md).

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No TBD, FIXME, or XXX markers in any phase-modified file. No stubs, placeholders, or empty returns in production paths. No hardcoded data substituting for real values. No `return null` / `return {}` / `return []` in live write path.

---

## Human Verification Required

None. All must-haves are verifiable statically or through the test suite. The full unit suite (1516 passed) and integration tests (2 passed under `APP_ENV=integration`) cover the functional behavior. No visual/UI/real-time/external-service behavior requiring human observation.

---

## Gaps Summary

No gaps. All 9 must-haves verified. All 4 requirement IDs (D-01, D-02, D-03, D-04) satisfied. All 7 required artifacts exist and are substantive and wired. Key links verified. Data flows through the write path with real runtime values. 5 commits confirmed in git log (6e5d7d5, d2bae40, f2ac698, 10f2254, 14d4756).

**Code review observations (from 17-REVIEW.md) noted for awareness — none block the goal:**

- WR-01: Early-return paths skip logging with no guarding test. Advisory quality gap; D-01 explicitly sanctions this design choice. Not a goal failure.
- WR-02: Failed-graph requests silently drop the log (exception before `add_task`). Undocumented and untested, but plausibly acceptable per D-01 intent. Advisory only.
- WR-03: `session_id` column always NULL in prod (D-02 sanctions nullable; `trace_id` is available but not threaded through). Advisory improvement; not a correctness defect.

These are advisory observations documented in 17-REVIEW.md, not goal-blocking gaps.

---

_Verified: 2026-06-16_
_Verifier: Claude (gsd-verifier)_
