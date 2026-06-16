---
phase: 17-query-logging-log
plan: "01"
subsystem: database-schema
tags: [alembic, migration, user_query_log, adaptive-data-loop, v2.3]
dependency_graph:
  requires: [e0cd7069bc8f migration (add_place_relations)]
  provides: [user_query_log table + idx_user_query_log_created_at index]
  affects: [Phase 18 GAP miner (downstream consumer of this table)]
tech_stack:
  added: []
  patterns: [alembic op.create_table/op.create_index, uuid_generate_v4() pk, TIMESTAMP(timezone=True) + NOW() default]
key_files:
  created:
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py
  modified: []
decisions:
  - D-02: 7-column schema exactly as spec — id (uuid pk), message (text not null), requested_primary_types (text[]), num_stops (int null), rag_label (text), created_at (timestamptz not null default now()), session_id (text null)
  - D-03: Brand-new user_query_log table; down_revision = e0cd7069bc8f; no existing table modified
  - D-04: Raw message stored verbatim, no PII scrubbing; documented in migration docstring
  - Scaffolded via plain alembic revision -m (non-autogenerate) to avoid live-DB dependency
metrics:
  duration: "117s (~2min)"
  completed: "2026-06-16"
  tasks_completed: 2
  files_changed: 1
---

# Phase 17 Plan 01: user_query_log Migration Summary

**One-liner:** Alembic migration creating `user_query_log` (7-column demand-signal table, uuid pk + created_at index) chained to head `e0cd7069bc8f`, with D-04 raw-store comment.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Generate and author the user_query_log migration | 6e5d7d5 | alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py |
| 2 | Apply the migration and verify the live schema round-trips | (no source changes; migration applied to local Docker DB) | — |

## Verification Results

- `make migrate` (via `DATABASE_URL=...city_concierge poetry run alembic upgrade head`) exits 0
- `information_schema.columns` for `user_query_log` returns exactly 7 columns: `created_at`, `id`, `message`, `num_stops`, `rag_label`, `requested_primary_types`, `session_id`
- `pg_indexes` contains `idx_user_query_log_created_at`
- `alembic downgrade -1` exits 0; `to_regclass('public.user_query_log')` returns NULL (table gone)
- `alembic upgrade head` re-applies cleanly; 7 columns + index confirmed again
- `make test-unit` passes: 1510 passed, 9 skipped (no regressions)

## Note on DATABASE_URL

The `.env` file has `DATABASE_URL=postgresql://...mlops-city-concierge` which does not match the running Docker DB name (`city_concierge`). The migration and verification were run with the explicit correct URL (`postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge`). This is a pre-existing `.env` drift — out of scope for this plan.

## Deviations from Plan

None — plan executed exactly as written. The plain `alembic revision -m` (non-autogenerate) path was used as specified. The scaffold correctly set `down_revision = "e0cd7069bc8f"`. All 7 D-02 columns + index authored per PATTERNS.md template.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary crossings introduced. The migration DDL is executed by a trusted operator job. The `user_query_log` table's write path (plan 17-02) is deferred and carries its own threat analysis.

## Self-Check: PASSED

- [x] `alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py` exists
- [x] Commit `6e5d7d5` exists and contains the migration
- [x] `down_revision = "e0cd7069bc8f"` confirmed in file
- [x] 7 D-02 columns confirmed via `information_schema.columns`
- [x] `idx_user_query_log_created_at` confirmed via `pg_indexes`
- [x] Round-trip (down + re-up) confirmed
- [x] Unit test suite green (1510 passed)
