---
phase: 18-gap-mining-gap
plan: "02"
subsystem: coverage_agent / demand extraction
tags: [gap-mining, demand-extraction, rag, pgvector, tdd, lexical, llm-batch]
dependency_graph:
  requires: [18-01]
  provides: [gather_demand, get_demand_conn, _types_to_cuisines, _lexical_cuisines, _lexical_neighborhoods, _extract_demand_batch, _build_demand_batch_prompt]
  affects: [scripts/coverage_agent.py, tests/unit/test_gap_miner.py]
tech_stack:
  added: [psycopg2 direct non-pooled connection, contextlib.closing pattern]
  patterns: [two-tier lexical+LLM extraction, symmetric cuisine/neighborhood extraction, cartesian demand accumulation, single combined batched LLM call]
key_files:
  created: [tests/unit/test_gap_miner.py]
  modified: [scripts/coverage_agent.py]
decisions:
  - "Cuisine extraction symmetric to neighborhood (lexical type-map → lexical message scan → combined batched LLM). Closes ROUND-3 HIGH: app/main.py returns requested_primary_types=[] for free-text, so cuisine must be recovered from message itself."
  - "Single combined batched LLM call for ALL lexical-miss rows on EITHER axis — no per-row, no per-axis second round-trip (D-01)."
  - "get_demand_conn uses contextlib.closing + set_session(readonly=True) — mirrors loop_falsifier._snapshot_ids_from_url pattern; never touches pool."
  - "Unmapped rows counted honestly; off-catalog LLM answers dropped by catalog-membership filter on both axes so premark_seed_isolation can never see garbage."
metrics:
  duration_seconds: 287
  completed: 2026-06-18
  tasks_completed: 2
  files_modified: 2
  tests_added: 39
---

# Phase 18 Plan 02: Demand Extraction Summary

Symmetric two-tier (lexical→batched-LLM) extraction on BOTH cuisine and neighborhood axes; `gather_demand` + `get_demand_conn` wired into `coverage_agent.py` additively with full regression preservation.

## What Was Built

### Task 1: Demand-extraction helpers

Five new functions added to `scripts/coverage_agent.py`:

- `_types_to_cuisines(primary_types)` — tier-1 cuisine extraction: strips " restaurant" suffix, lowercases, checks `_CUISINES_SET` membership. No LLM.
- `_lexical_cuisines(message)` — tier-2a cuisine extraction: case-insensitive scan of message text against all CUISINES catalog members. No LLM. Symmetric to `_lexical_neighborhoods`. Closes ROUND-3 HIGH (app/main.py returns `requested_primary_types=[]` for free-text).
- `_lexical_neighborhoods(message)` — tier-1 neighborhood extraction: case-insensitive scan of message text against all NEIGHBORHOODS catalog members. No LLM. Returns a list (multi-neighborhood messages return multiple hits).
- `_build_demand_batch_prompt(messages)` — builds the LLM extraction prompt with messages embedded via `json.dumps` (not raw f-string interpolation) to prevent prompt-format corruption (T-18-02-INJ).
- `_extract_demand_batch(messages, llm)` — single combined LLM call returning `(neighborhoods, cuisines)` per message; filters both axes to catalog membership; tolerates ```json fences via `_FENCE_RE`; returns all-empty pairs when `llm is None`.

Two module-level frozensets built on import: `_CUISINES_SET`, `_NEIGHBORHOODS_SET`.

### Task 2: gather_demand + get_demand_conn

- `get_demand_conn(url)` — `@contextmanager` using `contextlib.closing(psycopg2.connect(url))` with `set_session(readonly=True, autocommit=True)`. Direct non-pooled connection that never touches the shared pool's `DATABASE_URL` or its lru_cache settings. Mirrors `loop_falsifier._snapshot_ids_from_url`.
- `gather_demand(days, url=None)` — reads `user_query_log WHERE created_at >= %s` (parameterised cutoff, never interpolated). Applies the symmetric two-tier:
  - Cuisine: `_types_to_cuisines` → `_lexical_cuisines` → combined-batched LLM
  - Neighborhood: `_lexical_neighborhoods` → combined-batched LLM
  - One single `_extract_demand_batch` call for ALL lexical-miss rows (D-01 — no per-row/per-axis round-trip)
  - Accumulates `demand_counts[(neighborhood, cuisine)]` with cartesian product within each row (multi-intent messages count every pair)
  - Returns `(demand_counts, rows_scanned, unmapped_count)`

## Test Coverage

`tests/unit/test_gap_miner.py` — 39 new tests, all green. Covers all 8 behaviors from Task 1 and all 10 behaviors from Task 2, including:
- Types-to-cuisines mapping (incl. bars → no cuisine)
- Lexical cuisine message fallback (ROUND-3 HIGH)
- Lexical neighborhood multi-hit
- Single combined batch call — exactly 1 `llm.invoke` for multi-message misses
- Catalog constraint on both axes (off-catalog LLM output dropped)
- Fence-tolerant JSON parsing
- Prompt-injection guard (messages json.dumps-encoded, not raw)
- Judge-None graceful: lexical hits on both axes still map, LLM-needed rows degrade to empty
- Parameterised cutoff (SQLi guard asserted)
- Pool vs demand-conn routing
- Multi-intent cartesian: "italian in Mission and North Beach" → 2 demand pairs
- Free-text row with empty types maps via `_lexical_cuisines` (ROUND-3 HIGH, no unmapped)
- LLM-fallback for paraphrase cuisines (e.g. "pho place" → vietnamese via LLM)
- Judge-None with lexical rows: demand_counts non-empty, unmapped only for LLM-needed rows

## Regression Guardrail

`poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v`: **32/32 PASSED**. All 6 supply-only functions (`gather_stats`, `find_gaps`, `propose_queries`, `filter_already_covered`, `insert_pending`, `existing_query_texts`) unchanged.

## Deviations from Plan

None — plan executed exactly as written.

The ruff pre-commit hook auto-fixed formatting and flagged 5 lint issues (unused `os` import, unused loop variables, misnamed inner class, `PreResolved` type alias as uppercase variable). All fixed inline before GREEN commit.

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced. `get_demand_conn` is read-only by construction (`set_session(readonly=True)`) — write path remains on the pool. `_extract_demand_batch` sends user messages to `make_judge` LLM, but this is the same LLM already used by `propose_queries`; no new PII sink created. Both mitigations documented in 18-02-demand-extraction-PLAN.md threat model (T-18-02-INJ partial, T-18-02-PROD mitigated, T-18-02-SQLi mitigated).

## Self-Check

- [x] `tests/unit/test_gap_miner.py` exists and committed
- [x] `scripts/coverage_agent.py` modified and committed
- [x] 39 new tests pass
- [x] 32 regression tests pass
- [x] 6 supply-only functions unchanged (`grep -c` = 6)
- [x] 7 new demand functions present in `scripts/coverage_agent.py`
- [x] `ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes
- [x] No `--no-verify` used, no branch switching

## Self-Check: PASSED
