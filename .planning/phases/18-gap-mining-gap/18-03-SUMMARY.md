---
phase: 18-gap-mining-gap
plan: 03
subsystem: coverage_agent / gap_miner
tags: [gap-mining, demand-signal, pair-level-supply, sandbox-guard, mlflow, tdd]
dependency_graph:
  requires: [18-01, 18-02]
  provides: [DemandGap, gather_pair_supply, find_demand_gaps, gap_to_seed_query, ingested_query_texts, insert_pending(conn=), gap_mine_main, extended log_to_mlflow]
  affects: [scripts/coverage_agent.py, tests/unit/test_gap_miner.py]
tech_stack:
  added: []
  patterns: [TDD RED-GREEN, capturing-stub connections, same-connection guard+insert, prefix normalization, status-filter dedup]
key_files:
  created: []
  modified:
    - scripts/coverage_agent.py
    - tests/unit/test_gap_miner.py
decisions:
  - "SUPERSEDES RESEARCH Open Question #1 — per-cuisine/min(both) supply resolution is replaced by TRUE pair-level supply from place_query_hits (gather_pair_supply counts DISTINCT place_id WHERE query_text = seed for each demanded pair). A cuisine present city-wide but absent in the demanded neighborhood (Outer Sunset/Vietnamese canonical case) now correctly flags supply=0 for that pair."
  - "ingested_query_texts filters checkpoints to status='completed' (ROUND-3 MEDIUM — mirrors ingest get_completed_queries) AND normalizes the FIELD_MODE:: checkpoint prefix via split('::', 1)[1] (ROUND-2 NEW HIGH — completed all::seed checkpoint dedupes the matching raw seed proposal)."
  - "sandbox write guard runs on the SAME connection threaded into insert_pending(..., conn=conn) — gap_mine_main opens ONE pooled conn, runs assert_sandbox_write_target(conn), then passes that conn to insert_pending (ROUND-3 LOW)."
  - "assert_sandbox_write_target imported from scripts.sandbox_guard (Plan 01 shared module) — NOT redefined in coverage_agent.py (ROUND-2 MEDIUM-2)."
  - "Cold-start is keyed on empty demand_counts (D-04), not judge absence — lexically-mappable demand still mines with judge=None (ROUND-2 MEDIUM-3)."
  - "--top-n applied AFTER filter_already_covered (RESEARCH Open Question #2)."
metrics:
  duration: 421s
  completed: 2026-06-18T18:59:50Z
  tasks_completed: 2
  files_modified: 2
---

# Phase 18 Plan 03: Gap Scoring + CLI Summary

TRUE pair-level demand×supply gap definition with sandbox-guarded loop-seam write, deduplicated against ingested rows (completed-checkpoints-prefix-normalized union proposals), and MLflow demand metrics under `gap_mine_main` CLI.

## Tasks Completed

### Task 1: DemandGap + gather_pair_supply + find_demand_gaps + gap_to_seed_query

**RED commit:** `003c6a7` — 9 failing tests for DemandGap, gather_pair_supply, find_demand_gaps, gap_to_seed_query

**GREEN commit:** `d8da5ee` — implementation in scripts/coverage_agent.py

**What was added:**

- `DemandGap` dataclass with `neighborhood`, `cuisine`, `place_count` (pair-level supply), `demand_count` — explicit structure replacing any fragile `demand:{n}:{c}` bucket encoding.
- `gap_to_seed_query(neighborhood, cuisine) -> str` — returns exact `"{cuisine} restaurants in {neighborhood} San Francisco"` seed format with upfront catalog-membership assertions on both axes (raises on off-catalog inputs). Catalog-membership check against `build_seed_queries()` ensures every emitted seed passes `premark_seed_isolation` (D-03, T-18-03-SEED). Does NOT import or reference `loop_falsifier.GAP`.
- `gather_pair_supply(pairs, conn=None) -> dict` — ONE parameterised `SELECT query_text, COUNT(DISTINCT place_id) FROM place_query_hits WHERE query_text = ANY(%s) GROUP BY query_text`. Seed strings bound as `%s` param list (never interpolated — T-18-03-SQLi). Never-ingested pairs yield 0. This is TRUE pair-level supply (HIGH-1) — it counts places that matched the neighborhood-AND-cuisine seed in `place_query_hits`, so "Vietnamese everywhere in SF but zero in Outer Sunset" correctly scores 0 for `("Outer Sunset", "vietnamese")`.
- `find_demand_gaps(demand_counts, pair_supply, min_place_count=5) -> list[DemandGap]` — D-02 gate: `demand > 0 AND pair_supply < floor`, sorted by `demand_count` descending.

**Key correctness (HIGH-1):** The Outer Sunset / Vietnamese case is correctly flagged because `gather_pair_supply` looks at `place_query_hits.query_text = "vietnamese restaurants in Outer Sunset San Francisco"` — not the city-wide `cuisine:vietnamese` count that a per-cuisine resolver would use. This was the decisive difference in REVIEW HIGH-1.

**Supersedes RESEARCH Open Question #1:** The plan notes that RESEARCH suggested resolving gaps as `min(neighborhood_supply, cuisine_supply)` — this plan overrides that with TRUE pair-level supply from `place_query_hits` per the cross-AI HIGH-1 review finding. The per-cuisine approach would have missed the canonical case (city-wide supply existing, neighborhood supply absent).

### Task 2: ingested_query_texts + insert_pending conn= + gap_mine_main + MLflow demand metrics

**RED commit:** `d532e7d` — 14 failing tests for ingested_query_texts, insert_pending conn=, gap_mine_main, sandbox guard, cold-start, MLflow metrics (1 pre-passed since insert_pending already existed)

**GREEN commit:** `816ac95` — all Task 2 implementations

**What was added:**

- `ingested_query_texts(conn) -> set[str]` (HIGH-2 + ROUND-2 NEW HIGH + ROUND-3 MEDIUM):
  - Does NOT include `build_seed_queries()` static catalog — so valid NEW proposals survive `filter_already_covered` (HIGH-2 BLOCKER fix: the old `existing_query_texts` included the static catalog and would drop every valid proposal).
  - Checkpoint SELECT: `WHERE status = 'completed'` — mirrors ingest's `get_completed_queries()` (ingest_places_sf.py:487-497). An incomplete/budget-stopped checkpoint is retried by ingest and must NOT suppress the mined proposal (ROUND-3 MEDIUM — the "miner's already-ingested view must match ingest's real skip logic" class of bug, now in the status dimension). Opposite outcomes proven: Test 3 (completed prefixed checkpoint → deduped) vs Test 4 (incomplete checkpoint → NOT deduped, proposal lands in kept).
  - Prefix normalization: ingest's `checkpoint_key` stores `f"{FIELD_MODE}::{raw}"` (e.g. `"all::vietnamese restaurants in Outer Sunset San Francisco"`). The helper splits on the first `::` and takes the part after, so a completed `all::<seed>` checkpoint dedupes the raw `<seed>` proposal. A row without `::` is added as-is (defensive). (ROUND-2 NEW HIGH)
  - All proposals (`places_ingest_query_proposals`) are included regardless of their own status — already raw, any pending/processed proposal still dedupes.
- `insert_pending` extended with optional `conn=None` param (ROUND-3 LOW — additive, backward-compatible): when `conn` is provided, runs INSERT on THAT connection; when None (default), keeps the existing `with get_conn() as _conn` self-open. All existing supply-only callers pass nothing and are byte-for-byte unaffected.
- `from scripts.sandbox_guard import assert_sandbox_write_target` — imported from Plan 01 shared module, NOT redefined (ROUND-2 MEDIUM-2). The guard checks `SELECT current_database()` on the live write connection.
- `log_to_mlflow` extended with optional `demand_rows_scanned=0`, `unmapped_count=0`, `demand_gaps=None` kwargs and now logs `proposals_inserted`, `demand_rows_scanned`, `unmapped_count` metrics + `demand_gaps.json` ranked artifact. Existing supply-only callers pass no demand kwargs and get zero values.
- `gap_mine_main(argv=None) -> int` CLI with its own argparse (`--days`, `--dry-run`, `--min-places`, `--top-n`):
  - Reads `DEMAND_DATABASE_URL` env var for the demand read path.
  - Cold-start (empty `demand_counts`): logs `gaps_found=0`, returns 0, zero inserts (D-04). Cold-start is keyed on demand being empty, NOT on judge absence — lexically-mappable demand with `judge=None` still mines.
  - Proposal `query_text` is always `gap_to_seed_query(g.neighborhood, g.cuisine)` output (exact seed format, never free LLM text — T-18-03-SEED).
  - Write path: opens ONE pooled conn via `get_conn()`, builds `ingested_query_texts(conn)`, runs `filter_already_covered`, applies `--top-n` AFTER dedup, calls `assert_sandbox_write_target(write_conn)` THEN `insert_pending(kept, dry_run, conn=write_conn)` on that SAME conn (HIGH-3 + ROUND-3 LOW).
  - Does not touch the existing `main()` supply-only entrypoint.

## Test Results

- 94 total tests passing (71 baseline + 9 Task 1 + 14 Task 2)
- Regression: `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (32 supply-only tests — unchanged)
- `grep -c 'loop_falsifier' scripts/coverage_agent.py` == 1 (pre-existing docstring comment in `get_demand_conn`, not an import or reference to GAP — unchanged count vs prior commit)
- `grep -c "status = 'completed'" scripts/coverage_agent.py` == 1 (ROUND-3 MEDIUM filter confirmed)
- `grep -c 'from scripts.sandbox_guard import' scripts/coverage_agent.py` == 1 (imported, not redefined)
- `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` — all checks passed

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all connections are implemented; `gap_mine_main` wires demand→supply→gaps→proposals→insert end-to-end.

## Threat Flags

No new security-relevant surfaces beyond the plan's `<threat_model>`. All three STRIDE mitigations confirmed:
- T-18-03-SEED: `gap_to_seed_query` assertions prevent off-catalog seeds
- T-18-03-WRITE: guard on same connection prevents prod writes
- T-18-03-DEDUP: `ingested_query_texts` closes all three divergence directions (static catalog excluded, prefix normalized, status filtered)
- T-18-03-SQLi: seeds bound as `%s` param list in `gather_pair_supply`

## Summary Notes (per plan output spec)

- **SUPERSEDES RESEARCH Open Question #1:** TRUE pair-level supply from `place_query_hits` (gather_pair_supply) replaces the per-cuisine / min(both) approach. Pair-level is authoritative per REVIEW HIGH-1.
- **Status filter:** `ingested_query_texts` adds `WHERE status = 'completed'` to the checkpoint SELECT (ROUND-3 MEDIUM — incomplete checkpoints don't suppress retriable pairs).
- **Prefix normalization:** `ingested_query_texts` strips `FIELD_MODE::` prefix from completed checkpoints (ROUND-2 NEW HIGH — `all::seed` dedupes raw `seed`).
- **Same-conn guard+insert:** `gap_mine_main` opens ONE conn, runs `assert_sandbox_write_target(conn)`, passes that conn to `insert_pending(..., conn=conn)` (ROUND-3 LOW).

## Self-Check

- [x] FOUND: `.planning/phases/18-gap-mining-gap/18-03-SUMMARY.md`
- [x] FOUND commit: `816ac95` (feat Task 2 GREEN)
- [x] FOUND commit: `d8da5ee` (feat Task 1 GREEN)
- [x] FOUND commit: `d532e7d` (test Task 2 RED)
- [x] FOUND commit: `003c6a7` (test Task 1 RED)
- [x] 94 tests passing, 0 failures
- [x] ruff check passed

**Self-Check: PASSED**
