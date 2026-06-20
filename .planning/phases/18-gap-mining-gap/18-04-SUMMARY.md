---
phase: 18-gap-mining-gap
plan: "04"
subsystem: tests-make-docs
tags: [testing, smoke, integration, makefile, docs-sync, gap-miner]
dependency_graph:
  requires: [18-01, 18-02, 18-03]
  provides: [GAP-03-proof, GAP-04-ergonomics]
  affects: [tests/unit, tests/integration, Makefile, CLAUDE.md, AGENTS.md, copilot-instructions.md]
tech_stack:
  added: []
  patterns:
    - branched _StubCursor on SQL substrings for multi-SELECT gap_mine_main stubs
    - pytestmark skipif APP_ENV!=integration + fixture-level skip for table absence
    - ingested_query_texts() used at test-setup to pick a deterministic clean pair
key_files:
  created:
    - tests/unit/test_gap_miner_smoke.py
    - tests/integration/test_gap_miner.py
  modified:
    - Makefile
    - CLAUDE.md
    - AGENTS.md
    - .github/copilot-instructions.md
decisions:
  - "Branched _StubCursor on 4 SQL substrings (user_query_log, place_query_hits, places_ingest_query_checkpoints, places_ingest_query_proposals) — avoids cross-contamination across gap_mine_main's multiple selects"
  - "Integration test picks the FIRST catalog (neighborhood, cuisine) pair absent from both proposals AND ingested_query_texts() normalized dedup set — ensures checkpoint prefix normalization and completed-only filter cannot silently pre-dedupe the target (ROUND-2 MEDIUM-1 + ROUND-3 MEDIUM)"
  - "gap-mine Makefile target uses python -c entrypoint mirroring coverage-agent style; gap-mine-dry added as companion"
metrics:
  duration: "~15 minutes"
  completed: "2026-06-18"
  tasks_completed: 3
  tasks_total: 3
  files_changed: 6
---

# Phase 18 Plan 04: Tests + Make + Docs Summary

One-liner: Smoke/functional/integration test pyramid for gap_mine_main (cuisine-recall + checkpoint dedup paths), plus `make gap-mine` target and 3-way docs sync.

## What Was Built

**Task 1 — Smoke + functional tests (`tests/unit/test_gap_miner_smoke.py`)**

Four tests covering the demand-miner entrypoint:

1. `test_smoke_module_imports` — asserts `gap_mine_main`, `gather_demand`, `gather_pair_supply`, `find_demand_gaps`, `ingested_query_texts`, `gap_to_seed_query`, `get_demand_conn` are present and `assert_sandbox_write_target` is importable.

2. `test_smoke_cold_start_empty_db` — `gap_mine_main(["--dry-run","--days","1"])` on a stub with no `user_query_log` rows returns 0 and logs `gaps_found=0` (D-04 cold-start no-op).

3. `test_functional_cuisine_recall_emits_demand_gaps_artifact` — stub demand row with `requested_primary_types=[]` (free-text case) and `message="vietnamese restaurants in Outer Sunset"` triggers `_lexical_cuisines` fallback (ROUND-3 HIGH); stub COMPLETED prefixed `all::korean...` checkpoint survives dedup (ROUND-2 NEW HIGH + ROUND-3 MEDIUM); `mlflow.log_dict` called with `demand_gaps.json` containing `(Outer Sunset, vietnamese)`.

4. `test_functional_dry_run_no_write` — confirms `--dry-run` executes zero INSERT statements on the stub connection.

The `_StubCursor` branches on 4 SQL substrings (FROM user_query_log, FROM place_query_hits, FROM places_ingest_query_checkpoints, FROM places_ingest_query_proposals) preventing cross-contamination. `assert_sandbox_write_target` is monkeypatched to a no-op so no real sandbox connection is needed.

**Task 2 — Integration test (`tests/integration/test_gap_miner.py`)**

Gated on `APP_ENV=integration`. Two fixtures skip gracefully when `places_ingest_query_proposals` or `user_query_log` is absent/unwritable (schema-deploy safety, shared-DB convention). The main test:

- Builds the normalized dedup set via `ingested_query_texts(conn)` and reads existing proposals to find a `(neighborhood, cuisine)` pair free of BOTH sources (ROUND-2 MEDIUM-1 + ROUND-3 MEDIUM) or `pytest.skip`s with a clear reason.
- Seeds ONE demand row with `requested_primary_types=[]` and `message="{cuisine} restaurants in {neighborhood}"` — maps lexically on both axes with NO LLM (ROUND-3 cuisine-recall path against real DB).
- Stubs `vibe.make_judge` and mlflow.
- Runs `gap_mine_main(["--days","30","--min-places","100000"])` — forced gap regardless of real sandbox supply (REVIEW MEDIUM determinism).
- Asserts a `pending` proposal row with `query_text == gap_to_seed_query(n, c)`.
- Cleanup in `finally`: deletes demand rows by unique `session_id` marker; deletes the proposal ONLY if the test created it (pre-mining check); never blanket-deletes (REVIEW MEDIUM scoped cleanup).

Verified: `poetry run pytest tests/integration/test_gap_miner.py -v` without `APP_ENV=integration` exits 0 with 2 skipped, no DB pool leak.

**Task 3 — Makefile + docs sync**

Added to `Makefile` immediately after `coverage-agent-apply`:
- `.PHONY: gap-mine` — invokes `gap_mine_main` via `python -c` entrypoint (writes by default; D-04)
- `.PHONY: gap-mine-dry` — passes `--dry-run` flag

`make -n gap-mine` prints the `gap_mine_main` invocation without error.

Added identical `# Adaptive data loop (Phase 18 / v2.3)` section to:
- `CLAUDE.md` (## Commands block)
- `AGENTS.md` (## Commands block)
- `.github/copilot-instructions.md` (## Commands block)

Each states: reads sandbox by default or `DEMAND_DATABASE_URL` when set; writes proposals to sandbox guarded by `assert_sandbox_write_target / current_database()`.

## Verification Results

```
poetry run pytest tests/unit/test_gap_miner_smoke.py -v   → 4 passed
poetry run pytest tests/integration/test_gap_miner.py -v  → 2 skipped (exit 0, no pool leak)
Full regression suite (115 tests):                         → 115 passed
make -n gap-mine                                           → prints invocation, exit 0
gap-mine present in: Makefile (4×), CLAUDE.md (2×), AGENTS.md (2×), copilot-instructions.md (2×)
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — no placeholder text, hardcoded empties flowing to UI, or unresolved data sources.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced in this plan. The integration test targets the sandbox DB exclusively (fixture-level guard).

## Self-Check: PASSED

- `tests/unit/test_gap_miner_smoke.py` exists: confirmed
- `tests/integration/test_gap_miner.py` exists: confirmed
- Commit 2e94dbb: smoke+functional tests
- Commit 7d735af: integration test
- Commit a24ddfe: Makefile + docs sync
