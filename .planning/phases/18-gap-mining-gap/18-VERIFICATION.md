---
phase: 18-gap-mining-gap
verified: 2026-06-18T00:00:00Z
status: passed
score: 19/19 must-haves verified
overrides_applied: 0
---

# Phase 18: Gap Mining (GAP) Verification Report

**Phase Goal:** Gap Mining (GAP) — real demand/supply gap miner (replaces Phase 16's hardcoded gap constant).
**Verified:** 2026-06-18
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | GAP-01: `gather_demand()` reads user_query_log with symmetric two-tier extraction on BOTH axes (cuisine + neighborhood) | VERIFIED | `coverage_agent.py` lines 339-438: `_types_to_cuisines` (tier-1 cuisine), `_lexical_cuisines` (tier-2a cuisine, ROUND-3 HIGH), `_lexical_neighborhoods` (tier-1 neighborhood), `_extract_demand_batch` (single combined LLM batch for both-axis misses) |
| 2 | GAP-01 ROUND-3: Free-text rows with `requested_primary_types=[]` still produce demand via `_lexical_cuisines` message scan | VERIFIED | `gather_demand` lines 388-393: after `_types_to_cuisines` returns empty, immediately calls `_lexical_cuisines(message)`; test `TestGatherDemandRound3HighLexical` passes |
| 3 | GAP-01: Judge-None still maps lexical hits on both axes (`unmapped_count` honesty) | VERIFIED | `gather_demand` logic: lexically-resolved rows skip the LLM batch entirely; only LLM-miss rows degrade; `TestGatherDemandJudgeNone` passes |
| 4 | GAP-02: `find_demand_gaps` gates on `demand > 0 AND pair_place_count < min_places` at TRUE pair level via explicit `DemandGap` dataclass | VERIFIED | `DemandGap` dataclass at line 442; `find_demand_gaps` at line 530; `gather_pair_supply` counts `DISTINCT place_id FROM place_query_hits WHERE query_text = ANY(%s)` — TRUE pair-level, not city-wide |
| 5 | GAP-02: Pairs ranked by `demand_count` descending | VERIFIED | `find_demand_gaps` line 554: `gaps.sort(key=lambda g: g.demand_count, reverse=True)` |
| 6 | GAP-03: Miner emits exact `"{cuisine} restaurants in {neighborhood} San Francisco"` format, asserted against `build_seed_queries()` catalog | VERIFIED | `gap_to_seed_query` at line 449: f-string + upfront `assert seed in set(build_seed_queries())`; `TestSeedFormatExactness` passes |
| 7 | GAP-03: Dedup uses `ingested_query_texts` (NOT static catalog) — completed-checkpoints with FIELD_MODE:: prefix stripped, status='completed' only | VERIFIED | `ingested_query_texts` at line 618; SQL: `WHERE status = 'completed'`; prefix strip: `query_text.split("::", 1)[1] if "::" in query_text` (line 657); `TestIngestedQueryTextsHigh2`, `TestCheckpointPrefixDedup`, `TestCheckpointStatusFilter` all pass |
| 8 | GAP-03: Writes via `insert_pending` to `places_ingest_query_proposals`, NOT `loop_falsifier.GAP` | VERIFIED | `insert_pending` at line 678; `grep -n "loop_falsifier" coverage_agent.py` returns only a comment, zero imports; `gap_mine_main` line 864 calls `insert_pending` |
| 9 | GAP-03: Sandbox write guard (`assert_sandbox_write_target`) runs on SAME connection as `insert_pending`, BEFORE insert | VERIFIED | `gap_mine_main` lines 852-864: `with get_conn() as write_conn:` opens ONE connection; guard at line 863, insert at 864 with `conn=write_conn`; `TestSameConnectionGuardAndInsert` passes |
| 10 | GAP-04: `gap_mine_main` CLI with `--days/--dry-run/--min-places/--top-n` flags; writes by default | VERIFIED | `gap_mine_main` at line 768: argparse with all 4 flags; `--top-n` applied after dedup (line 858-859); all CLI tests pass |
| 11 | GAP-04: Cold start (empty demand) inserts nothing, logs `gaps_found=0`, exits 0 | VERIFIED | Lines 804-818: `if not demand_counts:` branch calls `log_to_mlflow(demand_gaps=[])` which logs `gaps_found=0` and returns 0; `TestColdStart` passes |
| 12 | GAP-04: MLflow metrics `gaps_found`, `proposals_inserted`, `demand_rows_scanned`, `unmapped_count` + `demand_gaps.json` artifact | VERIFIED | `log_to_mlflow` lines 744-758: all four metrics logged; `mlflow.log_dict(..., "demand_gaps.json")` at line 758; `TestMLflowDemandMetrics` passes |
| 13 | GAP-04: `make gap-mine` target exists and invokes `gap_mine_main` | VERIFIED | Makefile lines 110-112: `.PHONY: gap-mine` with `$(POETRY_RUN) python -c "from scripts.coverage_agent import gap_mine_main; ..."` |
| 14 | Docs sync: `gap-mine` present in CLAUDE.md, AGENTS.md, copilot-instructions.md | VERIFIED | All three files contain `gap-mine` with identical descriptions referencing `DEMAND_DATABASE_URL` and `assert_sandbox_write_target` |
| 15 | `sandbox_guard.py` shared module: ONE definition of `assert_sandbox_write_target`, checks live `SELECT current_database()`, NOT env-var parsing | VERIFIED | `sandbox_guard.py` line 52: `cur.execute("SELECT current_database()")`, pass condition on `live_dbname` value only; `_require_sandbox` helper contains no env-var logic |
| 16 | Both `seed_demand_log.py` and `coverage_agent.py` IMPORT (not redefine) from `scripts.sandbox_guard` | VERIFIED | `seed_demand_log.py` line 22 and `coverage_agent.py` line 39: `from scripts.sandbox_guard import assert_sandbox_write_target`; no redefinition found in either file |
| 17 | All SQL is parameterised — no interpolation of message/query_text into SQL strings | VERIFIED | `gather_demand` uses `[cutoff]` param list; `gather_pair_supply` uses `[seed_strings]` param list; `insert_demand_rows` uses `%s` placeholders; no f-string SQL found |
| 18 | Supply-only functions unchanged (additive guarantee) | VERIFIED | `grep -c "def gather_stats\|def find_gaps\|..."` returns 6 (all exist); existing tests `test_coverage_agent.py` + `test_coverage_agent_smoke.py`: 32 passed |
| 19 | Full test pyramid: unit (115) + smoke + functional + integration (skips cleanly) | VERIFIED | `test_gap_miner.py` 66 tests passed; `test_gap_miner_smoke.py` 4 tests passed (smoke + functional with cuisine-recall path); `tests/integration/test_gap_miner.py` 2 skipped cleanly without APP_ENV |

**Score:** 19/19 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/sandbox_guard.py` | Shared `assert_sandbox_write_target()` with `current_database()` check | VERIFIED | 85 lines; `SELECT current_database()` at line 52; `_require_sandbox` enforces sandbox name |
| `scripts/seed_demand_log.py` | Catalog-valid seed helper, sandbox-write-guarded | VERIFIED | 151 lines; `seed_demand_rows()` returns 5 fixture rows; `insert_demand_rows` calls guard before INSERT |
| `scripts/coverage_agent.py` | Full demand miner: `gather_demand`, `DemandGap`, `find_demand_gaps`, `gather_pair_supply`, `gap_to_seed_query`, `ingested_query_texts`, `insert_pending` (with conn=), `gap_mine_main`, extended `log_to_mlflow` | VERIFIED | 923 lines; all functions present |
| `tests/unit/test_gap_miner.py` | Unit tests for all demand extraction behaviors | VERIFIED | 66 tests, all pass |
| `tests/unit/test_gap_miner_smoke.py` | Smoke + functional tests with cuisine-recall path | VERIFIED | 4 tests (module smoke, cold-start, functional cuisine-recall, dry-run-no-write), all pass |
| `tests/integration/test_gap_miner.py` | Integration test (skips without APP_ENV) | VERIFIED | 2 skipped cleanly; gated on `APP_ENV=integration`; implements scoped cleanup, checkpoint-aware target selection |
| `Makefile` | `gap-mine:` and `sandbox-migrate:` targets | VERIFIED | Lines 62-69 (`sandbox-migrate`), lines 110-116 (`gap-mine`, `gap-mine-dry`) |
| `.env.example` | Commented `DEMAND_DATABASE_URL` block | VERIFIED | Line 31: `# DEMAND_DATABASE_URL=` |
| `CLAUDE.md` / `AGENTS.md` / `.github/copilot-instructions.md` | `gap-mine` documented identically in all three | VERIFIED | All three contain identical `gap-mine` + `gap-mine-dry` lines with DEMAND_DATABASE_URL and guard description |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `seed_demand_log.py` | `scripts.ingest_places_sf NEIGHBORHOODS/CUISINES` | `from scripts.ingest_places_sf import CUISINES, NEIGHBORHOODS` + catalog assertion at import | WIRED | Line 21; module-level assertion loop at lines 69-75 |
| `seed_demand_log.py` | `scripts.sandbox_guard.assert_sandbox_write_target` | import + call before INSERT | WIRED | Line 22; called at line 108 before INSERT at line 113 |
| `sandbox_guard.py assert_sandbox_write_target` | write connection `current_database()` | `SELECT current_database()` on passed/pool conn | WIRED | Line 52 |
| `coverage_agent.py gather_demand` | `user_query_log` | parameterised `SELECT ... FROM user_query_log WHERE created_at >= %s` | WIRED | Line 370; `[cutoff]` param list at line 373 |
| `coverage_agent.py demand extraction` | `scripts.ingest_places_sf CUISINES/NEIGHBORHOODS` | `from scripts.ingest_places_sf import CUISINES, NEIGHBORHOODS, build_seed_queries` | WIRED | Line 38 |
| `coverage_agent.py _lexical_cuisines` | `user_query_log.message` | case-insensitive scan against `CUISINES` | WIRED | Lines 203-214 |
| `coverage_agent.py get_demand_conn` | `DEMAND_DATABASE_URL` | `os.environ.get("DEMAND_DATABASE_URL")` in `gap_mine_main` | WIRED | Line 799; `get_demand_conn(url)` at line 371 |
| `coverage_agent.py gather_pair_supply` | `place_query_hits` | `COUNT(DISTINCT place_id) FROM place_query_hits WHERE query_text = ANY(%s)` | WIRED | Lines 503-519 |
| `coverage_agent.py ingested_query_texts` | `places_ingest_query_checkpoints + places_ingest_query_proposals` | Two SELECTs; checkpoints filtered `WHERE status = 'completed'`; prefix normalized | WIRED | Lines 649-663 |
| `coverage_agent.py gap_mine_main` | `scripts.sandbox_guard.assert_sandbox_write_target` | Import at line 39; called on write_conn at line 863 BEFORE `insert_pending` at line 864 | WIRED | Lines 863-864 |
| `Makefile gap-mine` | `scripts/coverage_agent.py gap_mine_main` | `python -c "from scripts.coverage_agent import gap_mine_main; ..."` | WIRED | Makefile line 112 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `coverage_agent.gap_mine_main` | `demand_counts` | `gather_demand(days, demand_url)` → `SELECT FROM user_query_log` | Real DB query with parameterised cutoff | FLOWING |
| `coverage_agent.gather_pair_supply` | `result` | `SELECT COUNT(DISTINCT place_id) FROM place_query_hits WHERE query_text = ANY(%s)` | Real DB query, never-ingested pairs → 0 | FLOWING |
| `coverage_agent.ingested_query_texts` | `ingested` set | `SELECT FROM places_ingest_query_checkpoints WHERE status='completed'` UNION proposals | Real DB query, not static catalog | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Unit test suite (gap miner) | `pytest tests/unit/test_gap_miner.py tests/unit/test_gap_miner_smoke.py -q` | 66 passed | PASS |
| Supply-only regression guard | `pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -q` | 32 passed | PASS |
| Integration test skips cleanly | `pytest tests/integration/test_gap_miner.py -q` (no APP_ENV) | 2 skipped | PASS |
| `gap-mine` target present in Makefile | `grep -c 'gap-mine:' Makefile` | 1 | PASS |
| `sandbox-migrate` target invokes alembic against SANDBOX_DATABASE_URL | `grep -n 'DATABASE_URL.*alembic upgrade' Makefile` | `DATABASE_URL=$${SANDBOX_DATABASE_URL} ... alembic upgrade head` | PASS |
| Guard not redefined in coverage_agent | `grep -n 'def assert_sandbox_write_target' coverage_agent.py` | no output | PASS |
| `loop_falsifier` not imported in coverage_agent | `grep 'import.*loop_falsifier' coverage_agent.py` | no output (only comment) | PASS |
| `status = 'completed'` in ingested_query_texts | `grep -c "status = 'completed'" coverage_agent.py` | 1 | PASS |
| Docs sync | `grep -c 'gap-mine' CLAUDE.md AGENTS.md .github/copilot-instructions.md` | 2 each | PASS |

### Probe Execution

Step 7c: SKIPPED — no probe scripts declared in any PLAN or SUMMARY for this phase; the phase's verification approach is unit/smoke/functional/integration tests, not probe scripts.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GAP-01 | 18-01, 18-02 | Demand extraction from `user_query_log` via symmetric two-tier extraction (neighborhood + cuisine) | SATISFIED | `gather_demand` + all tier helpers in `coverage_agent.py`; 115 unit tests cover all behaviors including ROUND-3 cuisine-recall |
| GAP-02 | 18-03 | Demand x supply gap definition at TRUE pair level via `DemandGap` dataclass, ranked by demand desc | SATISFIED | `DemandGap`, `find_demand_gaps`, `gather_pair_supply` in `coverage_agent.py` |
| GAP-03 | 18-03, 18-04 | Loop integration: exact seed format, `insert_pending` via proposals table, dedup against ingested (not static catalog), sandbox guard on same conn, no falsifier GAP touch | SATISFIED | All confirmed in code + unit tests + integration test structure |
| GAP-04 | 18-01, 18-03, 18-04 | CLI `gap_mine_main` with `--days/--dry-run/--min-places/--top-n`, MLflow demand metrics, cold-start no-op, `make gap-mine`, docs sync | SATISFIED | All flags present; all metrics logged; cold-start tested; Makefile + all 3 docs synced |

Note: REQUIREMENTS.md file was not found in `.planning/` (only ROADMAP.md and milestones exist). All 4 requirement IDs are declared in plan frontmatter and verified against the codebase above.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | No TBD/FIXME/XXX/placeholder markers found in phase-modified files |

The only comment referencing `loop_falsifier.py` in `coverage_agent.py` (line 327) is a docstring about a mirror pattern — no import or usage. No SQL interpolation of user content found. No stub returns or empty implementations in production paths.

### Human Verification Required

None. All verifiable claims are grounded in code that runs against controlled stubs. The integration test (`APP_ENV=integration`) is the only human-gateable item, and it is correctly structured to skip cleanly until a sandbox with `user_query_log` is available — this is by design (project convention), not a gap.

### Gaps Summary

No gaps. All 19 must-have truths are verified by direct code inspection and test execution. The phase achieved its goal: Phase 16's hardcoded gap constant is fully superseded by a real demand-signal miner that reads `user_query_log`, extracts catalog-constrained `(neighborhood, cuisine)` demand via symmetric two-tier extraction, cross-references true pair-level supply from `place_query_hits`, and writes loop-consumable pending proposals with sandbox-write protection.

---

_Verified: 2026-06-18_
_Verifier: Claude (gsd-verifier)_
