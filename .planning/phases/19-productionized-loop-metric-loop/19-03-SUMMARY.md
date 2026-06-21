---
phase: 19-productionized-loop-metric-loop
plan: "03"
subsystem: loop/orchestrator
tags: [loop-runner, orchestrator, gap-handoff, paraphrase-freeze, sandbox-coercion, mlflow]
dependency_graph:
  requires: [19-01]
  provides: [loop_runner.py, make-loop-runner]
  affects: [scripts/loop_runner.py]
tech_stack:
  added: []
  patterns:
    - "deferred-import pattern (all settings-touching imports inside main() after coercion)"
    - "D-07 coercion-ordering: DATABASE_URL coerce → cache_clear → close_pool → assert-sandbox → embedding-table-assert"
    - "D-08 deterministic one-gap set-diff on pending query_text (not created_at)"
    - "D-04 durable paraphrase freeze to disk BEFORE ingest subprocess"
    - "D-02 dual provisioning guards: places_raw GUARD A + v2 GUARD B as distinct INFRA signals"
    - "IN-05 artifacts-before-metrics in log_to_mlflow"
key_files:
  created:
    - scripts/loop_runner.py
  modified: []
decisions:
  - "helpers resolve_prod_url / assert_resolved_target / run_subprocess_or_infra / _snapshot_ids_from_url copied verbatim from loop_falsifier.py (NOT imported) so falsifier stays byte-for-byte unchanged (D-07)"
  - "paraphrase generation uses build_chat_model via DEFAULT_JUDGE_MODEL/PROVIDER from vibe.py (PARAPHRASE_PROVIDER/PARAPHRASE_MODEL env overrides available)"
  - "cold-start exit (len(new)==0) returns EXIT_PASS (not EXIT_FAIL) — no demand gap = no-op, not a failure"
  - "BASELINE/MINER GAP CONTRACT: when LOOP_GAP_NEIGHBORHOOD/CUISINE are set, miner-parsed pair must match or EXIT_INFRA (metric would silently measure wrong gap)"
metrics:
  duration: "301s"
  completed: "2026-06-21T00:29:01Z"
  tasks: 2
  files: 1
requirements: [LOOP-01, LOOP-02, LOOP-03, METRIC-01, METRIC-02, METRIC-03]
---

# Phase 19 Plan 03: Productionized Loop Runner Summary

**One-liner:** Build `scripts/loop_runner.py` — the capstone productionized loop orchestrator that chains gap-mine → LLM-paraphrase-freeze → ingest → embed-v2 → hit@k+recall@k → MLflow in the D-07 locked coercion order, scoring against a populated baseline.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Build loop_runner sandbox-coercion + gap-handoff + paraphrase-freeze stages | 8358e8a | scripts/loop_runner.py |
| 2 | Build loop_runner snapshot→ingest→embed→score→MLflow→exit stages | 8358e8a | scripts/loop_runner.py (continued) |

Both tasks landed in the same commit (single new file).

## What Was Built

### `scripts/loop_runner.py` (684 lines)

A new sibling to `loop_falsifier.py` that owns the **productionized loop's stage ordering** with all three LOCKED CONSTRAINTS from D-07/D-08:

**Stage 1: Coercion + Cache Invalidation (D-07 LOCKED CONSTRAINT)**

1. Read `SANDBOX_DATABASE_URL` → EXIT_INFRA if unset
2. `resolve_prod_url(sandbox_url)` BEFORE coercion (prod-safety guard)
3. `os.environ["DATABASE_URL"] = sandbox_url` → deferred-import `get_settings` + `close_db_pool` → `cache_clear()` + `close_db_pool()`
4. Assert BOTH `resolve_database_url(os.environ) == sandbox_url` AND `get_settings().resolved_database_url == sandbox_url`
5. Assert `settings.embedding_table == "place_embeddings_v2"` (D-07 embedding-table constraint)

**Stage 2: Gap Handoff (D-08 deterministic one-gap set-diff)**

Only NOW (after coercion) deferred-import `gap_mine_main`, `get_conn`, `semantic_search`, `assert_sandbox_write_target`.

- Open sandbox conn, `assert_sandbox_write_target(conn)`, clear ALL stale pending (UPDATE SET status='rejected' WHERE status='pending')
- Snapshot `pending_before`
- Call `gap_mine_main(["--top-n", "1"])`
- Snapshot `pending_after`
- Set-diff `new = pending_after - pending_before` on `query_text` (NOT `created_at`)
- Three branches: `len(new)==0` → EXIT_PASS (cold-start no-op); `len(new)>1` → EXIT_INFRA; `len(new)==1` → parse `(neighborhood, cuisine)` by reversing `"{cuisine} restaurants in {neighborhood} San Francisco"`
- BASELINE/MINER GAP CONTRACT: if `LOOP_GAP_NEIGHBORHOOD`/`LOOP_GAP_CUISINE` set, assert miner-parsed pair matches (else EXIT_INFRA with clear message)

**Stage 3: Paraphrase Generation + Durable Freeze (D-04 LOCKED CONSTRAINT)**

BEFORE before-snapshot/ingest:
- Generate N paraphrases via `build_chat_model` (configurable via `PARAPHRASE_PROVIDER`/`PARAPHRASE_MODEL` env)
- `check_non_circularity(paraphrases, [gap_seed_query])` → EXIT_INFRA on violation
- Build per-run artifact dict (extends `falsifier_paraphrases.json` shape with `gap_neighborhood`, `gap_cuisine`, `generation_timestamp`, `generation_model`)
- `json.dump(...)` to `loop_runner_artifacts/frozen_paraphrases_runner.json` — disk write is the freeze point; occurs at line 522, BEFORE ingest at line 541

**Stage 4: Before-Snapshot**

- `_snapshot_ids_from_url(sandbox_url, "places_raw")` + `_snapshot_ids_from_url(sandbox_url, "place_embeddings_v2")`
- Probe paraphrases: `before_topk = [[h.place_id for h in semantic_search(p, k=K)] for p in paraphrases]`
- `before_hit_result = compute_hit_rate(before_topk, before_v2_ids)` → 0.0 by construction (D-03)

**Stage 5: Ingest + Embed Subprocesses**

- `child_env = {**os.environ, "DATABASE_URL": sandbox_url}`
- `run_subprocess_or_infra([sys.executable, "scripts/ingest_places_sf.py"], child_env)`
- `run_subprocess_or_infra([sys.executable, "-m", "scripts.embed_places_pgvector_v2"], child_env)`

**Stage 6: DB-Diffs + Dual Provisioning Guards (D-02)**

- GUARD A (places_raw): `after_raw_ids = _snapshot_ids_from_url(sandbox_url, "places_raw")`; `new_raw_ids = db_diff(before_raw_ids, after_raw_ids)`; `new_place_count = len(new_raw_ids)`. If `new_place_count == 0` → EXIT_INFRA: "loop ran but ingested ZERO new places_raw rows"
- GUARD B (v2): `after_v2_ids = _snapshot_ids_from_url(sandbox_url, "place_embeddings_v2")`; `new_v2_ids = db_diff(before_v2_ids, after_v2_ids)`; `embed_added_count = len(new_v2_ids)`. If `embed_added_count == 0` → EXIT_INFRA: "rows ingested but NONE embedded"
- Scoring target is ONLY `new_v2_ids` (NOT `new_raw_ids`) per D-03

**Stage 7: After-Snapshot + Scoring**

- `after_topk = [[h.place_id for h in semantic_search(p, k=K)] for p in paraphrases]`
- `after_hit_result = compute_hit_rate(after_topk, new_v2_ids)` — delegates to falsifier_core
- `recall_result = compute_recall_at_k(after_topk, new_v2_ids)` — delegates to falsifier_core
- `floor = float(os.environ.get("LOOP_HIT_RATE_FLOOR", FLOOR))` (D-05)
- `fixture_mode = not bool(os.environ.get("DEMAND_DATABASE_URL"))` (D-01)

**Stage 8: Gate + MLflow + Verdict**

- `exit_code = decide_loop_exit(before_rate, after_rate, floor, guard_violation=None, embed_added_count=embed_added_count)` — delegates to falsifier_core pure function
- `log_to_mlflow(...)`: params first, then artifacts (IN-05): `frozen_paraphrases_runner.json`, `before_snapshot.json`, `after_snapshot.json`, `db_diff_v2_place_ids.json` BEFORE any `log_metric` call
- Metrics: `before_hit_at_k`, `after_hit_at_k`, `hit_rate_delta`, `recall_at_k`, `new_place_count`, `embed_added_count`
- Verdict banner with `='*60` separator reporting hit@k delta, recall@k, floor, fixture_mode, exit label (PASS/FAIL/INFRA)
- `raise SystemExit(exit_code)`

**Module-scope import discipline:** ONLY `stdlib` + `mlflow` + `dotenv` + `app.loop.falsifier_core` (stdlib-only). All `app.*` and `scripts.*` imports deferred inside `main()` after coercion (D-07).

**Copied helpers (NOT imported from loop_falsifier):** `resolve_prod_url`, `assert_resolved_target`, `run_subprocess_or_infra`, `_snapshot_ids_from_url` — copied verbatim so `loop_falsifier.py` stays byte-for-byte unchanged.

## Verification Results

- `python -c "import ast; ast.parse(...)"` — exit 0 (AST OK)
- `ruff check scripts/loop_runner.py` — clean (All checks passed)
- `mypy scripts/loop_runner.py` — no errors in loop_runner.py (pre-existing `datetime.UTC` error in `coverage_agent.py` is transitive, not caused by this plan)
- `git diff --quiet scripts/loop_falsifier.py` — exit 0 (untouched)
- Line count: 684 (exceeds 250 minimum)
- `grep -c 'embedding_table != "place_embeddings_v2"'` — 1
- `grep -c "pending_after\|pending_before"` — 3 (>= 2)
- `grep -c "LOOP_GAP_NEIGHBORHOOD\|LOOP_GAP_CUISINE"` — 4 (>= 1)
- `grep -c "frozen_paraphrases_runner.json"` — 2 (>= 1)
- json.dump at line 522 < ingest_places_sf.py at line 541 (freeze before ingest)
- `grep -c "new_v2_ids\|place_embeddings_v2"` — 16 (>= 3)
- `grep -c '_snapshot_ids_from_url(sandbox_url, "places_raw")'` — 2 (before + after)
- `grep -c "new_place_count = len(new_raw_ids)"` — 1
- `grep -c "new_place_count == 0\|new_raw_ids"` — 4 (>= 1)
- `grep -c "embed_added_count == 0"` — 1 (>= 1)
- `grep -c "after_hit_at_k\|recall_at_k"` — 16 (>= 2)
- `grep -c "decide_loop_exit("` — 1 (>= 1)
- All log_dict lines (269-276) before all log_metric lines (278-283) — IN-05 satisfied

## Deviations from Plan

None — plan executed exactly as written.

The minor adaptation: the pre-existing `coverage_agent.py` mypy error (`datetime.UTC` is Python 3.11+ and mypy flags it) appears when mypy analyzes loop_runner.py transitively. This is pre-existing (present before this plan, absent from `loop_falsifier.py` check which has no transitive coverage_agent dependency). No fix applied per CLAUDE.md scope-boundary rule.

## Threat Coverage

| Threat ID | Mitigation Implemented |
|-----------|------------------------|
| T-19-03-01 | Step 1-4 coerce DATABASE_URL=sandbox → cache_clear → close_db_pool → assert BOTH resolve_database_url AND get_settings().resolved_database_url == sandbox BEFORE any miner import/proposal mutation |
| T-19-03-02 | get_settings.cache_clear() + close_db_pool() immediately after coercion; both resolution paths asserted |
| T-19-03-03 | Step 4b: settings.embedding_table == 'place_embeddings_v2' asserted after cache-clear, before any semantic_search call |
| T-19-03-04 | gap_mine_main reads DEMAND_DATABASE_URL for demand signal (read-only); all writes target sandbox DATABASE_URL |
| T-19-03-05 | MLflow logs v2 diff IDs + fixture_mode flag — attribution is auditable |
| T-19-03-06 | Operator-run gate only; CI never runs loop_runner (19-04 unit-tests at zero key cost) |
| T-19-03-SC | No new package installs |

## Threat Flags

None — no new network endpoints, auth paths, or schema changes beyond what the plan's threat model covers.

## Self-Check: PASSED

- [x] `scripts/loop_runner.py` exists — FOUND (684 lines)
- [x] Commit 8358e8a exists in git log — FOUND
- [x] AST parses — confirmed
- [x] ruff clean — confirmed
- [x] loop_falsifier.py untouched — confirmed
- [x] All module-scope imports are stdlib/mlflow/dotenv/falsifier_core only — confirmed
- [x] get_settings.cache_clear() (347) < gap_mine_main import (390) < gap_mine_main call (418) — confirmed ordering
- [x] embedding_table assertion present — confirmed (1 match)
- [x] frozen_paraphrases_runner.json freeze (522) before ingest subprocess (541) — confirmed
- [x] Two places_raw snapshots (before + after) — confirmed
- [x] LOOP_GAP_NEIGHBORHOOD/CUISINE baseline-miner-gap-contract present — confirmed (4 matches)
- [x] decide_loop_exit gate used (not reimplemented) — confirmed
- [x] All log_dict before all log_metric in log_to_mlflow — confirmed (IN-05)
