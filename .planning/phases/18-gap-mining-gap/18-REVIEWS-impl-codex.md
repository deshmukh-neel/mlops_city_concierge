# Phase 18 Implementation Review — Codex

## HIGH

**CDX-H1 — `scripts/coverage_agent.py:678`, `scripts/coverage_agent.py:710`, `scripts/coverage_agent.py:914`**

`insert_pending(..., conn=None)` still self-opens `get_conn()` and writes without calling `assert_sandbox_write_target()` on that connection. `gap_mine_main()` uses the safe same-connection path, but the shipped function still exposes a real prod-write path via direct use or the legacy `main()` path, which calls `insert_pending(kept, args.dry_run)` with no guard. If `DATABASE_URL` points at prod, this can insert proposal rows into prod.

Suggested fix: move the guard into `insert_pending()` for every non-dry-run write path, immediately before the INSERT on the exact connection used. `gap_mine_main()` may keep its explicit guard, but `insert_pending()` should fail closed too.

## MEDIUM

**CDX-M1 — `scripts/coverage_agent.py:197`, `scripts/coverage_agent.py:214`, `app/main.py:81`**

Cuisine extraction is too literal and will misclassify common real demand. `_types_to_cuisines()` maps `"Steak House"` to `"steak house"`, but the catalog uses `"steakhouse"`, while the `/chat` intake prompt explicitly emits `Steak House`. `_lexical_cuisines()` also uses raw substring matching, so `"Thailand"` counts as `thai`, while `"dim sum"` does not count as catalog `dimsum`. This can create false gap demand or drop valid demand, especially when `judge=None`.

Suggested fix: add a catalog alias/canonicalization layer and use word/phrase-boundary regex matching. Include aliases like `steak house -> steakhouse`, `dim sum -> dimsum`, and avoid matching cuisine tokens inside unrelated words.

**CDX-M2 — `tests/integration/test_gap_miner.py:113`, `tests/integration/test_gap_miner.py:185`, `tests/integration/test_gap_miner.py:249`**

The integration test does not strictly clean up only its own proposal row. It checks that the proposal did not pre-exist, runs the miner, then deletes by exact `query_text`. If another process inserts the same proposal between the pre-check and cleanup, the test assertion can pass on that concurrent row and then delete it.

Suggested fix: serialize this test with a DB advisory lock, or make cleanup conditional on a test-owned marker that can be verified before deletion. If schema cannot identify ownership, prefer advisory locking around pair selection, miner execution, assertion, and cleanup.

## LOW

**CDX-L1 — `tests/unit/test_gap_miner.py:1025`**

`test_incomplete_checkpoint_does_not_dedupe` mocks checkpoint rows away instead of proving the implementation filters `WHERE status = 'completed'`. The test would still pass if `ingested_query_texts()` selected all checkpoints and the stub simply returned no rows.

Suggested fix: make the stub return an incomplete checkpoint unless the SQL includes the completed-status predicate, then assert the raw seed is absent.

**CDX-L2 — `scripts/seed_demand_log.py:83`**

`seed_demand_rows()` claims callers can mutate the returned rows safely, but it returns only a shallow list copy. Mutating a returned row dict or its `requested_primary_types` list mutates the module-level fixture.

Suggested fix: return deep copies, or at least copy each dict and nested list.

Verdict: DO-NOT-SHIP.

---

## Orchestrator disposition (2026-06-18)

Each finding evaluated against the code and the phase's locked decisions (D-05 write-policy split, "supply-only path UNCHANGED" guardrail). Codex's `DO-NOT-SHIP` verdict was driven by CDX-H1; on inspection that finding is mis-scoped (see below), so the corrected verdict is **SHIP-WITH-FIXES (all applied)**.

| ID | Severity | Disposition | Action |
|----|----------|-------------|--------|
| CDX-H1 | HIGH→re-scoped | Suggested fix DECLINED; defensive doc added | The demand path (`gap_mine_main`) IS guarded on the same connection (coverage_agent.py:863-864). The unguarded `insert_pending`/`main()` path is **pre-existing supply-only behavior** (verified at base `a39d8f7`) that intentionally writes to `DATABASE_URL` and is NOT sandbox-scoped by design (D-05). Guarding `insert_pending` internally (codex's fix) would BREAK the legitimate supply-only path and violate the "supply-only UNCHANGED" guardrail. Added an explicit WRITE-TARGET POLICY docstring on `insert_pending` so future demand callers guard their own conn. Not a shippable-blocking bug. |
| CDX-M1 | MEDIUM | FIXED | Real recall bug: `"Steak House"` (emitted by app/main.py slot intake) normalized to `"steak house"` and missed catalog `"steakhouse"`. Added `_CUISINE_ALIASES` canonicalization (`steak house→steakhouse`, `dim sum→dimsum`) wired into `_types_to_cuisines` + `_lexical_cuisines`. `"Fine Dining Restaurant"`→no cuisine confirmed CORRECT (not in catalog, like Bar). Tests added. |
| CDX-M2 | MEDIUM | ACCEPTED | Integration-test concurrent-insert race. Real but low-impact: single-operator APP_ENV-gated sandbox; advisory locking is over-engineering for a capstone. Marker-scoped cleanup + pre-existence check already mitigate the common case. |
| CDX-L1 | LOW | FIXED | `test_incomplete_checkpoint_does_not_dedupe` was vacuous (empty stub). Rewrote with a status-aware stub that leaks the incomplete row ONLY when the `status='completed'` predicate is absent. Mutation-verified: stripping the filter now FAILS the test. |
| CDX-L2 | LOW | FIXED | `seed_demand_rows()` returned a shallow `list()` copy (shared inner dicts/lists). Now `copy.deepcopy`. |

Verification after fixes: 1603 unit pass / 9 skipped (was 1601), ruff clean, integration still skips without APP_ENV.
