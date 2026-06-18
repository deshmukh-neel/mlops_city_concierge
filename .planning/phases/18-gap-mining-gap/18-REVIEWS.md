---
phase: 18
reviewers: [codex]
review_round: 2
reviewed_at: 2026-06-18T06:52:54Z
plans_reviewed:
  - 18-01-sandbox-prereqs-PLAN.md
  - 18-02-demand-extraction-PLAN.md
  - 18-03-gap-scoring-cli-PLAN.md
  - 18-04-tests-make-docs-PLAN.md
prior_round: 18-REVIEWS-round1-codex.md
---

# Cross-AI Plan Review — Phase 18 (Gap Mining) — ROUND 2

Re-review of the plans AFTER the round-1 `--reviews` replan. Round 1's three HIGH
findings (per-cuisine supply, static-catalog dedup blocker, unenforced sandbox
write) were the input to the revision; this round verifies those fixes and hunts
for anything the surgical edits introduced or round 1 missed.

## Codex Review (round 2)

### Fix Verification

**H1 pair-level supply: CONFIRMED.**
18-03 Task 1 correctly moves supply scoring to true `(neighborhood, cuisine)` pairs via `gather_pair_supply()` over `place_query_hits`, using exact canonical seed query text and `COUNT(DISTINCT place_id)`. That matches the ingest evidence table populated by `insert_query_hits()` and indexed on `query_text` in `scripts/db/init.sql:77`. The specified tests cover the important cases: city-wide cuisine presence no longer masks neighborhood absence, saturated pairs are excluded, zero-demand pairs are excluded, and output is structured as `DemandGap`.

**H2 static-catalog dedup blocker: CONFIRMED, with a new adjacent issue below.**
18-03 Task 2 fixes the original blocker: the demand path uses new `ingested_query_texts(conn)` and explicitly excludes `build_seed_queries()`, while `gap_to_seed_query()` separately asserts catalog membership. The acceptance tests that a catalog-valid seed survives `filter_already_covered(..., ingested_query_texts(conn))` directly prove the original "valid mined proposals all dropped" failure is addressed.

**H3 sandbox write guard: NOT RESOLVED.**
The plans correctly require guard calls before both demand seeding and proposal insertion: 18-01 Task 3 and 18-03 Task 2. The problem is the guard's pass condition: `dbname == city_concierge_sandbox OR dbname == SANDBOX_DATABASE_URL's dbname`. If `SANDBOX_DATABASE_URL` is mis-set to a non-sandbox/prod dbname, the guard whitelists that non-sandbox db. That violates the stated fix, "raises on a non-sandbox DB." Tighten this to require the active write target itself to be sandbox-safe, ideally by checking the actual write connection's `current_database()` and requiring `city_concierge_sandbox` or an explicitly sandbox-patterned normalized target, not merely equality to an env var.

### New Concerns

**HIGH: checkpoint dedup is probably wrong because checkpoints are prefixed.**
18-03 Task 2 says `ingested_query_texts()` returns `SELECT query_text FROM places_ingest_query_checkpoints`, but ingest writes checkpoints via `checkpoint_key(query)` as `FIELD_MODE::raw_query` in `scripts/ingest_places_sf.py:344` and `:784`. Proposals use raw query text. Literal set comparison will miss completed checkpoints like `all::vietnamese restaurants in Outer Sunset San Francisco`, insert a pending raw proposal, then ingest will skip it because `select_seed_queries_for_run()` checks `checkpoint_key(query)` in completed. Result: `proposals_inserted > 0` but the downstream loop may consume nothing. Fix: normalize checkpoint rows in `ingested_query_texts()` to include raw forms after `::`, and add a test with a real `all::<seed>` checkpoint.

**MEDIUM: Plan 04 integration determinism ignores checkpoint pre-existence.**
The integration test checks pre-existing proposal rows but not pre-existing checkpoint rows. Once checkpoint normalization is fixed, the hardcoded `Outer Sunset`/`vietnamese` seed may be deduped and the test will fail or skip unpredictably. The test should choose a catalog pair absent from both proposals and normalized checkpoints, or explicitly skip with a clear reason.

**MEDIUM: 18-01 depends on a future guard in 18-03.**
The lazy-import fallback note is directionally okay, but it is optional prose for a wave-0 plan that runs before the guard exists. Safer revision: move `assert_sandbox_write_target()` into 18-01 or a small shared module, then have both `seed_demand_log.py` and `coverage_agent.py` import the same guard. If keeping fallback, make lazy import mandatory and test the missing-guard path.

**MEDIUM: judge-unavailable behavior is internally inconsistent.**
18-02 correctly says lexical neighborhood extraction works without judge creds. 18-03 Task 2 Test 10 says `vibe.make_judge() returns None` should cold-start/no-proposal. That is only true for lexical misses. Clarify that judge absence must not suppress lexically mappable demand rows.

**LOW: 18-01 `.env.example` verification conflicts with commented default.**
The plan wants `DEMAND_DATABASE_URL` commented out, but the automated verify includes `grep -v '^#' .env.example | grep -c DEMAND_DATABASE_URL`, which fails if the line is correctly commented. Adjust the verify command.

### Strengths

- The `DemandGap` dataclass removes the earlier fragile `demand:n:c` bucket encoding.
- Multi-neighborhood extraction is now list-per-row and carried through to demand counting.
- `--top-n` after dedup is the right placement.
- Pair supply from `place_query_hits` is a defensible phase-18 signal and avoids stale `places_raw.source_query`.
- The plans preserve the existing supply-only `coverage_agent` path and tests.

### Risk Assessment

**Overall: HIGH as written.**
The core miner design is close, and H1/H2's original failures are substantially fixed. But H3 is still not a real sandbox guarantee, and the checkpoint-prefix mismatch can create false-positive pending proposals that the ingest immediately skips.

**Recommendation: another surgical revision before execution.** Fix the write guard, normalize checkpoint keys in `ingested_query_texts()`, and update the integration test to account for normalized checkpoints. After those changes, the plan set should be ready.

---

## Consensus Summary (orchestrator, round 2)

Single external reviewer (Codex). The orchestrator independently verified the new
HIGH finding and the H3 refinement against live source before recording them.

### Verified against code (orchestrator)

- **H1 / H2 original failures — fixes CONFIRMED.** Matches my own round-1 code
  trace; the revised plans address both correctly with proving tests.

- **NEW HIGH (checkpoint-prefix dedup mismatch) — CONFIRMED, execution-blocking.**
  Verified in `scripts/ingest_places_sf.py`:
  - `checkpoint_key(query)` (line 344) returns `f"{FIELD_MODE}::{query_text}"` —
    checkpoints are stored PREFIXED (e.g. `all::vietnamese restaurants in Outer
    Sunset San Francisco`), written via `checkpoint_key(query)` at line 784.
  - `get_completed_queries()` (line 487-493) reads `query_text` from the
    checkpoints table — i.e. the PREFIXED strings.
  - The ingest skip-check (line 742): `if checkpoint_key(query) not in completed
    and query not in completed` — ingest dedupes on the prefixed key.
  - The planned `ingested_query_texts()` would store the prefixed checkpoint
    string and compare it against the miner's RAW proposal `query_text`
    (`vietnamese restaurants in Outer Sunset San Francisco`) → NO match → the
    already-ingested pair is re-proposed as "new" → `insert_pending` writes it →
    ingest later computes `checkpoint_key` from the raw proposal, finds it IN
    `completed`, and SKIPS it. Net: `proposals_inserted > 0` but the loop ingests
    nothing for those pairs. This is the SAME class of silent-no-op as the
    original H2, just scoped to already-completed pairs. **Real bug.**
  - **Fix:** `ingested_query_texts()` must strip the `FIELD_MODE::` prefix from
    checkpoint rows (split on `::`, take the part after) so completed checkpoints
    dedupe the matching raw proposal. Proposals are already raw; only the
    checkpoint side needs normalizing. Add a test seeding a real `all::<seed>`
    checkpoint and asserting the matching gap is deduped (NOT re-proposed).

- **H3 refinement — VALID.** The guard accepting `SANDBOX_DATABASE_URL`'s dbname
  means a mis-set env var (pointed at prod) would whitelist a prod write. The
  robust form checks the ACTUAL write connection's `current_database()` and
  requires it to be the known sandbox name (or an explicitly sandbox-patterned
  target), not equality to a possibly-wrong env var. Cheap to do and closes the
  gap the round-1 fix left.

### Agreed strengths (carried from the revision)

- `DemandGap` dataclass (no string-keyed encoding); list-per-row multi-neighborhood
  extraction; `--top-n` after dedup; pair supply from `place_query_hits` (avoids
  stale `places_raw.source_query`); supply-only path + tests preserved.

### Must-fix before execute (round-2 blockers)

1. **Checkpoint-key normalization in `ingested_query_texts()`** (NEW HIGH) — strip
   `FIELD_MODE::` so completed checkpoints dedupe raw proposals; add the
   `all::<seed>` test.
2. **Write guard hardening** (H3 refinement) — check the write connection's
   `current_database()` against the known sandbox name, not just env-var equality.
3. **Integration test** (MEDIUM, coupled to #1) — pick a pair absent from BOTH
   proposals AND normalized checkpoints, or skip with a clear reason.

### Should-fix (cheap, fold into the same revision)

4. Guard ownership (MEDIUM) — move `assert_sandbox_write_target` into a shared
   module both `seed_demand_log.py` and `coverage_agent.py` import, removing the
   18-01→18-03 forward-reference; or make the lazy import mandatory + tested.
5. Judge-absence semantics (MEDIUM) — clarify that a `None` judge must NOT
   suppress lexically-mapped demand; only lexical misses are dropped.
6. `.env.example` verify command (LOW) — the `grep -v '^#'` check contradicts the
   commented-out `DEMAND_DATABASE_URL` default; fix the verify assertion.

### Divergent views

None — single reviewer. H1/H2 are settled (confirmed fixed). The round-2 items are
a tight, well-scoped second surgical pass — the design is sound; these are
correctness-of-wiring fixes, not a redesign.
