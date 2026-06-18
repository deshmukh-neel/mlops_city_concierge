---
phase: 18
reviewers: [codex]
review_round: 3
reviewed_at: 2026-06-18T07:57:12Z
plans_reviewed:
  - 18-01-sandbox-prereqs-PLAN.md
  - 18-02-demand-extraction-PLAN.md
  - 18-03-gap-scoring-cli-PLAN.md
  - 18-04-tests-make-docs-PLAN.md
prior_rounds:
  - 18-REVIEWS-round1-codex.md
  - 18-REVIEWS-round2-codex.md
---

# Cross-AI Plan Review — Phase 18 (Gap Mining) — ROUND 3

Third review round. Rounds 1-2 found and fixed 4 HIGHs (per-cuisine supply,
static-catalog dedup, unenforced sandbox write, checkpoint-prefix dedup) + the H3
env-var hole. This round verifies the round-2 fixes and hunts for anything new.

## Codex Review (round 3)

### Round-2 Fix Verification

| Item | Verdict | Proof in Plan |
|---|---|---|
| Checkpoint-prefix HIGH | **CONFIRMED**, with one related concern below | `18-03` Task 2 Test 3 requires `all::vietnamese restaurants in Outer Sunset San Francisco` to normalize to the raw seed and dedupe the mined proposal. `split("::", 1)[1]` handles raw queries containing later `::`, no-prefix rows as-is, and all `FIELD_MODE` values equally. |
| H3 write guard | **CONFIRMED** | `18-01` Task 3 Tests 1-5 require live `SELECT current_database()` and specifically prove a mis-set `SANDBOX_DATABASE_URL` naming prod cannot whitelist prod. `18-01` creates `scripts/sandbox_guard.py`; `18-01` Task 4 and `18-03` Task 2 import that one module. |
| MEDIUM-1 integration target | **CONFIRMED** | `18-04` Task 2 chooses a catalog pair absent from both existing proposals and `ingested_query_texts(conn)` normalized checkpoint set, or skips. It also forces the gap with high `--min-places`. |
| MEDIUM-3 judge absence | **CONFIRMED for the intended judge-none semantics** | `18-02` Task 1 Test 4, `18-02` Task 2 Test 8, and `18-03` Task 2 Test 11 prove lexical demand survives `llm=None`; cold-start fires only on empty mappable demand. |
| LOW env verify | **CONFIRMED** | `18-01` Task 1 verifies `grep -c '^# *DEMAND_DATABASE_URL' .env.example >= 1`, so the documented env var remains commented out by default. |

### New / Remaining Concerns

**HIGH: Free-text cuisine demand is still dropped when `requested_primary_types[]` is empty.**
`18-02` extracts cuisines only from `requested_primary_types[]` and uses the LLM only for neighborhoods. But the app intake prompt explicitly returns `[]` for free-text/no clear slot structure (`app/main.py:72`), so real rows like "vietnamese restaurants in Outer Sunset" can have an empty type list. Those rows would map neighborhood but no cuisine, increment `unmapped_count`, and never become demand gaps. This undercuts GAP-01's core "real demand" purpose. Minimal fix: add message-level cuisine fallback, preferably lexical CUISINES scan first, then batched LLM for unresolved cuisines, with tests for `requested_primary_types=[]`.

**MEDIUM: `ingested_query_texts` should filter checkpoints to completed only.**
The plans repeatedly say "completed checkpoint set," and ingest skips only completed checkpoints via `get_completed_queries(... WHERE status = 'completed')` (`scripts/ingest_places_sf.py:487`, `:738`). But `18-03` Task 2's action says select all rows from `places_ingest_query_checkpoints`. That can falsely suppress a proposal for an incomplete/budget-stopped query that ingest would retry. Minimal fix: `WHERE status = 'completed'` plus a regression test that an `incomplete` prefixed checkpoint does **not** dedupe the raw mined proposal.

**LOW: Guard and insert are not literally the same connection in `gap_mine_main`.**
Literal `18-03` calls `assert_sandbox_write_target()` and then `insert_pending(...)`; existing `insert_pending` opens its own pooled connection (`scripts/coverage_agent.py:236`). This is not a prod-whitelist bug because both borrow from the same pool target, and `ON CONFLICT` handles races, but it does not meet the strict wording "actual write connection." Cleaner: let `insert_pending` accept `conn=None`, call guard on that same `conn`, then insert on it.

### Strengths

The main round-1/round-2 structural fixes are solid: true pair-level supply from `place_query_hits`, exact seed-format emission, catalog assertion before proposal creation, dedup split from the static catalog, checkpoint-prefix normalization, `--top-n` after dedup, and a real test pyramid. Wave ordering is sound: `scripts/sandbox_guard.py` is created in wave 0 and imported in wave 2, with no same-wave ownership conflict.

### Execute-Readiness Verdict

**Risk: HIGH. Recommendation: REVISE before execution.**

Minimal must-fix list:
1. Add cuisine extraction from `message` when `requested_primary_types[]` maps to no CUISINES.
2. Restrict `ingested_query_texts` checkpoint dedup to `status='completed'`.
3. Prefer passing one write connection through guard + insert, or explicitly document/test the same-pool invariant.

---

## Consensus Summary (orchestrator, round 3)

Single external reviewer (Codex). All five round-2 fixes independently CONFIRMED.
The orchestrator verified the two new findings against live source before recording.

### Verified against code (orchestrator)

- **NEW HIGH (free-text cuisine dropped) — CONFIRMED, real recall gap.**
  - `app/main.py` slot-intake prompt (~line 77): *"If the message is free-text or
    has no clear slot structure, return []."* So `requested_primary_types[]` is
    legitimately empty for exactly the conversational queries the miner most wants.
  - The round-2 plan (18-02 line 17, `_types_to_cuisines` at line 92) maps cuisine
    **only** from `requested_primary_types[]` — there is a lexical→LLM two-tier for
    NEIGHBORHOODS but **no equivalent tier for CUISINE**. The asymmetry is the bug:
    a row like "vietnamese restaurants in Outer Sunset" with empty types maps the
    neighborhood, gets no cuisine, lands in `unmapped_count`, and never becomes a
    gap. That directly undercuts GAP-01 ("real demand"). Rounds 1-2 (and my own
    reviews) missed it because they focused on dedup/supply/safety mechanics, not
    extraction RECALL.
  - **Fix:** add a cuisine fallback symmetric to the neighborhood path — lexical
    scan of `message` against `CUISINES` first, then fold unresolved-cuisine rows
    into the existing batched LLM call (extend it to extract BOTH neighborhood and
    cuisine, or a second batched call), constrained to the catalog. Add tests with
    `requested_primary_types=[]` proving a free-text cuisine row becomes a gap.

- **MEDIUM (checkpoint status filter) — CONFIRMED, same recurring class.**
  - `get_completed_queries()` (ingest_places_sf.py:487) filters `WHERE status =
    'completed'`; the ingest skip-check (line ~742) dedupes only against completed
    checkpoints. A non-completed checkpoint (budget-stopped / `incomplete`) is NOT
    skipped by ingest — it WILL be retried.
  - The round-2 `ingested_query_texts` selects ALL checkpoint rows, so it would
    suppress a proposal for a pair the ingest would actually re-run. This is the
    SAME "miner's already-ingested view diverges from ingest's real skip logic"
    class that produced the round-2 HIGH — now in the status dimension.
  - **Fix:** `ingested_query_texts` checkpoint SELECT must add `WHERE status =
    'completed'`; add a test that an `incomplete` prefixed checkpoint does NOT
    dedupe the matching raw proposal (it should still be proposed).

- **LOW (guard vs insert connection) — VALID but minor.** `insert_pending` opens
  its own pooled connection; the guard runs on a separately-resolved connection.
  Both target the same pool `DATABASE_URL`, so this is not a prod-write hole, and
  `ON CONFLICT` handles races. It only fails the strict "same write connection"
  wording. Cheapest honest fix: thread one `conn` through guard → insert (give
  `insert_pending` a `conn=None` param), OR document + test the same-pool
  invariant explicitly. Acceptable to take the documentation route.

### Agreed strengths (now stable across 3 rounds)

True pair-level supply from `place_query_hits`; exact seed-format emission with
catalog assertion; dedup split from the static catalog + checkpoint-prefix
normalization; `current_database()`-enforced sandbox write guard; single shared
guard module with sound wave ordering; full unit/smoke/functional/integration
pyramid; supply-only path + W5 tests preserved; `loop_falsifier.GAP` untouched.

### Must-fix before execute (round-3)

1. **Cuisine recall** (NEW HIGH) — symmetric message-level cuisine fallback
   (lexical CUISINES scan → batched LLM) so free-text `requested_primary_types=[]`
   rows still become gaps; tests with empty types.
2. **Checkpoint status filter** (MEDIUM) — `ingested_query_texts` dedupes only
   `status='completed'` checkpoints; `incomplete`-checkpoint regression test.

### Should-fix (cheap)

3. **Guard/insert connection** (LOW) — thread one `conn` through guard + insert,
   or document/test the same-pool invariant.

### Divergent views

None — single reviewer. Note the trend: each round's finding count and severity is
falling (r1: 3 HIGH; r2: 1 new HIGH + 1 refine; r3: 1 new HIGH + 1 MEDIUM + 1 LOW),
and the r3 HIGH is a RECALL gap (miner finds fewer gaps than it could), not a
correctness/safety failure (miner does the wrong thing). After round 3's two
fixes, the remaining surface is small.
