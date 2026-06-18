---
phase: 18
reviewers: [codex]
review_round: 4
reviewed_at: 2026-06-18T17:07:03Z
verdict: EXECUTE NOW
plans_reviewed:
  - 18-01-sandbox-prereqs-PLAN.md
  - 18-02-demand-extraction-PLAN.md
  - 18-03-gap-scoring-cli-PLAN.md
  - 18-04-tests-make-docs-PLAN.md
prior_rounds:
  - 18-REVIEWS-round1-codex.md
  - 18-REVIEWS-round2-codex.md
  - 18-REVIEWS-round3-codex.md
---

# Cross-AI Plan Review — Phase 18 (Gap Mining) — ROUND 4 (FINAL GATE)

Fourth review round, run as an explicit final execute-readiness gate with a high
bar (do not inflate nitpicks; decisive execute-vs-revise verdict). Rounds 1-3
found and fixed 5 HIGH + supporting issues; this round checks the round-3 fixes
and looks for any genuine remaining must-fix.

**Verdict: EXECUTE NOW. Zero remaining HIGH or MEDIUM findings.**

## Codex Review (round 4)

### Round-3 Fix Verification

| Fix | Verdict | Proving Test |
|---|---|---|
| Cuisine-recall HIGH | CONFIRMED | `18-02 Task 1` Tests 2/8 and `18-02 Task 2` Tests 7/8/9 prove empty `requested_primary_types=[]` free-text rows still map cuisine from `message` via `_lexical_cuisines` or batched LLM. `18-04 Task 1 Functional 1` repeats this through `gap_mine_main`. |
| Checkpoint-status MEDIUM | CONFIRMED | `18-03 Task 2` Tests 3/4 prove completed prefixed checkpoints dedupe, but `incomplete` checkpoints do not. This matches ingest's real `WHERE status = 'completed'` skip logic. |
| Guard/insert same-connection LOW | CONFIRMED | `18-03 Task 2` Test 5 proves `assert_sandbox_write_target(conn)` and `insert_pending(..., conn=conn)` use the same connection object. Tests 7/8 prove guard failure prevents inserts. |

### Remaining Concerns

No remaining HIGH or MEDIUM findings.

**LOW:** Cartesian pairing can overcount noisy demand for messages like "sushi in Mission and tacos in Outer Sunset," producing false pairs. For this phase, that is acceptable because the output is a demand-count ranking signal, catalog-constrained, supply-gated, and deduped before insert. It is not a correctness/safety blocker.

**LOW:** `_extract_demand_batch` exception behavior is not explicitly tested. If `llm.invoke()` raises, the run should fail before writes, so there is no partial-write hazard. During execution, add a small `try/except` to degrade LLM-miss rows to unmapped while preserving lexical rows.

### Strengths

The recurring ingest-contract class is now well covered: static catalog is not used for demand dedup, checkpoint prefixes are normalized, checkpoint status matches ingest skip logic, and proposal writes are sandbox-guarded on the same connection.

Pair-level supply via `place_query_hits.query_text` is the right correction. It catches the real gap class: cuisine exists city-wide but not for the demanded neighborhood seed.

### Execute-Readiness Verdict

**EXECUTE NOW.**

The plans are sound. Remaining items are LOW implementation polish and can be handled during execution without another revision round.

---

## Consensus Summary (orchestrator, round 4 — FINAL)

Single external reviewer (Codex). All three round-3 fixes CONFIRMED. **Zero HIGH or
MEDIUM findings remain. Verdict: EXECUTE NOW.** No code re-verification was needed
this round — there were no HIGH/MEDIUM claims to ground-check, only two honestly-
classified LOW polish items.

### Review arc (4 rounds)

| Round | Findings | Character |
|-------|----------|-----------|
| 1 | 3 HIGH | Correctness/safety: per-cuisine→pair supply, static-catalog dedup blocker, unenforced sandbox write |
| 2 | 1 new HIGH + 1 refine | Correctness (checkpoint-prefix dedup) + safety (guard env-var hole) |
| 3 | 1 HIGH + 1 MEDIUM + 1 LOW | Recall (free-text cuisine) + status-filter edge + connection wording |
| 4 | 0 HIGH, 0 MEDIUM, 2 LOW | Polish only → **EXECUTE NOW** |

Severity and category both converged monotonically: "does the wrong thing" → "finds
less than it could" → "optional polish." The recurring failure class (miner's
already-ingested/covered view diverging from the ingest's real skip logic), which
produced a HIGH in rounds 2 and 3, is now explicitly closed on all three dimensions
the ingest cares about — static-catalog exclusion, `FIELD_MODE::` prefix
normalization, and `status='completed'` filtering.

### LOW items — handle during execution (NOT blocking, no replan)

1. **Cartesian over-pairing** — multi-slot messages can create false
   `(neighborhood, cuisine)` pairs. Acceptable: the signal is a demand-COUNT
   ranking, catalog-constrained + supply-gated + deduped before insert, so false
   pairs that have real supply are filtered and those without become low-ranked
   noise. A per-slot pairing refinement is a Phase-19+ improvement if the simple
   count proves too coarse (already noted in CONTEXT.md deferred ideas as "tuned
   demand/supply ratio scoring").
2. **`_extract_demand_batch` exception path** — wrap `llm.invoke()` in a
   `try/except` during execution so an LLM error degrades LLM-miss rows to
   `unmapped_count` while preserving lexically-resolved demand, rather than
   crashing the run. No partial-write risk either way (the error precedes all
   writes). The executor should add this when implementing the batch helper.

### Decision

Planning is complete and gate-passed. Recommend proceeding to
`/gsd-execute-phase 18`. The two LOW items are captured here for the executor;
neither warrants a fifth review round or another replan.
