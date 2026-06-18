---
phase: 18
reviewers: [codex]
reviewed_at: 2026-06-18T06:27:27Z
plans_reviewed:
  - 18-01-sandbox-prereqs-PLAN.md
  - 18-02-demand-extraction-PLAN.md
  - 18-03-gap-scoring-cli-PLAN.md
  - 18-04-tests-make-docs-PLAN.md
---

# Cross-AI Plan Review — Phase 18 (Gap Mining)

## Codex Review

**Summary**

The plans are well structured and mostly aligned with Phase 18, especially the additive `coverage_agent.py` strategy, the catalog-valid seed-format focus, cold-start behavior, MLflow metrics, and test layering. I would not approve them as-is, though. There are a few high-impact correctness and safety gaps that could make faithful execution fail the phase goal: the supply check does not actually measure `(neighborhood, cuisine)` coverage, exact catalog seed proposals may be filtered out as "already covered," and sandbox-only writes are documented but not enforced in code.

**Strengths**

- Strong phase boundaries: the plans correctly leave `loop_falsifier.py`'s hardcoded `GAP` constant alone and build the production-loop path separately.
- Good reuse: extending `scripts/coverage_agent.py` is the right integration point, and the plans repeatedly protect the existing supply-only contract.
- Good correctness emphasis on exact query format: `"{cuisine} restaurants in {neighborhood} San Francisco"` and static catalog membership are called out in 18-03 Task 1.
- Good operational behavior: cold start exits 0, logs `gaps_found=0`, and inserts nothing.
- Good two-DB read design: `DEMAND_DATABASE_URL` via direct read-only `psycopg2.connect` avoids pool retargeting.
- Good test ambition: unit, smoke, functional, and integration coverage are planned, with integration gated by `APP_ENV=integration`.

**Concerns**

- **HIGH — 18-03 Task 1 does not score true `(neighborhood, cuisine)` supply.** It gates on per-cuisine supply only. That can miss the core gap case: plenty of Vietnamese restaurants city-wide, but zero in Outer Sunset. Phase 18's goal is underserved `(neighborhood, cuisine)` buckets, so supply must be counted for the pair, not just `cuisine:X`.

- **HIGH — 18-03 Task 2 may filter out every correct proposal.** The plan emits exact seed-format query text, then runs `filter_already_covered`. Research says that filter considers `build_seed_queries()`. Since every valid exact seed is already in `build_seed_queries()`, the valid proposal may be dropped before insert. This directly threatens GAP-03.

- **HIGH — sandbox write safety is not enforced.** 18-03 Task 2 and 18-04 Task 3 rely on `DATABASE_URL` pointing at sandbox, but no code guard verifies that. `make gap-mine` or `seed_demand_log.py` could write proposals or seed rows to the wrong DB if the operator environment is wrong.

- **MEDIUM — 18-02 Task 1/2 has an extraction shape conflict.** The helper returns one `str | None` neighborhood per message, but Task 2 requires multi-intent messages to emit multiple neighborhoods. That needs a `list[list[str]]` or structured extraction result.

- **MEDIUM — demand extraction is too LLM-dependent.** Without judge credentials, all neighborhood extraction becomes unmapped, even when messages contain exact catalog neighborhood names. Add lexical neighborhood matching before LLM fallback.

- **MEDIUM — unbounded single-batch LLM call.** `--days 14` can eventually mean many rows. The plan needs chunking, row caps, message truncation, and preferably a timeout/failure path that logs partial extraction rather than failing or silently no-oping.

- **MEDIUM — `CoverageStat` is the wrong structure for demand gaps.** Encoding `demand:{neighborhood}:{cuisine}` in `bucket` risks fragile parsing and conflates supply count with demand count. This should be an explicit `DemandGap` dataclass.

- **MEDIUM — 18-04 Task 2 integration test may be flaky and stateful.** It seeds demand but does not guarantee supply is below threshold. It also asserts/inserts a static catalog query text that may already exist, and cleanup by query text could delete a legitimate pending proposal.

- **MEDIUM — 18-01 Task 1 is operational but not reproducible.** Applying the sandbox migration manually fixes the current sandbox, but the plan does not update provisioning or add a durable migration check. Future sandboxes can regress.

- **LOW — prompt-injection mitigation is overstated.** `json.dumps` prevents malformed prompt formatting, but it does not prevent the model from following malicious text inside a user message. The impact is limited, but the threat model should phrase this honestly.

**Suggestions**

- Replace 18-03 Task 1's supply logic with a true pair-level supply count for each demanded `(neighborhood, cuisine)` bucket. Do this either in SQL with a demand-bucket CTE joined to `places_raw`, or in Python after fetching relevant rows.
- Add a `DemandGap` dataclass: `neighborhood`, `cuisine`, `place_count`, `demand_count`, `last_ingest`. Use it through `find_demand_gaps`, `gap_to_seed_query`, MLflow artifact logging, and tests.
- Split catalog validation from dedup filtering. Assert `query_text in build_seed_queries()` for loop compatibility, but do not let "exists in static seed catalog" count as already covered. Dedup only against already-ingested checkpoints and existing proposal rows.
- Add a hard write-target guard before `insert_pending` and in `seed_demand_log.py`. For example, resolve the active DB name and require `city_concierge_sandbox` or equality with `SANDBOX_DATABASE_URL`, with an explicit override only if intentionally supported.
- Make neighborhood extraction return multiple neighborhoods per row. First do exact lexical matching against `NEIGHBORHOODS`; only send ambiguous rows to the LLM in bounded chunks.
- Make the integration test deterministic: force a gap with a high `--min-places`, use an isolated sandbox transaction if possible, detect preexisting proposal rows, and only clean up rows inserted by the test.
- Update sandbox provisioning or add a `make sandbox-migrate`/verification path so the Phase 17 migration prerequisite is repeatable, not just a one-time human action.

**Risk Assessment**

Overall risk: **HIGH as written**.

The plan quality is strong, but three issues are phase-critical: true pair-level supply is missing, valid exact seed proposals may be filtered out, and sandbox-only writes are not enforced. Those are not cosmetic concerns; they can cause the miner to either miss real gaps, insert nothing, or write to the wrong target. With those corrected, the remaining risks drop to medium and are mostly around extraction robustness and integration-test determinism.

---

## Consensus Summary

Single external reviewer (Codex). The orchestrator independently verified the two
most damaging HIGH findings against the live `scripts/coverage_agent.py` source
before recording them, since plan-prose review can over- or under-call.

### Verified against code (orchestrator)

- **HIGH-2 (filter drops every valid proposal) — CONFIRMED, phase-critical.**
  `existing_query_texts()` (coverage_agent.py:216) seeds its set with
  `set(build_seed_queries())`, then `filter_already_covered` (line 233) drops any
  proposal whose `query_text` is in that set. The plan's `gap_to_seed_query(n,c)`
  is asserted (18-03 Task 1, Test 4) to be a **member** of `build_seed_queries()`.
  Therefore every valid proposal the miner emits is in `existing` and is dropped
  → **the miner inserts nothing on every run** → GAP-03 fails silently (and the
  D-04 cold-start path would mask it as a "success"). This is the single highest
  risk in the phase and the plan-checker missed it (it verified the seed-format
  assertion in isolation, not its interaction with the dedup set). The fix is
  Codex's "split catalog validation from dedup": keep the `build_seed_queries()`
  membership *assertion* for loop-compatibility, but dedup ONLY against
  checkpoints + existing proposal rows — never against the static catalog.

- **HIGH-1 (per-cuisine supply, not pair-level) — CONFIRMED as a real semantic
  gap.** This stems from RESEARCH.md "Open Questions #1", which deliberately chose
  per-cuisine supply for the gate. Codex is right that this misses the canonical
  "cuisine exists city-wide but not in this neighborhood" gap — which is exactly
  the demand-driven case Phase 18 exists to surface. The RESEARCH resolution is
  defensible only if `places_raw` lacks a clean neighborhood column; the planner
  should re-examine whether a pair-level count is feasible (it likely is, via the
  same neighborhood CTE `gather_stats` already uses) and prefer it.

- **HIGH-3 (sandbox write not enforced in code) — CONFIRMED as documented-not-coded.**
  D-05's safety is currently a convention ("the pool targets sandbox") plus a
  read-only guard on the *demand* connection. There is no guard on the *write*
  path. A pre-`insert_pending` assertion that the active DB is the sandbox (name
  match or `SANDBOX_DATABASE_URL` equality) turns the Phase-16 prod-safety rule
  into an enforced invariant rather than an operator promise.

### Agreed strengths

- Additive extension of `coverage_agent.py` (no regression of the supply-only contract).
- `loop_falsifier.py` `GAP` constant left untouched (D-03 honored).
- Exact seed-format discipline + catalog membership asserted.
- Cold-start no-op (exit 0, `gaps_found=0`).
- Read-only two-DB demand read via `DEMAND_DATABASE_URL`.
- Four-layer test pyramid with `APP_ENV=integration` gating.

### Agreed / highest-priority concerns (action before execute)

1. **Dedup vs. catalog** (HIGH-2) — must fix or the miner is a no-op. **Blocker.**
2. **Pair-level supply** (HIGH-1) — must fix or gaps are mis-scored against phase goal.
3. **Sandbox write guard** (HIGH-3) — must add to make D-05 enforceable.
4. Extraction shape (single vs. multi neighborhood) + lexical-before-LLM matching (MEDIUM).
5. Integration-test determinism + scoped cleanup (MEDIUM).

### Divergent views

None — single reviewer. The MEDIUM/LOW items (DemandGap dataclass, batch
chunking, sandbox-migrate reproducibility, prompt-injection honesty) are
quality/robustness improvements, not blockers, and can be folded into a replan
or deferred with a note.
