# Phase 18: Gap Mining (GAP) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-17
**Phase:** 18-gap-mining-gap
**Areas discussed:** Miner shape & reuse, Demand→gap definition, Loop integration contract, Run mode & data source

---

## Miner shape & reuse

| Option | Description | Selected |
|--------|-------------|----------|
| Extend coverage_agent.py | Add a demand CTE to gather_stats + demand/supply scorer; reuse propose/filter/insert/MLflow | (recommended) |
| New gap_miner.py module | Fresh script, import coverage_agent helpers; cleaner separation, risks duplication | |
| You decide | Let researcher/planner pick after reading coverage_agent + W5 tests | ✓ |

**User's choice:** You decide
**Notes:** Captured as Claude's Discretion with a RECOMMENDATION to extend `coverage_agent.py` (DRY — the propose→filter→insert→MLflow pipeline is exactly what GAP needs; demand path is the smallest honest change), guarded by keeping the existing supply-only path + W5 tests intact. Net-new `gap_miner.py` is the acceptable fallback if the demand CTE doesn't slot cleanly.

---

## Demand→gap definition (1/3): Extraction strategy

| Option | Description | Selected |
|--------|-------------|----------|
| LLM extraction (reuse vibe judge) | Batched messages → vibe.make_judge() → (neighborhood, cuisine) tuples; handles messy text | ✓ |
| Deterministic parse | Regex/keyword match against NEIGHBORHOODS + CUISINES; deterministic + free, misses long tail | |
| Hybrid | Deterministic first, LLM fallback for unmatched | |

**User's choice:** LLM extraction (reuse vibe.make_judge())
**Notes:** Consistent with coverage_agent's existing LLM-proposal pattern. → D-01.

---

## Demand→gap definition (2/3): Gap score

| Option | Description | Selected |
|--------|-------------|----------|
| Demand>0 AND supply<threshold | Gap iff ≥1 user query AND place_count < min_places; rank by demand_count desc | ✓ |
| Demand/supply ratio score | score = demand/(place_count+1), top-N; nuanced but tuned formula | |
| Pure demand ranking | Rank by demand only, ignore supply | |

**User's choice:** Demand>0 AND supply<threshold
**Notes:** Most honest + explainable for the capstone; reuses coverage_agent's existing find_gaps threshold. → D-02.

---

## Demand→gap definition (3/3): Noise / unmappable demand

| Option | Description | Selected |
|--------|-------------|----------|
| Drop unmappable, count what maps | LLM returns only confident tuples; drop vague, log unmapped_count | (recommended) |
| Allow off-catalog buckets | Let LLM propose buckets outside the static catalogs; richer but breaks seed contract | |
| You decide | Let planner pick after seeing premark_seed_isolation's catalog requirement | ✓ |

**User's choice:** You decide
**Notes:** Captured as Claude's Discretion with a RECOMMENDATION to drop unmappable + constrain to catalog membership, logging `unmapped_count`. HARD REASON surfaced from code: `loop_falsifier.premark_seed_isolation` requires the seed to be in `build_seed_queries()` (exits INFRA otherwise) — off-catalog gaps would break the loop's seed contract. → Claude's Discretion + D-03 context.

---

## Loop integration contract

| Option | Description | Selected |
|--------|-------------|----------|
| Write proposals (canonical loop path) | Top-N gap seeds → pending rows in places_ingest_query_proposals (coverage_agent's path); ingest consumes them | ✓ |
| Replace falsifier's GAP constant | Falsifier imports get_top_gap(); couples deterministic gate to non-deterministic miner — likely wrong | |
| Both: proposals + a callable | Write proposals AND expose accessor behind a flag; most surface area | |

**User's choice:** Write proposals (canonical loop path)
**Notes:** Resolves the "replaces the constant" phrasing — the PRODUCTION loop's gap selection is now demand-driven; the falsifier's constant stays as its own deliberately-independent reproducible stub (Phase 16 D-01). → D-03.

---

## Loop integration — cold start

| Option | Description | Selected |
|--------|-------------|----------|
| Insert nothing, exit 0 | No demand → no gaps → zero proposals, log gaps_found=0, exit 0; honest no-op | (recommended) |
| Fall back to supply-only gaps | Degrade to coverage_agent's supply-only find_gaps; keeps loop productive, blurs demand story | |
| You decide | Let planner decide based on code sharing | ✓ |

**User's choice:** You decide
**Notes:** Captured as Claude's Discretion with a RECOMMENDATION for honest no-op (insert nothing, exit 0). Supply-only fallback rejected — it would re-create what coverage_agent already does and muddy the demand-driven story. → D-04.

---

## Run mode & data source (1/2): Data source

| Option | Description | Selected |
|--------|-------------|----------|
| Read demand from prod, write proposals to sandbox | Real demand lives in prod (read-only SELECT); all ingest stays sandboxed; needs two connections | (recommended) |
| Sandbox-only (mirror Phase 16) | Read + write all against sandbox; demand seeded into sandbox user_query_log; simplest + safest, thinner slice | |
| You decide | Let planner pick after checking prod row counts + two-DB threading feasibility | ✓ |

**User's choice:** You decide
**Notes:** Captured as D-05 (RECOMMENDED, researcher to confirm). Read prod demand (reads are allowed — Phase 16's rule is only "never WRITE prod places_raw"), write/measure against sandbox. CONSTRAINT flagged: Phase 16 D-10's single-`DATABASE_URL` injection model doesn't cleanly support two live connections; sandbox-only is the acceptable fallback if prod log is near-empty or two-DB threading proves invasive.

---

## Run mode & data source (2/2): CLI/ops conventions

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror coverage_agent CLI | --days, --dry-run, --min-places, +--top-n; argparse main()→exit | (Claude default) |
| make gap-mine target | Make wrapper documented in CLAUDE.md, like make loop-falsifier | (Claude default) |
| MLflow under coverage_agent exp | Log gaps_found, proposals_inserted, demand_rows_scanned, unmapped_count + gap artifact | (Claude default) |
| Dry-run default ON | Opt-in --apply to write; safer but diverges from coverage_agent | (not chosen) |

**User's choice:** "i honestly have no idea" — no preference
**Notes:** These are house-style conventions, not vision calls. Captured as Claude's Discretion using coverage_agent's established style: mirror its CLI (+--top-n), new `make gap-mine` target, MLflow under the coverage_agent experiment, and KEEP the `--dry-run` opt-out (writes by default) rather than inventing opt-in `--apply` — safety lives in the sandbox write-target guard (D-05), not the flag.

---

## Claude's Discretion

- **Miner shape:** RECOMMENDED extend `coverage_agent.py` (DRY); fallback net-new `gap_miner.py`. Guardrail: keep supply-only path + W5 tests intact.
- **Unmappable demand:** RECOMMENDED drop + constrain to catalog membership, log `unmapped_count`. Hard reason: loop seed-isolation requires catalog membership.
- **Cold start:** RECOMMENDED honest no-op (insert nothing, exit 0).
- **Data source (D-05):** RECOMMENDED read prod / write sandbox; researcher to confirm two-DB feasibility; sandbox-only fallback.
- **CLI/ops:** mirror coverage_agent CLI + `make gap-mine` + coverage_agent MLflow exp + keep `--dry-run` opt-out.

## Deferred Ideas

- Off-catalog gap discovery (blocked by the loop's catalog-membership seed contract) → Phase 19+.
- Productionized ingest→embed→metric loop (LOOP-01..03) + hit@k/recall@k scorer (METRIC) → Phase 19.
- Tuned demand/supply ratio scoring (vs. the simple gate in D-02) → revisit in Phase 19's metric.
- Wiring the miner into the falsifier's gap selection behind a flag → explicitly rejected for this phase (D-03).
