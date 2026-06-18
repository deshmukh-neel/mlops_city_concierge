# Phase 18: Gap Mining (GAP) - Research

**Researched:** 2026-06-17
**Domain:** Python script extension — demand/supply gap mining over PostgreSQL, psycopg2, MLflow, LangChain
**Confidence:** HIGH — all critical questions answered from live code inspection and direct DB probes

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** LLM extraction via `vibe.make_judge()` for turning `user_query_log` rows into `(neighborhood, cuisine)` demand tuples.
- **D-02:** Gap = `demand_count > 0` AND `place_count < min_places` (default 5). Rank by `demand_count` descending.
- **D-03:** Miner writes `pending` rows to `places_ingest_query_proposals` via `coverage_agent.insert_pending`'s existing path. Does NOT touch `loop_falsifier.py`'s `GAP` constant.
- **D-04:** Cold start (empty or zero-mappable `user_query_log`) = insert nothing, log `gaps_found=0`, exit 0.
- **D-05 (researcher to confirm feasibility):** Prod-read + sandbox-write split preferred. See Q1/Q2 answers below for the confirmed decision rule.

### Claude's Discretion

- **Miner shape:** RECOMMENDED: extend `scripts/coverage_agent.py`. Researcher confirms this below (Q3).
- **Unmappable demand:** Drop + log `unmapped_count`. Constrain to `NEIGHBORHOODS × CUISINES` catalog.
- **CLI / ops:** `--days`, `--dry-run` (opt-out), `--min-places`, `--top-n`. `make gap-mine` target. MLflow under `coverage_agent` experiment. Log `gaps_found`, `proposals_inserted`, `demand_rows_scanned`, `unmapped_count` + ranked gap artifact.
- **`--dry-run` stays opt-out** (writes by default); safety lives in sandbox write-target guard.

### Deferred Ideas (OUT OF SCOPE)

- Off-catalog gap discovery (Phase-19+ extension blocked by loop's catalog-membership seed contract).
- Productionized ingest→embed→metric loop (LOOP-01..03) + productionized hit@k/recall@k scorer (METRIC) — Phase 19.
- Tuned demand/supply ratio scoring.
- Wiring miner into falsifier's gap selection behind a flag.
</user_constraints>

---

## Summary

Phase 18 builds the demand-driven half of `coverage_agent.py`. The supply-side pipeline (`gather_stats → find_gaps → propose_queries → filter_already_covered → insert_pending → log_to_mlflow`) already exists and is fully tested. The new work is a **demand CTE over `user_query_log`** and a **demand-gated gap scorer** that merges with the existing supply buckets. Six critical open questions were all answered from live code inspection and direct DB probes.

The **local `user_query_log` is empty (0 rows) in both the `city_concierge` and `city_concierge_sandbox` DBs** — Phase 17 logging just shipped and no real `/chat` traffic has run against the local instance. The prod Cloud SQL DB cannot be reached from this environment (no IAP tunnel open). This makes the D-05 two-DB path unnecessary for local development, and the sandbox-only fallback (seed demand rows into sandbox `user_query_log` for testing) is the correct development path. The planner must include a Wave 0 task that adds the `user_query_log` migration to the sandbox.

The `get_conn()` / `db_pool` architecture uses a single `ThreadedConnectionPool` keyed to a single `DATABASE_URL`. Opening two simultaneous pools (one for prod read, one for sandbox write) requires either a new `get_conn(url=...)` overload that bypasses the pool, or two separate processes. The former is ~15 lines of new code; the decision rule is given in Q2.

The demand extraction path is clean: `requested_primary_types[]` stores Title-Case Google primary_type values (`"Vietnamese Restaurant"`, `"Bar"`) that can be lexically mapped to `CUISINES` strings (`"vietnamese"`) without an LLM call for most rows, with LLM fallback only for rows where `requested_primary_types` is empty and the `message` needs parsing. This is more efficient than the CONTEXT.md's one-call-per-run framing suggests and is documented as Q4.

**Primary recommendation:** Extend `scripts/coverage_agent.py` with a `gather_demand(days, conn_or_url)` function and a `merge_demand_supply_gaps` scorer. Add `user_query_log` migration to the sandbox in Wave 0. Use sandbox-only for local dev; enable prod-read path behind a `DEMAND_DATABASE_URL` env var (the clean two-DB pattern that avoids pool conflicts).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Demand signal read (user_query_log) | Database / Storage | Script layer | Raw SELECT over indexed `created_at`; pure SQL, no app logic |
| Supply signal read (places_raw counts) | Database / Storage | Script layer | Existing `gather_stats` CTEs; unchanged |
| Demand → (neighborhood, cuisine) mapping | Script layer (LLM + lexical) | — | `requested_primary_types` → CUISINES is lexical; `message` fallback is LLM |
| Gap ranking (demand × supply merge) | Script layer | — | Pure Python merge of two dicts; no DB round-trip |
| Proposal write (places_ingest_query_proposals) | Database / Storage | Script layer | Reuses `insert_pending`'s `ON CONFLICT DO NOTHING` INSERT |
| MLflow telemetry | External service | Script layer | Reuses `log_to_mlflow` pattern; same `coverage_agent` experiment |
| CLI / Make target | Script layer | — | `argparse main() → int` + `make gap-mine` |
| Sandbox/prod DB routing | Script layer | OS environment | `DATABASE_URL` coercion pattern from Phase 16 |

---

## Critical Open Questions — Answered

### Q1 — Prod `user_query_log` row count

**Finding:** Cannot reach prod Cloud SQL from this environment — no IAP tunnel is open (memory note: tunnel to localhost:5050, not 5000; AirPlay blocks 5000). Prod Cloud SQL instance is `mlops--city-concierge` (double-dash); the DB inside is single-dash. [VERIFIED: live DB probe]

**Local probe results:**
- `city_concierge` (local docker, the "prod-mirror" local DB): **0 rows** in `user_query_log`. Migration `d1be72aea7d4` is applied (table exists).
- `city_concierge_sandbox` (sandbox): `user_query_log` table does **not exist** — the Phase 17 migration has not been applied to the sandbox.

**Decision rule for planner:** Use this threshold to choose D-05 mode:

| Condition | Mode | Action |
|-----------|------|--------|
| Prod has ≥ 50 usable rows AND IAP tunnel is open | Prod-read + sandbox-write | Use `DEMAND_DATABASE_URL` pattern (see Q2) |
| Prod has < 50 rows OR tunnel unavailable | Sandbox-only | Seed test rows into sandbox `user_query_log`; miner reads + writes same sandbox |
| Local dev / CI | Sandbox-only always | No prod access in CI |

50 rows is a reasonable floor because with 30 neighborhoods × 45 cuisines = 1,350 catalog buckets, a demand signal needs at least O(50-100) queries to produce a non-trivial ranking. Below that, the cold-start path (D-04) will likely fire anyway.

**Current state:** Sandbox-only is the correct dev/CI path. The prod-read path should be implemented but behind a `DEMAND_DATABASE_URL` env var (see Q2) so it activates only when the tunnel is open and the table is populated.

---

### Q2 — Two-DB feasibility

**Finding:** The pool architecture is a single `ThreadedConnectionPool` keyed to `(url, min, max)` in `_pool_config`. Calling `get_conn()` a second time after the pool is initialized simply returns from the same pool — there is no way to open a second URL without reinitializing the pool. [VERIFIED: `app/db_pool.py` lines 36-51]

`init_db_pool` raises `RuntimeError` if called with different parameters than the existing pool. `close_db_pool()` + reinit would switch the pool, but would break the sandbox write path if done mid-run.

**Two clean options:**

**Option A — `DEMAND_DATABASE_URL` env var + direct `psycopg2.connect`** (recommended):
- Add a `get_demand_conn(url: str)` helper in `coverage_agent.py` that opens a **direct, non-pooled** `psycopg2.connect(url)` connection for read-only demand queries.
- The pool continues targeting `DATABASE_URL` (sandbox). Demand reads are outside the pool and closed immediately after the demand CTE.
- Diff surface: ~15 lines in `coverage_agent.py` (one helper function + one context manager call in `gather_demand`).
- If `DEMAND_DATABASE_URL` is unset, fall back to reading `user_query_log` from the same pool connection (sandbox).

**Option B — subprocess isolation** (Phase 16 D-10 pattern):
- Shell out a tiny SQL query to prod and capture stdout. Avoids any in-process pool conflict.
- Overkill for a SELECT; harder to test.

**Recommendation:** Option A. `psycopg2.connect` is already a transitive dependency (the pool uses it). The helper is ~15 lines and fully unit-testable with a mock. The prod-safety guard principle from Phase 16 still applies: the `DEMAND_DATABASE_URL` is read-only by convention (SELECT only in `gather_demand`); the write path (`insert_pending`) always uses the pool which targets sandbox `DATABASE_URL`.

**Estimated diff surface:** 15 lines new code in `coverage_agent.py` + 1 env var in `.env.example`.

---

### Q3 — Demand path slot-in analysis

**Finding:** `coverage_agent.py` is 328 lines. The pipeline is `gather_stats → find_gaps → propose_queries → filter_already_covered → insert_pending → log_to_mlflow`. Each function is independent and pure (takes inputs, returns outputs, no module-scope state). [VERIFIED: `scripts/coverage_agent.py`]

**Where the demand path attaches:**

The cleanest extension adds **one new function** `gather_demand(days, url=None) -> dict[tuple[str,str], int]` that returns `{(neighborhood, cuisine): demand_count}` — a dict of catalog-member buckets with their demand counts. This function is called in `main()` alongside `gather_stats`, not inside it.

Then `find_gaps` is extended (or a new `find_demand_gaps` is added) to take both `stats` (supply) and `demand_counts` (demand dict) and apply the D-02 filter: keep buckets where `demand_count > 0` AND `place_count < min_places`, ranked by `demand_count` descending.

**Supply-only contract preservation:** The existing `gather_stats` + `find_gaps` functions are **unchanged**. The demand path is additive: new parameter with a default (`demand_counts: dict | None = None`) or a separate function. Existing tests continue to pass unchanged.

**Guardrail check:** `tests/unit/test_coverage_agent.py` and `tests/unit/test_coverage_agent_smoke.py` test `gather_stats`, `find_gaps`, `propose_queries`, `filter_already_covered`, `insert_pending` independently via monkeypatch. None of these tests need to change if the demand path is additive. [VERIFIED: test file inspection]

**Verdict: Extend `coverage_agent.py`.** The demand CTE slots cleanly alongside `gather_stats` without touching the existing functions. A net-new `scripts/gap_miner.py` would be the right choice only if the demand path needed to REPLACE `find_gaps`'s logic — it doesn't; it gates the output of the existing logic.

**Rough diff surface:**
- `coverage_agent.py`: +60-80 lines (1 new dataclass or dict type, `gather_demand()`, updated `find_gaps()` or new `find_demand_gaps()`, updated `log_to_mlflow()` signature for new metrics, updated `main()` for new args + cold-start exit)
- `Makefile`: +4 lines (`gap-mine` target)
- `CLAUDE.md`: +2 lines (new command)
- Tests: +1 new test file `tests/unit/test_gap_miner.py` (unit tests for demand path) + new integration test

---

### Q4 — Demand extraction with `make_judge()`

**Finding:** `make_judge()` returns a `BaseChatModel | None`, constructed from `EVAL_JUDGE_PROVIDER` / `EVAL_JUDGE_MODEL` env vars (default: `gemini-3.1-flash-lite-preview` via Gemini). It returns `None` when credentials are missing. Used in `propose_queries` via `llm.invoke([HumanMessage(content=prompt)])`. [VERIFIED: `app/agent/critique/vibe.py` lines 101-129]

**`requested_primary_types[]` analysis:** This column stores Title-Case Google primary_type vocabulary strings like `"Vietnamese Restaurant"`, `"Bar"`, `"Italian Restaurant"`. These are validated by `family_of()` in `app/tools/filters.py` against `_PRIMARY_TYPE_FAMILIES` before being stored. [VERIFIED: `app/main.py` lines 737-738, `app/tools/filters.py` lines 168-296]

**Lexical mapping feasibility:** The mapping from `requested_primary_types` values to CUISINES catalog strings is straightforward for many cases:
- `"Vietnamese Restaurant"` → strip " Restaurant", lowercase → `"vietnamese"` ∈ CUISINES
- `"Bar"`, `"Cocktail Bar"` → map to a bar-family tag (but CUISINES has no "bar" — bars are captured by eatery_types/bar_types in the seed catalog, not by CUISINES)
- `"Italian Restaurant"` → `"italian"` ∈ CUISINES

**Key insight:** `requested_primary_types` captures **food category intent** but not **neighborhood**. The `message` field ("dinner in the Mission", "sushi in Japantown") captures neighborhood intent. This means:

- **Cuisine extraction**: `requested_primary_types` → lexical strip → CUISINES lookup. Fast, no LLM needed.
- **Neighborhood extraction**: `message` → LLM extraction OR simple regex ("in `<Neighborhood>`") → NEIGHBORHOODS lookup.
- **LLM fallback**: Used for rows where `requested_primary_types` is empty `[]` and `message` is the only signal.

**CONTEXT.md D-01 says "one LLM call per mining run"** — this means a single batch call, not one call per row. The recommended approach is:
1. Lexically extract cuisines from `requested_primary_types` for all rows (free, no LLM).
2. Batch all `message` values needing neighborhood extraction into ONE LLM call with a JSON-list prompt.
3. The LLM returns `[{"row_id": "...", "neighborhood": "...", "cuisine": "..."}]` or `null` per row.

This satisfies D-01's one-call-per-run framing, reduces cost, and handles the realistic scenario where `requested_primary_types` is populated but neighborhood is only in `message`.

**Unmappable rows:** Rows where neither `requested_primary_types` nor `message` maps to a catalog bucket are counted in `unmapped_count` and dropped.

---

### Q5 — Catalog membership + seed contract

**Finding:** `build_seed_queries()` generates 3,410 unique seed query strings. [VERIFIED: live Python invocation] The catalog is `NEIGHBORHOODS × CUISINES` (30 × 45 = 1,350 neighborhood+cuisine combinations) plus city-wide cuisine + eatery-type + bar-type + attraction-type seeds.

**The exact seed-query format for `(neighborhood, cuisine)` gaps is:**
```
"{cuisine} restaurants in {neighborhood} San Francisco"
```

Example: `"vietnamese restaurants in Outer Sunset San Francisco"` [VERIFIED: live Python check]

This is confirmed by `loop_falsifier.py` line 69:
```python
SEED_QUERY = f"{GAP[1]} restaurants in {GAP[0]} San Francisco"
```

**`premark_seed_isolation` contract:** [VERIFIED: `scripts/loop_falsifier.py` lines 173-199] The function calls `build_seed_queries()` and asserts `chosen_seed_query in set(catalog)`, exiting `EXIT_INFRA(2)` if not found. Therefore:

- The miner's output proposals for `(neighborhood, cuisine)` gaps MUST use the exact format: `f"{cuisine} restaurants in {neighborhood} San Francisco"`
- Both `neighborhood` must be in `NEIGHBORHOODS` and `cuisine` must be in `CUISINES`, otherwise `build_seed_queries()` will not have generated that string.
- Off-catalog proposals (e.g., `"burmese restaurants in Outer Sunset San Francisco"` where `"burmese"` is not in CUISINES) will pass `build_seed_queries()` filtering in `filter_already_covered` because they won't be in the existing set — but the loop falsifier will reject them at `premark_seed_isolation`.

**Conclusion:** The miner must constrain its output to `NEIGHBORHOODS × CUISINES` catalog buckets only. The `filter_already_covered` function already handles dedup against the full seed list; the new miner only needs to ensure it generates proposals in the correct format.

---

### Q6 — MLflow + CLI reuse

**Finding:** [VERIFIED: `scripts/coverage_agent.py` lines 261-323, `Makefile` lines 93-99]

**MLflow experiment name:** `"coverage_agent"` — set via `mlflow.set_experiment("coverage_agent")` [VERIFIED: line 269]

**`log_to_mlflow` current signature:**
```python
def log_to_mlflow(
    stats: list[CoverageStat],
    gaps: list[CoverageStat],
    proposals: list[ProposedQuery],
    dropped: list[ProposedQuery],
    inserted: int,
    dry_run: bool,
) -> None
```

**Current MLflow metrics logged:** `gaps_found`, `proposals_made`, `dropped_already_covered`, `inserted`, `dry_run` (param).

**New metrics required (from CONTEXT.md):** `demand_rows_scanned`, `unmapped_count`, `proposals_inserted` (rename of `inserted` or alias). The ranked gap artifact can be logged as `"demand_gaps.json"` alongside existing `"gaps.json"`.

**`log_to_mlflow` extension strategy:** Add new parameters `demand_rows_scanned: int = 0` and `unmapped_count: int = 0` with defaults so the existing supply-only call path continues to work unchanged.

**argparse `main()` current flags:** `--days` (default 14), `--dry-run` (action=store_true), `--min-places` (default 5). Returns `int` (always 0). [VERIFIED: lines 290-323]

**New flags required:** `--top-n` (cap inserted gaps, int, no default = no cap) — clean addition. Mirror the `--dry-run` opt-out convention (not opt-in `--apply`).

**Existing `make` targets:**
- `coverage-agent`: `python scripts/coverage_agent.py --dry-run`
- `coverage-agent-apply`: `python scripts/coverage_agent.py`
- `loop-falsifier`: `python scripts/loop_falsifier.py`

**New target:** `gap-mine` → `$(POETRY_RUN) python scripts/coverage_agent.py gap-mine --dry-run` OR the miner is a new entrypoint. Given the extend-not-reinvent decision, a clean approach is a new `gap-mine` subcommand or a new `gap_miner_main()` function alongside `main()`.

**Recommendation:** Add a `gap_mine_main(argv)` function to `coverage_agent.py` with its own `argparse` and a `gap-mine` Makefile target. This avoids touching the existing `main()` CLI contract at all and keeps backward compatibility.

---

## Standard Stack

### Core (all VERIFIED — already in pyproject.toml)

| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| psycopg2 | pinned via Poetry | PostgreSQL connection for demand CTE + insert | [VERIFIED: existing imports in coverage_agent.py] |
| mlflow | pinned via Poetry | Experiment tracking, existing `coverage_agent` experiment | [VERIFIED: coverage_agent.py line 28] |
| langchain-core | pinned via Poetry | `HumanMessage` for LLM batch prompt | [VERIFIED: coverage_agent.py line 29] |
| argparse | stdlib | CLI flag parsing, matching existing `main()` shape | [VERIFIED: coverage_agent.py line 19] |

### No New Dependencies Required

Phase 18 adds **zero new packages**. All required tools (psycopg2, MLflow, LangChain, vibe.make_judge) are already in the project's Poetry environment. The `psycopg2.connect()` direct connection for the optional prod-read path uses a package already present. [VERIFIED: `pyproject.toml` dependency list]

---

## Package Legitimacy Audit

No new packages are installed in this phase. Audit: N/A.

---

## Architecture Patterns

### System Architecture Diagram

```
DEMAND PATH                          SUPPLY PATH (existing)
user_query_log                       places_raw
  (DEMAND_DATABASE_URL or pool)        (pool → DATABASE_URL → sandbox)
       |                                      |
  gather_demand(days)               gather_stats(days)  [UNCHANGED]
       |                                      |
  {(nbhd, cuisine): count}          [CoverageStat list]
       \                                     /
        \                                   /
         merge_demand_supply_gaps()  [NEW]
               |
         D-02 filter: demand>0 AND supply<min_places
         rank by demand_count desc
               |
         [demand-gated CoverageStat list]
               |
         [cold-start guard: if empty → log gaps_found=0, exit 0]
               |
         propose_queries(gaps, llm)  [UNCHANGED]
               |
         filter_already_covered(proposals, existing)  [UNCHANGED]
               |
         insert_pending(kept, dry_run)  [UNCHANGED]
               |
         log_to_mlflow(... + demand_rows_scanned, unmapped_count)  [EXTENDED]
```

### Recommended Project Structure

No new files required. All changes land in:
```
scripts/
├── coverage_agent.py       # +60-80 lines: gather_demand(), merge_demand_supply_gaps(),
│                           #   gap_mine_main(), extended log_to_mlflow()
Makefile                    # +gap-mine target
CLAUDE.md                   # +make gap-mine documentation
.env.example                # +DEMAND_DATABASE_URL (optional, for prod-read path)
tests/
├── unit/
│   └── test_gap_miner.py   # NEW: unit tests for demand path functions
├── integration/
│   └── test_gap_miner.py   # NEW: integration tests (APP_ENV=integration)
```

### Pattern 1: `gather_demand` — Demand CTE over `user_query_log`

```python
# Source: coverage_agent.py gather_stats() pattern + Phase 17 schema
def gather_demand(
    days: int,
    url: str | None = None,
) -> tuple[dict[tuple[str, str], int], int, int]:
    """Read user_query_log for the last `days` days.

    Returns (demand_counts, rows_scanned, unmapped_count).
    demand_counts: {(neighborhood, cuisine): count} — catalog-constrained.

    If url is provided, opens a direct non-pooled connection to that URL
    (prod-read path). If None, uses the shared pool (sandbox path).
    """
    cutoff = datetime.now(UTC) - timedelta(days=days)
    sql = """
        SELECT message, requested_primary_types
        FROM user_query_log
        WHERE created_at >= %s
        ORDER BY created_at DESC
    """
    # ... extract rows, call _map_demand_rows(), return counts
```

### Pattern 2: Lexical cuisine mapping (avoiding LLM for typed rows)

```python
# Source: ingest_places_sf.py CUISINES list + app/tools/filters.py families
_CUISINE_FROM_PRIMARY_TYPE: dict[str, str] = {
    # primary_type.lower().removesuffix(" restaurant") → CUISINES member
    "vietnamese": "vietnamese",
    "italian": "italian",
    "thai": "thai",
    # ... generated from CUISINES list programmatically
}

def _types_to_cuisines(primary_types: list[str]) -> list[str]:
    """Map requested_primary_types to CUISINES catalog members.
    Returns [] for unmappable types (bar-family, dessert-family, etc.).
    """
    result = []
    for pt in primary_types:
        candidate = pt.lower().removesuffix(" restaurant")
        if candidate in CUISINES_SET:
            result.append(candidate)
    return result
```

### Pattern 3: Single-batch LLM neighborhood extraction

```python
# One LLM call per mining run (D-01) for rows where message needs parsing
def _extract_neighborhoods_batch(
    messages: list[str], llm: Any
) -> list[str | None]:
    """Batch all messages into ONE LLM call.
    Returns list of neighborhood strings or None (unmappable).
    """
    prompt = _build_neighborhood_extraction_prompt(messages, NEIGHBORHOODS)
    raw = llm.invoke([HumanMessage(content=prompt)]).content
    return _parse_neighborhood_batch(raw, NEIGHBORHOODS_SET)
```

### Pattern 4: `get_demand_conn` — Optional non-pooled prod connection

```python
# Source: psycopg2 direct connection; does NOT touch the shared pool
from contextlib import contextmanager

@contextmanager
def get_demand_conn(url: str):
    """Open a direct non-pooled read-only connection to the demand DB.
    Used when DEMAND_DATABASE_URL targets a different host than DATABASE_URL.
    Closes on exit — not returned to any pool.
    """
    conn = psycopg2.connect(url)
    try:
        conn.set_session(readonly=True, autocommit=True)
        yield conn
    finally:
        conn.close()
```

### Anti-Patterns to Avoid

- **Reinitializing the pool mid-run to switch DB targets.** `init_db_pool` raises `RuntimeError` if called with different params. Use a direct `psycopg2.connect` for the optional prod-read path instead.
- **Modifying `gather_stats()` to add the demand CTE.** Keep supply and demand paths separate; merge in Python, not SQL. This preserves the existing test contract.
- **Proposing off-catalog seeds.** Any `(neighborhood, cuisine)` pair where neighborhood ∉ `NEIGHBORHOODS` or cuisine ∉ `CUISINES` will fail `premark_seed_isolation` with EXIT_INFRA(2). Always validate against the catalog before calling `propose_queries`.
- **One LLM call per row.** Batch all messages into a single call. Cost and latency scale with row count otherwise.
- **Using `filter_already_covered` as the catalog-membership gate.** `filter_already_covered` filters against `build_seed_queries()` — it will NOT catch off-catalog proposals because those strings aren't in `build_seed_queries()` either. Catalog membership must be enforced upstream in `gather_demand`.
- **Forgetting `get_settings.cache_clear()` + `close_db_pool()` after env coercion.** This is the Phase 16 lru_cache footgun. If the miner ever coerces `DATABASE_URL` (e.g., to target sandbox), it must call both.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Proposals dedup | Custom SQL check | `filter_already_covered()` + `insert_pending(ON CONFLICT DO NOTHING)` | Already tested; handles checkpoints + prior proposals + static seeds |
| LLM construction | Custom model init | `vibe.make_judge()` | Provider-swappable, credential-safe, returns None gracefully |
| LLM output parsing | Custom JSON parser | `_parse_proposals()` (strip fences, skip invalid keys) | Battle-tested; handles all LLM quirkiness |
| MLflow logging | Custom tracking | `log_to_mlflow()` extended | Existing experiment; extends cleanly |
| DB URL resolution | Custom env parsing | `resolve_database_url(os.environ)` | Handles `DATABASE_URL` > `POSTGRES_*` precedence correctly |
| Seed query format | Custom string builder | `f"{cuisine} restaurants in {neighborhood} San Francisco"` | This is the exact format `build_seed_queries()` generates; drift will break `premark_seed_isolation` |

---

## Runtime State Inventory

Not applicable — this is a greenfield addition (new functions in an existing script). No rename, refactor, or migration of existing runtime state.

The sandbox `city_concierge_sandbox` database does NOT yet have the `user_query_log` table (the Phase 17 migration `d1be72aea7d4` has not been applied to it). This is a Wave 0 gap the planner must address.

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | `user_query_log` absent from `city_concierge_sandbox` | Apply migration `d1be72aea7d4` to sandbox: `SANDBOX_DATABASE_URL=... alembic upgrade head` |
| Stored data | `user_query_log` empty in `city_concierge` local (0 rows) | Seed test rows for local dev / CI integration tests |
| Live service config | None | — |
| OS-registered state | None | — |
| Secrets/env vars | `DEMAND_DATABASE_URL` (new, optional) — if set, miner reads demand from this URL rather than the pool | Add to `.env.example`; no existing keys renamed |
| Build artifacts | None | — |

---

## Common Pitfalls

### Pitfall 1: Sandbox `user_query_log` missing → miner exits cold-start immediately
**What goes wrong:** `gather_demand()` queries a table that doesn't exist → psycopg2 raises `UndefinedTable` → miner logs `gaps_found=0` (or crashes if not handled).
**Why it happens:** The Phase 17 migration was applied to `city_concierge` but not `city_concierge_sandbox`. The sandbox was provisioned at Phase 16 when `user_query_log` didn't exist.
**How to avoid:** Wave 0 task: `SANDBOX_DATABASE_URL=... alembic upgrade head`. Verify with `\dt` in psql.
**Warning signs:** `ERROR: relation "user_query_log" does not exist` in miner output.

### Pitfall 2: lru_cache footgun on `get_settings()` with env coercion
**What goes wrong:** If `DATABASE_URL` is overridden in env after module import, `get_settings()` returns stale prod settings because `@lru_cache` froze the result at first call.
**Why it happens:** `settings = get_settings()` at module scope in `app/config.py` line 160 populates the cache.
**How to avoid:** If the miner ever coerces `DATABASE_URL`, follow Phase 16 pattern: `get_settings.cache_clear()` + `close_db_pool()` immediately after coercion, then all imports of `app.*` settings-touching code AFTER coercion.
**Warning signs:** Miner writes to prod despite `SANDBOX_DATABASE_URL` being set; Phase 16 D-10 history.

### Pitfall 3: Off-catalog proposals pass `filter_already_covered` but fail `premark_seed_isolation`
**What goes wrong:** `filter_already_covered` keeps a proposal because `"burmese restaurants in Outer Sunset San Francisco"` is not in `build_seed_queries()`. The proposal gets inserted as `pending`. Later, when `loop_falsifier.py` calls `premark_seed_isolation(conn, chosen_seed_query)` with this query, it exits `EXIT_INFRA(2)`.
**Why it happens:** `filter_already_covered` filters for dedup (already done), not catalog membership (new requirement). These are different things.
**How to avoid:** Enforce catalog membership in `gather_demand` / `_map_demand_rows`: only yield `(neighborhood, cuisine)` pairs where both are in `NEIGHBORHOODS` and `CUISINES`. Never pass off-catalog pairs to `propose_queries`.
**Warning signs:** Loop falsifier exits 2 with "chosen_seed_query not in catalog" message.

### Pitfall 4: Neighborhood extraction from `message` is ambiguous for multi-neighborhood queries
**What goes wrong:** "dinner in the Mission and drinks in North Beach" → LLM extracts two neighborhoods → which to use?
**Why it happens:** `user_query_log` stores the full message verbatim; multi-intent messages are common.
**How to avoid:** For multi-neighborhood extractions, either (a) emit one row per `(neighborhood, cuisine)` pair found, or (b) take the first neighborhood. Option (a) is more informative for demand counting; document the choice in the prompt design.
**Warning signs:** Demand counts spike unrealistically for a single message.

### Pitfall 5: `requested_primary_types` stores bar/dessert families that don't map to CUISINES
**What goes wrong:** `["Bar", "Cocktail Bar"]` → lexical strip → `"bar"` ∉ CUISINES → dropped as unmapped. `unmapped_count` spikes.
**Why it happens:** CUISINES covers food cuisines; bars and desserts are in EATERY_TYPES and BAR_TYPES but not CUISINES. The demand signal captures mixed intent.
**How to avoid:** This is expected behavior — `unmapped_count` is intended to capture this. Document in the honesty metric. Phase-19+ extension could add bar/dessert buckets to the catalog.
**Warning signs:** Very high `unmapped_count` relative to `demand_rows_scanned` — investigate whether neighborhood queries are being dropped unnecessarily.

### Pitfall 6: Cold-start during local dev misleads about implementation correctness
**What goes wrong:** Developer runs `make gap-mine` locally, sees `gaps_found=0`, thinks the miner is broken.
**Why it happens:** `user_query_log` is empty locally (Phase 17 just shipped; no real traffic).
**How to avoid:** Include a `make gap-mine-seed-test` target (or document a SQL seed command) in Wave 0 that inserts a few representative rows into sandbox `user_query_log` for testing.
**Warning signs:** `demand_rows_scanned=0` in MLflow output.

---

## Code Examples

### Complete demand CTE query

```python
# Source: inferred from coverage_agent.gather_stats() pattern + user_query_log schema
# (alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py)
def _fetch_demand_rows(days: int, conn) -> list[tuple[str, list[str]]]:
    """Fetch recent user_query_log rows as (message, requested_primary_types)."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    sql = """
        SELECT message, COALESCE(requested_primary_types, '{}')
        FROM user_query_log
        WHERE created_at >= %s
        ORDER BY created_at DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, [cutoff])
        return [(row[0], list(row[1])) for row in cur.fetchall()]
```

### Catalog-constrained bucket mapping

```python
# Source: ingest_places_sf.CUISINES (line 194) + NEIGHBORHOODS (line 161)
# Seed query format: f"{cuisine} restaurants in {neighborhood} San Francisco"
# (confirmed from loop_falsifier.py line 69)
_CUISINES_SET = set(CUISINES)
_NEIGHBORHOODS_SET = set(NEIGHBORHOODS)

def _types_to_cuisines(primary_types: list[str]) -> list[str]:
    """Map requested_primary_types to CUISINES catalog members (lexical, no LLM)."""
    result = []
    for pt in primary_types:
        # "Vietnamese Restaurant" → "vietnamese"
        candidate = pt.lower().removesuffix(" restaurant")
        if candidate in _CUISINES_SET:
            result.append(candidate)
    return result
```

### D-02 gap merge (demand gates supply)

```python
def find_demand_gaps(
    stats: list[CoverageStat],
    demand_counts: dict[tuple[str, str], int],
    min_place_count: int = 5,
) -> list[tuple[CoverageStat, int]]:
    """Return [(gap_stat, demand_count)] for under-served, demand-confirmed buckets.

    A bucket is a gap iff:
      - It is a neighborhood+cuisine combo (not recent_query)
      - supply: place_count < min_place_count
      - demand: demand_count > 0
    Sorted by demand_count descending (D-02).
    """
    supply_map: dict[str, int] = {
        s.bucket: s.place_count
        for s in stats
        if s.bucket.startswith(("neighborhood:", "cuisine:"))
    }
    results = []
    for (nbhd, cuisine), demand in demand_counts.items():
        if demand == 0:
            continue
        # Only neighborhood+cuisine combos are valid gaps
        # (city-wide cuisine gaps are handled by supply-only coverage_agent)
        nbhd_bucket = f"neighborhood:{nbhd}"
        cuisine_bucket = f"cuisine:{cuisine}"
        nbhd_supply = supply_map.get(nbhd_bucket, 0)
        cuisine_supply = supply_map.get(cuisine_bucket, 0)
        # Use the more constrained supply: neighborhood-level if available
        supply = min(nbhd_supply, cuisine_supply) if nbhd_supply > 0 else cuisine_supply
        if supply < min_place_count:
            gap_stat = CoverageStat(
                bucket=f"demand:{nbhd}:{cuisine}",
                place_count=supply,
                distinct_queries=demand,
                last_ingest=None,
            )
            results.append((gap_stat, demand))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
```

### Seed query generation for mined gaps

```python
# Source: loop_falsifier.py line 69 — exact format confirmed
def gap_to_seed_query(neighborhood: str, cuisine: str) -> str:
    """Generate the canonical seed query string for a (neighborhood, cuisine) gap.
    MUST match build_seed_queries() output exactly or premark_seed_isolation exits INFRA.
    """
    assert neighborhood in _NEIGHBORHOODS_SET, f"off-catalog neighborhood: {neighborhood!r}"
    assert cuisine in _CUISINES_SET, f"off-catalog cuisine: {cuisine!r}"
    return f"{cuisine} restaurants in {neighborhood} San Francisco"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded `GAP = ("Outer Sunset", "vietnamese")` constant | Demand-mined gap from `user_query_log` | Phase 18 (this phase) | Production loop gap selection becomes data-driven |
| `coverage_agent` supply-only gap detection | Supply-side as before + demand-gated filter | Phase 18 | Gaps that users aren't asking for are not proposed |
| No `user_query_log` | Phase 17 shipped `user_query_log` with `created_at` index | Phase 17 (PR #112, 2026-06-16) | Time-window demand mining is cheap and anticipated |

**Deprecated / outdated (relative to Phase 18):**
- Hardcoded `GAP` constant in `loop_falsifier.py` stays put (it is the falsifier's own deterministic mechanism stub, Phase 16 D-01). Phase 18 does NOT delete it. "Replaces" means the production loop driver, not the test gate.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Prod Cloud SQL `user_query_log` row count is unknown (IAP tunnel not open in this environment) | Q1 | If prod has many rows, the prod-read path becomes higher priority; decision rule still applies |
| A2 | `requested_primary_types` lexical strip (`.removesuffix(" restaurant")`) maps most food types to CUISINES | Q4 | If many primary_types use different suffixes (e.g. "Ramen Restaurant" → "ramen" ∈ CUISINES ✓), the mapping still works; test with the actual vocabulary |
| A3 | One batch LLM call for neighborhood extraction from `message` is feasible within token limits for typical `--days` windows | Q4 | If `demand_rows_scanned` grows to thousands of rows, the batch prompt may exceed context limits; add row-capping logic |

---

## Open Questions (RESOLVED)

> Both questions were answered at research time with the recommendations below and are
> implemented by the plans (18-03-T1 cites "RESEARCH Open Question #1 resolution"; 18-03-T2
> cites "RESEARCH Open Question #2"). Retained here for provenance.

1. **Exact supply join for neighborhood+cuisine cross-buckets**
   - What we know: `gather_stats` returns `neighborhood:X` and `cuisine:Y` as separate rows, not cross-product rows. There is no `(neighborhood, cuisine)` combined row in the current stats.
   - What's unclear: Should the D-02 supply check use neighborhood supply, cuisine supply, or min(both)? The code example above uses min(both) but this may be wrong — a neighborhood might have many Italian places but zero Vietnamese, and the cuisine bucket might show 0 for Vietnamese overall (all of SF).
   - RESOLVED: Use per-cuisine supply for ranking (covers the "no Vietnamese anywhere in SF" case) AND per-neighborhood for the proposal text. This matches the intent of D-02 ("place_count < min_places" means both axes are thin).

2. **`--top-n` cap placement**
   - What we know: CONTEXT.md says `--top-n` caps inserted gaps; the ranked list is by `demand_count` descending.
   - What's unclear: Does `--top-n` cap before or after `filter_already_covered`? Capping before means you might insert 0 if the top N are all already covered.
   - RESOLVED: Apply `--top-n` after `filter_already_covered` (cap the kept proposals list) so you always insert up to N truly-new gaps.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PostgreSQL (Docker `city_concierge_sandbox`) | Sandbox write path | ✓ | pg16 via pgvector/pgvector:pg16, port 5433 | — |
| `user_query_log` migration in sandbox | Demand read from sandbox | ✗ (table absent) | — | Wave 0: apply migration `d1be72aea7d4` to sandbox |
| Prod Cloud SQL via IAP tunnel | Prod-read D-05 path | ✗ (tunnel not open) | — | Sandbox-only path (D-04/D-05 fallback) |
| MLflow tracking server | `log_to_mlflow` | Unknown (tunnel to localhost:5050) | — | Skip MLflow if unreachable (log warning) |
| `EVAL_JUDGE_PROVIDER` / `EVAL_JUDGE_MODEL` / key | `vibe.make_judge()` for LLM extraction | Conditional on `.env` | — | Returns None → cold-start path fires |
| `DEMAND_DATABASE_URL` (new, optional) | Prod-read path | Not set yet | — | Falls back to pool (sandbox) |

**Missing dependencies with no fallback:**
- `user_query_log` table in sandbox — blocks Wave 0 integration tests. Must apply Phase 17 migration to sandbox.

**Missing dependencies with fallback:**
- Prod IAP tunnel → sandbox-only mode
- MLflow → log warning, exit 0 (consistent with existing pattern)
- `vibe.make_judge()` → cold-start path (D-04)

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A (script, no user auth) |
| V3 Session Management | No | N/A |
| V4 Access Control | Yes | DB credentials via env vars; prod-read is read-only by convention (`set_session(readonly=True)`) |
| V5 Input Validation | Yes | Raw `message` text from users piped to LLM; parameterised SQL (already in `log_user_query`) |
| V6 Cryptography | No | N/A |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| PII in `message` to LLM | Information Disclosure | `user_query_log` stores raw `message` verbatim (documented, acknowledged in migration comment). LLM receives user messages. For a private capstone DB this is accepted risk; for production would require a PII scrubbing step. Document in threat model. |
| Prompt injection via `message` | Tampering | Batch extraction prompt must never interpolate `message` directly into a format string that escapes the JSON boundary. Pass messages as a JSON-encoded array in the prompt, not raw string interpolation. |
| SQL injection in demand CTE | Tampering | Already handled: `gather_demand` uses parameterised `%s` placeholder for the `cutoff` datetime; no user input is interpolated into SQL. |
| Prod-read credential exposure | Information Disclosure | `DEMAND_DATABASE_URL` contains prod credentials. Store only in `.env` (gitignored). Add to `.env.example` with placeholder value. |
| Write to prod `places_raw` | Elevation of Privilege | Hard guard: `insert_pending` uses the pool which targets `DATABASE_URL` = sandbox. The prod-read connection (`get_demand_conn`) is opened `readonly=True` and never used for writes. |

---

## Project Constraints (from CLAUDE.md)

- Python 3.10+; ruff line-length 100, rules: E, F, I, N, UP, B, SIM
- `app` is editable-installed via Poetry — `from app.xxx import yyy` works; never re-add `sys.path` hacks
- Tests: pytest, `asyncio_mode = "auto"` (from `pyproject.toml`), integration tests gated on `APP_ENV=integration`
- Test layering preference: unit/mock + smoke + functional + integration (from memory `feedback_test_layering`)
- Small focused commits; single-line commit messages (from memory `feedback_small_focused_commits`)
- Never run `gh pr merge`; hand back to user when CI is green (from memory `feedback_user_merges_prs`)
- `nyquist_validation: false` in `.planning/config.json` — skip Validation Architecture section
- `make test` runs full suite with coverage; pre-commit hook runs ruff automatically (no manual ruff before commit)
- AGENTS.md and `.github/copilot-instructions.md` must be kept in sync with CLAUDE.md when adding commands

---

## Sources

### Primary (HIGH confidence — VERIFIED from live code/DB)

- `scripts/coverage_agent.py` — full pipeline inspection (lines 1-328); gather_stats CTE; find_gaps; propose_queries; filter_already_covered; insert_pending; log_to_mlflow; main() argparse
- `scripts/ingest_places_sf.py` — NEIGHBORHOODS (line 161), CUISINES (line 194), build_seed_queries() (line 319); confirmed 3,410 seeds, exact format `"{cuisine} restaurants in {neighborhood} San Francisco"`
- `scripts/loop_falsifier.py` — GAP constant (line 68-69), SEED_QUERY format, premark_seed_isolation contract (lines 173-199)
- `app/db_pool.py` — single-pool architecture, init_db_pool raises on different params; get_connection/close_db_pool
- `app/db.py` — get_conn() context manager uses shared pool
- `app/config.py` — resolve_database_url precedence; Settings.resolved_database_url; lru_cache on get_settings()
- `app/agent/critique/vibe.py` — make_judge() signature, JUDGE_PROVIDER/MODEL env vars, returns None on missing creds
- `app/query_log.py` — log_user_query() signature, all 5 stored fields
- `alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py` — user_query_log schema (7 columns, idx on created_at)
- `app/main.py` lines 60-89 — requested_primary_types extraction; Title-Case Google primary_type vocabulary
- `tests/unit/test_coverage_agent.py`, `tests/unit/test_coverage_agent_smoke.py`, `tests/integration/test_coverage_agent.py` — existing test contracts
- Live DB probe: `city_concierge` local = 0 rows in user_query_log; `city_concierge_sandbox` = no user_query_log table; sandbox has 20 rows in places_raw from Phase 16 FALSIFY-01 run
- `Makefile` lines 53-99 — coverage-agent and sandbox-provision targets

### Secondary (MEDIUM confidence)

- `.planning/phases/16-loop-falsifier/16-CONTEXT.md` D-10 — sandbox DATABASE_URL injection pattern; rejection of threaded db_url params
- `.planning/phases/18-gap-mining-gap/18-CONTEXT.md` — all decisions, Claude's Discretion, Deferred scope
- `.planning/STATE.md` — Phase 17 shipped PR #112; v2.3 active

---

## Metadata

**Confidence breakdown:**
- Pipeline extension approach (extend vs. new file): HIGH — based on direct code inspection of 328-line file, clean function separation, and test contracts
- Two-DB feasibility and `get_demand_conn` pattern: HIGH — confirmed from pool source code; psycopg2.connect is already available
- Seed query format contract: HIGH — confirmed by live Python invocation (3,410 seeds, exact string verified)
- Prod row count: CANNOT VERIFY (IAP tunnel not open) — decision rule provided
- LLM batch extraction feasibility: MEDIUM — pattern is logical but token-limit ceiling for large windows is assumed
- `requested_primary_types` → CUISINES lexical mapping coverage: MEDIUM — vocabulary confirmed, edge cases for non-standard types assumed to drop to `unmapped_count`

**Research date:** 2026-06-17
**Valid until:** 2026-07-17 (stable codebase; DB schema and vocabulary are locked; only risk is prod row count growing to change the D-05 recommendation)
