---
phase: 18-gap-mining-gap
plan: 03
type: execute
wave: 2
depends_on: [18-02]
files_modified:
  - scripts/coverage_agent.py
  - tests/unit/test_gap_miner.py
autonomous: true
requirements: [GAP-02, GAP-03, GAP-04]
must_haves:
  truths:
    - "A bucket is a gap iff `demand_count > 0` AND TRUE PAIR-LEVEL supply `pair_place_count < min_places` (default 5), where pair supply counts places matching BOTH the neighborhood AND the cuisine — ranked by `demand_count` descending. This surfaces the canonical demand case (a cuisine present city-wide but ABSENT in the demanded neighborhood, e.g. Vietnamese everywhere in SF but zero in Outer Sunset). Demand gates and orders; pair supply is the existing absolute floor applied at the (neighborhood, cuisine) level (D-02, REVIEW HIGH-1 fix — supersedes RESEARCH Open Question #1's per-cuisine resolution)."
    - "Pair-level supply is counted from `place_query_hits` (the indexed query→place evidence table) by `COUNT(DISTINCT place_id)` for the exact seed `query_text` of each demanded pair — never-ingested pairs return 0 and are correctly flagged. `gather_stats` and the supply-only path are UNCHANGED (additive `gather_pair_supply`)."
    - "Gaps flow through an explicit `DemandGap(neighborhood, cuisine, place_count, demand_count)` dataclass — structured data is NEVER encoded as a parseable `demand:{n}:{c}` string in `CoverageStat.bucket` (REVIEW MEDIUM — no fragile re-parsing, no supply/demand conflation)."
    - "The miner emits proposal `query_text` in the EXACT seed format `\"{cuisine} restaurants in {neighborhood} San Francisco\"` so inserted pending rows are loop-consumable and pass `loop_falsifier.premark_seed_isolation` catalog membership (D-03)."
    - "Catalog validation is SEPARATE from dedup filtering: `gap_to_seed_query` ASSERTS the emitted text is a `build_seed_queries()` member (loop-compat for `premark_seed_isolation`/D-03), but dedup runs against a NEW `ingested_query_texts(conn)` helper returning ONLY `places_ingest_query_checkpoints` ∪ existing `places_ingest_query_proposals` rows — NOT the static `build_seed_queries()` catalog. This prevents the demand path from dropping EVERY valid proposal (REVIEW HIGH-2 BLOCKER fix). The supply-only path keeps using `existing_query_texts` unchanged (no regression)."
    - "The miner writes `pending` rows to `places_ingest_query_proposals` via the existing `insert_pending` path (the productionized loop seam) and does NOT touch `loop_falsifier.py`'s `GAP` constant (D-03 — production path supersedes, falsifier stub stays)."
    - "A hard `assert_sandbox_write_target()` guard runs BEFORE `insert_pending` in the demand path: it resolves the active write DB and refuses unless the resolved dbname is `city_concierge_sandbox` OR equals `SANDBOX_DATABASE_URL`'s dbname — turning D-05's sandbox-write convention into an ENFORCED invariant (REVIEW HIGH-3 fix)."
    - "Cold start (empty/zero-mappable `user_query_log`) inserts NOTHING, logs `gaps_found=0`, and exits 0 (D-04 honest no-op)."
    - "`gap_mine_main` mirrors `coverage_agent`'s CLI/MLflow: argparse `main()->int` with `--days/--dry-run/--min-places/--top-n` (writes by default, `--top-n` applied AFTER dedup filtering); logs `gaps_found`, `proposals_inserted`, `demand_rows_scanned`, `unmapped_count` + ranked-gap artifact under the `coverage_agent` experiment (D-04 CLI/ops discretion)."
  artifacts:
    - path: "scripts/coverage_agent.py"
      provides: "DemandGap dataclass, find_demand_gaps(), gather_pair_supply(), gap_to_seed_query(), ingested_query_texts(), assert_sandbox_write_target(), gap_mine_main(), extended log_to_mlflow()"
      contains: "def gap_mine_main"
    - path: "tests/unit/test_gap_miner.py"
      provides: "Unit tests for pair-level supply gating, D-02 ranking, dedup-survives-catalog-membership (HIGH-2), sandbox write guard (HIGH-3), seed-format exactness, cold-start exit, --top-n placement, MLflow metrics"
  key_links:
    - from: "scripts/coverage_agent.py gap_mine_main"
      to: "places_ingest_query_proposals"
      via: "insert_pending (reused, unchanged) guarded by assert_sandbox_write_target"
      pattern: "insert_pending\\("
    - from: "scripts/coverage_agent.py gather_pair_supply"
      to: "place_query_hits"
      via: "COUNT(DISTINCT place_id) per seed query_text"
      pattern: "FROM place_query_hits"
    - from: "scripts/coverage_agent.py ingested_query_texts"
      to: "places_ingest_query_checkpoints + places_ingest_query_proposals"
      via: "dedup set WITHOUT build_seed_queries catalog"
      pattern: "def ingested_query_texts"
    - from: "scripts/coverage_agent.py gap_to_seed_query"
      to: "build_seed_queries() catalog format"
      via: "exact f-string match + membership assertion"
      pattern: "restaurants in .* San Francisco"
    - from: "scripts/coverage_agent.py gap_mine_main"
      to: "MLflow coverage_agent experiment"
      via: "extended log_to_mlflow with demand metrics"
      pattern: "demand_rows_scanned"
---

<objective>
Wire the demand signal into gaps and the loop: `find_demand_gaps` applies the D-02 filter at TRUE PAIR LEVEL (`demand>0 AND pair_place_count<min_places`, ranked by demand desc) using `gather_pair_supply` over `place_query_hits`; `gap_to_seed_query` emits the exact loop-consumable seed format with a catalog-membership assertion; a NEW `ingested_query_texts` helper dedups proposals against ingested rows ONLY (not the static catalog); `assert_sandbox_write_target` enforces sandbox-only writes; and `gap_mine_main` is the CLI entrypoint that composes these, handles the cold-start no-op, and logs demand metrics to MLflow.

Purpose: This closes GAP-02 (demand×supply gap definition + pair-level ranking), GAP-03 (loop-integration write via the proposals seam — the productionized replacement for the hardcoded constant), and GAP-04 (CLI/ops/MLflow/cold-start). It composes the existing supply-side pipeline (unchanged) with the demand path from Plan 02 and incorporates the three cross-AI-review HIGH fixes (pair-level supply, dedup-split, enforced sandbox write guard).

Output: `DemandGap`, `gather_pair_supply`, `find_demand_gaps`, `gap_to_seed_query`, `ingested_query_texts`, `assert_sandbox_write_target`, `gap_mine_main`, and an extended `log_to_mlflow` in `coverage_agent.py`; new unit tests in `tests/unit/test_gap_miner.py`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/18-gap-mining-gap/18-CONTEXT.md
@.planning/phases/18-gap-mining-gap/18-RESEARCH.md
@.planning/phases/18-gap-mining-gap/18-REVIEWS.md
@scripts/coverage_agent.py
@scripts/ingest_places_sf.py
@scripts/loop_falsifier.py
@app/loop/falsifier_core.py
@app/config.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: DemandGap + gather_pair_supply (TRUE pair-level supply) + find_demand_gaps (D-02 ranking) + gap_to_seed_query</name>
  <files>scripts/coverage_agent.py, tests/unit/test_gap_miner.py</files>
  <read_first>
    - scripts/coverage_agent.py (the file being modified — the `CoverageStat`/`ProposedQuery` dataclasses to sit `DemandGap` beside; `gather_stats`'s parameterised `cur.execute(sql, [...])` shape and its `datetime.now(UTC) - timedelta(days=days)` cutoff to mirror in `gather_pair_supply`; `find_gaps`'s `place_count < min_place_count` floor to reuse as the supply gate; the `get_conn` import)
    - scripts/ingest_places_sf.py (build_seed_queries line ~319 — the canonical seed string `f"{cuisine} restaurants in {neighborhood} San Francisco"`; CUISINES line ~194 / NEIGHBORHOODS line ~161 catalogs; the ON-CONFLICT upsert at lines ~639-672 that does NOT update source_query, which is why `place_query_hits.query_text` — NOT `places_raw.source_query` — is the authoritative per-pair evidence)
    - scripts/db/init.sql lines ~64-78 (the `place_query_hits` schema: `place_id`, `query_text` indexed via `idx_place_query_hits_query_text`, `seen_at` — the evidence table that records EVERY (query, place) match, so COUNT(DISTINCT place_id) WHERE query_text = the seed string is the true pair-supply count)
    - scripts/loop_falsifier.py lines ~64-69 (the GAP constant + `SEED_QUERY = f"{GAP[1]} restaurants in {GAP[0]} San Francisco"` — the EXACT format gap_to_seed_query must match; DO NOT delete or import this constant) and lines ~173-205 (premark_seed_isolation's catalog-membership assertion the seed format must satisfy)
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § "HIGH-1" + "Verified against code (orchestrator) HIGH-1" (the authoritative direction to score supply at the PAIR level, not per-cuisine; the Outer-Sunset-Vietnamese canonical case)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § "Open Questions" #1 (the SUPERSEDED per-cuisine/min(both) resolution — note in the SUMMARY that this plan overrides it per the review; the code example's `min(nbhd_supply, cuisine_supply)` approach is NOT used)
  </read_first>
  <behavior>
    - Test 1 (pair-level gate — REVIEW HIGH-1): given `gather_pair_supply` stubbed so `("Outer Sunset","vietnamese")` has pair_place_count 0 but the city-wide cuisine bucket `cuisine:vietnamese` would be high, and demand `{("Outer Sunset","vietnamese"): 5}` with min_places=5, the Outer-Sunset bucket IS a gap. A cuisine present city-wide but absent in the demanded neighborhood is flagged — the per-cuisine resolution would have MISSED this.
    - Test 2 (pair supply excludes saturated pair): given `("Mission District","italian")` pair_place_count 40 and demand 3 with min_places=5, that bucket is NOT a gap (pair supply ≥ floor).
    - Test 3 (demand gates): a pair with pair_place_count=0 but `demand_count=0` is NOT a gap — demand must be > 0.
    - Test 4 (ranking): with two qualifying gaps demand 5 and demand 9, the result is ordered demand-descending (9 before 5).
    - Test 5 (gather_pair_supply SQL shape): `gather_pair_supply([("Outer Sunset","vietnamese")])` issues a parameterised SELECT `FROM place_query_hits` that counts `DISTINCT place_id` grouped by `query_text`, with the seed `query_text` value(s) passed as a param list (never string-interpolated — SQLi guard); a pair with no rows yields count 0.
    - Test 6 (DemandGap structure — REVIEW MEDIUM): `find_demand_gaps` returns a list of `DemandGap` instances with explicit `neighborhood`, `cuisine`, `place_count`, `demand_count` fields — NOT `CoverageStat` rows with `bucket="demand:..."` strings (assert `isinstance(g, DemandGap)` and `g.neighborhood == "Outer Sunset"`, no string parsing of a bucket).
    - Test 7 (seed format exactness): `gap_to_seed_query("Outer Sunset", "vietnamese") == "vietnamese restaurants in Outer Sunset San Francisco"` AND that exact string is a member of `set(build_seed_queries())`.
    - Test 8 (catalog assertion): `gap_to_seed_query` rejects off-catalog inputs (neighborhood ∉ NEIGHBORHOODS or cuisine ∉ CUISINES) — raises rather than emitting an un-ingestable seed.
  </behavior>
  <action>
    In `scripts/coverage_agent.py` add a `@dataclass DemandGap` with fields `neighborhood: str`, `cuisine: str`, `place_count: int` (pair-level supply), `demand_count: int` — the explicit structure that replaces encoding `demand:{n}:{c}` inside `CoverageStat.bucket` (REVIEW MEDIUM). Add `gather_pair_supply(pairs: list[tuple[str, str]], conn=None) -> dict[tuple[str, str], int]` that, for the demanded `(neighborhood, cuisine)` pairs, computes each pair's exact seed `query_text` via `gap_to_seed_query` and runs ONE parameterised `SELECT query_text, COUNT(DISTINCT place_id) FROM place_query_hits WHERE query_text = ANY(%s) GROUP BY query_text` (seed strings passed as the param list — never interpolated), mapping the result back to `{(neighborhood, cuisine): count}` with 0 for pairs that returned no rows (never-ingested pairs are correctly zero); use the pooled `get_conn()` when `conn` is None. This is TRUE pair-level supply (REVIEW HIGH-1) — it counts places that matched the neighborhood-AND-cuisine seed, so "Vietnamese everywhere in SF but zero in Outer Sunset" registers as supply 0 for that pair. Do NOT modify `gather_stats` or `find_gaps`. Add `find_demand_gaps(demand_counts: dict[tuple[str,str], int], pair_supply: dict[tuple[str,str], int], min_place_count: int = 5) -> list[DemandGap]` implementing D-02: for each pair with `demand > 0`, gate on `pair_supply.get(pair, 0) < min_place_count`, build a `DemandGap(neighborhood, cuisine, pair_supply.get(pair, 0), demand)`, and return the list sorted by `demand_count` descending. Add `gap_to_seed_query(neighborhood: str, cuisine: str) -> str` returning `f"{cuisine} restaurants in {neighborhood} San Francisco"` with upfront assertions that both are catalog members (raise on off-catalog — never emit an un-ingestable seed). Do NOT import or modify `loop_falsifier.GAP`. Add the tests to `tests/unit/test_gap_miner.py`.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner.py -v && poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/coverage_agent.py` contains `class DemandGap`, `def gather_pair_supply`, `def find_demand_gaps`, and `def gap_to_seed_query`.
    - A test proves TRUE pair-level gating (REVIEW HIGH-1): a cuisine present city-wide but with pair_place_count 0 in the demanded neighborhood IS flagged as a gap.
    - `gather_pair_supply` SELECTs `FROM place_query_hits` counting `DISTINCT place_id` per seed `query_text`, with seed strings passed as a `%s` param list (assert no string interpolation; assert never-ingested pair → 0).
    - `find_demand_gaps` returns `DemandGap` instances (assert `isinstance(..., DemandGap)`), NOT `CoverageStat` rows with a `demand:` bucket string (REVIEW MEDIUM — no fragile re-parse).
    - Test asserts `gap_to_seed_query("Outer Sunset","vietnamese")` equals the exact string `"vietnamese restaurants in Outer Sunset San Francisco"` AND that it is in `set(build_seed_queries())`; off-catalog inputs raise.
    - `find_demand_gaps` excludes pairs with `demand_count == 0` and pairs with pair supply ≥ `min_place_count`; output is demand-descending.
    - `loop_falsifier.py`'s `GAP` constant is NOT referenced, imported, or modified by `coverage_agent.py` (`grep -c 'loop_falsifier' scripts/coverage_agent.py` == 0) — D-03 falsifier-stub-stays guardrail.
    - `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (REGRESSION GUARDRAIL — supply-only path untouched).
  </acceptance_criteria>
  <done>Demand-gated gap scoring uses TRUE pair-level supply from place_query_hits (D-02, HIGH-1), flows through the explicit `DemandGap` dataclass (no string parsing, MEDIUM), and emits exact loop-consumable seed text with a catalog assertion; the falsifier's GAP constant is untouched and the supply-only tests still pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: ingested_query_texts (dedup-split, HIGH-2) + assert_sandbox_write_target (HIGH-3) + gap_mine_main CLI + cold-start no-op + extended MLflow demand metrics</name>
  <files>scripts/coverage_agent.py, tests/unit/test_gap_miner.py</files>
  <read_first>
    - scripts/coverage_agent.py (the file being modified — `existing_query_texts` lines ~209-223 which UNIONs `set(build_seed_queries())` with checkpoints+proposals: the demand path must NOT use this because every valid mined proposal IS in build_seed_queries() and would be dropped — REVIEW HIGH-2; `filter_already_covered` lines ~226-234; `main()` argparse lines ~290-323 to mirror; `propose_queries`/`insert_pending` to REUSE unchanged; `log_to_mlflow` lines ~261-281 to EXTEND with demand metrics; the `mlflow.set_experiment("coverage_agent")` line)
    - app/loop/falsifier_core.py lines ~168-201 (`_normalize_url` returns `(host, port, dbname, cloud_sql_instance)` — REUSE it to extract the active write dbname for the sandbox guard) and lines ~209-290 (`check_prod_safety` + `GuardResult` — the established prod-safety pattern; the sandbox dbname convention is the `_sandbox` suffix / `city_concierge_sandbox`)
    - app/config.py lines ~60-85, ~137-152 (`resolve_database_url` precedence `DATABASE_URL` > `POSTGRES_*`; `settings.resolved_database_url` — how to resolve the ACTIVE write URL the pool targets; `SANDBOX_DATABASE_URL` is read from `os.environ` directly, it is NOT a Settings field)
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § "HIGH-2" + "HIGH-3" + the "Verified against code" subsection (authoritative: split catalog-validation from dedup; add a hard write-target guard before insert_pending)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § Q6 + "Open Questions" #2 (--top-n AFTER dedup filtering; metric names gaps_found/proposals_inserted/demand_rows_scanned/unmapped_count; keep --dry-run opt-out)
    - .planning/phases/18-gap-mining-gap/18-CONTEXT.md D-04 (cold start = insert nothing, log gaps_found=0, exit 0) + D-03 (write to proposals, don't touch falsifier) + D-05 (write target MUST be sandbox)
  </read_first>
  <behavior>
    - Test 1 (HIGH-2 — valid proposal SURVIVES filtering): a freshly-mined catalog-valid gap whose seed `query_text` IS a member of `build_seed_queries()` but is NOT in checkpoints or existing proposals SURVIVES `filter_already_covered(proposals, ingested_query_texts(conn))` and reaches `insert_pending`. (Contrast: passing `existing_query_texts(conn)` — which includes the static catalog — would drop it. Assert the proposal is in `kept`, not `dropped`.)
    - Test 2 (ingested_query_texts excludes catalog): `ingested_query_texts(conn)` returns ONLY the union of `places_ingest_query_checkpoints` + existing `places_ingest_query_proposals` query_texts — it does NOT include `build_seed_queries()` members (assert a known catalog-only seed string is ABSENT from the returned set, while a stubbed checkpoint/proposal row IS present).
    - Test 3 (HIGH-3 — write guard raises off-sandbox): `assert_sandbox_write_target()` raises (SystemExit or a clear RuntimeError) when the resolved active write URL's dbname is NOT `city_concierge_sandbox` and does not equal `SANDBOX_DATABASE_URL`'s dbname (e.g. resolved dbname `city_concierge`); it returns/None-passes when the dbname is `city_concierge_sandbox`.
    - Test 4 (HIGH-3 — guard runs before insert in demand path): in the happy-path `gap_mine_main` run, `assert_sandbox_write_target` is invoked BEFORE `insert_pending` (assert call order or that a guard failure prevents any insert).
    - Test 5 (cold start, D-04): with `gather_demand` stubbed to return `({}, 0, 0)` (empty user_query_log), `gap_mine_main([])` inserts nothing, logs `gaps_found=0` and `demand_rows_scanned=0`, and returns 0.
    - Test 6 (happy path): with stubbed demand counts + stubbed `gather_pair_supply` producing one gap, `gap_mine_main([])` calls `insert_pending` with a proposal whose `query_text` is the exact `gap_to_seed_query` output, and returns 0.
    - Test 7 (dry-run opt-out): `gap_mine_main(["--dry-run"])` runs the full path but `insert_pending` inserts nothing; MLflow `proposals_inserted` is 0.
    - Test 8 (--top-n after dedup filter): with three demand gaps surviving `filter_already_covered` and `--top-n 2`, only the top-2-by-demand proposals are inserted (cap applied AFTER dedup filtering — RESEARCH Open Question #2).
    - Test 9 (MLflow demand metrics): `log_to_mlflow` is called such that `mlflow.log_metric` receives `demand_rows_scanned`, `unmapped_count`, `gaps_found`, `proposals_inserted`, and a ranked-gap artifact (`demand_gaps.json`) is logged via `mlflow.log_dict`.
    - Test 10 (judge None): when `vibe.make_judge()` returns None, the miner degrades to the cold-start/no-proposal path and still exits 0 (no crash).
  </behavior>
  <action>
    In `scripts/coverage_agent.py` add `ingested_query_texts(conn) -> set[str]` (REVIEW HIGH-2): it returns ONLY the union of `SELECT query_text FROM places_ingest_query_checkpoints` and `SELECT query_text FROM places_ingest_query_proposals` — it does NOT seed the set with `build_seed_queries()`. The existing `existing_query_texts` stays UNCHANGED so the supply-only `main()` path keeps its current behavior (no regression). Add `assert_sandbox_write_target() -> None` (REVIEW HIGH-3): resolve the active write URL (`settings.resolved_database_url`, i.e. what the pool targets), extract its dbname via the reused `_normalize_url` from `app.loop.falsifier_core`, and raise a clear error (SystemExit with a non-zero code and an explicit message, or RuntimeError) UNLESS the dbname equals `city_concierge_sandbox` OR equals the dbname parsed from `os.environ.get("SANDBOX_DATABASE_URL")`. Never write on mismatch. Add `gap_mine_main(argv: list[str] | None = None) -> int` with its OWN argparse (do NOT touch the existing `main()`): flags `--days` (default 14), `--dry-run` (store_true, opt-out — writes by default), `--min-places` (default 5), `--top-n` (int, default None = no cap). It reads `DEMAND_DATABASE_URL` from `os.environ` (None if unset → pooled sandbox read), calls `gather_demand(days, url)`, derives the demanded pairs, calls `gather_pair_supply(pairs)`, computes gaps via `find_demand_gaps(demand_counts, pair_supply, min_places)`, and on cold start (no demand gaps) logs `gaps_found=0` and returns 0 (D-04). Otherwise it builds proposals — construct `ProposedQuery(gap_to_seed_query(g.neighborhood, g.cuisine), "enriched", rationale)` directly from the catalog-valid `DemandGap`s so the seed format is guaranteed exact (the LLM `propose_queries` step may run for rationale richness, but the `query_text` MUST be `gap_to_seed_query` output, never free LLM text) — runs `filter_already_covered(proposals, ingested_query_texts(conn))` (the HIGH-2 dedup set, NOT `existing_query_texts`), applies `--top-n` to the KEPT list AFTER filtering, calls `assert_sandbox_write_target()` and THEN `insert_pending(kept[:top_n], dry_run)` (guard strictly before the write, even though the guard is a no-op under --dry-run since nothing is written), and logs to MLflow. Extend `log_to_mlflow` with new keyword params `demand_rows_scanned: int = 0` and `unmapped_count: int = 0` (defaults preserve the existing supply-only call) and log a `demand_gaps.json` ranked artifact + a `proposals_inserted` metric. Keep the existing `if __name__ == "__main__"` for `main`; `gap_mine_main` is invoked by the Makefile in Plan 04. Add the tests to `tests/unit/test_gap_miner.py`.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner.py -v && poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/coverage_agent.py` contains `def ingested_query_texts`, `def assert_sandbox_write_target`, and `def gap_mine_main`.
    - HIGH-2 test: a catalog-valid, not-yet-ingested gap (in `build_seed_queries()` but absent from checkpoints/proposals) SURVIVES `filter_already_covered(..., ingested_query_texts(conn))` and reaches `insert_pending` (asserted in `kept`, not `dropped`). `ingested_query_texts` does NOT include `build_seed_queries()` members.
    - HIGH-3 test: `assert_sandbox_write_target` raises when the resolved write dbname is non-sandbox (e.g. `city_concierge`) and passes for `city_concierge_sandbox`; the guard is invoked before `insert_pending` in the demand path.
    - `gap_mine_main` exposes `--days`, `--dry-run`, `--min-places`, `--top-n` (assert via parse/inspection); it does NOT add an opt-in `--apply` flag.
    - Cold-start test: stubbed empty demand → `gaps_found=0` logged, return 0, zero inserts (D-04).
    - `--top-n` cap is applied to the post-`filter_already_covered` kept list (Test 8).
    - Inserted proposals' `query_text` equals `gap_to_seed_query(...)` output (exact seed format — never raw LLM text), so they are loop-consumable.
    - `log_to_mlflow` logs `demand_rows_scanned`, `unmapped_count`, `gaps_found`, `proposals_inserted` and a `demand_gaps.json` artifact under the `coverage_agent` experiment.
    - The existing `main()` argparse + `existing_query_texts` + supply-only `log_to_mlflow` call still work: `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (REGRESSION GUARDRAIL).
    - `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes.
  </acceptance_criteria>
  <done>`gap_mine_main` mines demand→pair-supply→gaps→pending proposals, dedups ONLY against ingested rows (HIGH-2 fix — valid proposals survive), enforces sandbox-only writes via `assert_sandbox_write_target` before `insert_pending` (HIGH-3 fix), handles cold start, caps with --top-n after filtering, and logs demand metrics to MLflow; the existing CLI + supply-only tests still pass.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| miner → places_ingest_query_proposals | The only loop side-effect; pending rows feed ingest. Off-catalog/malformed seeds would break the downstream loop |
| miner write target → DB | Writes go through the pool (`DATABASE_URL` = sandbox by convention); HIGH-3 fix turns this into an ENFORCED invariant — must never write prod `places_raw`/proposals |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-18-03-SEED | Tampering | proposal `query_text` emission | mitigate | `query_text` is `gap_to_seed_query(n,c)` output with upfront catalog assertions — never raw LLM free text — so every inserted seed is in `build_seed_queries()` and passes `premark_seed_isolation`. Off-catalog inputs raise rather than insert. |
| T-18-03-WRITE | Elevation of Privilege | `insert_pending` write target | mitigate | **UPGRADED from convention to ENFORCED (REVIEW HIGH-3):** `assert_sandbox_write_target()` runs BEFORE `insert_pending` and resolves the active write dbname via `_normalize_url`, raising unless it is `city_concierge_sandbox` or equals `SANDBOX_DATABASE_URL`'s dbname. A misconfigured operator environment can no longer write proposals to prod — the guard refuses. The demand read path (`get_demand_conn`) remains read-only and separate. |
| T-18-03-DEDUP | Repudiation | `filter_already_covered` dedup set | mitigate | **REVIEW HIGH-2:** the demand path dedups against `ingested_query_texts` (checkpoints ∪ proposals) ONLY, never the static `build_seed_queries()` catalog, so valid mined proposals are not silently dropped — the miner can no longer be a silent no-op masked by the D-04 cold-start path. A unit test proves a catalog-valid, not-yet-ingested gap survives to insert. |
| T-18-03-COLD | Denial of Service | cold-start path | accept | D-04 honest no-op: empty demand inserts nothing and exits 0 — no resource exhaustion, no partial loop pollution. |
| T-18-03-SQLi | Tampering | `gather_pair_supply` SELECT | mitigate | Seed `query_text` strings are bound via a `%s` param list (`query_text = ANY(%s)`); no value is interpolated into the SQL string (asserted by Task 1 Test 5). |
| T-18-03-SC | Tampering | npm/pip/cargo installs | accept | No new packages (RESEARCH Package Legitimacy Audit = N/A). |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_gap_miner.py -v` exits 0.
- `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (guardrail).
- `DemandGap`, `gather_pair_supply`, `find_demand_gaps`, `gap_to_seed_query`, `ingested_query_texts`, `assert_sandbox_write_target`, `gap_mine_main` exist; `log_to_mlflow` extended with demand metrics.
- `grep -c 'loop_falsifier' scripts/coverage_agent.py` == 0 (falsifier GAP constant untouched, D-03).
- `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes.
</verification>

<success_criteria>
- GAP-02: demand×supply gap definition (D-02 gate) at TRUE pair level (REVIEW HIGH-1) + demand-descending ranking implemented via the `DemandGap` dataclass (REVIEW MEDIUM).
- GAP-03: miner writes pending proposals in exact loop-consumable seed format via `insert_pending`, dedups against ingested rows only so valid proposals survive (REVIEW HIGH-2), enforces sandbox-only writes (REVIEW HIGH-3); falsifier `GAP` constant untouched.
- GAP-04: `gap_mine_main` CLI mirrors `coverage_agent` conventions (`--days/--dry-run/--min-places/--top-n`, opt-out), logs demand metrics under the `coverage_agent` experiment, and cold-starts to a clean exit 0 (D-04).
</success_criteria>

<output>
Create `.planning/phases/18-gap-mining-gap/18-03-SUMMARY.md` when done. In the SUMMARY, explicitly note that this plan SUPERSEDES RESEARCH "Open Question #1" (per-cuisine / min(both) supply) with TRUE pair-level supply from `place_query_hits` per the cross-AI review HIGH-1 finding.
</output>
