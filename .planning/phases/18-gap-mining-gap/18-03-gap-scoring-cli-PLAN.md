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
    - "A bucket is a gap iff `demand_count > 0` AND `place_count < min_places` (default 5), ranked by `demand_count` descending — demand gates and orders, supply is the existing absolute floor (D-02)."
    - "The miner emits proposal `query_text` in the EXACT seed format `\"{cuisine} restaurants in {neighborhood} San Francisco\"` so inserted pending rows are loop-consumable and pass `loop_falsifier.premark_seed_isolation` catalog membership (D-03)."
    - "The miner writes `pending` rows to `places_ingest_query_proposals` via the existing `insert_pending` path (the productionized loop seam) and does NOT touch `loop_falsifier.py`'s `GAP` constant (D-03 — production path supersedes, falsifier stub stays)."
    - "Cold start (empty/zero-mappable `user_query_log`) inserts NOTHING, logs `gaps_found=0`, and exits 0 (D-04 honest no-op)."
    - "`gap_mine_main` mirrors `coverage_agent`'s CLI/MLflow: argparse `main()->int` with `--days/--dry-run/--min-places/--top-n` (writes by default, `--top-n` applied AFTER `filter_already_covered`); logs `gaps_found`, `proposals_inserted`, `demand_rows_scanned`, `unmapped_count` + ranked-gap artifact under the `coverage_agent` experiment (D-04 CLI/ops discretion)."
  artifacts:
    - path: "scripts/coverage_agent.py"
      provides: "find_demand_gaps(), gap_to_seed_query(), gap_mine_main(), extended log_to_mlflow()"
      contains: "def gap_mine_main"
    - path: "tests/unit/test_gap_miner.py"
      provides: "Unit tests for D-02 ranking, seed-format exactness, cold-start exit, --top-n placement, MLflow metrics"
  key_links:
    - from: "scripts/coverage_agent.py gap_mine_main"
      to: "places_ingest_query_proposals"
      via: "insert_pending (reused, unchanged)"
      pattern: "insert_pending\\("
    - from: "scripts/coverage_agent.py gap_to_seed_query"
      to: "build_seed_queries() catalog format"
      via: "exact f-string match"
      pattern: "restaurants in .* San Francisco"
    - from: "scripts/coverage_agent.py gap_mine_main"
      to: "MLflow coverage_agent experiment"
      via: "extended log_to_mlflow with demand metrics"
      pattern: "demand_rows_scanned"
---

<objective>
Wire the demand signal into gaps and the loop: `find_demand_gaps` applies the D-02 filter (`demand>0 AND supply<min_places`, ranked by demand desc), `gap_to_seed_query` emits the exact loop-consumable seed format, and `gap_mine_main` is the new CLI entrypoint that reuses `propose_queries`/`filter_already_covered`/`insert_pending` to write `pending` proposals, handles the cold-start no-op, and logs demand metrics to MLflow.

Purpose: This closes GAP-02 (demand×supply gap definition + ranking), GAP-03 (loop-integration write via the proposals seam — the productionized replacement for the hardcoded constant), and GAP-04 (CLI/ops/MLflow/cold-start). It composes the existing supply-side pipeline (unchanged) with the demand path from Plan 02.

Output: `find_demand_gaps`, `gap_to_seed_query`, `gap_mine_main`, and an extended `log_to_mlflow` in `coverage_agent.py`; new unit tests in `tests/unit/test_gap_miner.py`.
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
@scripts/coverage_agent.py
@scripts/ingest_places_sf.py
@scripts/loop_falsifier.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: find_demand_gaps (D-02 ranking) + gap_to_seed_query (exact seed format)</name>
  <files>scripts/coverage_agent.py, tests/unit/test_gap_miner.py</files>
  <read_first>
    - scripts/coverage_agent.py (the file being modified — `CoverageStat` dataclass, `find_gaps`'s `place_count < min_place_count` floor to reuse as the supply gate, the existing `gather_stats` bucket strings `neighborhood:X`/`cuisine:Y`)
    - scripts/ingest_places_sf.py (build_seed_queries line ~319 — the canonical seed string `f"{cuisine} restaurants in {neighborhood} San Francisco"`; CUISINES/NEIGHBORHOODS catalogs)
    - scripts/loop_falsifier.py lines ~64-69 (the GAP constant + `SEED_QUERY = f"{GAP[1]} restaurants in {GAP[0]} San Francisco"` — the EXACT format gap_to_seed_query must match; DO NOT delete or import this constant) and lines ~173-205 (premark_seed_isolation's catalog-membership assertion the seed format must satisfy)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § Q5 + "Open Questions" #1 (supply join choice: use per-cuisine supply for the gate to cover "no Vietnamese anywhere in SF", per-neighborhood for proposal text) + "Code Examples" find_demand_gaps
  </read_first>
  <behavior>
    - Test 1 (D-02 gate): given supply stats where `cuisine:vietnamese` has place_count 2 and `cuisine:italian` has 100, and demand counts `{("Outer Sunset","vietnamese"): 5, ("Mission District","italian"): 3}` with min_places=5, only the vietnamese bucket is a gap (italian supply ≥ floor → excluded).
    - Test 2 (demand gates): a bucket with `place_count=0` (thin supply) but `demand_count=0` is NOT a gap — demand must be > 0.
    - Test 3 (ranking): with two qualifying gaps demand 5 and demand 9, the result is ordered demand-descending (9 before 5).
    - Test 4 (seed format exactness): `gap_to_seed_query("Outer Sunset", "vietnamese") == "vietnamese restaurants in Outer Sunset San Francisco"` AND that exact string is a member of `set(build_seed_queries())` (so premark_seed_isolation would accept it).
    - Test 5 (catalog assertion): `gap_to_seed_query` rejects off-catalog inputs (neighborhood ∉ NEIGHBORHOODS or cuisine ∉ CUISINES) — raises rather than emitting an un-ingestable seed.
  </behavior>
  <action>
    In `scripts/coverage_agent.py` add `find_demand_gaps(stats: list[CoverageStat], demand_counts: dict[tuple[str,str], int], min_place_count: int = 5) -> list[tuple[CoverageStat, int]]` implementing D-02: build a supply map from the existing `cuisine:`/`neighborhood:` stat buckets; for each `(neighborhood, cuisine)` with `demand > 0`, gate on per-cuisine supply `< min_place_count` (covers the city-wide-thin-cuisine case; use neighborhood supply only for the proposal text), produce a `CoverageStat` carrying the demand in `distinct_queries`, and return the list sorted by demand descending (RESEARCH Open Question #1 resolution: per-cuisine supply for the gate). Add `gap_to_seed_query(neighborhood: str, cuisine: str) -> str` returning `f"{cuisine} restaurants in {neighborhood} San Francisco"` with upfront assertions that both are catalog members (raise on off-catalog — never emit an un-ingestable seed). Do NOT import or modify `loop_falsifier.GAP`. Add the tests to `tests/unit/test_gap_miner.py`.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner.py -v && poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/coverage_agent.py` contains `def find_demand_gaps` and `def gap_to_seed_query`.
    - Test asserts `gap_to_seed_query("Outer Sunset","vietnamese")` equals the exact string `"vietnamese restaurants in Outer Sunset San Francisco"` AND that it is in `set(build_seed_queries())`.
    - `find_demand_gaps` excludes buckets with `demand_count == 0` and buckets with supply ≥ `min_place_count`; output is demand-descending.
    - `loop_falsifier.py`'s `GAP` constant is NOT referenced, imported, or modified by `coverage_agent.py` (`grep -c 'loop_falsifier' scripts/coverage_agent.py` == 0) — D-03 falsifier-stub-stays guardrail.
    - `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (REGRESSION GUARDRAIL).
  </acceptance_criteria>
  <done>Demand-gated gap scoring (D-02) and exact loop-consumable seed-format emission exist and are unit-tested; the falsifier's GAP constant is untouched.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: gap_mine_main CLI + cold-start no-op + extended MLflow demand metrics</name>
  <files>scripts/coverage_agent.py, tests/unit/test_gap_miner.py</files>
  <read_first>
    - scripts/coverage_agent.py (the file being modified — `main()` argparse shape lines ~290-323 to mirror; `propose_queries`/`existing_query_texts`/`filter_already_covered`/`insert_pending` to REUSE unchanged; `log_to_mlflow` lines ~261-281 to EXTEND with demand metrics; the `mlflow.set_experiment("coverage_agent")` line)
    - scripts/loop_falsifier.py log_to_mlflow lines ~335-385 (the same `coverage_agent` experiment is shared — confirm metric names don't collide)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § Q6 + "Open Questions" #2 (--top-n AFTER filter_already_covered; metric names gaps_found/proposals_inserted/demand_rows_scanned/unmapped_count; keep --dry-run opt-out)
    - .planning/phases/18-gap-mining-gap/18-CONTEXT.md D-04 (cold start = insert nothing, log gaps_found=0, exit 0) + D-03 (write to proposals, don't touch falsifier)
  </read_first>
  <behavior>
    - Test 1 (cold start, D-04): with `gather_demand` stubbed to return `({}, 0, 0)` (empty user_query_log), `gap_mine_main([])` inserts nothing, logs `gaps_found=0` and `demand_rows_scanned=0`, and returns 0.
    - Test 2 (happy path): with stubbed demand counts + supply stats producing one gap and a stub LLM proposing the matching seed string, `gap_mine_main([])` calls `insert_pending` with a proposal whose `query_text` is the exact seed format, and returns 0.
    - Test 3 (dry-run opt-out): `gap_mine_main(["--dry-run"])` runs the full path but `insert_pending` inserts nothing (the existing dry_run contract); MLflow `proposals_inserted` is 0.
    - Test 4 (--top-n after filter): with three demand gaps surviving `filter_already_covered` and `--top-n 2`, only the top-2-by-demand proposals are inserted (cap applied AFTER filter_already_covered — RESEARCH Open Question #2).
    - Test 5 (MLflow demand metrics): `log_to_mlflow` is called such that `mlflow.log_metric` receives `demand_rows_scanned` and `unmapped_count` and `gaps_found`, and a ranked-gap artifact (`demand_gaps.json`) is logged via `mlflow.log_dict`.
    - Test 6 (judge None): when `vibe.make_judge()` returns None, the miner degrades to the cold-start/no-proposal path and still exits 0 (no crash).
  </behavior>
  <action>
    In `scripts/coverage_agent.py` add `gap_mine_main(argv: list[str] | None = None) -> int` with its OWN argparse (do NOT touch the existing `main()`): flags `--days` (default 14), `--dry-run` (store_true, opt-out — writes by default), `--min-places` (default 5), `--top-n` (int, default None = no cap). It reads `DEMAND_DATABASE_URL` from `os.environ` (None if unset → pooled sandbox read), calls `gather_demand(days, url)` and `gather_stats(days)`, computes gaps via `find_demand_gaps`, and on cold start (no demand gaps) logs `gaps_found=0` and returns 0 (D-04). Otherwise it builds proposals — prefer constructing `ProposedQuery(gap_to_seed_query(n,c), "enriched", rationale)` directly from the catalog-valid gaps so the seed format is guaranteed exact (the LLM `propose_queries` step may still run for rationale richness, but the `query_text` MUST be `gap_to_seed_query` output, never free LLM text, to preserve loop-consumability) — runs `filter_already_covered(proposals, existing_query_texts(conn))`, applies `--top-n` to the KEPT list AFTER filtering, calls `insert_pending(kept[:top_n], dry_run)`, and logs to MLflow. Extend `log_to_mlflow` with new keyword params `demand_rows_scanned: int = 0` and `unmapped_count: int = 0` (defaults preserve the existing supply-only call) and log a `demand_gaps.json` ranked artifact + a `proposals_inserted` metric. Add a `if __name__ == "__main__"` guard option or keep the existing one for `main`; `gap_mine_main` is invoked by the Makefile in Plan 04. Add the tests to `tests/unit/test_gap_miner.py`.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner.py -v && poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/coverage_agent.py` contains `def gap_mine_main`.
    - `gap_mine_main` exposes `--days`, `--dry-run`, `--min-places`, `--top-n` (assert via `--help` parse or by inspecting the ArgumentParser in a test); it does NOT add an opt-in `--apply` flag.
    - Cold-start test: stubbed empty demand → `gaps_found=0` logged, return 0, zero inserts (D-04).
    - `--top-n` cap is applied to the post-`filter_already_covered` kept list (asserted by Test 4).
    - Inserted proposals' `query_text` equals `gap_to_seed_query(...)` output (exact seed format — never raw LLM text), so they are loop-consumable.
    - `log_to_mlflow` logs `demand_rows_scanned`, `unmapped_count`, `gaps_found`, `proposals_inserted` and a `demand_gaps.json` artifact under the `coverage_agent` experiment.
    - The existing `main()` argparse contract and `log_to_mlflow` supply-only call still work: `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (REGRESSION GUARDRAIL).
    - `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes.
  </acceptance_criteria>
  <done>`gap_mine_main` mines demand→gaps→pending proposals via the reused pipeline, handles cold start, caps with --top-n after filtering, and logs demand metrics to MLflow; existing CLI + supply-only tests still pass.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| miner → places_ingest_query_proposals | The only loop side-effect; pending rows feed ingest. Off-catalog/malformed seeds would break the downstream loop |
| miner write target → DB | Writes go through the pool (`DATABASE_URL` = sandbox); must never write prod `places_raw` |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-18-03-SEED | Tampering | proposal `query_text` emission | mitigate | `query_text` is `gap_to_seed_query(n,c)` output with upfront catalog assertions — never raw LLM free text — so every inserted seed is in `build_seed_queries()` and passes `premark_seed_isolation`. Off-catalog inputs raise rather than insert. |
| T-18-03-WRITE | Elevation of Privilege | `insert_pending` write target | mitigate | `insert_pending` (reused, unchanged) writes via the pool → `DATABASE_URL` (sandbox in dev/CI). The demand read path (`get_demand_conn`) is read-only and separate. No write ever targets prod `places_raw`. |
| T-18-03-COLD | Denial of Service | cold-start path | accept | D-04 honest no-op: empty demand inserts nothing and exits 0 — no resource exhaustion, no partial loop pollution. |
| T-18-03-SC | Tampering | npm/pip/cargo installs | accept | No new packages (RESEARCH Package Legitimacy Audit = N/A). |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_gap_miner.py -v` exits 0.
- `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (guardrail).
- `find_demand_gaps`, `gap_to_seed_query`, `gap_mine_main` exist; `log_to_mlflow` extended with demand metrics.
- `grep -c 'loop_falsifier' scripts/coverage_agent.py` == 0 (falsifier GAP constant untouched, D-03).
- `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes.
</verification>

<success_criteria>
- GAP-02: demand×supply gap definition (D-02 gate) + demand-descending ranking implemented.
- GAP-03: miner writes pending proposals in exact loop-consumable seed format via `insert_pending`; falsifier `GAP` constant untouched.
- GAP-04: `gap_mine_main` CLI mirrors `coverage_agent` conventions (`--days/--dry-run/--min-places/--top-n`, opt-out), logs demand metrics under the `coverage_agent` experiment, and cold-starts to a clean exit 0 (D-04).
</success_criteria>

<output>
Create `.planning/phases/18-gap-mining-gap/18-03-SUMMARY.md` when done.
</output>
