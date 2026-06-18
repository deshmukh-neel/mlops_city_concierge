---
phase: 18-gap-mining-gap
plan: 04
type: execute
wave: 3
depends_on: [18-03]
files_modified:
  - tests/unit/test_gap_miner_smoke.py
  - tests/integration/test_gap_miner.py
  - Makefile
  - CLAUDE.md
  - AGENTS.md
  - .github/copilot-instructions.md
autonomous: true
requirements: [GAP-03, GAP-04]
must_haves:
  truths:
    - "The full demand-mining path is covered at all four test layers the project owner requires: unit/mock (Plans 02/03) + smoke (module imports, gap_mine_main invokable) + functional (stubbed DB+LLM walks gather_demand→gather_pair_supply→find_demand_gaps→insert_pending→MLflow) + integration (real seeded-sandbox demand → real pending proposal row), so the new module is not unit-only."
    - "The integration test is DETERMINISTIC and STATEFUL-SAFE (REVIEW MEDIUM): it forces a gap by passing a high `--min-places` so the seeded bucket is guaranteed under threshold, and it selects a catalog `(neighborhood, cuisine)` pair that is absent from BOTH `places_ingest_query_proposals` AND the NORMALIZED completed-checkpoint set BEFORE asserting it gets inserted — so the new checkpoint-prefix normalization (Plan 03 ROUND-2 NEW HIGH) cannot cause the target pair to be silently deduped (REVIEW ROUND-2 MEDIUM-1). It cleans up ONLY the rows it inserted (tracked by a unique rationale marker / returned id) — it NEVER blanket-deletes by `query_text`."
    - "Integration tests are skipped unless `APP_ENV=integration` (project convention) and seed the sandbox via the Plan 01 demand-seed helper, then assert a real `pending` proposal row in `places_ingest_query_proposals` with the exact loop-consumable seed format (D-03). If no catalog pair is free of both proposals and normalized checkpoints, the test SKIPS with an explicit reason rather than asserting against a pre-deduped pair (REVIEW ROUND-2 MEDIUM-1)."
    - "A `make gap-mine` target wraps `gap_mine_main` mirroring `make coverage-agent`/`make loop-falsifier` house style (D-04 CLI/ops)."
    - "CLAUDE.md, AGENTS.md, and `.github/copilot-instructions.md` are kept in sync (project sync notice) with the new `make gap-mine` command and a one-line description of the demand-driven gap miner."
  artifacts:
    - path: "tests/unit/test_gap_miner_smoke.py"
      provides: "Smoke + functional tests for the demand-mining entrypoint"
    - path: "tests/integration/test_gap_miner.py"
      provides: "Deterministic, scoped-cleanup integration test: seeded sandbox demand → real pending proposal, picks a pair free of proposals AND normalized checkpoints"
    - path: "Makefile"
      provides: "gap-mine target"
      contains: "gap-mine:"
  key_links:
    - from: "Makefile gap-mine"
      to: "scripts/coverage_agent.py gap_mine_main"
      via: "python -c entrypoint or module invocation"
      pattern: "gap_mine_main\\|coverage_agent"
    - from: "tests/integration/test_gap_miner.py"
      to: "scripts/seed_demand_log.py"
      via: "seeds sandbox user_query_log before mining"
      pattern: "seed_demand"
    - from: "tests/integration/test_gap_miner.py"
      to: "scripts.coverage_agent ingested_query_texts"
      via: "normalized completed-checkpoint set used to pick a clean target pair"
      pattern: "ingested_query_texts"
---

<objective>
Complete the test pyramid for the demand miner (smoke + functional + integration, on top of the unit tests from Plans 02/03), add the `make gap-mine` target, and sync the new command into the three agent-instruction files (CLAUDE.md, AGENTS.md, copilot-instructions.md).

Purpose: The project owner's non-negotiable test-layering preference requires unit/mock + smoke + functional + integration for new modules. Plans 02/03 delivered unit/mock; this plan adds the remaining three layers and the operator-facing ergonomics (Make target + docs). The integration test is the honest end-to-end proof that seeded demand produces a real loop-consumable proposal row (GAP-03) — made DETERMINISTIC and STATEFUL-SAFE per the cross-AI review MEDIUMs, and (ROUND-2 MEDIUM-1) made robust to the new checkpoint-prefix normalization by choosing a target pair that is absent from both proposals AND the normalized completed-checkpoint set.

Output: `tests/unit/test_gap_miner_smoke.py` (smoke + functional), `tests/integration/test_gap_miner.py` (real-DB, deterministic, scoped cleanup, checkpoint-aware target selection), `make gap-mine`, and the three synced docs.
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
@tests/unit/test_coverage_agent_smoke.py
@tests/integration/test_coverage_agent.py
@Makefile
@CLAUDE.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Smoke + functional tests for the demand-mining entrypoint</name>
  <files>tests/unit/test_gap_miner_smoke.py</files>
  <read_first>
    - tests/unit/test_coverage_agent_smoke.py (the EXACT smoke+functional pattern to mirror — `_StubCursor`/`_StubConn` branching on SQL substrings, mlflow MagicMock setup, the `test_smoke_module_imports` and `test_functional_dry_run_emits_artifacts` structure)
    - scripts/coverage_agent.py (the demand entrypoint `gap_mine_main` + `gather_demand`/`gather_pair_supply`/`find_demand_gaps`/`ingested_query_texts`/`log_to_mlflow` from Plans 02/03; the `_StubCursor` must now branch for each distinct SELECT — user_query_log rows for the demand SELECT (`FROM user_query_log`), pair-supply rows for the pair-supply SELECT (`FROM place_query_hits`), prefixed checkpoint rows for `ingested_query_texts`'s checkpoint SELECT (`FROM places_ingest_query_checkpoints`), and an explicit empty result for `ingested_query_texts`'s proposals SELECT (`FROM places_ingest_query_proposals`) — so the reads don't cross-contaminate and BOTH `ingested_query_texts` branches are exercised)
    - scripts/sandbox_guard.py (Plan 01's `assert_sandbox_write_target` — monkeypatch it to a no-op in the unit/smoke context, since there is no real sandbox connection to query `current_database()` against)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § "Pitfall 6" (cold-start is the default local state; the functional test must stub demand rows so it exercises the populated path, and a separate test must exercise the cold-start no-op)
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § HIGH-1/HIGH-2 + ROUND-2 NEW HIGH (so the functional stubs exercise pair-supply and the ingested-only/prefix-normalized dedup correctly — the checkpoint stub returns a PREFIXED `all::...` row to confirm the smoke path tolerates it)
  </read_first>
  <behavior>
    - Smoke 1: `scripts.coverage_agent` imports cleanly and exposes `gap_mine_main`, `gather_demand`, `gather_pair_supply`, `find_demand_gaps`, `gap_to_seed_query`, `get_demand_conn`, `ingested_query_texts` (module contract); `assert_sandbox_write_target` is importable from `scripts.sandbox_guard`.
    - Smoke 2: `gap_mine_main(["--dry-run","--days","1"])` against an empty stubbed DB (no user_query_log rows) returns 0 and logs `gaps_found=0` — the cold-start no-op (D-04) runs cleanly.
    - Functional 1: with a stubbed connection returning user_query_log rows that map to a thin catalog bucket, stubbed `place_query_hits` showing that bucket's pair supply below min_places, ingested-query reads returning a PREFIXED `all::...` checkpoint row that does NOT match the mined seed (so it survives dedup — HIGH-2 + ROUND-2 NEW HIGH), and a stub LLM, `gap_mine_main(["--dry-run","--days","1"])` returns 0 and `mlflow.log_dict` is called with a `demand_gaps.json` artifact containing the ranked gap.
    - Functional 2: the dry-run functional path never calls the real `insert_pending` write (no INSERT execute captured on the stub conn).
  </behavior>
  <action>
    Create `tests/unit/test_gap_miner_smoke.py` mirroring `tests/unit/test_coverage_agent_smoke.py`'s structure. Extend the stub cursor to branch on SQL substrings: return seeded demand tuples (`message`, `requested_primary_types`) when the executed SQL contains `FROM user_query_log`; return pair-supply rows (`query_text`, count) when it contains `FROM place_query_hits`; return checkpoint rows when it contains `FROM places_ingest_query_checkpoints` (return a PREFIXED `all::...` checkpoint string that does NOT normalize to the mined seed, so the proposal survives `ingested_query_texts` dedup — HIGH-2 + ROUND-2 NEW HIGH); return an explicit empty list `[]` when it contains `FROM places_ingest_query_proposals` (so `ingested_query_texts`'s second SELECT branch is exercised, not silently caught by the default); return `[]` otherwise. Monkeypatch `vibe.make_judge` (stub LLM for neighborhood extraction returning a catalog neighborhood, and for any proposal step), monkeypatch `scripts.sandbox_guard.assert_sandbox_write_target` (and/or its import in coverage_agent) to a no-op (unit context has no real sandbox), and monkeypatch all `mlflow.*` calls. Cover the four behaviors above: module smoke (assert all the demand names present + the guard importable), cold-start smoke, populated functional with `demand_gaps.json` artifact assertion, and dry-run-no-write. Do NOT hit a real DB or LLM.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `tests/unit/test_gap_miner_smoke.py` exists with a module-smoke test asserting `gap_mine_main`/`gather_demand`/`gather_pair_supply`/`find_demand_gaps`/`ingested_query_texts` are present and `assert_sandbox_write_target` is importable from `scripts.sandbox_guard`.
    - A cold-start smoke test asserts `gap_mine_main(["--dry-run","--days","1"])` on an empty stubbed DB returns 0 and logs `gaps_found=0` (D-04).
    - A functional test asserts the populated path (pair supply below threshold, a PREFIXED `all::...` checkpoint stub that does NOT match the mined seed) logs a `demand_gaps.json` ranked artifact and returns 0.
    - `poetry run pytest tests/unit/test_gap_miner_smoke.py -v` exits 0.
    - `poetry run ruff check tests/unit/test_gap_miner_smoke.py` passes.
  </acceptance_criteria>
  <done>Smoke + functional layers exist: the demand entrypoint imports, cold-starts cleanly, and walks the populated path (pair supply + ingested-only/prefix-normalized dedup) emitting demand artifacts — all with stubs, no real services.</done>
</task>

<task type="auto">
  <name>Task 2: Deterministic integration test — seeded sandbox demand → real pending proposal row, checkpoint-aware target pick, scoped cleanup</name>
  <files>tests/integration/test_gap_miner.py</files>
  <read_first>
    - tests/integration/test_coverage_agent.py (the EXACT integration pattern to mirror — `pytestmark = pytest.mark.skipif(APP_ENV != integration)`, the `_proposals_table_or_skip` fixture that checks table existence + INSERT privilege, the `_purge_test_proposals` marker-cleanup, the `_stub_mlflow` helper)
    - scripts/seed_demand_log.py (Plan 01 — `seed_demand_rows`/`insert_demand_rows` to populate sandbox `user_query_log` with a catalog-valid overlap row before mining; the helper is sandbox-write-guarded and supports passing the connection to guard)
    - scripts/coverage_agent.py (`gap_mine_main` — the entrypoint under test; `gap_to_seed_query` for the expected proposal text; `ingested_query_texts(conn)` — the NORMALIZED dedup set the test uses to verify the chosen target pair is not already covered; `--min-places` for forcing the gap)
    - scripts/ingest_places_sf.py (NEIGHBORHOODS / CUISINES / build_seed_queries — the catalog the test iterates to find a pair whose seed is absent from both proposals and normalized checkpoints)
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py (the user_query_log columns — the integration test must skip gracefully if user_query_log is absent, mirroring the proposals-table skip, since the shared CI DB may not have it)
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § MEDIUM "Integration-test determinism" + ROUND-2 "MEDIUM-1" (authoritative: force the gap with high --min-places; choose a catalog pair absent from BOTH proposals AND the NORMALIZED completed-checkpoint set so the new prefix normalization can't pre-dedupe the target; clean up ONLY test-inserted rows by unique marker; skip with a clear reason if no clean pair exists)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § "Environment Availability" (integration runs need a real DATABASE_URL with user_query_log + proposals; skip-on-absent is the contract)
  </read_first>
  <action>
    Create `tests/integration/test_gap_miner.py` gated on `APP_ENV=integration` (mirror `test_coverage_agent.py`'s `pytestmark`). Add a `_user_query_log_or_skip` fixture mirroring `_proposals_table_or_skip` that skips when `user_query_log` is absent OR the role lacks INSERT on it (so the test starts passing once the sandbox/CI DB has the Phase 17 migration — the known shared-DB skip pattern from memory project_full_suite_db_pool_contamination and the existing integration test). The test, in order: (1) builds the NORMALIZED dedup set via `ingested_query_texts(conn)` (which strips the `FIELD_MODE::` checkpoint prefix per Plan 03 ROUND-2 NEW HIGH) AND reads the current `places_ingest_query_proposals` query_texts, then iterates `[(n, c) for n in NEIGHBORHOODS for c in CUISINES]` to pick the FIRST pair whose `gap_to_seed_query(n, c)` is in `set(build_seed_queries())` (catalog-valid) AND is absent from BOTH the normalized dedup set AND the existing proposals — if NO such pair exists, `pytest.skip` with an explicit reason ("no catalog pair free of proposals and normalized checkpoints") rather than asserting against a pre-deduped pair (REVIEW ROUND-2 MEDIUM-1); (2) records the expected seed `query_text = gap_to_seed_query(chosen_n, chosen_c)` and confirms (defensively) no pre-existing proposal row has that text (so cleanup never deletes a legitimate pre-existing proposal — REVIEW MEDIUM); (3) seeds demand rows for the CHOSEN pair via `insert_demand_rows` with a UNIQUE per-run rationale/marker AND a `message` containing the chosen neighborhood + a `requested_primary_types` mapping to the chosen cuisine (so the demand path maps it lexically — no LLM dependency); (4) stubs `vibe.make_judge` so neighborhood extraction is deterministic (return the chosen neighborhood) — but the lexical pre-pass should already cover it; (5) stubs mlflow; (6) runs `gap_mine_main(["--days","30","--min-places","100000"])` — the deliberately huge `--min-places` FORCES the seeded bucket under threshold so the gap is GUARANTEED regardless of real sandbox supply (REVIEW MEDIUM determinism); (7) asserts a real `pending` row exists in `places_ingest_query_proposals` whose `query_text == gap_to_seed_query(chosen_n, chosen_c)`; (8) in a `finally`, cleans up ONLY the rows the test created — delete the seeded demand rows by their unique marker, and delete the proposal row ONLY because step (1)/(2) confirmed it did NOT pre-exist (track by the row id returned at insert if available, else by the exact chosen query_text). NEVER blanket-delete by query_text when a pre-existing proposal was present. Use parameterised DELETEs for cleanup (no string interpolation). Mirror `_purge_test_proposals` for the marker-scoped delete.
  </action>
  <verify>
    <automated>APP_ENV=integration poetry run pytest tests/integration/test_gap_miner.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `tests/integration/test_gap_miner.py` exists, gated on `APP_ENV=integration`, and skips gracefully when `user_query_log` or `places_ingest_query_proposals` is absent / unwritable (so the default `make test` run skips it without error per the shared-DB convention).
    - The test selects a catalog `(neighborhood, cuisine)` pair absent from BOTH `places_ingest_query_proposals` AND the NORMALIZED completed-checkpoint set from `ingested_query_texts(conn)`, or `pytest.skip`s with an explicit reason when none is free (REVIEW ROUND-2 MEDIUM-1 — the new prefix normalization cannot silently pre-dedupe the target).
    - The test FORCES the gap via a high `--min-places` so it does not depend on real sandbox supply (REVIEW MEDIUM determinism).
    - The test detects a pre-existing proposal for the chosen seed BEFORE mining and, in cleanup, deletes the proposal ONLY if the test created it — it never blanket-deletes by `query_text` when a legitimate pre-existing proposal was present (REVIEW MEDIUM scoped cleanup).
    - The test cleans up its seeded demand rows (by unique marker) and any test-created proposal in a `finally` (no residue).
    - `poetry run pytest tests/integration/test_gap_miner.py -v` (without APP_ENV) skips cleanly (exit 0, skipped).
    - `poetry run ruff check tests/integration/test_gap_miner.py` passes.
  </acceptance_criteria>
  <done>A deterministic integration test proves seeded sandbox demand produces a real loop-consumable pending proposal (GAP-03 end-to-end) using a forced gap and a target pair guaranteed free of proposals AND normalized checkpoints (ROUND-2 MEDIUM-1), scopes its cleanup to only test-created rows, and skips gracefully on a DB without the Phase 17 migration.</done>
</task>

<task type="auto">
  <name>Task 3: make gap-mine target + sync CLAUDE.md / AGENTS.md / copilot-instructions.md</name>
  <files>Makefile, CLAUDE.md, AGENTS.md, .github/copilot-instructions.md</files>
  <read_first>
    - Makefile lines ~93-99 (the `coverage-agent`/`coverage-agent-apply` targets and `$(POETRY_RUN)` style to mirror; line ~227 `loop-falsifier` for the help-comment style; the `sandbox-migrate` target added in Plan 01 for adjacency)
    - scripts/coverage_agent.py (how `gap_mine_main` is invoked — a `python -c "from scripts.coverage_agent import gap_mine_main; import sys; sys.exit(gap_mine_main())"` wrapper, or a module entrypoint; choose the invocation that matches how the script exposes gap_mine_main from Plan 03)
    - CLAUDE.md lines ~11-40 (the `## Commands` block where `make coverage-agent`-style commands live; the "Sync notice" at the top requiring AGENTS.md + copilot-instructions.md mirroring)
    - AGENTS.md + .github/copilot-instructions.md (the corresponding command sections to keep in sync — find the coverage-agent / loop-falsifier command listing)
  </read_first>
  <action>
    Add a `.PHONY: gap-mine` / `gap-mine:` target to the Makefile wrapping `gap_mine_main` (default writes; per D-04 the primary `gap-mine` writes by default — optionally add a `gap-mine-dry` that passes `--dry-run`). Use `$(POETRY_RUN)` and a `## ` help comment in the established style ("Mine real demand×supply gaps from user_query_log → pending proposals (GAP)"). Then mirror the new command into CLAUDE.md's `## Commands` block AND AGENTS.md AND `.github/copilot-instructions.md` (the project sync notice mandates all three move together) with a one-line description of the demand-driven gap miner and a note that it reads demand from sandbox by default or `DEMAND_DATABASE_URL` when set, and writes proposals to the sandbox (guarded by `assert_sandbox_write_target`, which checks the live `current_database()`). Do NOT update the `implementation_plan/james/` workstream tables (that bookkeeping is for the agent product roadmap, not the v2.3 loop scripts).
  </action>
  <verify>
    <automated>grep -c 'gap-mine' Makefile && grep -c 'gap-mine' CLAUDE.md && grep -c 'gap-mine' AGENTS.md && grep -c 'gap-mine' .github/copilot-instructions.md</automated>
  </verify>
  <acceptance_criteria>
    - `Makefile` contains a `gap-mine:` target that invokes `gap_mine_main` via `$(POETRY_RUN)`.
    - `make -n gap-mine` (dry print) shows the `gap_mine_main` invocation without error.
    - The string `gap-mine` appears in CLAUDE.md, AGENTS.md, AND `.github/copilot-instructions.md` (all three synced per the sync notice).
    - Each doc's new line states the miner reads demand (sandbox by default / `DEMAND_DATABASE_URL` opt-in) and writes pending proposals to the sandbox.
    - `make help` (if present) lists `gap-mine` with its `##` description.
  </acceptance_criteria>
  <done>`make gap-mine` runs the miner; the command is documented identically across CLAUDE.md, AGENTS.md, and copilot-instructions.md (sync notice honored).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| integration test → sandbox DB | Seeds demand + reads/writes proposals against a real DB; must target sandbox, must clean up only its own rows |
| make gap-mine → DB | Operator-run; writes proposals to the pool (`DATABASE_URL` = sandbox), enforced by the `current_database()`-based `assert_sandbox_write_target` |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-18-04-INT | Tampering | integration test seed/cleanup | mitigate | Test is gated on `APP_ENV=integration`, FORCES the gap deterministically (high `--min-places`), picks a target pair absent from BOTH proposals AND the NORMALIZED checkpoint set (ROUND-2 MEDIUM-1) or skips with a clear reason, detects pre-existing proposals, and purges ONLY test-created demand + proposal rows in a `finally` with parameterised DELETEs scoped by unique marker — never a blanket query_text delete (REVIEW MEDIUM). Skips gracefully when the table is absent (shared-DB safety). |
| T-18-04-WRITE | Elevation of Privilege | `make gap-mine` write target | mitigate | The target invokes `gap_mine_main`, which calls `assert_sandbox_write_target()` (shared `scripts.sandbox_guard`, checks live `current_database()`) before `insert_pending` (Plan 03, REVIEW HIGH-3 + ROUND-2 H3) — writes go via the pool → `DATABASE_URL` and are refused unless the resolved dbname is the sandbox. No prod `places_raw` write path. Docs state the write target is the sandbox. |
| T-18-04-DOC | Information Disclosure | docs sync | accept | Docs reference `DEMAND_DATABASE_URL` by name only (no credential value); the placeholder lives in `.env.example` (gitignored real `.env`). |
| T-18-04-SC | Tampering | npm/pip/cargo installs | accept | No new packages (RESEARCH Package Legitimacy Audit = N/A). |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_gap_miner_smoke.py -v` exits 0.
- `poetry run pytest tests/integration/test_gap_miner.py -v` skips cleanly without `APP_ENV=integration`.
- `make -n gap-mine` prints the `gap_mine_main` invocation.
- `gap-mine` present in Makefile + CLAUDE.md + AGENTS.md + .github/copilot-instructions.md.
- `make test` (full suite) passes — watch the known full-suite DB-pool contamination gotcha (real-graph tests need mocks); ensure the new integration test skips, not leaks a live pool.
</verification>

<success_criteria>
- All four test layers present for the demand miner (unit from 02/03 + smoke + functional + integration here) — project test-layering preference satisfied.
- GAP-03 proven end-to-end: seeded sandbox demand → real pending proposal in exact seed format, via a deterministic forced-gap test that picks a target pair free of proposals AND normalized checkpoints (ROUND-2 MEDIUM-1) with scoped cleanup (REVIEW MEDIUM).
- GAP-04 operator ergonomics: `make gap-mine` exists and is documented identically in CLAUDE.md, AGENTS.md, and copilot-instructions.md.
</success_criteria>

<output>
Create `.planning/phases/18-gap-mining-gap/18-04-SUMMARY.md` when done.
</output>
