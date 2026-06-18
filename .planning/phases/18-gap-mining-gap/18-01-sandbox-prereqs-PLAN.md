---
phase: 18-gap-mining-gap
plan: 01
type: execute
wave: 0
depends_on: []
files_modified:
  - .env.example
  - Makefile
  - scripts/sandbox_guard.py
  - tests/unit/test_sandbox_guard.py
  - scripts/seed_demand_log.py
  - tests/unit/test_seed_demand_log.py
autonomous: false
requirements: [GAP-01, GAP-04]
must_haves:
  truths:
    - "The sandbox `city_concierge_sandbox` DB has the `user_query_log` table (Phase 17 migration d1be72aea7d4 applied) so the demand miner can read it (D-05 sandbox-only default)."
    - "Sandbox-migrate is REPRODUCIBLE, not a one-time human action: a `make sandbox-migrate` target applies `alembic upgrade head` against `SANDBOX_DATABASE_URL`, AND `scripts/seed_demand_log.py` guards that `user_query_log` exists, erroring with the exact `make sandbox-migrate` fix command if absent (REVIEW MEDIUM — future sandboxes can't silently regress)."
    - "`.env.example` documents the optional `DEMAND_DATABASE_URL` prod-read env var that activates the prod-read demand path (D-05)."
    - "A deterministic seed helper inserts representative catalog-valid rows into sandbox `user_query_log` so functional/integration tests of the demand path (D-01) have real data and the cold-start path (D-04) can be distinguished from a broken miner."
    - "The sandbox-write guard `assert_sandbox_write_target()` lives in ONE shared module `scripts/sandbox_guard.py` (created here in wave 0) that BOTH `seed_demand_log.py` (this plan) and `coverage_agent.py` (Plan 03) import — no forward-reference, one DRY definition (REVIEW ROUND-2 MEDIUM-2 — guard ownership)."
    - "The guard checks the ACTUAL write connection's `current_database()` (it opens or reuses the write connection and runs `SELECT current_database()`), requiring the result to be the known sandbox name `city_concierge_sandbox` (or a name CONTAINING `sandbox` and NOT a known-prod name) — it does NOT pass merely because `SANDBOX_DATABASE_URL`'s parsed dbname matches, so a mis-set `SANDBOX_DATABASE_URL` pointed at prod can never whitelist a prod write (REVIEW ROUND-2 H3 refinement)."
    - "The seed helper ENFORCES the sandbox write target before inserting: it calls the shared `assert_sandbox_write_target()` guard so seed rows can never be written to a non-sandbox DB (REVIEW HIGH-3 — applies to seed_demand_log.py as well as the miner)."
  artifacts:
    - path: ".env.example"
      provides: "DEMAND_DATABASE_URL documentation block"
      contains: "DEMAND_DATABASE_URL"
    - path: "Makefile"
      provides: "sandbox-migrate target (reproducible Phase 17 migration into the sandbox)"
      contains: "sandbox-migrate:"
    - path: "scripts/sandbox_guard.py"
      provides: "Shared assert_sandbox_write_target() guard (current_database()-based) imported by both seed_demand_log.py and coverage_agent.py"
      contains: "def assert_sandbox_write_target"
    - path: "tests/unit/test_sandbox_guard.py"
      provides: "Unit test that the guard RAISES when current_database() is non-sandbox even if SANDBOX_DATABASE_URL names that db (H3 refinement)"
    - path: "scripts/seed_demand_log.py"
      provides: "Deterministic user_query_log seed helper for sandbox demand testing, sandbox-write-guarded"
      contains: "def seed_demand_rows"
    - path: "tests/unit/test_seed_demand_log.py"
      provides: "Unit test that the seed rows are catalog-valid and the sandbox write guard fires off-sandbox"
  key_links:
    - from: "scripts/seed_demand_log.py"
      to: "scripts.ingest_places_sf NEIGHBORHOODS/CUISINES"
      via: "import + membership assertion"
      pattern: "from scripts.ingest_places_sf import"
    - from: "scripts/seed_demand_log.py"
      to: "scripts.sandbox_guard.assert_sandbox_write_target"
      via: "write-target guard before INSERT"
      pattern: "assert_sandbox_write_target"
    - from: "scripts/sandbox_guard.py assert_sandbox_write_target"
      to: "write connection current_database()"
      via: "SELECT current_database() requiring the sandbox name"
      pattern: "current_database"
    - from: "Makefile sandbox-migrate"
      to: "SANDBOX_DATABASE_URL"
      via: "alembic upgrade head against the sandbox URL"
      pattern: "alembic upgrade head"
---

<objective>
Lay down the Wave 0 prerequisites the demand miner needs before any demand code runs: apply the Phase 17 `user_query_log` migration to the sandbox DB (REPRODUCIBLY via a new `make sandbox-migrate` target, not a one-off), document the optional `DEMAND_DATABASE_URL` prod-read env var (D-05), create the ONE shared sandbox-write guard `scripts/sandbox_guard.py` (so both the seed helper here and the miner in Plan 03 import a single DRY definition — REVIEW ROUND-2 MEDIUM-2), and provide a deterministic, sandbox-write-guarded seed helper that inserts catalog-valid demand rows into sandbox `user_query_log` for functional/integration testing.

Purpose: The research probe found `user_query_log` is ABSENT from `city_concierge_sandbox` (the Phase 17 migration was never applied there) and EMPTY (0 rows) in local `city_concierge`. Without the table the miner crashes on `UndefinedTable`; without seed rows every local run hits the cold-start no-op (D-04) and developers cannot tell a working miner from a broken one (RESEARCH Pitfall 1, Pitfall 6). The cross-AI review additionally flagged that a manual one-time migration lets future sandboxes silently regress — so this plan makes the migration reproducible and guarded — and that the write guard must be owned by ONE shared module (no 18-01→18-03 forward-reference) and must check the live `current_database()`, not a possibly-mis-set env var (REVIEW ROUND-2 MEDIUM-2 + H3 refinement).

Output: `make sandbox-migrate` target, applied sandbox migration (operator-verified), `.env.example` `DEMAND_DATABASE_URL` block, `scripts/sandbox_guard.py` (the shared `current_database()`-based guard) + its unit test, and `scripts/seed_demand_log.py` (sandbox-write-guarded) + its unit test.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/18-gap-mining-gap/18-CONTEXT.md
@.planning/phases/18-gap-mining-gap/18-RESEARCH.md
@.planning/phases/18-gap-mining-gap/18-REVIEWS.md
@alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py
@scripts/ingest_places_sf.py
@app/query_log.py
@app/config.py
@app/loop/falsifier_core.py
@Makefile
@.env.example
</context>

<note_on_ownership>
The sandbox-write guard `assert_sandbox_write_target()` is DEFINED ONCE in `scripts/sandbox_guard.py`, created in this wave-0 plan. Both consumers import it: `seed_demand_log.py` here, and `coverage_agent.py` in Plan 03 (wave 2). Because `scripts/sandbox_guard.py` exists from wave 0, there is NO forward-reference and NO lazy-import fallback needed — Plan 03 imports the same module (REVIEW ROUND-2 MEDIUM-2). The guard's correctness check uses the live write connection's `current_database()`, not URL/env-var parsing (REVIEW ROUND-2 H3 refinement). Document the chosen module path in the SUMMARY.
</note_on_ownership>

<tasks>

<task type="auto">
  <name>Task 1: make sandbox-migrate target + DEMAND_DATABASE_URL in .env.example</name>
  <files>Makefile, .env.example</files>
  <read_first>
    - Makefile lines ~around the `sandbox-provision` / `db-up` / `migrate` targets (mirror the `$(POETRY_RUN)` + `##` help-comment house style; find how `migrate` invokes `alembic upgrade head` and how sandbox targets read `SANDBOX_DATABASE_URL`)
    - scripts/provision_sandbox.sh (the sandbox is provisioned by Alembic layering; this migration was added AFTER provisioning so the sandbox is one revision behind — confirm fresh provisions already layer it)
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py (the migration to apply — creates user_query_log + idx_user_query_log_created_at)
    - app/config.py (resolve_database_url — Alembic reads DATABASE_URL; sandbox-migrate must inject SANDBOX_DATABASE_URL as DATABASE_URL for the alembic run)
    - .env.example lines ~17-26 (the DATABASE_URL / SANDBOX_DATABASE_URL block to mirror in style for the new DEMAND_DATABASE_URL block)
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § MEDIUM "Sandbox-migrate reproducibility" (authoritative: add a durable `make sandbox-migrate` / verification path so future sandboxes don't silently regress)
    - .planning/phases/18-gap-mining-gap/18-CONTEXT.md D-05 (prod-read opt-in, write-target always sandbox)
  </read_first>
  <action>
    Add a `.PHONY: sandbox-migrate` / `sandbox-migrate:` target to the Makefile that runs `DATABASE_URL=$$SANDBOX_DATABASE_URL $(POETRY_RUN) alembic upgrade head` (inject the sandbox URL as DATABASE_URL so Alembic targets the sandbox — a DDL on `_sandbox` only, never prod) with a `## Apply all migrations to the sandbox DB (SANDBOX_DATABASE_URL)` help comment in the established style. Guard the target so it errors clearly if `SANDBOX_DATABASE_URL` is unset (mirror how existing sandbox targets check it). This makes the Phase 17 migration into the sandbox REPRODUCIBLE (REVIEW MEDIUM). Then add a commented `DEMAND_DATABASE_URL` block to `.env.example` directly after the `SANDBOX_DATABASE_URL` block, mirroring its comment style: (1) OPTIONAL and unset/commented by default; (2) when set, the gap miner reads `user_query_log` demand from this URL via a direct non-pooled read-only connection while STILL writing proposals to the pool (sandbox) target — D-05's prod-read + sandbox-write split; (3) it must point at a database that has `user_query_log` (prod Cloud SQL); (4) leave it commented (line begins with `#`) and empty so default behavior is sandbox-only demand reads. Placeholder only — no real value. This is per D-05.
  </action>
  <verify>
    <automated>grep -c 'sandbox-migrate:' Makefile && test "$(grep -c '^# *DEMAND_DATABASE_URL' .env.example)" -ge 1</automated>
  </verify>
  <acceptance_criteria>
    - `Makefile` contains a `sandbox-migrate:` target that runs `alembic upgrade head` against `SANDBOX_DATABASE_URL` and errors when it is unset.
    - `make -n sandbox-migrate` (dry print) shows the alembic invocation with the sandbox URL injected as DATABASE_URL.
    - `.env.example` contains a COMMENTED `DEMAND_DATABASE_URL` line (the line begins with `#`, no real credential value), after the `SANDBOX_DATABASE_URL` block, explaining the opt-in prod-read / sandbox-write split (D-05). `grep -c '^# *DEMAND_DATABASE_URL' .env.example` is ≥ 1 (REVIEW ROUND-2 LOW — the verify asserts the COMMENTED form, not an uncommented one).
    - The DEMAND_DATABASE_URL line is commented out so a fresh `.env` copy defaults to sandbox-only demand reads.
  </acceptance_criteria>
  <done>`make sandbox-migrate` reproducibly applies the Phase 17 migration to the sandbox; `.env.example` documents `DEMAND_DATABASE_URL` as an optional, COMMENTED prod-read override (D-05), and the verify asserts the commented form (REVIEW ROUND-2 LOW).</done>
</task>

<task type="checkpoint:human-action" gate="blocking">
  <name>Task 2: Apply the user_query_log migration to the sandbox DB (via make sandbox-migrate)</name>
  <files>(no source files — operator runs `make sandbox-migrate` against the sandbox DB)</files>
  <read_first>
    - Makefile (the new `sandbox-migrate` target from Task 1)
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py (the migration to apply — creates user_query_log + idx_user_query_log_created_at)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § "Runtime State Inventory" / "Environment Availability" (confirms sandbox lacks the table)
  </read_first>
  <action>
    The sandbox `city_concierge_sandbox` (port 5433, see memory project_local_postgres_port_collision) is missing the Phase 17 `user_query_log` table. Apply it by running `SANDBOX_DATABASE_URL=$SANDBOX_DATABASE_URL make sandbox-migrate` (the Task 1 target injects the sandbox URL as DATABASE_URL — this is a DDL on sandbox only, never prod). This is operator-run because it mutates a real DB and requires the SANDBOX_DATABASE_URL secret from the operator's shell. Do NOT coerce settings in-process. If the operator does not have a sandbox running, surface the `make sandbox-provision` path (which layers this migration automatically on a fresh DB) as the alternative.
  </action>
  <verify>
    <automated>MISSING — operator-verified: after migrate, `docker exec city_concierge_db psql -U postgres -d city_concierge_sandbox -c '\dt user_query_log'` shows one row (the table exists). No automated test can run this without the sandbox secret.</automated>
    <human-check>Operator confirms `\dt user_query_log` against the sandbox returns the table and `\d user_query_log` shows the `idx_user_query_log_created_at` index.</human-check>
  </verify>
  <acceptance_criteria>
    - `psql -d city_concierge_sandbox -c '\dt user_query_log'` lists the table (operator pastes output).
    - `psql -d city_concierge_sandbox -c "SELECT count(*) FROM user_query_log"` returns a number (table queryable, not UndefinedTable).
    - No DDL was run against any prod or shared `places_raw` database (the command targeted the sandbox URL only).
  </acceptance_criteria>
  <resume-signal>Paste the `\dt user_query_log` output against the sandbox, or "approved".</resume-signal>
  <done>The sandbox `user_query_log` table exists and is queryable via the reproducible `make sandbox-migrate` path; no prod DB was touched.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Shared current_database()-based sandbox-write guard (scripts/sandbox_guard.py) + unit test</name>
  <files>scripts/sandbox_guard.py, tests/unit/test_sandbox_guard.py</files>
  <read_first>
    - app/loop/falsifier_core.py lines ~168-201 (`_normalize_url` returns `(host, port, dbname, cloud_sql_instance)` — the established URL-parse helper; the guard MAY reuse it to extract a fallback dbname for messaging, but the PASS condition is the live `current_database()` value, not URL parsing) and lines ~209-293 (`check_prod_safety` + `GuardResult` — the established prod-safety pattern + the `_sandbox` suffix / `city_concierge_sandbox` convention)
    - app/config.py lines ~60-85, ~137-152 (`resolve_database_url`; `settings.resolved_database_url` — the active write URL the pool targets; `SANDBOX_DATABASE_URL` is read from `os.environ` directly, NOT a Settings field)
    - app/db.py (get_conn context manager — how to open the active write connection so the guard can run `SELECT current_database()` against the SAME pool target the writes use)
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § ROUND-2 "H3 refinement" + the "Verified against code (orchestrator)" H3 subsection (authoritative: check the write connection's `current_database()`, require the known sandbox name, do NOT pass solely on equality to `SANDBOX_DATABASE_URL`'s parsed dbname)
    - .planning/phases/18-gap-mining-gap/18-CONTEXT.md D-05 (write target MUST be sandbox)
  </read_first>
  <behavior>
    - Test 1 (sandbox passes): when the write connection's `current_database()` returns `city_concierge_sandbox`, `assert_sandbox_write_target()` returns/None-passes (no raise).
    - Test 2 (prod raises): when `current_database()` returns `city_concierge` (a known-prod / non-sandbox name), `assert_sandbox_write_target()` RAISES a clear error (SystemExit non-zero or RuntimeError) and the message names the offending dbname.
    - Test 3 (H3 refinement — mis-set env var cannot whitelist prod): even when `os.environ["SANDBOX_DATABASE_URL"]` is mis-set to a URL whose dbname is `city_concierge` (prod), if `current_database()` returns `city_concierge` the guard STILL RAISES. The pass decision depends on the live `current_database()` result, never on equality to `SANDBOX_DATABASE_URL`'s parsed dbname.
    - Test 4 (sandbox-pattern name passes if you support a configurable name): a `current_database()` of e.g. `city_concierge_sandbox` or a name CONTAINING `sandbox` and NOT a known-prod name passes; a name that does NOT contain `sandbox` (and is not the explicit known-sandbox name) raises.
    - Test 5 (connection reuse): when a connection is passed in, the guard runs `SELECT current_database()` on THAT connection (no new pool open); when none is passed, it opens the active write connection via `get_conn()` (the same pool target writes use) and closes it.
  </behavior>
  <action>
    Create `scripts/sandbox_guard.py` with `def assert_sandbox_write_target(conn=None) -> None`: it runs `SELECT current_database()` against the ACTIVE write connection (the passed `conn` if supplied, otherwise a connection from `get_conn()`), reads the returned dbname, and RAISES a clear error (SystemExit with a non-zero code, or RuntimeError) UNLESS the live dbname is the known sandbox name `city_concierge_sandbox` OR (configurable-name support) a name that CONTAINS the substring `sandbox` AND is NOT a known-prod name (`city_concierge`). The decision MUST be based on the value returned by `SELECT current_database()` on the real write connection — do NOT pass merely because `SANDBOX_DATABASE_URL`'s parsed dbname matches (REVIEW ROUND-2 H3 refinement: a mis-set env var pointed at prod must not whitelist a prod write). You MAY reuse `app.loop.falsifier_core._normalize_url` only to enrich the error message with the URL-parsed dbname; it must NOT be the pass condition. Create `tests/unit/test_sandbox_guard.py` covering the five behaviors with a capturing-stub connection whose cursor returns a controllable `current_database()` value (reuse the `_InsertConn`/`_InsertCursor` stub pattern from test_coverage_agent.py) and monkeypatched `os.environ["SANDBOX_DATABASE_URL"]`; the unit test must NOT touch a real DB.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_sandbox_guard.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/sandbox_guard.py` contains `def assert_sandbox_write_target` and runs `SELECT current_database()` on the active write connection.
    - The guard RAISES when `current_database()` returns a non-sandbox name (e.g. `city_concierge`) EVEN WHEN `SANDBOX_DATABASE_URL` is mis-set to a URL naming that same db (REVIEW ROUND-2 H3 refinement — Test 3).
    - The guard passes for `current_database() == "city_concierge_sandbox"` and for a sandbox-pattern name; it does NOT pass solely on `SANDBOX_DATABASE_URL` env-var equality.
    - `grep -c 'current_database' scripts/sandbox_guard.py` is ≥ 1.
    - `poetry run pytest tests/unit/test_sandbox_guard.py -v` exits 0.
    - `poetry run ruff check scripts/sandbox_guard.py tests/unit/test_sandbox_guard.py` passes.
  </acceptance_criteria>
  <done>A single shared sandbox-write guard exists in `scripts/sandbox_guard.py` whose pass condition is the live `SELECT current_database()` value (known sandbox name / sandbox-pattern, never a mis-settable env var), unit-tested to RAISE on a prod dbname even when `SANDBOX_DATABASE_URL` names it (REVIEW ROUND-2 H3 + MEDIUM-2).</done>
</task>

<task type="auto" tdd="true">
  <name>Task 4: Deterministic, sandbox-write-guarded demand-log seed helper + unit test</name>
  <files>scripts/seed_demand_log.py, tests/unit/test_seed_demand_log.py</files>
  <read_first>
    - app/query_log.py (the production INSERT shape into user_query_log — columns message, requested_primary_types, num_stops, rag_label, session_id; mirror its parameterised %s style)
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py (the exact column set + types)
    - scripts/ingest_places_sf.py (NEIGHBORHOODS line ~161, CUISINES line ~194 — the catalog the seed rows MUST stay inside so the mined gaps are catalog-valid)
    - scripts/sandbox_guard.py (Task 3's `assert_sandbox_write_target` — the shared write-target guard the seed helper imports and calls before INSERT; it EXISTS from wave 0, so import it directly — no lazy fallback, no forward-reference per REVIEW ROUND-2 MEDIUM-2)
    - app/db.py (get_conn context manager — the seed helper writes via get_conn so it targets whatever DATABASE_URL is set, i.e. the sandbox in CI)
    - tests/unit/test_coverage_agent.py (the _InsertConn/_InsertCursor capturing-stub pattern to reuse for the unit test — do NOT hit a real DB in the unit test)
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § HIGH-3 + ROUND-2 MEDIUM-2 (the write guard must apply to seed_demand_log.py too, via the ONE shared module)
  </read_first>
  <behavior>
    - Test 1: `seed_demand_rows()` produces a non-empty list of row dicts whose every `requested_primary_types` entry lexically maps to a member of `CUISINES` (e.g. "Vietnamese Restaurant" → "vietnamese") AND whose `message` contains a substring that is a member of `NEIGHBORHOODS` (e.g. "Outer Sunset"). This guarantees the seeded demand is mappable to catalog buckets, so the miner produces real gaps rather than only `unmapped_count`.
    - Test 2: At least one seeded row targets a known thin bucket — `("Outer Sunset", "vietnamese")` — so the demand path and the supply gate overlap and a functional test downstream sees a real gap.
    - Test 3: The INSERT path is exercised against a capturing-stub connection (monkeypatched `get_conn`): each row produces one `INSERT INTO user_query_log` execute with parameters passed as a list (parameterised, never string-interpolated — T-18-SQLi guard), and exactly one commit after the loop.
    - Test 4 (HIGH-3 — write guard): `insert_demand_rows` calls `assert_sandbox_write_target()` (imported from `scripts.sandbox_guard`) BEFORE any INSERT; when the guard is monkeypatched to raise (simulating a non-sandbox write target), `insert_demand_rows` raises and performs ZERO inserts (assert no execute on the capturing stub).
  </behavior>
  <action>
    Create `scripts/seed_demand_log.py` with `def seed_demand_rows() -> list[dict]` returning a small deterministic catalog-valid demand fixture (each dict: `message`, `requested_primary_types`, `num_stops`, `rag_label`, `session_id`) and `def insert_demand_rows(rows: list[dict], conn=None) -> int` that: FIRST calls `assert_sandbox_write_target()` (imported directly from `scripts.sandbox_guard` — the shared module created in Task 3; no lazy import, no inline fallback, per REVIEW ROUND-2 MEDIUM-2) so seed rows are never written off-sandbox (REVIEW HIGH-3), THEN INSERTs each via parameterised `%s` placeholders (mirror `app/query_log.log_user_query`'s INSERT shape and never interpolate `message` into SQL) using `get_conn()` when no conn is supplied, committing once after the loop and returning the inserted count. When a `conn` is passed, the guard runs against THAT connection so the integration test guards the same connection it writes to. Add a thin `main(argv) -> int` so `python scripts/seed_demand_log.py` seeds whatever DATABASE_URL points at (operator sets sandbox; the guard enforces it). Constrain every fixture row to `NEIGHBORHOODS × CUISINES` and include the `("Outer Sunset", "vietnamese")` overlap row. Create `tests/unit/test_seed_demand_log.py` covering the four behaviors above with a capturing-stub connection (reuse the `_InsertConn` pattern from test_coverage_agent.py) and a monkeypatched `assert_sandbox_write_target`; the unit test must NOT touch a real DB.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_seed_demand_log.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/seed_demand_log.py` contains `def seed_demand_rows` and `def insert_demand_rows`, and imports `assert_sandbox_write_target` from `scripts.sandbox_guard` (the shared module — no redefinition, REVIEW ROUND-2 MEDIUM-2).
    - `insert_demand_rows` calls `assert_sandbox_write_target()` before any INSERT, and a monkeypatched raising guard prevents all inserts (REVIEW HIGH-3 — asserted by Test 4).
    - `poetry run pytest tests/unit/test_seed_demand_log.py -v` exits 0.
    - The unit test asserts every fixture `requested_primary_types` value maps into `CUISINES` and every `message` contains a `NEIGHBORHOODS` member (catalog-valid).
    - The unit test asserts the INSERT uses parameter lists (no f-string SQL interpolation of `message`).
    - `poetry run ruff check scripts/seed_demand_log.py tests/unit/test_seed_demand_log.py` passes.
  </acceptance_criteria>
  <done>A catalog-valid deterministic demand seed helper exists with parameterised INSERTs, importing the ONE shared sandbox-write guard (MEDIUM-2) that blocks off-sandbox writes (HIGH-3), and a green unit test; running it against the sandbox populates `user_query_log` with mappable rows.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| operator shell → sandbox DB | Operator-supplied SANDBOX_DATABASE_URL drives DDL via `make sandbox-migrate`; must never resolve to prod/shared `places_raw` |
| seed fixture → sandbox DB write | The seed helper writes rows; if pointed at prod it would pollute real demand telemetry — now blocked by the `current_database()`-based write guard |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-18-01-MIG | Elevation of Privilege | `make sandbox-migrate` target | mitigate | Task 2 is a blocking operator checkpoint; the `make sandbox-migrate` target injects the sandbox URL explicitly and is verified against `city_concierge_sandbox` only — never prod. DDL on a `_sandbox`-suffixed DB only. Reproducible (REVIEW MEDIUM) so future sandboxes layer it the same way. |
| T-18-01-SEED | Tampering | `insert_demand_rows` write target | mitigate | **ENFORCED via the shared guard (REVIEW HIGH-3 + ROUND-2 H3 refinement):** `insert_demand_rows` calls `assert_sandbox_write_target()` (from `scripts.sandbox_guard`) before any INSERT — it runs `SELECT current_database()` on the write connection and refuses unless the live dbname is the known sandbox name / a sandbox-pattern name. A mis-set `SANDBOX_DATABASE_URL` pointed at prod can no longer whitelist a prod seed. |
| T-18-01-GUARD | Spoofing | `assert_sandbox_write_target` pass condition | mitigate | **ROUND-2 H3 refinement:** the guard's decision is the live `current_database()` result, NOT equality to a possibly-mis-set `SANDBOX_DATABASE_URL` dbname — closing the env-var-spoof gap the round-1 guard left. Unit-tested to RAISE on a prod dbname even when the env var names it. |
| T-18-01-SQLi | Tampering | `insert_demand_rows` INSERT | mitigate | Parameterised `%s` placeholders (asserted by unit test); `message` content never interpolated into the SQL string. |
| T-18-01-SC | Tampering | npm/pip/cargo installs | accept | No new packages installed this plan (psycopg2/mlflow already present per RESEARCH Package Legitimacy Audit = N/A). |
</threat_model>

<verification>
- `make -n sandbox-migrate` prints the alembic-against-sandbox invocation; sandbox `user_query_log` table exists and is queryable (operator-verified).
- `.env.example` documents `DEMAND_DATABASE_URL` as a COMMENTED line (`grep -c '^# *DEMAND_DATABASE_URL' .env.example` ≥ 1 — REVIEW ROUND-2 LOW).
- `scripts/sandbox_guard.py` exists with a `current_database()`-based `assert_sandbox_write_target`; `poetry run pytest tests/unit/test_sandbox_guard.py -v` exits 0 (incl. the H3 mis-set-env-var raise test).
- `poetry run pytest tests/unit/test_seed_demand_log.py -v` exits 0.
- `poetry run ruff check scripts/sandbox_guard.py tests/unit/test_sandbox_guard.py scripts/seed_demand_log.py tests/unit/test_seed_demand_log.py` passes.
</verification>

<success_criteria>
- Sandbox has `user_query_log` (Phase 17 migration applied via the reproducible `make sandbox-migrate` target — REVIEW MEDIUM) — the miner can read demand from it.
- `DEMAND_DATABASE_URL` documented (COMMENTED) in `.env.example` (D-05 prod-read opt-in); the verify asserts the commented form (REVIEW ROUND-2 LOW).
- ONE shared sandbox-write guard `scripts/sandbox_guard.py` exists, checks the live `current_database()` (REVIEW ROUND-2 H3 refinement), and is imported by both this plan's seed helper and Plan 03's miner (REVIEW ROUND-2 MEDIUM-2) — no forward-reference, DRY.
- Catalog-valid deterministic demand seed helper exists, sandbox-write-guarded (REVIEW HIGH-3), unit-tested green, ready to populate the sandbox for downstream functional/integration tests.
</success_criteria>

<output>
Create `.planning/phases/18-gap-mining-gap/18-01-SUMMARY.md` when done. Note the chosen shared-guard module path (`scripts/sandbox_guard.py`) so Plan 03 imports from the same place.
</output>
