---
phase: 18-gap-mining-gap
plan: 01
type: execute
wave: 0
depends_on: []
files_modified:
  - .env.example
  - scripts/seed_demand_log.py
  - tests/unit/test_seed_demand_log.py
autonomous: false
requirements: [GAP-01, GAP-04]
must_haves:
  truths:
    - "The sandbox `city_concierge_sandbox` DB has the `user_query_log` table (Phase 17 migration d1be72aea7d4 applied) so the demand miner can read it (D-05 sandbox-only default)."
    - "`.env.example` documents the optional `DEMAND_DATABASE_URL` prod-read env var that activates the prod-read demand path (D-05)."
    - "A deterministic seed helper inserts representative catalog-valid rows into sandbox `user_query_log` so functional/integration tests of the demand path (D-01) have real data and the cold-start path (D-04) can be distinguished from a broken miner."
  artifacts:
    - path: ".env.example"
      provides: "DEMAND_DATABASE_URL documentation block"
      contains: "DEMAND_DATABASE_URL"
    - path: "scripts/seed_demand_log.py"
      provides: "Deterministic user_query_log seed helper for sandbox demand testing"
      contains: "def seed_demand_rows"
    - path: "tests/unit/test_seed_demand_log.py"
      provides: "Unit test that the seed rows are catalog-valid"
  key_links:
    - from: "scripts/seed_demand_log.py"
      to: "scripts.ingest_places_sf NEIGHBORHOODS/CUISINES"
      via: "import + membership assertion"
      pattern: "from scripts.ingest_places_sf import"
---

<objective>
Lay down the Wave 0 prerequisites the demand miner needs before any demand code runs: apply the Phase 17 `user_query_log` migration to the sandbox DB, document the optional `DEMAND_DATABASE_URL` prod-read env var (D-05), and provide a deterministic seed helper that inserts catalog-valid demand rows into sandbox `user_query_log` for functional/integration testing.

Purpose: The research probe found `user_query_log` is ABSENT from `city_concierge_sandbox` (the Phase 17 migration was never applied there) and EMPTY (0 rows) in local `city_concierge`. Without the table the miner crashes on `UndefinedTable`; without seed rows every local run hits the cold-start no-op (D-04) and developers cannot tell a working miner from a broken one (RESEARCH Pitfall 1, Pitfall 6). This plan removes both blockers.

Output: Applied sandbox migration (operational, verified via psql), `.env.example` `DEMAND_DATABASE_URL` block, `scripts/seed_demand_log.py` + its unit test.
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
@alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py
@scripts/ingest_places_sf.py
@app/query_log.py
@.env.example
</context>

<tasks>

<task type="checkpoint:human-action" gate="blocking">
  <name>Task 1: Apply the user_query_log migration to the sandbox DB</name>
  <files>(no source files — operator runs `alembic upgrade head` against the sandbox DB)</files>
  <read_first>
    - scripts/provision_sandbox.sh (the sandbox is provisioned by Alembic layering; this migration was added AFTER provisioning so the sandbox is one revision behind)
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py (the migration to apply — creates user_query_log + idx_user_query_log_created_at)
    - app/config.py (resolve_database_url — Alembic reads DATABASE_URL)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § "Runtime State Inventory" (confirms sandbox lacks the table)
  </read_first>
  <action>
    The sandbox `city_concierge_sandbox` (port 5433, see memory project_local_postgres_port_collision) is missing the Phase 17 `user_query_log` table. Apply it with `SANDBOX_DATABASE_URL=$SANDBOX_DATABASE_URL DATABASE_URL=$SANDBOX_DATABASE_URL poetry run alembic upgrade head` (Alembic resolves DATABASE_URL, so the run must inject the sandbox URL as DATABASE_URL — this is a READ/DDL on sandbox only, never prod). This is operator-run because it mutates a real DB and requires the SANDBOX_DATABASE_URL secret from the operator's shell. Do NOT coerce settings in-process; this is a CLI invocation. If the operator does not have a sandbox running, surface the `make sandbox-provision` path (which now layers this migration automatically on a fresh DB) as the alternative.
  </action>
  <verify>
    <automated>MISSING — operator-verified: after upgrade, `docker exec city_concierge_db psql -U postgres -d city_concierge_sandbox -c '\dt user_query_log'` shows one row (the table exists). No automated test can run this without the sandbox secret.</automated>
    <human-check>Operator confirms `\dt user_query_log` against the sandbox returns the table and `\d user_query_log` shows the `idx_user_query_log_created_at` index.</human-check>
  </verify>
  <acceptance_criteria>
    - `psql -d city_concierge_sandbox -c '\dt user_query_log'` lists the table (operator pastes output).
    - `psql -d city_concierge_sandbox -c "SELECT count(*) FROM user_query_log"` returns a number (table queryable, not UndefinedTable).
    - No DDL was run against any prod or shared `places_raw` database (the command targeted the sandbox URL only).
  </acceptance_criteria>
  <resume-signal>Paste the `\dt user_query_log` output against the sandbox, or "approved".</resume-signal>
  <done>The sandbox `user_query_log` table exists and is queryable; no prod DB was touched.</done>
</task>

<task type="auto">
  <name>Task 2: Document DEMAND_DATABASE_URL in .env.example</name>
  <files>.env.example</files>
  <read_first>
    - .env.example (lines 15-30 — the DATABASE_URL / SANDBOX_DATABASE_URL block to mirror in style)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § Q2 + "Security Domain" (DEMAND_DATABASE_URL is opt-in, read-only prod credentials, gitignored)
    - .planning/phases/18-gap-mining-gap/18-CONTEXT.md D-05 (prod-read opt-in, write-target always sandbox)
  </read_first>
  <action>
    Add a commented `DEMAND_DATABASE_URL` block to `.env.example` directly after the `SANDBOX_DATABASE_URL` block, mirroring its comment style. The comment must state: (1) it is OPTIONAL and unset by default; (2) when set, the gap miner reads `user_query_log` demand from this URL via a direct non-pooled read-only connection while STILL writing proposals to the pool (sandbox) target — implementing D-05's prod-read + sandbox-write split; (3) it must point at a database that has `user_query_log` (prod Cloud SQL); (4) leave it commented/empty so the default behavior is sandbox-only demand reads (mirrors Phase 16). Do NOT set a real value — placeholder only. This is per D-05.
  </action>
  <verify>
    <automated>grep -v '^#' .env.example | grep -c DEMAND_DATABASE_URL; test "$(grep -c DEMAND_DATABASE_URL .env.example)" -ge 1</automated>
  </verify>
  <acceptance_criteria>
    - `.env.example` contains the string `DEMAND_DATABASE_URL` (commented, no real credential value).
    - The block appears after the `SANDBOX_DATABASE_URL` block and explains the opt-in prod-read / sandbox-write split (D-05).
    - The line is commented out (a `#`-prefixed example), so a fresh `.env` copy defaults to sandbox-only demand reads.
  </acceptance_criteria>
  <done>`.env.example` documents `DEMAND_DATABASE_URL` as an optional, commented, read-only prod-demand override per D-05.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Deterministic sandbox demand-log seed helper + unit test</name>
  <files>scripts/seed_demand_log.py, tests/unit/test_seed_demand_log.py</files>
  <read_first>
    - app/query_log.py (the production INSERT shape into user_query_log — columns message, requested_primary_types, num_stops, rag_label, session_id; mirror its parameterised %s style)
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py (the exact column set + types)
    - scripts/ingest_places_sf.py (NEIGHBORHOODS line ~161, CUISINES line ~194 — the catalog the seed rows MUST stay inside so the mined gaps are catalog-valid)
    - app/db.py (get_conn context manager — the seed helper writes via get_conn so it targets whatever DATABASE_URL is set, i.e. the sandbox in CI)
    - tests/unit/test_coverage_agent.py (the _InsertConn/_InsertCursor capturing-stub pattern to reuse for the unit test — do NOT hit a real DB in the unit test)
  </read_first>
  <behavior>
    - Test 1: `seed_demand_rows()` produces a non-empty list of row dicts whose every `requested_primary_types` entry lexically maps to a member of `CUISINES` (e.g. "Vietnamese Restaurant" → "vietnamese") AND whose `message` contains a substring that is a member of `NEIGHBORHOODS` (e.g. "Outer Sunset"). This guarantees the seeded demand is mappable to catalog buckets, so the miner produces real gaps rather than only `unmapped_count`.
    - Test 2: At least one seeded row targets a known thin bucket — `("Outer Sunset", "vietnamese")` — so the demand path and the supply gate overlap and a functional test downstream sees a real gap.
    - Test 3: The INSERT path is exercised against a capturing-stub connection (monkeypatched `get_conn`): each row produces one `INSERT INTO user_query_log` execute with parameters passed as a list (parameterised, never string-interpolated — T-18-SQLi guard), and exactly one commit after the loop.
  </behavior>
  <action>
    Create `scripts/seed_demand_log.py` with `def seed_demand_rows() -> list[dict]` returning a small deterministic catalog-valid demand fixture (each dict: `message`, `requested_primary_types`, `num_stops`, `rag_label`, `session_id`) and `def insert_demand_rows(rows: list[dict], conn=None) -> int` that INSERTs each via parameterised `%s` placeholders (mirror `app/query_log.log_user_query`'s INSERT shape and never interpolate `message` into SQL) using `get_conn()` when no conn is supplied, committing once after the loop and returning the inserted count. Add a thin `main(argv) -> int` so `python scripts/seed_demand_log.py` seeds whatever DATABASE_URL points at (operator sets sandbox). Constrain every fixture row to `NEIGHBORHOODS × CUISINES` and include the `("Outer Sunset", "vietnamese")` overlap row. Create `tests/unit/test_seed_demand_log.py` covering the three behaviors above with a capturing-stub connection (reuse the `_InsertConn` pattern from test_coverage_agent.py); the unit test must NOT touch a real DB.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_seed_demand_log.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/seed_demand_log.py` contains `def seed_demand_rows` and `def insert_demand_rows`.
    - `poetry run pytest tests/unit/test_seed_demand_log.py -v` exits 0.
    - The unit test asserts every fixture `requested_primary_types` value maps into `CUISINES` and every `message` contains a `NEIGHBORHOODS` member (catalog-valid).
    - The unit test asserts the INSERT uses parameter lists (no f-string SQL interpolation of `message`).
    - `poetry run ruff check scripts/seed_demand_log.py tests/unit/test_seed_demand_log.py` passes.
  </acceptance_criteria>
  <done>A catalog-valid deterministic demand seed helper exists with parameterised INSERTs and a green unit test; running it against the sandbox populates `user_query_log` with mappable rows.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| operator shell → sandbox DB | Operator-supplied SANDBOX_DATABASE_URL drives DDL; must never resolve to prod/shared `places_raw` |
| seed fixture → sandbox DB write | The seed helper writes rows; if pointed at prod it would pollute real demand telemetry |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-18-01-MIG | Elevation of Privilege | `alembic upgrade head` target | mitigate | Task 1 is a blocking operator checkpoint; the command injects the sandbox URL explicitly and is verified against `city_concierge_sandbox` only — never prod. DDL on a `_sandbox`-suffixed DB only. |
| T-18-01-SEED | Tampering | `insert_demand_rows` write target | mitigate | `seed_demand_log.py` writes via `get_conn()` which targets `DATABASE_URL`; in CI/dev that is sandbox. Document that the seed must only run with sandbox `DATABASE_URL`. No prod write path. |
| T-18-01-SQLi | Tampering | `insert_demand_rows` INSERT | mitigate | Parameterised `%s` placeholders (asserted by unit test); `message` content never interpolated into the SQL string. |
| T-18-01-SC | Tampering | npm/pip/cargo installs | accept | No new packages installed this plan (psycopg2/mlflow already present per RESEARCH Package Legitimacy Audit = N/A). |
</threat_model>

<verification>
- Sandbox `user_query_log` table exists and is queryable (operator-verified).
- `.env.example` documents `DEMAND_DATABASE_URL` (grep).
- `poetry run pytest tests/unit/test_seed_demand_log.py -v` exits 0.
- `poetry run ruff check scripts/seed_demand_log.py tests/unit/test_seed_demand_log.py` passes.
</verification>

<success_criteria>
- Sandbox has `user_query_log` (Phase 17 migration applied) — the miner can read demand from it.
- `DEMAND_DATABASE_URL` documented in `.env.example` (D-05 prod-read opt-in).
- Catalog-valid deterministic demand seed helper exists, unit-tested green, ready to populate the sandbox for downstream functional/integration tests.
</success_criteria>

<output>
Create `.planning/phases/18-gap-mining-gap/18-01-SUMMARY.md` when done.
</output>
