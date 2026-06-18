---
phase: 18-gap-mining-gap
plan: 02
type: execute
wave: 1
depends_on: [18-01]
files_modified:
  - scripts/coverage_agent.py
  - tests/unit/test_gap_miner.py
autonomous: true
requirements: [GAP-01]
must_haves:
  truths:
    - "The miner turns raw `user_query_log` rows into catalog-constrained `(neighborhood, cuisine)` demand tuples using the existing `vibe.make_judge()` LLM in a SINGLE batched call for free-text neighborhood extraction (D-01)."
    - "Cuisine intent is mapped lexically from `requested_primary_types[]` Title-Case primary types to `CUISINES` members with NO LLM call (the two-tier optimization the research confirmed)."
    - "Demand buckets are constrained to `NEIGHBORHOODS × CUISINES` catalog membership; rows that map to nothing in-catalog are counted in `unmapped_count` and dropped (Claude's-Discretion: unmappable-demand catalog-constraint), so off-catalog gaps can never break `loop_falsifier.premark_seed_isolation`."
    - "A direct non-pooled `get_demand_conn(url)` reads demand from `DEMAND_DATABASE_URL` when set, otherwise the shared pool (sandbox `DATABASE_URL`) is used — the ~15-line two-DB plumbing for D-05's prod-read + sandbox-write split, leaving the existing pool/`get_conn()` untouched for writes."
    - "The existing supply-only `coverage_agent` functions (`gather_stats`, `find_gaps`, `propose_queries`, `filter_already_covered`, `insert_pending`) and their W5 tests are UNCHANGED — the demand path is strictly additive (guardrail: extend, do not regress; Claude's-Discretion miner-shape/extend)."
  artifacts:
    - path: "scripts/coverage_agent.py"
      provides: "gather_demand() + get_demand_conn() + lexical/LLM demand extraction helpers"
      contains: "def gather_demand"
    - path: "tests/unit/test_gap_miner.py"
      provides: "Unit tests for the demand-extraction path (lexical map, batch LLM parse, catalog constraint, unmapped_count)"
  key_links:
    - from: "scripts/coverage_agent.py gather_demand"
      to: "user_query_log table"
      via: "parameterised SELECT over created_at window"
      pattern: "FROM user_query_log"
    - from: "scripts/coverage_agent.py demand extraction"
      to: "scripts.ingest_places_sf CUISINES/NEIGHBORHOODS"
      via: "catalog-membership filter"
      pattern: "from scripts.ingest_places_sf import"
    - from: "scripts/coverage_agent.py get_demand_conn"
      to: "DEMAND_DATABASE_URL"
      via: "direct non-pooled psycopg2.connect when set"
      pattern: "DEMAND_DATABASE_URL"
---

<objective>
Build the demand-extraction half of the gap miner inside `scripts/coverage_agent.py`: a `gather_demand(days, url=None)` function that reads `user_query_log`, maps rows to catalog-constrained `(neighborhood, cuisine)` demand tuples via a two-tier strategy (lexical cuisine map from `requested_primary_types`, single batched LLM call for free-text neighborhoods per D-01), and returns demand counts plus honesty metrics (`rows_scanned`, `unmapped_count`). Add the `get_demand_conn(url)` non-pooled helper that enables D-05's prod-read / sandbox-write split.

Purpose: This is GAP-01 — Phase 18's distinctive contribution. `coverage_agent` already does supply-side gap detection; what it lacks is a real demand signal. This plan adds that signal additively without touching any existing supply-only function or its W5 tests (the guardrail).

Output: New `gather_demand`, `get_demand_conn`, and demand-mapping helpers in `coverage_agent.py`; new `tests/unit/test_gap_miner.py` covering the demand path with mocks (no real DB/LLM).
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
@app/config.py
@app/query_log.py
@alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Lexical cuisine map + single-batch LLM neighborhood extraction + catalog constraint</name>
  <files>scripts/coverage_agent.py, tests/unit/test_gap_miner.py</files>
  <read_first>
    - scripts/coverage_agent.py (the file being modified — study `_parse_proposals` fence-tolerant JSON parsing to mirror for the LLM batch parse; study `propose_queries`'s `llm.invoke([HumanMessage(content=prompt)])` shape to reuse the same vibe.make_judge LLM accessor)
    - scripts/ingest_places_sf.py (CUISINES line ~194 and NEIGHBORHOODS line ~161 — the exact catalog strings; build module-level frozensets `_CUISINES_SET`/`_NEIGHBORHOODS_SET` from them)
    - app/main.py lines ~60-89, ~737-738 (provenance of `requested_primary_types` — validated Title-Case Google primary_type vocabulary like "Vietnamese Restaurant", "Bar")
    - app/tools/filters.py lines ~168-296 (the family vocabulary so you know which primary types map to no CUISINE, e.g. "Bar"/"Cocktail Bar" → unmapped)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § Q4 + "Pattern 2/3" + "Security Domain" (prompt-injection mitigation: pass messages as a JSON-encoded array, never raw string interpolation)
  </read_first>
  <behavior>
    - Test 1 (lexical map): `_types_to_cuisines(["Vietnamese Restaurant", "Italian Restaurant"])` returns `["vietnamese", "italian"]`; `_types_to_cuisines(["Bar", "Cocktail Bar"])` returns `[]` (bars are off-CUISINES and become unmapped).
    - Test 2 (batch LLM call count): extracting neighborhoods for N messages calls `llm.invoke` exactly ONCE (single batched call per D-01), not once per row.
    - Test 3 (batch LLM parse): the batch extractor tolerates ```json fences and returns one neighborhood-or-None per input message, mapping only `NEIGHBORHOODS` members and yielding None for off-catalog or vague answers.
    - Test 4 (catalog constraint): a row whose cuisine is in CUISINES but whose neighborhood is off-catalog (e.g. "Berkeley") contributes nothing to demand counts and increments `unmapped_count`.
    - Test 5 (prompt-injection safety): the batch prompt embeds messages as a JSON-encoded array element, so a message containing `"]}` or prompt-injection text cannot escape the JSON boundary of the prompt (assert the message appears json.dumps-encoded in the built prompt, not raw).
    - Test 6 (LLM None): when `vibe.make_judge()` returns None (missing creds), neighborhood extraction yields all-None and those rows fall to `unmapped_count` (graceful degrade, no crash).
  </behavior>
  <action>
    In `scripts/coverage_agent.py` add: module-level `_CUISINES_SET = frozenset(CUISINES)` and `_NEIGHBORHOODS_SET = frozenset(NEIGHBORHOODS)` (import `NEIGHBORHOODS` alongside the existing `CUISINES` import). Add `_types_to_cuisines(primary_types: list[str]) -> list[str]` doing `pt.lower().removesuffix(" restaurant")` then `_CUISINES_SET` membership (no LLM). Add `_build_neighborhood_batch_prompt(messages: list[str]) -> str` that json.dumps-encodes the messages into a JSON array inside the prompt (prompt-injection guard — never raw f-string interpolation of message text) and asks the LLM for a parallel JSON list of neighborhood-or-null constrained to `NEIGHBORHOODS`. Add `_extract_neighborhoods_batch(messages: list[str], llm) -> list[str | None]` that makes ONE `llm.invoke([HumanMessage(content=prompt)])` call, reuses fence-tolerant JSON parsing (mirror `_parse_proposals`'s `_FENCE_RE` approach), and returns one `NEIGHBORHOODS`-member-or-None per message; returns all-None when `llm is None`. Do NOT modify any existing function. Add the new unit tests to `tests/unit/test_gap_miner.py` (create the file).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner.py -v && poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/coverage_agent.py` contains `def _types_to_cuisines` and `def _extract_neighborhoods_batch` and `def _build_neighborhood_batch_prompt`.
    - `grep -c 'def gather_stats\|def find_gaps\|def propose_queries\|def filter_already_covered\|def insert_pending' scripts/coverage_agent.py` is unchanged (5) — no existing function deleted.
    - `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (existing supply-only contract preserved — REGRESSION GUARDRAIL).
    - `poetry run pytest tests/unit/test_gap_miner.py -v` exits 0 with the six behaviors above.
    - The batch prompt builder embeds messages via `json.dumps` (assert in test), not raw interpolation.
    - `_extract_neighborhoods_batch` makes exactly one `llm.invoke` call for a multi-message input (assert call_count == 1).
  </acceptance_criteria>
  <done>Demand rows map to catalog-constrained cuisines lexically and neighborhoods via a single batched LLM call; off-catalog answers are dropped; existing supply-only tests still pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: gather_demand() over user_query_log + get_demand_conn() two-DB plumbing</name>
  <files>scripts/coverage_agent.py, tests/unit/test_gap_miner.py</files>
  <read_first>
    - scripts/coverage_agent.py (the file being modified — `gather_stats`'s cutoff/`datetime.now(UTC) - timedelta(days=days)` pattern and its parameterised `cur.execute(sql, [...])` shape to mirror; the existing `get_conn` import)
    - app/config.py (resolve_database_url; the lru_cache footgun note — gather_demand must NOT coerce DATABASE_URL, it opens a SEPARATE connection so no cache_clear is needed)
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py (columns to SELECT: message, requested_primary_types, created_at indexed)
    - app/db.py (get_conn — the pooled sandbox-targeting connection used when DEMAND_DATABASE_URL is unset)
    - scripts/loop_falsifier.py lines ~650-665 (`_snapshot_ids_from_url` — the existing direct-psycopg2 `contextlib.closing(psycopg2.connect(url))` pattern to mirror for get_demand_conn)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § Q2 + "Pattern 1/4" + "Code Examples" (gather_demand signature returns (demand_counts, rows_scanned, unmapped_count); get_demand_conn is read-only set_session(readonly=True))
  </read_first>
  <behavior>
    - Test 1 (shape): `gather_demand(days=14, ...)` returns a 3-tuple `(demand_counts: dict[tuple[str,str], int], rows_scanned: int, unmapped_count: int)`.
    - Test 2 (counting): given fetched rows where two rows map to `("Outer Sunset","vietnamese")` and one to `("Mission District","thai")`, `demand_counts` is `{("Outer Sunset","vietnamese"): 2, ("Mission District","thai"): 1}` and `rows_scanned == 3`.
    - Test 3 (unmapped): a row that maps to no catalog bucket increments `unmapped_count` and contributes nothing to `demand_counts`.
    - Test 4 (windowing): the SELECT is parameterised with a `created_at >= %s` cutoff computed from `days` (assert the cutoff param is passed as a list element, never string-interpolated — SQLi guard).
    - Test 5 (pool path): when `url is None`, gather_demand reads via the shared pooled `get_conn()` (sandbox); when `url` is provided it opens `get_demand_conn(url)` (direct non-pooled, read-only) and the pool is NOT touched.
    - Test 6 (multi-intent): a message mapping to multiple neighborhoods emits one demand tuple per `(neighborhood, cuisine)` pair (RESEARCH Pitfall 4 — option a, more informative for counting).
  </behavior>
  <action>
    In `scripts/coverage_agent.py` add `get_demand_conn(url: str)` as a `@contextmanager` that does `conn = psycopg2.connect(url); conn.set_session(readonly=True, autocommit=True); yield conn; finally conn.close()` (mirror loop_falsifier's direct-connection pattern; import `psycopg2` and `contextmanager`). Add `gather_demand(days: int, url: str | None = None) -> tuple[dict[tuple[str,str], int], int, int]` that: computes `cutoff = datetime.now(UTC) - timedelta(days=days)`; runs a parameterised `SELECT message, COALESCE(requested_primary_types, '{}') FROM user_query_log WHERE created_at >= %s ORDER BY created_at DESC` (cutoff passed as a param, never interpolated) using `get_demand_conn(url)` when `url` is set else the pooled `get_conn()`; for each row lexically maps cuisines via `_types_to_cuisines`, batch-extracts neighborhoods via `_extract_neighborhoods_batch` (ONE call for all rows' messages, D-01), and accumulates `demand_counts[(neighborhood, cuisine)] += 1` only for `(n, c)` where both are catalog members; counts `unmapped_count` for rows yielding no catalog bucket; returns `(demand_counts, rows_scanned, unmapped_count)`. Read `DEMAND_DATABASE_URL` from `os.environ` inside `gap_mine_main` (not here) — `gather_demand` takes the url as a param so it stays unit-testable. Do NOT modify `gather_stats` or any other existing function. Add the tests to `tests/unit/test_gap_miner.py`.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner.py -v && poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/coverage_agent.py` contains `def gather_demand(` and `def get_demand_conn(`.
    - `gather_demand` SELECTs `FROM user_query_log` with a parameterised `created_at >= %s` cutoff (assert the cutoff is a param-list element, not interpolated).
    - `get_demand_conn` opens a direct `psycopg2.connect(url)` with `set_session(readonly=True, ...)` and is NEVER used for writes (read-only by construction — T-18-PROD mitigation).
    - When `url is None`, gather_demand uses the pooled `get_conn()`; the pool's `DATABASE_URL` (sandbox) is never coerced/retargeted (no `init_db_pool` reinit, no `cache_clear`).
    - `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` still exits 0 (REGRESSION GUARDRAIL).
    - `poetry run pytest tests/unit/test_gap_miner.py -v` exits 0 with the six behaviors above.
    - `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes.
  </acceptance_criteria>
  <done>`gather_demand` reads `user_query_log` (pool or DEMAND_DATABASE_URL), returns catalog-constrained demand counts + rows_scanned + unmapped_count via a single batched LLM call; supply-only path untouched; tests green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| user_query_log.message → LLM | Raw, unscrubbed user free-text (Phase 17 stores verbatim) crosses into an external LLM (`make_judge`) during neighborhood extraction |
| DEMAND_DATABASE_URL → demand read | Prod credentials used for a read-only SELECT against prod `user_query_log` |
| message text → SQL | Demand CTE over user_query_log — user content must never be interpolated into SQL |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-18-02-PII | Information Disclosure | `_extract_neighborhoods_batch` sends `message` to make_judge | accept | This is dev/eval telemetry on a private capstone DB; the LLM is the SAME judge already in the pipeline (`vibe.make_judge`, used by `propose_queries`); no NEW PII sink is created and nothing is persisted beyond the already-existing user_query_log. Documented per RESEARCH Security Domain. LOW severity — gate does not block. |
| T-18-02-INJ | Tampering | batch neighborhood prompt | mitigate | Messages are embedded into the prompt via `json.dumps` as a JSON array element, never raw f-string interpolation, so message content cannot escape the JSON boundary and rewrite the instruction (asserted by Task 1 Test 5). |
| T-18-02-SQLi | Tampering | `gather_demand` SELECT | mitigate | Only the `cutoff` datetime is bound, via a `%s` parameter list; no user `message` content is interpolated into SQL (asserted by Task 2 Test 4). |
| T-18-02-PROD | Information Disclosure | `get_demand_conn(DEMAND_DATABASE_URL)` | mitigate | Connection opened `set_session(readonly=True)`, used only for SELECT in `gather_demand`, closed immediately; DEMAND_DATABASE_URL lives only in gitignored `.env` (documented in 18-01). Never used for writes — the write path stays on the pool (sandbox). Opt-in (unset by default). |
| T-18-02-SC | Tampering | npm/pip/cargo installs | accept | No new packages (psycopg2/mlflow/langchain-core already present; RESEARCH Package Legitimacy Audit = N/A). |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_gap_miner.py -v` exits 0.
- `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (guardrail: supply-only contract preserved).
- `gather_demand`, `get_demand_conn`, `_types_to_cuisines`, `_extract_neighborhoods_batch` exist in `scripts/coverage_agent.py`.
- `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes.
</verification>

<success_criteria>
- GAP-01 demand extraction implemented: `user_query_log` rows → catalog-constrained `(neighborhood, cuisine)` demand counts via lexical cuisine map + single batched LLM neighborhood call (D-01).
- Two-DB plumbing (`get_demand_conn` + `DEMAND_DATABASE_URL`) supports D-05's prod-read / sandbox-write split with ~15 lines, no pool retargeting.
- `unmapped_count` honesty metric captured; catalog constraint enforced upstream so off-catalog gaps can never reach `premark_seed_isolation`.
- Existing supply-only functions and their tests are byte-for-byte unaffected (guardrail).
</success_criteria>

<output>
Create `.planning/phases/18-gap-mining-gap/18-02-SUMMARY.md` when done.
</output>
