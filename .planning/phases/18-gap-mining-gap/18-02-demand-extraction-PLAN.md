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
    - "The miner turns raw `user_query_log` rows into catalog-constrained `(neighborhood, cuisine)` demand tuples. Neighborhood extraction is TWO-TIER: first exact lexical matching of `message` against `NEIGHBORHOODS` (case-insensitive), and ONLY rows with no lexical neighborhood hit are sent to the existing `vibe.make_judge()` LLM in a SINGLE batched call (D-01). This makes extraction work even when judge credentials are absent (REVIEW MEDIUM — lexical-before-LLM)."
    - "Cuisine extraction is SYMMETRIC to the neighborhood two-tier (REVIEW ROUND-3 NEW HIGH — cuisine-recall asymmetry): tier-1 is the lexical map from `requested_primary_types[]` (`_types_to_cuisines`); when that yields NO catalog cuisine for a row, tier-2 falls back to the `message` itself — FIRST `_lexical_cuisines(message)` (case-insensitive scan of `message` against the `CUISINES` catalog, mirroring `_lexical_neighborhoods`), THEN, only for rows STILL unresolved, the same SINGLE batched LLM call (extended to return cuisines alongside neighborhoods). This closes the asymmetry where `app/main.py`'s slot-intake prompt returns `requested_primary_types=[]` for free-text (app/main.py:76-77 — \"If the message is free-text or has no clear slot structure, return []\"), which is exactly the conversational demand the miner most wants to capture (strengthens GAP-01, still D-01 LLM use)."
    - "Judge absence does NOT suppress demand: when `vibe.make_judge()` returns None, every row whose neighborhood AND cuisine resolved via the lexical pre-passes (`_lexical_neighborhoods` and `_types_to_cuisines`/`_lexical_cuisines`) STILL produces a `(neighborhood, cuisine)` demand tuple; only the rows that NEEDED the LLM (lexical-MISS on either axis) degrade to empty and are counted in `unmapped_count`. This is the upstream half of the judge-absence semantics consumed by Plan 03's cold-start logic (REVIEW ROUND-2 MEDIUM-3 + ROUND-3 — judge None ≠ blanket no demand; the cuisine path honors the same invariant)."
    - "Neighborhood AND cuisine extraction both return a LIST of catalog members per message (not one `str | None`), so a multi-intent message contributes demand to EACH `(neighborhood, cuisine)` pair it implies. The demand tuple build pairs each extracted neighborhood with each extracted cuisine for the row (CARTESIAN within the row), staying catalog-constrained (REVIEW MEDIUM — extraction-shape fix; RESEARCH Pitfall 4 option a; ROUND-3 — cuisine cross-product)."
    - "All LLM-returned neighborhoods AND cuisines are CONSTRAINED to their respective catalogs (`NEIGHBORHOODS` / `CUISINES`); demand buckets are constrained to `NEIGHBORHOODS × CUISINES` catalog membership; rows that map to no in-catalog cuisine (even after the lexical+LLM cuisine fallback) or no in-catalog neighborhood are counted in `unmapped_count` and dropped (Claude's-Discretion: unmappable-demand catalog-constraint; honest per ROUND-3), so off-catalog gaps can never break `loop_falsifier.premark_seed_isolation`."
    - "A direct non-pooled `get_demand_conn(url)` reads demand from `DEMAND_DATABASE_URL` when set, otherwise the shared pool (sandbox `DATABASE_URL`) is used — the ~15-line two-DB plumbing for D-05's prod-read + sandbox-write split, leaving the existing pool/`get_conn()` untouched for writes."
    - "The existing supply-only `coverage_agent` functions (`gather_stats`, `find_gaps`, `propose_queries`, `filter_already_covered`, `insert_pending`, `existing_query_texts`) and their W5 tests are UNCHANGED — the demand path is strictly additive (guardrail: extend, do not regress; Claude's-Discretion miner-shape/extend)."
  artifacts:
    - path: "scripts/coverage_agent.py"
      provides: "gather_demand() + get_demand_conn() + two-tier (lexical+LLM) multi-neighborhood extraction + SYMMETRIC two-tier cuisine extraction (`_types_to_cuisines` → `_lexical_cuisines` → batched LLM) helpers"
      contains: "def gather_demand"
    - path: "tests/unit/test_gap_miner.py"
      provides: "Unit tests for the demand-extraction path (lexical cuisine map, lexical message-cuisine fallback, LLM-only cuisine fallback, judge-None+lexical-cuisine still maps, lexical-then-LLM multi-neighborhood extraction, combined batch LLM call-count, catalog constraint on both axes, unmapped_count, multi-intent cartesian demand)"
  key_links:
    - from: "scripts/coverage_agent.py gather_demand"
      to: "user_query_log table"
      via: "parameterised SELECT over created_at window"
      pattern: "FROM user_query_log"
    - from: "scripts/coverage_agent.py demand extraction"
      to: "scripts.ingest_places_sf CUISINES/NEIGHBORHOODS"
      via: "catalog-membership filter + lexical match (both axes)"
      pattern: "from scripts.ingest_places_sf import"
    - from: "scripts/coverage_agent.py cuisine fallback"
      to: "user_query_log.message"
      via: "_lexical_cuisines message scan + batched-LLM fallback when requested_primary_types yields no cuisine"
      pattern: "def _lexical_cuisines"
    - from: "scripts/coverage_agent.py get_demand_conn"
      to: "DEMAND_DATABASE_URL"
      via: "direct non-pooled psycopg2.connect when set"
      pattern: "DEMAND_DATABASE_URL"
---

<objective>
Build the demand-extraction half of the gap miner inside `scripts/coverage_agent.py`: a `gather_demand(days, url=None)` function that reads `user_query_log`, maps rows to catalog-constrained `(neighborhood, cuisine)` demand tuples via SYMMETRIC two-tier strategies on BOTH axes (cuisine: lexical `_types_to_cuisines` → lexical `_lexical_cuisines(message)` → batched LLM; neighborhood: lexical `_lexical_neighborhoods(message)` → batched LLM), and returns demand counts plus honesty metrics (`rows_scanned`, `unmapped_count`). Add the `get_demand_conn(url)` non-pooled helper that enables D-05's prod-read / sandbox-write split.

Purpose: This is GAP-01 — Phase 18's distinctive contribution. `coverage_agent` already does supply-side gap detection; what it lacks is a real demand signal. This plan adds that signal additively without touching any existing supply-only function or its W5 tests (the guardrail). It incorporates the review's extraction MEDIUMs (lexical-before-LLM, list-per-row) AND the ROUND-3 NEW HIGH: cuisine extraction is now SYMMETRIC to the neighborhood path — because `app/main.py`'s slot-intake prompt returns `requested_primary_types=[]` for free-text (app/main.py:76-77), a row like "vietnamese restaurants in Outer Sunset" would otherwise map the neighborhood but NO cuisine and land in `unmapped_count`. The cuisine message-level fallback (lexical CUISINES scan → batched LLM) recovers exactly that conversational demand. Per ROUND-2 MEDIUM-3 (extended to cuisine in ROUND-3), judge absence must NOT suppress lexically-mappable demand on EITHER axis — that is the upstream contract Plan 03's cold-start logic relies on.

Output: New `gather_demand`, `get_demand_conn`, and demand-mapping helpers in `coverage_agent.py` (including the symmetric cuisine fallback `_lexical_cuisines` + combined batched extractor); new `tests/unit/test_gap_miner.py` covering the demand path with mocks (no real DB/LLM).
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
@app/main.py
@app/config.py
@app/query_log.py
@alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Lexical cuisine map + SYMMETRIC message-cuisine fallback + lexical-then-LLM MULTI-neighborhood extraction + combined batched extractor + catalog constraint</name>
  <files>scripts/coverage_agent.py, tests/unit/test_gap_miner.py</files>
  <read_first>
    - scripts/coverage_agent.py (the file being modified — study `_parse_proposals` fence-tolerant JSON parsing and its `_FENCE_RE` at line 126 to mirror for the LLM batch parse; study `propose_queries`'s `llm.invoke([HumanMessage(content=prompt)])` shape (line 202) to reuse the same `vibe.make_judge` LLM accessor; the `from scripts.ingest_places_sf import CUISINES, build_seed_queries` at line 33)
    - scripts/ingest_places_sf.py (CUISINES line ~194 — lowercase cuisine strings like "italian","vietnamese"; NEIGHBORHOODS line ~161 — Title-Case strings — the exact catalog strings; build module-level frozensets `_CUISINES_SET`/`_NEIGHBORHOODS_SET` from them)
    - app/main.py lines ~64-88 (the `_SLOT_INTAKE_PROMPT_TEMPLATE` — SOURCE OF TRUTH for the ROUND-3 HIGH: line 76-77 instructs the LLM "If the message is free-text or has no clear slot structure, return []", so `requested_primary_types` is legitimately EMPTY for conversational rows; the Title-Case primary_type vocabulary like "Vietnamese Restaurant","Bar")
    - app/tools/filters.py lines ~168-296 (the family vocabulary so you know which primary types map to no CUISINE, e.g. "Bar"/"Cocktail Bar" → no cuisine)
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § ROUND-3 "NEW HIGH (free-text cuisine demand is dropped)" + the orchestrator "Verified against code" HIGH subsection (authoritative: add a cuisine fallback SYMMETRIC to the neighborhood two-tier — lexical message scan first, batched LLM only for misses, catalog-constrained) + the round-2 MEDIUMs "Extraction shape conflict" + "Lexical-before-LLM neighborhood matching" + "MEDIUM-3"
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § Q4 + "Pattern 2/3" + "Security Domain" + "Pitfall 4" (prompt-injection mitigation: pass messages as a JSON-encoded array; multi-neighborhood option (a) = one demand tuple per pair)
  </read_first>
  <behavior>
    - Test 1 (lexical cuisine map — tier 1): `_types_to_cuisines(["Vietnamese Restaurant", "Italian Restaurant"])` returns `["vietnamese", "italian"]`; `_types_to_cuisines(["Bar", "Cocktail Bar"])` returns `[]` (bars are off-CUISINES → no cuisine from types).
    - Test 2 (lexical message-cuisine fallback — tier 2a, ROUND-3 HIGH): `_lexical_cuisines("vietnamese restaurants in Outer Sunset")` returns `["vietnamese"]` with NO LLM call (case-insensitive scan of `message` against `CUISINES`, mirroring `_lexical_neighborhoods`); a message naming two catalog cuisines ("italian or thai tonight") returns both. A message naming no catalog cuisine ("somewhere fun") returns `[]`.
    - Test 3 (lexical neighborhood pre-pass — REVIEW MEDIUM): `_lexical_neighborhoods("dinner in Outer Sunset")` returns `["Outer Sunset"]` with NO LLM call; a message naming two catalog neighborhoods ("dinner in the Mission District and drinks in North Beach") returns `["Mission District", "North Beach"]` (multi-neighborhood, case-insensitive).
    - Test 4 (combined batched extractor — LLM only for misses): given three messages where two resolve BOTH neighborhood AND cuisine lexically and one resolves neither, only the ONE lexical-miss message is sent to `_extract_demand_batch`, and `llm.invoke` is called exactly ONCE (single combined batched call returning BOTH neighborhoods AND cuisines per message per D-01 — no second LLM round-trip).
    - Test 5 (catalog constraint on LLM output): the combined batch extractor filters each returned element to `NEIGHBORHOODS` members (for the neighborhood axis) and `CUISINES` members (for the cuisine axis) — an LLM-returned off-catalog name (e.g. "Berkeley" / "fusion") is dropped, and tolerates ```json fences.
    - Test 6 (cuisine cross-product within row — ROUND-3): a row whose extraction yields neighborhoods `["Outer Sunset"]` and cuisines `["vietnamese","thai"]` produces BOTH `("Outer Sunset","vietnamese")` and `("Outer Sunset","thai")` demand tuples (cartesian within the row), all catalog-constrained.
    - Test 7 (prompt-injection safety): the batch prompt embeds messages as a JSON-encoded array element, so a message containing `"]}` or prompt-injection text cannot escape the JSON BOUNDARY of the prompt (assert the message appears `json.dumps`-encoded in the built prompt, not raw). NOTE: this prevents prompt-FORMAT corruption only — it does NOT prevent the model from following instructions inside the message text (residual risk, see threat model T-18-02-INJ).
    - Test 8 (LLM None graceful, BOTH axes — ROUND-2 MEDIUM-3 + ROUND-3): when `vibe.make_judge()` returns None: (a) a message resolving its neighborhood lexically AND its cuisine via `_types_to_cuisines` or `_lexical_cuisines` STILL maps (judge absence does not suppress lexically-mappable demand on either axis); (b) a lexical-MISS message (needs the LLM on either axis) degrades to empty without crashing. Assert the lexical-cuisine-hit row maps even when `llm is None`.
  </behavior>
  <action>
    In `scripts/coverage_agent.py` add: module-level `_CUISINES_SET = frozenset(CUISINES)` and `_NEIGHBORHOODS_SET = frozenset(NEIGHBORHOODS)` (import `NEIGHBORHOODS` alongside the existing `CUISINES` import at line 33). Add `_types_to_cuisines(primary_types: list[str]) -> list[str]` doing `pt.lower().removesuffix(" restaurant")` then `_CUISINES_SET` membership (no LLM — tier 1). Add `_lexical_cuisines(message: str) -> list[str]` (ROUND-3 cuisine fallback tier 2a, SYMMETRIC to `_lexical_neighborhoods`): case-insensitive scan of `message` for each `CUISINES` member as a word/substring match, returning the LIST of all catalog cuisines named (empty list if none) — no LLM. Add `_lexical_neighborhoods(message: str) -> list[str]` (REVIEW MEDIUM lexical-before-LLM): case-insensitive scan of `message` for each `NEIGHBORHOODS` member, returning the LIST of catalog neighborhoods named — no LLM. Add `_build_demand_batch_prompt(messages: list[str]) -> str` that `json.dumps`-encodes the messages into a JSON array inside the prompt (prompt-injection guard — never raw f-string interpolation of message text) and asks the LLM for a parallel JSON list where each element is an object with a `"neighborhoods"` LIST (constrained to `NEIGHBORHOODS`) AND a `"cuisines"` LIST (constrained to `CUISINES`). Add `_extract_demand_batch(messages: list[str], llm) -> list[tuple[list[str], list[str]]]` that returns one `(neighborhoods, cuisines)` pair per input message: it makes ONE `llm.invoke([HumanMessage(content=prompt)])` call (only for the messages passed to it — callers pass only the rows where BOTH lexical tiers missed the axis still needed), reuses fence-tolerant JSON parsing (mirror `_FENCE_RE`/`_parse_proposals`), filters neighborhoods to `_NEIGHBORHOODS_SET` and cuisines to `_CUISINES_SET`, and returns all-empty pairs when `llm is None`. The lexical pre-passes guarantee a row resolved lexically on both axes maps even when `llm is None` (ROUND-2 MEDIUM-3 + ROUND-3 — judge absence does not suppress lexically-mappable demand on either axis). Do NOT modify any existing supply-only function (this is additive to the demand path). Add the new unit tests to `tests/unit/test_gap_miner.py` (create the file).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner.py -v && poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/coverage_agent.py` contains `def _types_to_cuisines`, `def _lexical_cuisines`, `def _lexical_neighborhoods`, `def _extract_demand_batch`, and `def _build_demand_batch_prompt`.
    - `_lexical_cuisines` maps a free-text message naming a catalog cuisine ("vietnamese restaurants in Outer Sunset") to `["vietnamese"]` with NO LLM call and returns `[]` for a message naming no catalog cuisine (ROUND-3 NEW HIGH — cuisine fallback symmetric to `_lexical_neighborhoods`).
    - `_lexical_neighborhoods` maps an exact-catalog-name message with NO LLM call and returns a LIST (multi-neighborhood).
    - `_extract_demand_batch` returns `list[tuple[list[str], list[str]]]` (a `(neighborhoods, cuisines)` pair per message), makes exactly ONE `llm.invoke` call for a multi-message input (combined extraction — no second LLM round-trip per ROUND-3), filters BOTH axes to their catalogs, and returns all-empty pairs when `llm is None`.
    - A test proves judge==None + a lexical cuisine hit (via `_lexical_cuisines` or `_types_to_cuisines`) still maps (ROUND-2 MEDIUM-3 + ROUND-3 — the judge-absence invariant holds for the cuisine path too — Test 8).
    - `grep -c 'def gather_stats\|def find_gaps\|def propose_queries\|def filter_already_covered\|def insert_pending\|def existing_query_texts' scripts/coverage_agent.py` is unchanged (6) — no existing function deleted or its signature changed in THIS task.
    - `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (existing supply-only contract preserved — REGRESSION GUARDRAIL).
    - `poetry run pytest tests/unit/test_gap_miner.py -v` exits 0 with the eight behaviors above.
    - The batch prompt builder embeds messages via `json.dumps` (assert in test), not raw interpolation.
  </acceptance_criteria>
  <done>Demand rows map to catalog-constrained cuisines via a SYMMETRIC two-tier (types → message-lexical → batched LLM) and to a LIST of neighborhoods via a lexical pre-pass + the SAME single combined batched LLM call for misses; the cuisine cross-product is built within the row; multi-intent messages keep every pair; off-catalog answers are dropped; extraction works without judge creds for exact catalog names on BOTH axes (judge None still maps lexical cuisine hits — ROUND-3); existing supply-only tests still pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: gather_demand() over user_query_log + symmetric cuisine/neighborhood two-tier wiring + get_demand_conn() two-DB plumbing</name>
  <files>scripts/coverage_agent.py, tests/unit/test_gap_miner.py</files>
  <read_first>
    - scripts/coverage_agent.py (the file being modified — `gather_stats`'s cutoff/`datetime.now(UTC) - timedelta(days=days)` pattern (line ~60) and its parameterised `cur.execute(sql, [...])` shape to mirror; the existing `get_conn` import; the Task 1 helpers `_types_to_cuisines`/`_lexical_cuisines`/`_lexical_neighborhoods`/`_extract_demand_batch`)
    - app/config.py (resolve_database_url; the lru_cache footgun note — gather_demand must NOT coerce DATABASE_URL, it opens a SEPARATE connection so no cache_clear is needed)
    - alembic/versions/2026_06_16_1137-d1be72aea7d4_add_user_query_log.py (columns to SELECT: message, requested_primary_types, created_at indexed)
    - app/main.py lines ~64-88 (the slot-intake prompt that returns `requested_primary_types=[]` for free-text — ground truth for why the cuisine fallback MUST run on the message when types is empty; ROUND-3 HIGH)
    - app/db.py (get_conn — the pooled sandbox-targeting connection used when DEMAND_DATABASE_URL is unset)
    - scripts/loop_falsifier.py lines ~650-665 (`_snapshot_ids_from_url` — the existing direct-psycopg2 `contextlib.closing(psycopg2.connect(url))` pattern to mirror for get_demand_conn)
    - .planning/phases/18-gap-mining-gap/18-RESEARCH.md § Q2 + "Pattern 1/4" + "Code Examples" (gather_demand signature returns (demand_counts, rows_scanned, unmapped_count); get_demand_conn is read-only set_session(readonly=True))
    - .planning/phases/18-gap-mining-gap/18-REVIEWS.md § ROUND-3 "NEW HIGH" (cuisine fallback) + round-2 "Extraction shape conflict" + "MEDIUM-3" (multi-neighborhood + cartesian; judge None still counts lexically-mapped rows on BOTH axes)
  </read_first>
  <behavior>
    - Test 1 (shape): `gather_demand(days=14, ...)` returns a 3-tuple `(demand_counts: dict[tuple[str,str], int], rows_scanned: int, unmapped_count: int)`.
    - Test 2 (counting): given fetched rows where two rows map to `("Outer Sunset","vietnamese")` and one to `("Mission District","thai")`, `demand_counts` is `{("Outer Sunset","vietnamese"): 2, ("Mission District","thai"): 1}` and `rows_scanned == 3`.
    - Test 3 (unmapped): a row that maps to no catalog bucket on EITHER axis (after the cuisine lexical+LLM fallback and the neighborhood two-tier) increments `unmapped_count` and contributes nothing to `demand_counts`.
    - Test 4 (windowing): the SELECT is parameterised with a `created_at >= %s` cutoff computed from `days` (assert the cutoff param is passed as a list element, never string-interpolated — SQLi guard).
    - Test 5 (pool path): when `url is None`, gather_demand reads via the shared pooled `get_conn()` (sandbox); when `url` is provided it opens `get_demand_conn(url)` (direct non-pooled, read-only) and the pool is NOT touched.
    - Test 6 (multi-intent cartesian — REVIEW MEDIUM + ROUND-3): a single message mapping to multiple neighborhoods AND/OR multiple cuisines emits one demand increment per `(neighborhood, cuisine)` pair in the cross-product — e.g. message "italian in Mission District and North Beach" increments BOTH `("Mission District","italian")` and `("North Beach","italian")`.
    - Test 7 (ROUND-3 HIGH — empty requested_primary_types, message cuisine recall): a row with `requested_primary_types=[]` (the free-text case `app/main.py` produces) whose `message` is "vietnamese restaurants in Outer Sunset" STILL produces the `("Outer Sunset","vietnamese")` demand tuple — the cuisine is recovered from the message via `_lexical_cuisines` (NO LLM), the neighborhood via `_lexical_neighborhoods`. It does NOT land in `unmapped_count`.
    - Test 8 (ROUND-3 HIGH — cuisine resolved only via LLM): a row with `requested_primary_types=[]` and a `message` whose cuisine is NOT in the lexical CUISINES scan (e.g. a paraphrase) is folded into the single batched `_extract_demand_batch` call and STILL maps when the LLM returns a catalog cuisine; assert the batched extractor is invoked for that row and the resulting pair is counted.
    - Test 9 (judge None still counts lexical on BOTH axes — ROUND-2 MEDIUM-3 + ROUND-3): with the LLM accessor returning None but rows whose neighborhood AND cuisine both resolve lexically (incl. via `_lexical_cuisines`), `gather_demand` returns NON-EMPTY `demand_counts`; only rows that needed the LLM on some axis land in `unmapped_count`.
    - Test 10 (single batched call): for a set of rows where some resolve both axes lexically and some don't, `_extract_demand_batch` is invoked AT MOST ONCE and only with the lexical-miss messages (no per-row and no per-axis second LLM round-trip — ROUND-3).
  </behavior>
  <action>
    In `scripts/coverage_agent.py` add `get_demand_conn(url: str)` as a `@contextmanager` that does `conn = psycopg2.connect(url); conn.set_session(readonly=True, autocommit=True); yield conn; finally conn.close()` (mirror loop_falsifier's direct-connection pattern; import `psycopg2` and `contextmanager`). Add `gather_demand(days: int, url: str | None = None) -> tuple[dict[tuple[str,str], int], int, int]` that: computes `cutoff = datetime.now(UTC) - timedelta(days=days)`; runs a parameterised `SELECT message, COALESCE(requested_primary_types, '{}') FROM user_query_log WHERE created_at >= %s ORDER BY created_at DESC` (cutoff passed as a param, never interpolated) using `get_demand_conn(url)` when `url` is set else the pooled `get_conn()`. For each row resolve CUISINES two-tier (REVIEW ROUND-3 NEW HIGH): tier-1 `_types_to_cuisines(requested_primary_types)`; if that returns NO catalog cuisine, tier-2a `_lexical_cuisines(message)`; rows whose cuisine is STILL unresolved are collected into the LLM-miss batch. Resolve NEIGHBORHOODS two-tier: tier-1 `_lexical_neighborhoods(message)`; rows with no lexical neighborhood are collected into the SAME LLM-miss batch. Send the union of cuisine-miss and neighborhood-miss messages to `_extract_demand_batch(misses, llm)` in ONE combined call (it returns `(neighborhoods, cuisines)` per message — fill in whichever axis was missing). Then for each row accumulate `demand_counts[(neighborhood, cuisine)] += 1` for EVERY `(neighborhood, cuisine)` in the CROSS-PRODUCT of the row's neighborhood LIST and cuisine LIST where both are catalog members (REVIEW MEDIUM + ROUND-3 cuisine cross-product); count `unmapped_count` for rows yielding no catalog bucket on either axis; returns `(demand_counts, rows_scanned, unmapped_count)`. When the LLM accessor is None, lexical hits on BOTH axes still count (ROUND-2 MEDIUM-3 + ROUND-3 — only rows that needed the LLM become unmapped). Read `DEMAND_DATABASE_URL` from `os.environ` inside `gap_mine_main` (Plan 03), not here — `gather_demand` takes the url as a param so it stays unit-testable. Do NOT modify `gather_stats` or any other existing function. Add the tests to `tests/unit/test_gap_miner.py`.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_gap_miner.py -v && poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/coverage_agent.py` contains `def gather_demand(` and `def get_demand_conn(`.
    - `gather_demand` SELECTs `FROM user_query_log` with a parameterised `created_at >= %s` cutoff (assert the cutoff is a param-list element, not interpolated).
    - ROUND-3 HIGH test: a row with `requested_primary_types=[]` and a `message` naming a catalog cuisine ("vietnamese restaurants in Outer Sunset") produces the `("Outer Sunset","vietnamese")` demand tuple and does NOT land in `unmapped_count` (cuisine recovered from the message — Test 7).
    - ROUND-3 HIGH test: a row with empty types whose cuisine resolves only via the LLM still maps via the single batched `_extract_demand_batch` (Test 8).
    - A multi-neighborhood/multi-cuisine message increments demand for EVERY (neighborhood, cuisine) pair in the cross-product (REVIEW MEDIUM + ROUND-3 — assert all pairs counted).
    - `_extract_demand_batch` is called at most once and only with lexical-miss messages (ROUND-3 — single combined batched call, no second round-trip).
    - Judge None still yields non-empty `demand_counts` for rows lexically-resolved on BOTH axes (incl. `_lexical_cuisines`); only LLM-needed rows become unmapped (ROUND-2 MEDIUM-3 + ROUND-3 — Test 9).
    - `get_demand_conn` opens a direct `psycopg2.connect(url)` with `set_session(readonly=True, ...)` and is NEVER used for writes (read-only by construction — T-18-02-PROD mitigation).
    - When `url is None`, gather_demand uses the pooled `get_conn()`; the pool's `DATABASE_URL` (sandbox) is never coerced/retargeted (no `init_db_pool` reinit, no `cache_clear`).
    - `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` still exits 0 (REGRESSION GUARDRAIL).
    - `poetry run pytest tests/unit/test_gap_miner.py -v` exits 0 with the ten behaviors above.
    - `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes.
  </acceptance_criteria>
  <done>`gather_demand` reads `user_query_log` (pool or DEMAND_DATABASE_URL), returns catalog-constrained demand counts + rows_scanned + unmapped_count via a SYMMETRIC two-tier on BOTH axes — cuisine (types → message-lexical → batched LLM) recovers free-text `requested_primary_types=[]` rows (ROUND-3 HIGH), neighborhood (lexical → batched LLM) — with a single combined batched call; multi-intent rows count every pair in the cross-product; judge None still counts lexical hits on both axes; supply-only path untouched; tests green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| user_query_log.message → LLM | Raw, unscrubbed user free-text (Phase 17 stores verbatim) crosses into an external LLM (`make_judge`) during the combined neighborhood+cuisine extraction (only for lexical-miss rows on either axis) |
| DEMAND_DATABASE_URL → demand read | Prod credentials used for a read-only SELECT against prod `user_query_log` |
| message text → SQL | Demand SELECT over user_query_log — user content must never be interpolated into SQL |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-18-02-PII | Information Disclosure | `_extract_demand_batch` sends `message` to make_judge | accept | This is dev/eval telemetry on a private capstone DB; the LLM is the SAME judge already in the pipeline (`vibe.make_judge`, used by `propose_queries`); no NEW PII sink is created and nothing is persisted beyond the already-existing user_query_log. The lexical pre-passes (neighborhood AND cuisine, ROUND-3) mean only lexical-miss rows reach the LLM, shrinking the surface. LOW severity — gate does not block. |
| T-18-02-INJ | Tampering | combined demand batch prompt | mitigate (partial) | Messages are embedded via `json.dumps` as a JSON array element, never raw f-string interpolation, so message content cannot CORRUPT the prompt FORMAT / escape the JSON boundary (asserted by Task 1 Test 7). **RESIDUAL RISK (honest framing per REVIEW MEDIUM):** `json.dumps` does NOT stop the model from FOLLOWING instructions embedded in the message text — a hostile message could still steer the extraction answer. Impact is LOW: the output is constrained to `NEIGHBORHOODS` and `CUISINES` catalog members on BOTH axes (off-catalog answers are dropped to `unmapped_count`), so the worst case is a mis-attributed in-catalog neighborhood/cuisine for one demand row — it cannot inject SQL, off-catalog seeds, or writes. |
| T-18-02-SQLi | Tampering | `gather_demand` SELECT | mitigate | Only the `cutoff` datetime is bound, via a `%s` parameter list; no user `message` content is interpolated into SQL (asserted by Task 2 Test 4). |
| T-18-02-PROD | Information Disclosure | `get_demand_conn(DEMAND_DATABASE_URL)` | mitigate | Connection opened `set_session(readonly=True)`, used only for SELECT in `gather_demand`, closed immediately; DEMAND_DATABASE_URL lives only in gitignored `.env` (documented in 18-01). Never used for writes — the write path stays on the pool (sandbox), and Plan 03's `assert_sandbox_write_target` (shared `scripts.sandbox_guard`, `current_database()`-based) enforces the sandbox write target. Opt-in (unset by default). |
| T-18-02-SC | Tampering | npm/pip/cargo installs | accept | No new packages (psycopg2/mlflow/langchain-core already present; RESEARCH Package Legitimacy Audit = N/A). |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_gap_miner.py -v` exits 0.
- `poetry run pytest tests/unit/test_coverage_agent.py tests/unit/test_coverage_agent_smoke.py -v` exits 0 (guardrail: supply-only contract preserved).
- `gather_demand`, `get_demand_conn`, `_types_to_cuisines`, `_lexical_cuisines`, `_lexical_neighborhoods`, `_extract_demand_batch` exist in `scripts/coverage_agent.py`.
- A test proves a free-text row with `requested_primary_types=[]` and a catalog cuisine in `message` becomes a `(neighborhood, cuisine)` demand tuple (ROUND-3 HIGH — cuisine recall).
- `poetry run ruff check scripts/coverage_agent.py tests/unit/test_gap_miner.py` passes.
</verification>

<success_criteria>
- GAP-01 demand extraction implemented: `user_query_log` rows → catalog-constrained `(neighborhood, cuisine)` demand counts via SYMMETRIC two-tier extraction on BOTH axes — cuisine (types → message-lexical → batched LLM) and neighborhood (lexical → batched LLM) — in a single combined batched call (D-01 + REVIEW MEDIUMs + ROUND-3 HIGH).
- Free-text rows where `app/main.py` returns `requested_primary_types=[]` still produce demand gaps because cuisine is recovered from the message (ROUND-3 HIGH — closes the cuisine-recall asymmetry; strengthens GAP-01).
- Neighborhood AND cuisine extraction return a LIST per row and work without judge creds for exact catalog names (REVIEW MEDIUMs + ROUND-3); judge None still counts lexically-mapped demand on both axes (ROUND-2 MEDIUM-3 + ROUND-3 — the upstream half Plan 03's cold-start logic relies on); multi-intent rows count every pair in the cross-product.
- Two-DB plumbing (`get_demand_conn` + `DEMAND_DATABASE_URL`) supports D-05's prod-read / sandbox-write split with ~15 lines, no pool retargeting.
- `unmapped_count` honesty metric captured; catalog constraint enforced upstream so off-catalog gaps can never reach `premark_seed_isolation`.
- Existing supply-only functions and their tests are byte-for-byte unaffected (guardrail).
</success_criteria>

<output>
Create `.planning/phases/18-gap-mining-gap/18-02-SUMMARY.md` when done. Note that this plan closes the ROUND-3 cuisine-recall HIGH by making cuisine extraction symmetric to the neighborhood two-tier (lexical message scan → combined batched LLM) so free-text rows with empty `requested_primary_types` still become demand gaps.
</output>
