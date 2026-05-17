# Future-watch — concerns to revisit, not act on now

This file collects design concerns from the planning round that are real but
not load-bearing today. Keep them documented so we don't rediscover them
mid-implementation. Each item lists what to watch for, what would trigger
action, and where in the plan it touches.

## Verify before implementation

### LangGraph ↔ Pydantic AI integration story

**What:** W2 currently writes a small adapter that wraps Pydantic AI tool
functions as LangChain `Tool` instances so `llm.bind_tools(...)` works inside
the LangGraph node (`app/agent/tools.py`, `_to_lc_tool` helper).

**Concern:** The author of the plan was working from training data through
January 2026. The LangGraph and Pydantic AI ecosystems were both moving fast
at the time. By the time W2 is implemented, one of three things may be true:

1. LangGraph added first-class Pydantic AI tool support → delete the adapter.
2. Pydantic AI added a way to plug into `StateGraph` directly → maybe simplify W2's graph.
3. Nothing changed → ship the adapter as planned.

**Trigger:** before starting W2 implementation, check
  - `langchain-ai/langgraph` GitHub releases
  - `pydantic/pydantic-ai` docs and changelog

If either now offers native interop, update W2's `app/agent/tools.py` section
to use it; the rest of the workstream is unaffected.

### Apache AGE on Cloud SQL

**What:** W7 explicitly rules out Apache AGE for the knowledge graph because
it isn't supported on Cloud SQL for Postgres.

**Concern:** This was true as of the planning round. If GCP starts supporting
the AGE extension on Cloud SQL, we could in principle gain Cypher queries
for free.

**Trigger:** Before starting W7 implementation, check Cloud SQL extension list
(`SELECT * FROM pg_available_extensions WHERE name = 'age';`). If it's there,
reassess — but plain edge tables are likely still simpler at this scale.

## Watch as the codebase grows

### `PlaceCard` is missing address / rating / price_level

**What:** `state_to_cards` (`app/agent/io.py:38`) projects `Stop` → `PlaceCard`,
but `Stop` never carried `address`, `rating`, or `price_level` (W2 didn't add
them). Every card the frontend renders has `address: null, rating: null,
price_level: null` even though `places_raw` has all three. `PlaceCard` declares
the fields as optional, so the gap is silent.

**Concern:** the frontend gets a degraded card — name + booking link but no
location, no rating, no price tier. Most place-card UIs surface those.

**Resolved (2026-05-16):** Fixed via Approach 1 on branch
`fix/placecard-address-rating-price`. `Stop` now carries
`address`/`rating`/`price_level`; populated in `enrich_stops_with_booking`
(`app/agent/graph.py`) from the already-fetched `PlaceDetails` — zero extra DB
calls, reusing the existing batched `get_details_many` read. `price_level` is
mapped enum→int 0..4 via `price_level_to_rank` (`app/agent/state.py`, mirrors
SQL `price_level_rank()`). Card fields are stamped *before* the `when is None`
booking skip so timeless stops still render full cards. `state_to_cards`
(`app/agent/io.py`) passes the fields through. Frontend contract
(`docs/api/chat_contract.md`) already declared these fields — no shape change,
the backend just stopped shipping them as `null`.

### `app/agent/` directory size

**What:** W2 + W3 + W4 + W7 all add files under `app/agent/`. After everything
lands, expect: `state.py`, `tools.py`, `prompts.py`, `graph.py`, `planning.py`,
`critique/checks.py`, `critique/vibe.py`, plus W4's `booking.py`.

**Concern:** Single-purpose files are good, but if any one (especially
`graph.py`) pushes past ~400 lines or `app/agent/` exceeds ~10 top-level
files, navigation gets harder.

**Trigger:** when `app/agent/graph.py` exceeds ~400 lines OR a flat directory
has 10+ files, split: `app/agent/critique/` is already its own dir; consider
`app/agent/booking/` (W4) and `app/agent/kg/` (W7 wrapping) at that point. Do
NOT pre-split — empty subdirs hurt readability more than a flat list of 6
files.

### W7 KG builder is unscoped O(n²); integration test slow + non-isolated

**What:** `build_near` / `build_same_neighborhood` in `scripts/build_place_relations.py`
self-join the entire `places_raw` table with no fixture/place scoping (~10 min
per full build at 5,855 SF places → ~2.0M edges). `tests/integration/test_build_place_relations.py`
seeds a 10-row `KGT_`-prefixed fixture but the builder ignores that scope, so
each of the suite's 8 `main()` invocations rebuilds the *entire production
graph* → ~60–90 min wall time, and the run mutates/repopulates the real
~1.3M-edge graph as a side effect (teardown only deletes `KGT_%` rows).

**Concern:** The integration suite is impractically slow against shared Cloud
SQL and not isolated — running it pollutes production edges. The builder
itself is correct and idempotent (verified at full 5,855-place scale during
W7 UAT); this is a test-isolation / scoping gap, not a correctness bug. Also
note: Cloud SQL IAM users are read-only (`has_schema_privilege CREATE = f`) —
`make migrate` / `make build-relations` need the `postgres` role, not a
per-user IAM token. CI/prod migration steps need a CREATE-capable role.

**Trigger:** before wiring the W7 integration suite into CI, or before anyone
needs to run it routinely. Fix by scoping the builder to a fixture (place_id
prefix filter or a temp-schema run) so the integration test is fast and
isolated. Until then, the suite stays gated behind `APP_ENV=integration` and
W7's contract is covered by unit/smoke/functional tests + the full-scale UAT
evidence in `.planning/.../01-HUMAN-UAT.md`. Not load-bearing for the demo
(the live `kg_traverse` query is an indexed 0.3 ms lookup; only the offline
builder is slow).

### v1/v2 embedding script duplication

**What:** `scripts/embed_places_pgvector.py` and `_v2.py` share ~80% of their
code intentionally (W0a keeps them parallel during the v1→v2 migration).

**Concern:** If the duplication outlives the migration window (e.g. v1 is
never retired because we lose track), the two scripts will drift and
re-running v1 may write inconsistent embeddings.

**Trigger:** Once W6 evals confirm v2 wins (`EMBEDDING_TABLE=place_embeddings_v2`
is the production alias and stable), open a follow-up PR that deletes
`scripts/embed_places_pgvector.py` and the `place_embeddings` table. Do NOT
refactor into a shared base class during the migration — that adds an
abstraction layer at exactly the moment when both code paths need to be
independently inspectable.

### `SearchFilters` field count

**What:** After W1's expansion, `SearchFilters` has ~25 fields (price/rating/
hours/neighborhood/types + 16 boolean amenities).

**Concern:** Flat option models past ~30 fields stop being readable in a
single screen. The LLM also sees this schema in the tool description; very
large schemas can degrade tool-call quality.

**Trigger:** if we add 5+ more amenity booleans (e.g. for a non-SF city's
attribute set), split into `SearchFilters` (general numeric / string) and
`AmenityFilters` (booleans), composed in tool signatures. Until then, the
flat shape is fine.

### Planning duration defaults

**What:** `DEFAULT_STOP_DURATION_MIN` in `app/agent/state.py` (W2) hardcodes
per-`primary_type` durations: restaurant=90, bar=60, etc.

**Concern:** Numbers are reasonable guesses, not measured.

**Trigger:** after the first eval pass with real users, reset these from
logged user-overrides in `state.scratch` — not from intuition. Add a small
analysis script in `scripts/analyze_durations.py` if/when this matters.

## Notes that supersede merged plans

### Judge default flipped to Gemini 3.1 Flash Lite preview (W5 PR)

**What:** W3 (merged) shipped `app/agent/critique/vibe.py` with
`DEFAULT_JUDGE_PROVIDER='openai'` / `DEFAULT_JUDGE_MODEL='gpt-4o-mini'`. As
part of W5 the defaults flipped to `provider='gemini'` /
`model='gemini-3.1-flash-lite-preview'`. W3's plan text and W6's plan text
still mention `gpt-4o-mini` — those references are historical, not
prescriptive.

**Why:** user request — `gemini-3.1-flash-lite-preview` is the cheaper
current-generation Flash tier; same role (cheap rubric judge) and the env
override mechanism (`EVAL_JUDGE_PROVIDER` / `EVAL_JUDGE_MODEL`) is
unchanged.

**Trigger:** before W6 implementation, treat the W6 plan's references to
`gpt-4o-mini` / `gemini-2.5-flash` as out-of-date. The judge is whatever
`vibe.make_judge()` returns — don't re-derive provider/model in W6.

### Coverage agent does not change runtime agent driver model

**What:** W5 only flipped the **judge** model (vibe / coverage / W6 taste
rubric). The runtime agent driver stays MLflow-registry-selected via
`parse_active_model_config` (`app/main.py:99-110`).

**Trigger:** if/when the user wants to swap the runtime agent driver,
that's a separate change — register a new model version in MLflow and
promote via `set-production-alias`, no code edit needed.

### CI SA grants live in alembic, not Terraform

**What:** PR #78 established the pattern: when a workstream creates a
new table that the CI SA (`github-actions-deployer@mlops-491820.iam`)
needs to write to, ship a follow-up alembic migration that does
`GRANT INSERT, DELETE ON <table> TO "<sa>"` wrapped in a `DO $$ … END $$`
block guarded by `IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = ...)`.
That guard makes the migration a no-op on local dev DBs and the ephemeral
CI Postgres where the SA isn't provisioned, so it applies cleanly
everywhere.

**Why not Terraform:** Postgres-level GRANTs aren't a Terraform-native
resource, and adding the `cyrilgdn/postgresql` provider for one
cross-cutting concern that's already adjacent to schema management was
considered out-of-scope. The alembic-migration approach lives next to
the schema it's granting on, runs through the same `migrate` job that
deploys the table, and survives downgrade-cleanup.

**Trigger:** any future workstream where the integration job needs to
write to a new table. Reference the W5 footer + `alembic/versions/2026_05_08_1100-a1b2c3d4e5f6_grant_ci_sa_proposals.py`
for the template. **Note:** the CI SA only needs writes for tables that
integration tests touch directly. Read-only retrieval tables don't need
a grant — `SELECT` on `public` is already there.

### Integration test skip gates should probe privileges, not just existence

**What:** PR #77's first cut of `_proposals_table_or_skip` only checked
`information_schema.tables`. After merge, the W5 schema deployed (via the
`migrate` job in `docker.yml`) before the GRANT migration existed, so the
SA could see the table but not write to it — main went red. PR #78 hardened
the gate to also check `has_table_privilege(current_user, ..., 'INSERT, DELETE')`.

**Trigger:** any future integration-test fixture that gates on schema
readiness should follow the same pattern: existence + privilege probe.
Schema-deploy and grant-deploy can land in different commits / different
migration runs; assume there's a window where the table exists without
the grant and skip cleanly.

### Coverage agent's runtime DB role (when it gets scheduled)

**What:** Today the W5 agent runs from a developer laptop via
`make coverage-agent[-apply]` with whatever creds are in `.env`. When it
moves to a scheduled runner (Cloud Run job, cron VM, etc.), it needs its
own DB role with **least privilege**: `SELECT` on `places_raw`, `place_query_hits`,
`places_ingest_query_checkpoints`, `places_ingest_query_proposals`, plus
`INSERT` on `places_ingest_query_proposals`. No DDL.

**Why this matters:** reusing the CI SA or the ingest SA gives the agent
access it doesn't need (principle of least privilege). The agent also
runs LLM-generated payloads through SQL parameters — narrowing privileges
limits blast radius if a prompt-injection attack ever produced a malicious
proposal.

**Trigger:** when W5 moves from "James runs it manually" to a scheduled
job. Likely paired with whatever workstream introduces the scheduler
(possibly a sibling of W7, or its own follow-up). Provision the SA in
Terraform (it's a GCP-side concern), then GRANT the privileges via an
alembic migration following the pattern above.

### `gemini-3.1-flash-lite-preview` not in cost PRICING table

**What:** `app/observability/cost.py:24-32` PRICING dict only knows
`gemini-2.5-flash`. Calls made with the new judge model record
`est_cost_usd = 0.0` until pricing is added.

**Trigger:** when judge usage volume matters for cost telemetry (or
before any cost dashboard is shown to users), add the
`gemini-3.1-flash-lite-preview` row to PRICING with current per-MTok
input/output rates from
<https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview>.

## Architectural items deliberately deferred

### Supervisor + specialized subagents

**What:** Considered and deferred during the planning round. See
`w3_self_correction.md`'s "Future direction: supervisor / specialized
subagents" section for reasoning.

**Trigger:** add a supervisor / subagent split if any of:
- W4 booking grows beyond URL deep-links into actual API calls (different
  ToS regime, different rate limits, very different prompt).
- We add open-web search as a first-class tool with a dramatically different
  prompt regime than place-lookup.
- Eval data shows the single planner is losing quality on multi-domain queries.

Until then, the W2 `plan → act → critique` loop is sufficient.

### Constraint-extractor pre-pass

**What:** A cheap-LLM node that turns user free-text into a structured
`UserConstraints` before the main planning agent runs. Sketched in W3's
"Future direction" section.

**Trigger:** if W6 evals show the planner spending tokens on constraint
parsing rather than planning, OR if `gpt-4o-mini`-tier models start
underperforming on the planner role due to NLU load. Add as a node BEFORE
`plan` that writes `state.constraints` directly.

### LLM-extracted KG edges

**What:** W7 (merged in [#83](https://github.com/deshmukh-neel/mlops_city_concierge/pull/83))
only seeds free / computed edges (NEAR, SAME_NEIGHBORHOOD, CONTAINED_IN,
NEAR_LANDMARK, SIMILAR_VECTOR). LLM-extracted edges (OPERATED_BY,
MENTIONED_WITH, SAME_CHEF) and editorial-source edges (Eater / Infatuation
scrape) were explicitly out-of-scope in #83 and remain deferred. The shipped
`place_relations` schema is forward-compatible with both.

**Trigger:** once the editorial scrape (Eater / Infatuation) lands and we
have prose that mentions chefs / owners / sister-restaurants, an extraction
PR adds those edges with `source = 'editorial_llm'` and a confidence in
`metadata`. The schema is forward-compatible.

### Apache AGE / Cypher

**What:** Pure SQL edge tables in W7 vs. Cypher via AGE. AGE not on Cloud SQL
today.

**Trigger:** if/when AGE becomes available on Cloud SQL AND our edge count
or traversal complexity outgrows what plain SQL handles cleanly (multi-hop
traversals with weighted paths). Today's NEAR / SIMILAR_VECTOR queries are
plain SELECTs and don't need Cypher.

### Streaming `/chat` responses

**What:** Frontend currently waits for full response. LangGraph supports
`astream_events()`.

**Trigger:** when latency feedback from real users shows multi-second waits
hurt UX. Layer streaming on top of W2's `/chat` without changing the
response contract.

### Managed eval UI (Langfuse / Braintrust)

**What:** W6 logs to MLflow only. MLflow's UI is okay but not great for
diffing N agent runs across M queries with full traces.

**Trigger:** once we have ≥3 evals worth of run-to-run data and the
diff-and-debug workflow is painful in MLflow's UI. Mirror the same metrics
to Langfuse (free self-host tier) or Braintrust without removing the MLflow
gate.
