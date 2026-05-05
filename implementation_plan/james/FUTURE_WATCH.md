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

**What:** W7 only seeds free / computed edges (NEAR, SAME_NEIGHBORHOOD,
CONTAINED_IN, NEAR_LANDMARK, SIMILAR_VECTOR). LLM-extracted edges
(OPERATED_BY, MENTIONED_WITH, SAME_CHEF) are deferred.

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
