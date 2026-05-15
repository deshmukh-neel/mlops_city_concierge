<!-- refreshed: 2026-05-14 -->
# Architecture

**Analysis Date:** 2026-05-14

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                      HTTP / Frontend                         │
├──────────────────┬──────────────────┬───────────────────────┤
│  React (Vite)    │   FastAPI app    │   /health, /health/db │
│  `frontend/src/` │  `app/main.py`   │   `app/main.py`       │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │ POST /chat       │ POST /predict      │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                         │
│  Agent graph (LangGraph)        Legacy RAG chain             │
│  `app/agent/graph.py`           `app/chain.py`               │
│   plan → act → critique          RetrievalQA "stuff"         │
└────────┬─────────────────────────────────┬──────────────────┘
         │ tool calls                       │ retriever.invoke
         ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool / Retrieval Layer                    │
│  `app/agent/tools.py`  (LLM-facing StructuredTool wrappers)  │
│  `app/tools/retrieval.py`  semantic_search / nearby /        │
│                            get_details / get_details_many    │
│  `app/tools/booking.py`    deterministic deep-link builder   │
│  `app/tools/filters.py`    SearchFilters → SQL fragments     │
│  `app/retriever.py`        PgVectorRetriever (legacy chain)  │
└────────┬─────────────────────────────────┬──────────────────┘
         │ get_conn() (pooled psycopg2)     │ OpenAIEmbeddings
         ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Persistence              External LLM / Embedding APIs      │
│  Postgres + pgvector      OpenAI / Gemini / Langfuse         │
│  `places_raw`, `place_embeddings(_v2)`,                       │
│  `place_documents(_v2)` view, `places_ingest_query_proposals` │
│  Migrations: `alembic/versions/`                              │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  MLflow Registry (shared GCP server) — alias=production      │
│  Loaded at lifespan startup → params drive chat_model/k/...  │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| FastAPI app + lifespan | HTTP entry, startup loads RAG chain + agent graph from MLflow, manages DB pool | `app/main.py` |
| Settings | Pydantic-settings env loader, embedding-table allowlist, DB URL resolution | `app/config.py` |
| Alembic URL helper | Resolve DB URL for migrations, fail loudly | `app/db_url.py` |
| DB pool | Process-local `ThreadedConnectionPool`, init/close, lazy ensure | `app/db_pool.py` |
| DB facade | `get_db` (FastAPI dep) and `get_conn` (CM) with safe rollback-and-return | `app/db.py` |
| Legacy RAG chain | LangChain `RetrievalQA` over `PgVectorRetriever` | `app/chain.py` |
| Vector retriever (legacy) | Embed query, cosine search via raw SQL | `app/retriever.py` |
| Agent graph | LangGraph StateGraph: plan → act → critique, with revision budget | `app/agent/graph.py` |
| Agent state | `ItineraryState`, `Stop`, `UserConstraints`, `RevisionHint` | `app/agent/state.py` |
| Agent tools | LLM-facing `StructuredTool` wrappers with Pydantic args schema | `app/agent/tools.py` |
| Agent IO | History ↔ BaseMessage; stops → frontend `PlaceCard` | `app/agent/io.py` |
| Agent prompts | System prompt + revision guidance | `app/agent/prompts.py` |
| Critique checks | Deterministic itinerary checks (geographic/temporal/budget/hallucination) | `app/agent/critique/checks.py` |
| Vibe critique | Optional cheap-LLM cross-stop coherence judge | `app/agent/critique/vibe.py` |
| Planning math | Haversine, walking time, default durations | `app/agent/planning.py` |
| Retrieval tools | `semantic_search`, `nearby`, `get_details`, `get_details_many` over `place_documents(_v2)` | `app/tools/retrieval.py` |
| Booking | Provider detection (Resy/Tock/OpenTable) + URL deep-link, no LLM | `app/tools/booking.py`, `app/tools/booking_types.py` |
| SearchFilters | Pydantic filters → parameterized SQL fragments with quality floors | `app/tools/filters.py` |
| Tracing | Langfuse client + LangChain callbacks; no-op when env unset | `app/observability/__init__.py` |
| Cost telemetry | Per-call token/USD logging | `app/observability/cost.py` |

## Pattern Overview

**Overall:** Layered FastAPI + agentic LangGraph orchestration over a pgvector-backed retrieval layer, with MLflow-registry-driven runtime configuration.

**Key Characteristics:**
- **Two parallel inference paths** sharing one retrieval/embedding stack: a legacy `RetrievalQA` chain (`/predict`) and a LangGraph agent (`/chat`). Both LLMs are constructed once at startup from the same `production`-aliased MLflow run params.
- **Plan → Act → Critique** agent loop with a structured `ItineraryState` (Pydantic), bounded revisions per failure category (`MAX_REVISIONS_PER_REASON = 2`), and deterministic checks before any LLM critique.
- **Tool surface = thin LangChain wrappers** over plain Python functions in `app/tools/`, so the same retrieval primitives are used by the agent, the legacy chain, evals, and tests.
- **Deterministic enrichment** — booking deep-links are pure functions of `(PlaceDetails, when, party_size)` and never go through the LLM.
- **Single-source-of-truth env resolution** in `app/config.py:resolve_database_url`, reused by app, scripts, and Alembic.

## Layers

**HTTP / API (`app/main.py`):**
- Purpose: Wire transport, validate request/response Pydantic models, dispatch to chain or graph, hold app-level state on `request.app.state`.
- Depends on: `app.agent.graph`, `app.chain`, `app.config`, `app.db_pool`, `app.observability`.
- Used by: Uvicorn (port 8000), GitHub Actions Cloud Run deploy, frontend.

**Orchestration (`app/agent/`, `app/chain.py`):**
- Purpose: Decide the next LLM/tool action and assemble the final reply.
- Location: `app/agent/graph.py` (LangGraph), `app/chain.py` (LangChain `RetrievalQA`).
- Depends on: tool layer, LLM SDKs (`langchain_openai`, `langchain_google_genai`).
- Used by: HTTP layer at request time, MLflow logging script for example outputs.

**Tools / Retrieval (`app/tools/`, `app/retriever.py`):**
- Purpose: Encapsulate every database read and every external SDK call needed to answer the user.
- Location: `app/tools/retrieval.py`, `app/tools/booking.py`, `app/tools/filters.py`, `app/retriever.py`, `app/agent/tools.py`.
- Depends on: `app/db.py`, OpenAI embeddings, Postgres/pgvector.
- Used by: agent graph, legacy chain, eval scripts, tests.

**Persistence (`app/db_pool.py`, `app/db.py`, `alembic/`):**
- Purpose: Borrow/return pooled psycopg2 connections with safe rollback; manage schema via hand-written SQL migrations (autogenerate is off — see `alembic/env.py:34`).
- Location: `app/db_pool.py`, `app/db.py`, `app/db_url.py`, `alembic/env.py`, `alembic/versions/*.py`.

**Configuration / Observability:**
- `app/config.py` — `Settings` (BaseSettings) + `resolve_database_url` + `resolve_llm_api_key` + `ALLOWED_EMBEDDING_TABLES` validator.
- `app/observability/__init__.py` — Langfuse callbacks gated on `LANGFUSE_SECRET_KEY`.
- `app/observability/cost.py` — token/USD logging.

## Data Flow

### `/chat` Agent Request Path

1. Frontend POSTs `{message, history}` to `/chat` (`app/main.py:275`).
2. Lifespan-loaded `agent_graph` is fetched from `request.app.state` (raises 503 if MLflow load failed at startup, `app/main.py:277-279`).
3. Request wrapped in `trace_request("chat", ...)` Langfuse span (`app/main.py:282`).
4. History + new `HumanMessage` packed into `ItineraryState(messages=[...])` (`app/main.py:283`).
5. `graph.ainvoke(state, config={callbacks: langgraph_callbacks(), metadata: {trace_id}})` runs the loop (`app/main.py:289`).
6. Each loop iteration (`app/agent/graph.py:478-579`):
   - `plan` — prepend system prompt on step 0; prune ToolMessages older than the last 2 tool exchanges (`_prune_for_llm`); LLM with bound tools emits next AIMessage.
   - Edge `route_after_plan` — if AIMessage has tool_calls → `act`, else → `critique`.
   - `act` — for each tool_call: `commit_itinerary` is intercepted (`_commit_stops` validates place_ids against `_grounded_place_ids(scratch)` and rejects hallucinations, then `enrich_stops_with_booking` batched via `get_details_many`). Other tools dispatched via `asyncio.to_thread(tool.invoke, args)`. Results serialized through `_serialize_tool_result` and appended as ToolMessage.
   - `critique` — if `step_count >= max_steps` short-circuit; else if AIMessage is finalizing and `state.stops` exist run `_critique_final_with_stops` (deterministic `itinerary_violations` then optional vibe judge); else `_critique_step` diagnoses last tool result (`empty_results`/`all_closed`/`low_similarity`/`tool_error`).
   - Critique either emits a `RevisionHint` + `HumanMessage` to drive another `plan` round (bounded by `_can_retry`/`MAX_REVISIONS_PER_REASON`), or sets `done=True` with a final reply (optionally with caveats).
7. `state_to_cards(final_state)` projects committed `Stop`s into the `PlaceCard` shape (`app/agent/io.py:38`).
8. Response: `{reply, places, ragLabel}` where `ragLabel = "{provider}:{model}"` (`app/main.py:179`).

### `/predict` Legacy RAG Path

1. POST `/predict` with `{query, limit}` (`app/main.py:304`).
2. `rag_chain.invoke({"query": ...})` runs `RetrievalQA` (`app/chain.py:50-55`).
3. `PgVectorRetriever._get_relevant_documents` (`app/retriever.py:50`):
   - `OpenAIEmbeddings.embed_query(query)` → vector.
   - Raw SQL: cosine `embedding <=> %s::vector` over `{settings.embedding_table}` JOIN `places_raw` (`app/retriever.py:68-82`).
4. LLM generates answer over retrieved Documents; sources serialized via `serialize_sources` (`app/main.py:159`).

### Startup / Lifespan

1. `create_app()` builds FastAPI with CORS for `localhost:5173/3000` and `*.vercel.app` (`app/main.py:228-243`).
2. `lifespan` (`app/main.py:182`):
   - `init_db_pool(database_url, min, max)` — single process-local `ThreadedConnectionPool`.
   - `load_registered_rag_chain()` — `MlflowClient.get_model_version_by_alias(model_name, "production")`, parse params (`llm_provider`/`chat_model`/`k`/`temperature`), build chain + LLM.
   - `build_agent_graph(loaded.llm)` — bind tools to LLM, optionally construct `judge_llm` via `vibe.make_judge()`.
   - On any failure: log warning, set `app.state.rag_chain/agent_graph = None` — endpoints return 503 in degraded mode.
3. On shutdown: `close_db_pool()`.

### Ingestion / Offline Flow

1. `scripts/ingest_places_sf.py` — pull Google Places v1 search results into `places_raw` (uses `resolve_database_url(os.environ)`, `MAX_PAGES_PER_QUERY = 1`).
2. `scripts/embed_places_pgvector.py` / `scripts/embed_places_pgvector_v2.py` — generate OpenAI embeddings into `place_embeddings` / `place_embeddings_v2`.
3. `scripts/log_model_to_mlflow.py --config configs/experiments.yaml` — for each `RunConfig`, build chain, run sample queries, log params + sample outputs to MLflow registry.
4. `scripts/set_production_alias.py --version N` — promote a registered model version to the `production` alias the API loads at startup.
5. `scripts/coverage_agent.py` — read `place_query_hits` + `places_raw`, ask `vibe.make_judge()` LLM for new seed queries, insert into `places_ingest_query_proposals(status='pending')`.

**State Management:**
- App-level: `app.state.rag_chain`, `app.state.agent_graph`, `app.state.active_model_config`, `app.state.rag_label`.
- Per-request: `ItineraryState` (Pydantic) flows through every graph node; `add_messages` reducer merges new BaseMessages.
- DB pool + `Settings` are module-level singletons (`@lru_cache` on `get_settings`).

## Key Abstractions

**`ItineraryState` (`app/agent/state.py:127`):**
- Purpose: Single piece of state passed through every graph node; carries messages, structured constraints, committed stops, per-tool scratch, retry budgets, revision hints.
- Pattern: Pydantic model with `add_messages` annotated reducer for LangGraph merging.

**`Stop` / `PlaceCard` (`app/agent/state.py:113`, `:152`):**
- Purpose: Internal stop record vs. frontend-facing card; cards are projected from stops at response time in `state_to_cards`.

**`RevisionHint` (`app/agent/state.py:46`):**
- Purpose: Structured cue from `critique` to `plan` — `(reason, detail, suggested_action, target)`. Stored on state for tracing and retry-budget bookkeeping (`revision_counts`).

**`SearchFilters` (`app/tools/filters.py:16`):**
- Purpose: Pydantic-validated filters compiled by `compile_filters()` to parameterized SQL fragments. Quality floors (`min_user_rating_count=50`, `business_status='OPERATIONAL'`) apply by default.

**`PlaceHit` / `PlaceDetails` (`app/tools/retrieval.py:32`, `:47`):**
- Purpose: Unified row shape for retrieval tools; `PlaceDetails` extends `PlaceHit` with `types`, `regular_opening_hours`, `website_uri`, etc.

**`Settings` + `resolve_database_url` (`app/config.py:75`, `:47`):**
- Purpose: Single source of truth for env config; reused by FastAPI lifespan, scripts, and Alembic.

## Entry Points

**FastAPI (`app/main.py`):**
- Triggers: Uvicorn (`docker-compose.yml:43`, `Dockerfile:18`).
- Routes:
  - `GET  /root` (`app/main.py:249`)
  - `GET  /health` (`app/main.py:254`) — reports degraded if no chain loaded.
  - `GET  /health/db` (`app/main.py:267`) — `SELECT 1` via pool.
  - `POST /chat` (`app/main.py:275`) — agent graph.
  - `POST /predict` (`app/main.py:304`) — legacy RAG chain.

**CLI scripts (Makefile-backed, all under `scripts/`):**
- `scripts/ingest_places_sf.py` — `make ingest-places` — Google Places ingest.
- `scripts/embed_places_pgvector.py` / `embed_places_pgvector_v2.py` — `make embed-places` / `make embed-v2`.
- `scripts/log_model_to_mlflow.py` — `make log-mlflow` — register experiment runs.
- `scripts/set_production_alias.py --version N` — `make set-production-alias VERSION=N`.
- `scripts/coverage_agent.py` — `make coverage-agent` (dry-run) / `coverage-agent-apply` (W5).
- `scripts/diagnose_chunks.py`, `scripts/smoke_w1.py`, `scripts/smoke_w3.py` — diagnostics & smoke tests.

**Alembic (`alembic/env.py`):**
- Triggers: `make migrate` → `alembic upgrade head`; `make migration MSG=...` for new revisions.
- DB URL resolved via `app.db_url.resolve_alembic_database_url()`; `%` escaped for IAM tokens (`alembic/env.py:32`).
- `target_metadata = None` — autogenerate is OFF; all migrations are hand-written `op.execute()` SQL.

**Frontend (`frontend/src/main.jsx` → `App.jsx`):**
- Vite dev server on port 5173 (`docker-compose.yml:48-62`); calls `/chat` via `frontend/src/api/chat.js`.

**CI/CD (`.github/workflows/`):**
- `ci.yml` — tests/lint on push.
- `docker.yml` — build & deploy `city-concierge-api` to Cloud Run on every push to `main`.
- `terraform-plan.yml` / `terraform-apply.yml` — infra changes.

## Architectural Constraints

- **Threading:** FastAPI is async; tool calls in the agent's `act` node are sync (psycopg2 + OpenAI), so they're offloaded via `asyncio.to_thread(tool.invoke, args)` (`app/agent/graph.py:534`) to keep the event loop responsive.
- **Global state:** `app/db_pool.py` holds a process-local `ThreadedConnectionPool` singleton (`_pool`, `_pool_config`, guarded by `_pool_lock`). `get_settings` is `@lru_cache`d. `app/retriever.py` caches embeddings via `@lru_cache(maxsize=4096)` on `_embed_cached`. `app/observability/__init__.py` keeps `_warned_missing_package` flag.
- **Embedding-table allowlist:** `app/config.py:10` defines `ALLOWED_EMBEDDING_TABLES = {"place_embeddings", "place_embeddings_v2"}`; `Settings._validate_embedding_table` rejects anything else, so the `f"... FROM {settings.embedding_table}"` SQL in `app/retriever.py:77` and `_view_name()` in `app/tools/retrieval.py:56` are safe.
- **MLflow degraded mode:** If `load_registered_rag_chain()` fails at startup, the app boots with `app.state.rag_chain = None` and both `/chat` and `/predict` return 503 (`RAG_UNAVAILABLE_DETAIL`/`AGENT_UNAVAILABLE_DETAIL`). Recovery requires opening the IAP tunnel and restarting.
- **Alembic autogenerate disabled:** `target_metadata = None` (`alembic/env.py:35`); every migration is hand-written `op.execute()` SQL — schema is not derived from SQLAlchemy models.
- **Place-id grounding:** `_grounded_place_ids(scratch)` (`app/agent/graph.py:45`) is the only set of place_ids `commit_itinerary` will accept — every committed `Stop` must have appeared in a prior tool result, blocking hallucination.
- **Retry budget:** `MAX_REVISIONS_PER_REASON = 2` (`app/agent/graph.py:42`); after that the agent ships with caveats rather than loop further. `max_steps = 8` default in `build_agent_graph`.
- **Token budget:** `_prune_for_llm` keeps only the last `_RECENT_TOOL_EXCHANGES_KEPT = 2` tool exchanges (`app/agent/graph.py:150`) so cost stays linear in tool *kinds*, not tool *calls*.
- **Booking is not a tool:** `app/tools/booking.py` is a plain library; `enrich_stops_with_booking` (`app/agent/graph.py:98`) runs deterministically on commit. Adding new providers is a single edit to `_PROVIDER_SPECS`.
- **No circular imports:** `Provider` literal lives in `app/tools/booking_types.py` precisely so `app.agent.state` (consumer) and `app.tools.booking` (producer) can both import it without cycling.

## Anti-Patterns

### Bypassing the tool layer for ad-hoc SQL inside agent code

**What happens:** A new feature inlines `psycopg2.connect(...)` or raw SQL into `app/agent/graph.py` or a new node module.
**Why it's wrong:** Bypasses the pooled connection (`get_conn()` in `app/db.py`), the embedding-table allowlist, and breaks the contract that every DB read goes through `app/tools/retrieval.py` so evals, tests, and the legacy chain see the same primitives.
**Do this instead:** Add a function to `app/tools/retrieval.py` (or a sibling module under `app/tools/`) and call it via `get_conn()` like `get_details_many` does (`app/tools/retrieval.py:182`).

### Calling `commit_itinerary` with hallucinated place_ids

**What happens:** The LLM emits `commit_itinerary(stops=[...])` with a `place_id` it invented or copy-pasted instead of one returned by a prior tool result.
**Why it's wrong:** `_commit_stops` rejects ungrounded place_ids (`app/agent/graph.py:84`), so the commit fails and burns a revision turn for no reason.
**Do this instead:** Only commit place_ids that appear in `state.scratch[<tool>][*].result` — `_grounded_place_ids(scratch)` is the canonical set. The system prompt warns the LLM about this; revision hint `hallucinated_place_id` recovers it.

### N+1 DB reads from per-stop enrichment

**What happens:** Calling `get_details(place_id)` once per committed stop inside a loop.
**Why it's wrong:** Used to be the booking-enrichment shape; replaced because it generated N round-trips for an N-stop itinerary.
**Do this instead:** Use `get_details_many(place_ids)` (`app/tools/retrieval.py:182`) for batched lookups; see `enrich_stops_with_booking` (`app/agent/graph.py:98`).

### Reading `os.environ` inside request handlers

**What happens:** A handler or tool calls `os.getenv(...)` directly.
**Why it's wrong:** Bypasses the validated, cached `Settings` object — embedding-table allowlist, DB-URL resolution, and provider-key checks all live there.
**Do this instead:** `from app.config import get_settings; settings = get_settings()` (Langfuse env reads in `app/observability/__init__.py` are the only intentional exception, gated behind module-level helpers).

### Adding new SQL to migrations via `--autogenerate`

**What happens:** Running `alembic revision --autogenerate` and committing the diff.
**Why it's wrong:** `target_metadata = None` (`alembic/env.py:35`) — there is no SQLAlchemy model registry; autogenerate would emit empty migrations.
**Do this instead:** Use `make migration MSG="..."` to scaffold an empty revision, then write `op.execute("...")` SQL by hand. See `alembic/versions/*_create_place_documents_view.py` for the canonical pattern.

## Error Handling

**Strategy:** Fail loudly on misconfiguration (missing DB URL, missing API key, invalid embedding table); fall back gracefully on transient external failures (Langfuse, MLflow load, booking enrichment); convert tool exceptions into structured `{"error": ...}` payloads the LLM can reason about.

**Patterns:**
- Lifespan catches MLflow load failures and continues in degraded mode (`app/main.py:197-203`); endpoints return 503 with explicit detail strings (`RAG_UNAVAILABLE_DETAIL`, `AGENT_UNAVAILABLE_DETAIL`).
- Agent `act` node wraps tool invocation in `try/except` and serializes `{"error": str(e)}` back to the LLM as a `ToolMessage` (`app/agent/graph.py:535-539`); `_diagnose_one` then emits a `tool_error` revision hint.
- DB connection return is defensive: `_return_connection_safely` (`app/db.py:12`) rolls back, marks closed connections to be discarded, swallows `conn.close()` errors during shutdown.
- Booking enrichment swallows `psycopg2.Error` and skips per-stop enrichment rather than 500-ing the request (`app/agent/graph.py:117-127`).
- Langfuse client construction and trace creation are wrapped — observability never breaks the request path (`app/observability/__init__.py:60-67`, `:90-97`).

## Cross-Cutting Concerns

**Logging:** Stdlib `logging` with module-named loggers (`logger = logging.getLogger(__name__)` or `"city_concierge.cost"`/`"city_concierge.observability"`); structured JSON output for cost telemetry to be picked up by Cloud Run's log router.
**Validation:** Pydantic everywhere — request bodies (`RecommendationRequest`, `ChatRequest`), state (`ItineraryState`), tool args (auto-built via `_args_schema_for`), filters (`SearchFilters` with `field_validator`), settings (`BaseSettings`).
**Authentication:** None at the app layer (Cloud Run handles ingress); MLflow access is via IAP tunnel; Cloud SQL uses IAM-DB auth via the proxy in `make test-integration-cloud`.
**Tracing:** Per-request Langfuse spans via `trace_request("chat", ...)` + LangGraph `callbacks=langgraph_callbacks()`; no-op when `LANGFUSE_SECRET_KEY` unset.
**Cost telemetry:** `app/observability/cost.py` — JSON log line per LLM call with model/tokens/USD/latency; pricing table at `PRICING` dict (refresh quarterly).
**MLflow-driven config:** Runtime LLM provider, chat model, retriever `k`, and temperature all come from the `production`-aliased registered model's run params, not env vars (`app/main.py:99-118`).

---

*Architecture analysis: 2026-05-14*
