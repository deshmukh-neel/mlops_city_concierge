# Codebase Structure

**Analysis Date:** 2026-05-14

## Directory Layout

```
mlops_city_concierge/
├── app/                              # FastAPI application package
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, lifespan, routes (/chat, /predict, /health)
│   ├── config.py                     # Pydantic Settings, DB URL resolution, embedding allowlist
│   ├── db_url.py                     # Alembic-facing DB URL resolver
│   ├── db_pool.py                    # ThreadedConnectionPool singleton
│   ├── db.py                         # get_db (FastAPI dep) + get_conn (CM)
│   ├── chain.py                      # Legacy RAG chain (RetrievalQA)
│   ├── retriever.py                  # PgVectorRetriever (legacy chain) + build_embedding
│   ├── agent/                        # Agentic core (LangGraph)
│   │   ├── __init__.py
│   │   ├── graph.py                  # plan/act/critique StateGraph
│   │   ├── state.py                  # ItineraryState, Stop, PlaceCard, RevisionHint
│   │   ├── tools.py                  # LangChain StructuredTool wrappers
│   │   ├── io.py                     # history ↔ BaseMessage; stops → PlaceCard
│   │   ├── prompts.py                # SYSTEM_PROMPT + REVISION_GUIDANCE
│   │   ├── planning.py               # haversine, walking time, durations
│   │   └── critique/
│   │       ├── __init__.py           # CRITIQUE_STEP/ITINERARY/VIBE prefixes
│   │       ├── checks.py             # Deterministic itinerary checks
│   │       └── vibe.py               # Optional cheap-LLM judge
│   ├── tools/                        # Plain-Python tool implementations
│   │   ├── __init__.py
│   │   ├── retrieval.py              # semantic_search/nearby/get_details(_many)
│   │   ├── filters.py                # SearchFilters → SQL fragments
│   │   ├── booking.py                # Provider detection + deep-link URLs
│   │   └── booking_types.py          # Shared Provider Literal
│   └── observability/
│       ├── __init__.py               # Langfuse trace_request + callbacks
│       └── cost.py                   # Per-call token + USD logging
├── alembic/                          # Hand-written SQL migrations (autogenerate OFF)
│   ├── env.py                        # Loads .env, escapes % for IAM tokens
│   ├── script.py.mako
│   └── versions/                     # 6 revisions: baseline, embeddings_v2, view, helpers, W5 proposals
├── alembic.ini
├── scripts/                          # CLI entry points (Makefile-backed)
│   ├── ingest_places_sf.py           # Google Places → places_raw
│   ├── embed_places_pgvector.py      # → place_embeddings
│   ├── embed_places_pgvector_v2.py   # → place_embeddings_v2
│   ├── log_model_to_mlflow.py        # Register chain runs in MLflow
│   ├── set_production_alias.py       # Promote a version to "production"
│   ├── coverage_agent.py             # W5: propose new seed queries
│   ├── diagnose_chunks.py
│   ├── smoke_w1.py / smoke_w3.py     # Workstream smoke tests
│   └── db/init.sql                   # docker compose init for pgvector
├── tests/
│   ├── unit/                         # Mocked unit tests + functional + smoke
│   └── integration/                  # APP_ENV=integration, requires DB
├── configs/
│   └── experiments.yaml              # MLflow run configurations
├── frontend/                         # React + Vite UI (separate node container)
│   ├── src/
│   │   ├── App.jsx, main.jsx
│   │   ├── components/               # ChatPanel, RightPanel, PlaceCard, MapView, ...
│   │   ├── api/chat.js               # POST /chat client
│   │   └── styles/
│   └── dist/                         # Vite build output
├── infra/                            # Terraform — Cloud SQL, MLflow VM, firewall
│   ├── providers.tf, network.tf, sql.tf, compute.tf, backend.tf
│   ├── README.md, CI_SETUP.md
│   └── tfplan
├── implementation_plan/james/        # Workstream roadmap (W0..W7 + README index)
├── notebooks/                        # Exploration (simple.model.ipynb)
├── docs/api/chat_contract.md         # Frontend ↔ /chat shape contract
├── .github/workflows/                # ci.yml, docker.yml, terraform-{plan,apply}.yml
├── docker-compose.yml                # db (pgvector/pg16) + app + frontend
├── Dockerfile                        # python:3.11-slim, poetry install --only main
├── Makefile                          # All dev/test/migration/ingest commands
├── pyproject.toml                    # Poetry deps, ruff, pytest, mypy config
├── poetry.lock
├── .env.example                      # Template (real .env is gitignored)
├── .pre-commit-config.yaml           # ruff hook
├── README.md
├── CLAUDE.md / AGENTS.md / .github/copilot-instructions.md  # Synced agent guidance
└── .planning/codebase/               # This document set
```

## Directory Purposes

**`app/`:**
- Purpose: The FastAPI application package — every runtime import path starts here.
- Contains: HTTP layer, agent graph, tools, retrieval, DB plumbing, observability.
- Key files: `main.py`, `config.py`, `agent/graph.py`, `agent/state.py`, `tools/retrieval.py`.

**`app/agent/`:**
- Purpose: LangGraph agent — state, nodes, tools, prompts, critique.
- Contains: One module per concern (graph topology vs. state vs. critique vs. prompts).
- Key files: `graph.py` (the loop), `state.py` (the data model), `critique/checks.py` (deterministic gates).

**`app/tools/`:**
- Purpose: Plain-Python tool implementations callable by the agent, the legacy chain, evals, and tests.
- Key files: `retrieval.py` (DB-backed), `booking.py` (deterministic deep-links), `filters.py` (SearchFilters → SQL).

**`app/observability/`:**
- Purpose: Cross-cutting tracing + cost telemetry, all best-effort/no-op safe.
- Key files: `__init__.py` (Langfuse), `cost.py` (per-call USD logging).

**`alembic/versions/`:**
- Purpose: Hand-written SQL migrations. Autogenerate is disabled (`target_metadata = None`).
- Files: numbered/dated revisions (e.g. `2026_05_06_1356-b932216bf431_baseline.py`).

**`scripts/`:**
- Purpose: One-shot CLI entry points for ingestion, embedding, MLflow registration, ops, and smoke tests. All wired through Makefile targets.
- Contains: Ingest, embed, MLflow logger, coverage agent (W5), diagnose, smoke.

**`tests/unit/` and `tests/integration/`:**
- Purpose: Pytest with `asyncio_mode = "auto"`. Unit tests mock external deps via `conftest`; integration tests require running DB and `APP_ENV=integration`.

**`configs/`:**
- Purpose: Declarative experiment definitions consumed by `scripts/log_model_to_mlflow.py`.

**`frontend/`:**
- Purpose: React + Vite SPA served by a separate `node:20-alpine` container in `docker-compose.yml`. Calls `/chat`.
- Generated: `dist/` (Vite build).
- Committed: `dist/` is checked in, `node_modules/` is volume-mounted.

**`infra/`:**
- Purpose: Terraform source of truth for Cloud SQL, the `mlflow-server` GCE VM, and the `allow-mlflow` firewall rule.
- NOT here: Cloud Run (`city-concierge-api`) — deployed by GitHub Actions, intentionally not Terraform-managed.

**`implementation_plan/james/`:**
- Purpose: Workstream roadmap (W0..W7) with a `README.md` status index. Updated only on PR merge.

**`docs/api/`:**
- Purpose: Frontend ↔ backend contract documentation.

**`.github/workflows/`:**
- Purpose: CI (lint+tests), Docker build & Cloud Run deploy, Terraform plan/apply.

## Key File Locations

**Entry Points:**
- `app/main.py`: FastAPI app, lifespan, all routes.
- `alembic/env.py`: Alembic env (loads `.env`, resolves DB URL via `app.db_url`).
- `scripts/*.py`: All CLI tools.

**Configuration:**
- `app/config.py`: Pydantic `Settings`, `resolve_database_url`, `ALLOWED_EMBEDDING_TABLES`, `resolve_llm_api_key`.
- `pyproject.toml`: Poetry deps, ruff (line-length 100, rules E/F/I/N/UP/B/SIM/S), pytest (`asyncio_mode = "auto"`), mypy.
- `.env.example`: Required env var template.
- `Makefile`: Canonical commands (dev, test, migrate, ingest, lint).
- `docker-compose.yml`: Local dev services (db pg16, app, frontend node).
- `Dockerfile`: Production image (python:3.11-slim).
- `configs/experiments.yaml`: MLflow run configurations.
- `alembic.ini`: Alembic config.

**Core Logic:**
- `app/agent/graph.py`: Plan → Act → Critique LangGraph.
- `app/agent/state.py`: `ItineraryState`, `Stop`, `RevisionHint`.
- `app/agent/tools.py`: LLM-facing `StructuredTool` registry.
- `app/tools/retrieval.py`: `semantic_search`, `nearby`, `get_details`, `get_details_many`.
- `app/tools/booking.py`: Provider deep-link builder.
- `app/tools/filters.py`: `SearchFilters` + `compile_filters`.
- `app/chain.py`: Legacy `RetrievalQA` chain.
- `app/retriever.py`: `PgVectorRetriever` + cached `build_embedding`.
- `app/db_pool.py`: `ThreadedConnectionPool` lifecycle.
- `app/db.py`: `get_db` (FastAPI dep) + `get_conn` (context manager).

**Testing:**
- `tests/unit/`: Mocked unit + functional + smoke tests (one file per module, e.g. `test_agent_graph.py`).
- `tests/integration/`: DB-required tests; gated by `APP_ENV=integration`.
- `tests/conftest.py`: env-var patches so unit tests never hit real services.

## Naming Conventions

**Files:**
- snake_case Python modules: `db_pool.py`, `embed_places_pgvector_v2.py`, `coverage_agent.py`.
- Test files mirror module name: `tests/unit/test_<module>.py`, with suffixes `_smoke`, `_functional` for layered variants (`test_agent_self_correct_functional.py`, `test_chat_functional.py`).
- Alembic revisions: `YYYY_MM_DD_HHMM-<rev_id>_<slug>.py` (e.g. `2026_05_06_1356-b932216bf431_baseline.py`).
- Workstream docs: `wN_<slug>.md` under `implementation_plan/james/`.

**Directories:**
- snake_case packages: `app/agent/critique/`, `app/observability/`, `app/tools/`.
- Lowercase top-level: `app/`, `scripts/`, `tests/`, `infra/`, `configs/`, `frontend/`.

**Symbols:**
- Modules and functions: `snake_case`.
- Classes and Pydantic models: `PascalCase` (`ItineraryState`, `PgVectorRetriever`, `BookingProposal`).
- Constants: `UPPER_SNAKE` (`MAX_REVISIONS_PER_REASON`, `WALKING_SPEED_M_PER_MIN`, `ALLOWED_EMBEDDING_TABLES`, `CRITIQUE_STEP`).
- Private helpers: leading underscore (`_commit_stops`, `_prune_for_llm`, `_VIEW_FOR_TABLE`).
- Frontend response field `ragLabel`: camelCase to match frontend contract; ruff `N815` ignored at point of use.

## Where to Add New Code

**New API endpoint:**
- Route handler: append to `app/main.py` next to `/chat`/`/predict`.
- Request/response models: declare as Pydantic classes near the top of `app/main.py` (or extract to `app/schemas.py` if the file grows).
- Tests: `tests/unit/test_<endpoint>_endpoint.py` + `tests/unit/test_<endpoint>_functional.py` (mock graph/chain) + `tests/integration/` if it touches the DB.

**New retrieval tool / DB read:**
- Implementation: add a function to `app/tools/retrieval.py` using `_execute(sql, params)` and `_view_name()` so it respects the embedding-table allowlist.
- Expose to the LLM: add a wrapper in `app/agent/tools.py` and register in `_TOOLS` (the `_args_schema_for` helper builds the Pydantic schema from annotations).
- Tests: `tests/unit/test_tools_retrieval.py` + `tests/integration/test_<tool>.py`.

**New filter:**
- Add a field to `SearchFilters` in `app/tools/filters.py` and extend `compile_filters` to emit a parameterized SQL fragment.
- The view it filters against must already have the column (see `alembic/versions/*_create_place_documents_view.py`).

**New booking provider:**
- Single-source-of-truth edit: add a `Provider` literal value in `app/tools/booking_types.py` and a `_ProviderSpec` entry in `app/tools/booking.py:_PROVIDER_SPECS`. No other files need updating.

**New agent node / critique check:**
- Node: add a function inside `build_agent_graph` in `app/agent/graph.py` and wire `g.add_node` / `g.add_edge`.
- Deterministic check: add to `app/agent/critique/checks.py`, register a threshold in `CRITIQUE_THRESHOLDS`, and map it in `_hint_for_violation` (`app/agent/graph.py:319`).
- LLM-based check: model after `app/agent/critique/vibe.py` — gated by env var, bounded to one call per request.

**New Alembic migration:**
- Scaffold: `make migration MSG="add foo column"` → empty file under `alembic/versions/`.
- Body: write `op.execute("...")` SQL by hand (autogenerate is OFF — `alembic/env.py:35`).
- Apply locally: `make migrate`.

**New ingestion / ops script:**
- Place under `scripts/`, add a Makefile target alongside `make ingest` / `make embed-places` / `make coverage-agent`.
- Use `from app.config import resolve_database_url` and `load_dotenv()` at top of file (every existing script does this).

**New env var / setting:**
- Add a typed field to `Settings` in `app/config.py` with a sensible default; document in `.env.example`. Never read `os.environ` directly from request handlers.

**New experiment configuration:**
- Append a `RunConfig` block to `configs/experiments.yaml`; `make log-mlflow` will pick it up.

**New frontend component:**
- `frontend/src/components/<Name>.jsx`; wire from `App.jsx`. API calls via `frontend/src/api/chat.js`.

**New workstream:**
- New file `implementation_plan/james/wN_<slug>.md` and add a row to `implementation_plan/james/README.md`. Update only on PR merge.

## Special Directories

**`alembic/versions/`:**
- Purpose: Schema migrations.
- Generated: No (autogenerate disabled).
- Committed: Yes — every revision is hand-written SQL.

**`frontend/dist/`:**
- Purpose: Vite build output.
- Generated: Yes (`vite build`).
- Committed: Yes (checked in for static hosting deploys).

**`frontend/node_modules/`:**
- Purpose: Frontend deps.
- Generated: Yes (`npm install` inside the container).
- Committed: No — volume-mounted in `docker-compose.yml`.

**`infra/.terraform/`:**
- Purpose: Terraform provider plugins.
- Generated: Yes (`terraform init`).
- Committed: No.

**`infra/tfplan`:**
- Purpose: Saved plan output.
- Generated: Yes (`terraform plan -out tfplan`).
- Committed: Tracked but transient — refreshed per PR.

**`.planning/codebase/`:**
- Purpose: GSD codebase maps (this document set), consumed by `/gsd-plan-phase` and `/gsd-execute-phase`.
- Generated: By the `/gsd-map-codebase` workflow.
- Committed: Yes.

**`notebooks/`:**
- Purpose: Exploratory data analysis (`simple.model.ipynb`, `feature_importances.csv`).
- Generated: No (manual).
- Committed: Yes.

**`.coverage`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `__pycache__/`:**
- Purpose: Tool caches.
- Generated: Yes.
- Committed: No (gitignored).

---

*Structure analysis: 2026-05-14*
