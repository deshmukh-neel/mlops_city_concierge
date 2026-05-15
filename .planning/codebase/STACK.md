# Technology Stack

**Analysis Date:** 2026-05-14

## Languages

**Primary:**
- Python `^3.10` (pinned to 3.10 in CI / mypy / ruff `target-version = "py310"`) — application, scripts, tests, Alembic migrations
- HCL (Terraform `~> 1.9`) — GCP infrastructure under `infra/`

**Secondary:**
- JavaScript (ES modules) + JSX — frontend in `frontend/src/` (React 18)
- SQL — Postgres baseline schema in `scripts/db/init.sql` and Alembic versions in `alembic/versions/`
- Bash — `Makefile`, GitHub Actions inline scripts
- YAML — workflow definitions in `.github/workflows/`, experiment configs in `configs/`

## Runtime

**Environment:**
- Application container: `python:3.11-slim` base image (`Dockerfile:1`) running uvicorn
- Local development: Python 3.10 (CI matrix), Python 3.11 inside Docker
- Frontend container: `node:20-alpine` (`docker-compose.yml:49`)

**Package Manager:**
- Poetry `1.8.3` (pinned in `.github/workflows/ci.yml:14` and `.github/workflows/docker.yml:159`)
- Lockfile: `poetry.lock` (present, 674 KB — committed)
- Frontend: npm with `frontend/package-lock.json`

## Frameworks

**Core:**
- FastAPI `>=0.111.0,<1.0.0` — HTTP API (`app/main.py`)
- Uvicorn `>=0.29.0,<1.0.0` with `[standard]` extras — ASGI server, port 8000 (`Dockerfile:18`)
- LangChain `>=0.2.0,<1.0.0` + `langchain-community`, `langchain-openai`, `langchain-google-genai` — RAG chain construction (`app/chain.py`)
- LangGraph `>=0.2.0,<1.0.0` — agent state graph (`app/agent/graph.py`)
- SQLAlchemy `>=2.0.29,<3.0.0` — used by Alembic only (the runtime app talks to Postgres directly via psycopg2)
- Alembic `>=1.13.1,<2.0.0` — schema migrations (`alembic.ini`, `alembic/versions/`)
- Pydantic `>=2.7.0,<3.0.0` + `pydantic-settings >=2.2.1,<3.0.0` — request/response models, settings (`app/config.py`)

**Frontend:**
- React `^18.2.0` + react-dom `^18.2.0` (`frontend/package.json`)
- Vite `^5.1.4` with `@vitejs/plugin-react ^4.2.1` — dev server on port 5173

**Testing:**
- pytest `>=8.2.0,<9.0.0` with `asyncio_mode = "auto"` (`pyproject.toml:62-64`)
- pytest-asyncio, pytest-cov, pytest-mock
- factory-boy `>=3.3.0,<4.0.0` — test fixtures

**Build/Dev:**
- ruff `>=0.4.4,<1.0.0` — lint + format, line-length 100, rules `E, F, I, N, UP, B, SIM, S` (`pyproject.toml:66-76`)
- mypy `>=1.10.0,<2.0.0` — `strict = false`, `ignore_missing_imports = true`
- pre-commit `^4.6.0` — hooks pinned in `.pre-commit-config.yaml` to `ruff-pre-commit v0.15.8` (must match poetry.lock)
- types-requests, types-psycopg2 — type stubs

## Key Dependencies

**Critical:**
- `psycopg2-binary >=2.9.9,<3.0.0` — Postgres driver used directly by `app/db.py`, `app/db_pool.py`, ingestion scripts
- `pgvector >=0.3.0,<1.0.0` — pgvector Python helpers; vector(1536) columns for OpenAI `text-embedding-3-small`
- `openai >=1.30.0,<2.0.0` — OpenAI SDK (chat + embeddings)
- `google-genai >=0.7.0,<1.0.0` — Gemini SDK (alternative LLM provider)
- `mlflow >=2.12.0,<3.0.0` — experiment tracking + model registry; production model loaded at FastAPI startup (`app/main.py:121-156`)
- `tiktoken >=0.7.0,<1.0.0` — tokenizer
- `langfuse >=2.0.0,<3.0.0` — per-request agent tracing (`app/observability/__init__.py`); degrades to no-op when `LANGFUSE_SECRET_KEY` unset

**Infrastructure / utility:**
- `httpx >=0.27.0,<1.0.0` — async HTTP client
- `tenacity >=8.3.0,<9.0.0` — retries (used in ingestion)
- `pyyaml >=6.0.1,<7.0.0` — config loading (`configs/experiments.yaml`)
- `python-dotenv >=1.0.1,<2.0.0` — `.env` loading in scripts (settings use pydantic-settings instead)
- `requests` (transitive) — used by `scripts/ingest_places_sf.py` to call the Google Places HTTP API directly

## Configuration

**Environment:**
- All runtime config flows through pydantic-settings `Settings` (`app/config.py:75-127`), backed by `.env`
- `.env.example` (74 lines) is the canonical list of required vars; copied to `.env` via `make env`
- Settings include: `DATABASE_URL` or `POSTGRES_*`, `CLOUD_SQL_INSTANCE_CONNECTION_NAME`, `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`), `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`), `GEMINI_API_KEY`, `GEMINI_CHAT_MODEL` (default `gemini-2.5-flash`), `MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_NAME` (default `city-concierge-rag`), `EMBEDDING_TABLE` (validated against `{place_embeddings, place_embeddings_v2}`), `RETRIEVER_K`, `AGENT_MAX_STEPS`, `LANGFUSE_*`, `ANTHROPIC_API_KEY`, `MLFLOW_TRACKING_TOKEN`, `DB_POOL_MIN_CONNECTIONS`, `DB_POOL_MAX_CONNECTIONS`
- Cloud Run deploy injects secrets from GCP Secret Manager and merges env vars (`.github/workflows/docker.yml:281-289`)

**Build:**
- `Dockerfile` — single-stage, installs prod-only Poetry deps, copies `app/` only
- `pyproject.toml` — Poetry config, ruff/mypy/pytest tool tables
- `alembic.ini` — leaves `sqlalchemy.url` blank; resolved at runtime by `app.db_url.resolve_alembic_database_url()`
- `.pre-commit-config.yaml` — ruff (lint with `--fix` + format)
- `.trivyignore` — accepted CVEs (mostly client-side MLflow + transitive)

## Platform Requirements

**Development:**
- Docker + Docker Compose for local Postgres (pgvector/pgvector:pg16) and the FastAPI container
- Poetry installed locally for `make test` / `make migrate` / `make lint`
- `gcloud` + `cloud-sql-proxy` for Cloud SQL integration testing (`make test-integration-cloud`, `Makefile:103-128`)
- gcloud IAP SSH tunnel for the private MLflow VM (`gcloud compute ssh mlflow-server --tunnel-through-iap -- -L 5000:localhost:5000`)

**Production:**
- Google Cloud Run (`city-concierge-api`) in `us-central1` — image deployed from Artifact Registry by GitHub Actions on push to `main`
- Cloud Run sizing: `min-instances=1`, `max-instances=10`, `concurrency=15`, `cpu=2`, `memory=2Gi`, `timeout=120s` (`.github/workflows/docker.yml:274-280`)
- Cloud SQL Postgres 18 instance `mlops-491820:us-central1:mlops--city-concierge`, tier `db-perf-optimized-N-8`, edition `ENTERPRISE_PLUS` (`infra/sql.tf`)
- MLflow tracking server on GCE VM `mlflow-server` (e2-small, Ubuntu 22.04 LTS) in default VPC at `10.128.0.2:5000` (`infra/compute.tf`)
- Artifact Registry: `us-central1-docker.pkg.dev/mlops-491820/ml-repo/city-concierge`

---

*Stack analysis: 2026-05-14*
