# External Integrations

**Analysis Date:** 2026-05-14

## APIs & External Services

**LLM providers:**
- **OpenAI** — chat completions (`gpt-4o-mini` default) and embeddings (`text-embedding-3-small`, vector dim 1536)
  - SDK: `openai`, `langchain-openai` (`ChatOpenAI`, `OpenAIEmbeddings`)
  - Wired in: `app/chain.py:39-40`, `app/retriever.py:24`
  - Auth: env `OPENAI_API_KEY` (resolved via `resolve_llm_api_key("openai")` in `app/config.py:138-152`)
- **Google Gemini** — alternative chat provider (`gemini-2.5-flash` default)
  - SDK: `google-genai`, `langchain-google-genai` (`ChatGoogleGenerativeAI`)
  - Wired in: `app/chain.py:41-46`
  - Auth: env `GEMINI_API_KEY`
- **Anthropic Claude** — agent driver (W0+; selectable at runtime via MLflow registry params)
  - Auth: env `ANTHROPIC_API_KEY` (mounted as a Secret Manager secret in Cloud Run, `.github/workflows/docker.yml:286`)

**Google Places API:**
- HTTP endpoint: `https://places.googleapis.com/v1/places:searchText` (`scripts/ingest_places_sf.py:37`)
- Client: plain `requests` (no SDK)
- Auth: env `GOOGLE_PLACES_API_KEY`
- Rate-limit handling: `MIN_REQUEST_INTERVAL_SECONDS = 0.25`, `API_MAX_RETRIES = 4`, exponential backoff up to 20s

**MLflow tracking server:**
- Self-hosted on GCE VM `mlflow-server` (`infra/compute.tf:1-54`); listens on port 5000, only reachable from `10.128.0.0/9` (`infra/compute.tf:56-71`)
- SDK: `mlflow >=2.12.0,<3.0.0` — used as **client only** (CVEs in `.trivyignore` reflect this)
- Loaded at FastAPI startup to fetch the production-aliased model version + run params (`app/main.py:121-156`)
- Local dev access: `gcloud compute ssh mlflow-server --tunnel-through-iap -- -L 5000:localhost:5000`
- Docker access: `host.docker.internal:5000` (`docker-compose.yml:39`)
- Auth: optional bearer `MLFLOW_TRACKING_TOKEN` (mounted as Cloud Run secret)

**Langfuse:**
- LLM observability / per-request agent tracing
- Hosted at `https://cloud.langfuse.com` by default (env `LANGFUSE_HOST`); self-hosting supported
- Wired in: `app/observability/__init__.py` — degrades to a no-op when `LANGFUSE_SECRET_KEY` is unset
- Auth: env `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` (Cloud Run secrets)

**Booking providers (deep-link only, no API calls):**
- Resy, Tock, OpenTable, Google Maps, "unknown" fallback (`app/tools/booking_types.py:12`)
- Implementation builds deterministic deep-link URLs in `app/tools/booking.py`; user confirms the reservation on the provider's site

## Data Storage

**Databases:**
- **PostgreSQL 18 + pgvector** (production: Cloud SQL `mlops-491820:us-central1:mlops--city-concierge`, db `mlops-city-concierge`)
  - Local dev: `pgvector/pgvector:pg16` Docker image (`docker-compose.yml:3`) — note prod is PG18 per `infra/sql.tf:3`
  - Tier: `db-perf-optimized-N-8`, edition `ENTERPRISE_PLUS`, zonal availability, 100 GB PD_SSD
  - IAM DB authentication enabled (`cloudsql.iam_authentication = on`, `infra/sql.tf:25-28`)
  - Backups currently disabled (`infra/sql.tf:30-40`)
  - Connection: `DATABASE_URL` (preferred) or `POSTGRES_*` env vars assembled by `app/config.py:build_database_url()`
  - Driver: `psycopg2-binary` directly (no SQLAlchemy ORM at runtime)
  - Connection pool: `app/db_pool.py` initialized at FastAPI startup (`app/main.py:189-193`); sized by `DB_POOL_MIN_CONNECTIONS` / `DB_POOL_MAX_CONNECTIONS`
  - Vector tables: `place_embeddings`, `place_embeddings_v2` (selected by `EMBEDDING_TABLE`, validated against `ALLOWED_EMBEDDING_TABLES` allowlist in `app/config.py:10`)
  - Other tables: `places_raw` (Google Places snapshot), `city_chunks`, `places_ingest_query_proposals` (`scripts/db/init.sql`, alembic versions)
  - HNSW cosine indexes (`scripts/db/init.sql:26-28`)

**File Storage:**
- GCS bucket `mlops-491820-terraform-state` — Terraform remote state (`infra/backend.tf`)
- MLflow artifacts URI: `mlflow-artifacts://localhost:5000` (proxied through tracking server)
- No application-level object storage; uploads not used

**Caching:**
- In-process `functools.lru_cache` on OpenAI embeddings (`app/retriever.py:19-25`, `maxsize=4096`) keyed on `(query, model)`
- pydantic-settings `get_settings()` is `@lru_cache`'d (`app/config.py:130-132`)
- No Redis / Memcached

## Authentication & Identity

**Application-level auth:**
- None on `/chat`, `/predict`, `/health` — Cloud Run service is public; CORS restricted to `localhost:5173`, `localhost:3000`, and `*.vercel.app` (`app/main.py:235-242`)

**Database auth:**
- Local: password (`POSTGRES_PASSWORD` from `.env`)
- Production: GCP IAM-DB-auth via Cloud SQL Auth Proxy with short-lived tokens (`gcloud sql generate-login-token`); used in `Makefile:117-128` and `.github/workflows/ci.yml:222-237` (integration tests) and `.github/workflows/docker.yml:170-216` (migrations)
- The CI service account is a `CLOUD_IAM_SERVICE_ACCOUNT` DB user; `@` in the username is URL-encoded to `%40`

**GCP auth (CI):**
- GitHub Actions → GCP via Workload Identity Federation
- Provider: `vars.GCP_WORKLOAD_IDENTITY_PROVIDER`
- Service accounts:
  - `vars.GCP_SERVICE_ACCOUNT` — for app deploys + integration tests
  - `terraform-ci@mlops-491820.iam.gserviceaccount.com` — for `terraform plan/apply` (`.github/workflows/terraform-plan.yml:30`)

## Monitoring & Observability

**LLM tracing:**
- Langfuse (per-request, agent only) — see APIs section
- Trace context plumbed via `langgraph_callbacks()` and `metadata={"trace_id": …}` (`app/main.py:289-295`)

**Cost tracking:**
- `app/observability/cost.py` (token / cost accounting helpers)

**Error tracking:**
- Standard `logging` (logger names like `city_concierge.observability`); no Sentry / Rollbar
- Cloud Run / GCP Cloud Logging picks up stdout (uvicorn)

**Metrics / dashboards:**
- MLflow experiment runs (model registry-driven runtime selection)
- No Prometheus / Grafana / Datadog

## CI/CD & Deployment

**Hosting:**
- Backend: Google Cloud Run service `city-concierge-api` in `us-central1` (NOT Terraform-managed — deployed by GHA)
- Database: Cloud SQL (Terraform-managed in `infra/sql.tf`)
- MLflow: GCE VM (Terraform-managed in `infra/compute.tf`)
- Frontend: Vercel (CORS allows `*.vercel.app`)

**CI Pipeline (`.github/workflows/`):**
- `ci.yml` — lint (ruff), typecheck (mypy), unit tests with coverage, alembic migrations round-trip on a pgvector service container, integration tests against Cloud SQL via the proxy
- `docker.yml` — build image, Trivy scan (HIGH/CRITICAL, `.trivyignore`), push to Artifact Registry, run `alembic upgrade head` against prod, deploy to Cloud Run
- `terraform-plan.yml` — `terraform fmt/init/validate/plan` on `infra/**` PRs; comments plan back on the PR
- `terraform-apply.yml` — applies Terraform from `main` after merge

**Container registry:**
- `us-central1-docker.pkg.dev/mlops-491820/ml-repo/city-concierge` (Artifact Registry)
- Tagged with `:latest` (or semver on `v*` tag) plus `:<sha-7>`

**Pre-commit hooks:**
- ruff (`--fix`) + ruff-format, pinned to `v0.15.8` in `.pre-commit-config.yaml` to match `poetry.lock`

## Environment Configuration

**Required env vars (production-critical):**
- Database: `DATABASE_URL` **or** the trio `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` (+ `CLOUD_SQL_INSTANCE_CONNECTION_NAME` for Unix-socket Cloud SQL)
- LLMs: `OPENAI_API_KEY` (always), `GEMINI_API_KEY` (if used), `ANTHROPIC_API_KEY` (agent)
- MLflow: `MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_NAME`, optional `MLFLOW_TRACKING_TOKEN`
- Langfuse: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`
- App tuning: `EMBEDDING_TABLE`, `RETRIEVER_K`, `AGENT_MAX_STEPS`, `APP_ENV`, `LOG_LEVEL`, `DB_POOL_MIN_CONNECTIONS`, `DB_POOL_MAX_CONNECTIONS`
- Ingestion: `GOOGLE_PLACES_API_KEY`, `DATA_SOURCE_PATH`

**Secrets location:**
- Local: `.env` (gitignored; template in `.env.example`)
- Production: GCP Secret Manager — `OPENAI_API_KEY`, `GEMINI_API_KEY`, `POSTGRES_PASSWORD`, `ANTHROPIC_API_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `MLFLOW_TRACKING_TOKEN` (mounted into Cloud Run via `secrets:` block in `.github/workflows/docker.yml:285-289`; read in the migrate job at `.github/workflows/docker.yml:194-201`)

## Webhooks & Callbacks

**Incoming HTTP endpoints (`app/main.py`):**
- `GET /root` — banner
- `GET /health` — RAG chain status
- `GET /health/db` — DB connection probe
- `POST /chat` — LangGraph agent (`ChatRequest` → `ChatResponse`)
- `POST /predict` — legacy RAG chain (`RecommendationRequest` → `RecommendationResponse`)

**Outgoing:**
- OpenAI / Gemini / Anthropic API calls (per chat request)
- MLflow client calls at startup and from `scripts/log_model_to_mlflow.py`, `scripts/coverage_agent.py`, `scripts/set_production_alias.py`
- Google Places HTTP API from `scripts/ingest_places_sf.py` (batch ingestion only)
- Langfuse trace events (when enabled)

**No webhook receivers** — the service is request/response only.

---

*Integration audit: 2026-05-14*
