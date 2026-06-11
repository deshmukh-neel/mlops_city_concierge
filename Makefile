.DEFAULT_GOAL := help
SHELL := /bin/bash

# ─── Variables ────────────────────────────────────────────────────────────────
DOCKER_COMPOSE := docker compose
POETRY_RUN     := poetry run

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Environment ──────────────────────────────────────────────────────────────
.PHONY: env
env: ## Copy .env.example to .env (skips if .env already exists)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env created — fill in your secrets before running."; \
	else \
		echo ".env already exists, skipping."; \
	fi

# ─── Development ──────────────────────────────────────────────────────────────
.PHONY: dev
dev: ## Start all services in development mode (Docker Compose)
	$(DOCKER_COMPOSE) up --build

.PHONY: dev-detached
dev-detached: ## Start all services in the background
	$(DOCKER_COMPOSE) up --build -d

.PHONY: down
down: ## Stop and remove all containers
	$(DOCKER_COMPOSE) down

.PHONY: logs
logs: ## Tail logs from all running services
	$(DOCKER_COMPOSE) logs -f

# ─── Database ─────────────────────────────────────────────────────────────────
.PHONY: db-up
db-up: ## Start only the database service
	$(DOCKER_COMPOSE) up -d db

.PHONY: migrate
migrate: ## Run Alembic database migrations
	$(POETRY_RUN) alembic upgrade head

.PHONY: migration
migration: ## Create a new Alembic migration (usage: make migration MSG="add cities table")
	$(POETRY_RUN) alembic revision --autogenerate -m "$(MSG)"

# ─── Ingestion ────────────────────────────────────────────────────────────────
.PHONY: ingest
ingest: ## Run the data ingestion pipeline
	$(POETRY_RUN) python scripts/ingest.py

.PHONY: ingest-places
ingest-places: ## Pull SF Google Places data into Postgres
	$(POETRY_RUN) python scripts/ingest_places_sf.py

.PHONY: embed-places
embed-places: ## Generate pgvector embeddings for places
	$(POETRY_RUN) python -m scripts.embed_places_pgvector

.PHONY: embed-v2
embed-v2: ## Generate cleaned pgvector embeddings into place_embeddings_v2
	$(POETRY_RUN) python -m scripts.embed_places_pgvector_v2

.PHONY: build-relations
build-relations: ## Build the place_relations knowledge graph (args: ARGS="--only NEAR")
	$(POETRY_RUN) python scripts/build_place_relations.py $(ARGS)

.PHONY: mlflow-tunnel
mlflow-tunnel: ## Open IAP tunnel to the private MLflow VM (localhost:5050; run in a separate terminal)
	gcloud compute start-iap-tunnel mlflow-server 5000 \
	  --local-host-port=localhost:5050 \
	  --zone=us-central1-a --project=mlops-491820

.PHONY: log-mlflow
log-mlflow: ## Log a RAG configuration and sample outputs to MLflow (needs `make mlflow-tunnel` running)
	$(POETRY_RUN) python scripts/log_model_to_mlflow.py --config configs/experiments.yaml

.PHONY: coverage-agent
coverage-agent: ## Dry-run the coverage-gap ingestion agent (W5)
	$(POETRY_RUN) python scripts/coverage_agent.py --dry-run

.PHONY: coverage-agent-apply
coverage-agent-apply: ## Run the coverage-gap agent and insert proposals (W5)
	$(POETRY_RUN) python scripts/coverage_agent.py

.PHONY: set-production-alias
set-production-alias: ## Promote a registered model version to production (usage: make set-production-alias VERSION=42)
	$(POETRY_RUN) python scripts/set_production_alias.py --version $(VERSION)

.PHONY: train-simple-model
train-simple-model: ## Train a simple baseline model from places data
	$(POETRY_RUN) python scripts/train_simple_model.py

# ─── Eval (Plan 03-05 / EVAL-10) ──────────────────────────────────────────────
# Parameter variables — override on the command line.
#   make eval-agent PROVIDER=openai MODEL=gpt-4o-mini QUERIES=1 SCENARIOS=omakase_mission_open_ended
#   make eval-matrix RUNS=3
#   make eval-matrix LLM_OVERRIDE=scripted   (CI mode — no APP_ENV gate needed)
PROVIDER ?= scripted
MODEL ?= placeholder
RUNS ?= 1
QUERIES ?= 1
SCENARIOS ?=
LLM_OVERRIDE ?=

.PHONY: eval-agent
eval-agent: ## Run scripts/eval_agent.py once (PROVIDER/MODEL/QUERIES/SCENARIOS params)
	$(POETRY_RUN) python scripts/eval_agent.py \
	  --llm-provider $(PROVIDER) \
	  --chat-model $(MODEL) \
	  $(if $(SCENARIOS),--scenario-ids $(SCENARIOS),) \
	  --max-queries $(QUERIES)

.PHONY: eval-matrix
eval-matrix: ## Run cross-provider matrix (LLM_OVERRIDE=scripted for CI; RUNS=3 default)
	$(POETRY_RUN) python scripts/eval_matrix.py \
	  --matrix-config configs/eval_matrix.yaml \
	  --runs $(RUNS) \
	  $(if $(LLM_OVERRIDE),--llm-provider-override $(LLM_OVERRIDE),)

# Phase 6 plan 06-07 / D-06-10: refinement-only matrix runner.
# LIVE target — used for human baseline generation (Task 2a). Runs
# REFINEMENT_STRUCTURED_PLAN_ENABLED=true on both provider entries and
# fans out to real-provider subprocesses. Requires APP_ENV=eval +
# OPENAI_API_KEY + DEEPSEEK_API_KEY for non-scripted runs.
.PHONY: eval-matrix-refinement
eval-matrix-refinement: ## Run the Phase 6 refinement-only matrix (LIVE; APP_ENV=eval required)
	$(POETRY_RUN) python scripts/eval_matrix.py \
	  --matrix-config configs/eval_matrix_refinement.yaml \
	  --runs $(RUNS) \
	  $(if $(LLM_OVERRIDE),--llm-provider-override $(LLM_OVERRIDE),)

# Phase 6 plan 06-07 / NEW HIGH-A: structural-check target used by CI as
# a HARD gate (no continue-on-error). Validates the matrix loads,
# iter_cells produces cells, env override propagates through
# _apply_override, DETERMINISTIC_CHECKS contains 'refinement_minimal_edit',
# and build_refinement_prompt_message is functional. Does NOT invoke any
# subprocess — sidesteps the SCRIPTED_SCENARIOS-empty problem.
.PHONY: eval-matrix-refinement-structural-check
eval-matrix-refinement-structural-check: ## Phase 6 refinement matrix structural check (CI hard gate; no subprocess)
	$(POETRY_RUN) python scripts/eval_matrix.py \
	  --matrix-config configs/eval_matrix_refinement.yaml \
	  --structural-check

# Phase 10 / EVAL-05 / D-10-11: live probe writes one redacted AIMessage fixture per provider
# to tests/fixtures/provider_payloads/{provider}.json.
# MANDATORY PRE-MATRIX STEP: run this before make eval-matrix to ensure adapter tests
# execute against real-wire shapes, not just synthetic cases.
# NO CI/CRON: live provider keys are NOT in CI (D-10-14 / D-09-10). Adapter tests in
# tests/unit/test_adapters.py SKIP gracefully when fixtures are absent — CI stays green.
# Run this locally whenever a provider's SDK is upgraded or the wire shape may have changed.
.PHONY: probe-providers
probe-providers: ## MANDATORY pre-matrix: run live probes for all four providers, write redacted fixtures (no CI/cron — D-10-14)
	$(POETRY_RUN) python scripts/probe_provider_capture.py --provider openai
	$(POETRY_RUN) python scripts/probe_provider_capture.py --provider deepseek
	$(POETRY_RUN) python scripts/probe_provider_capture.py --provider anthropic
	$(POETRY_RUN) python scripts/probe_provider_capture.py --provider gemini

# Phase 10 / EVAL-03 / D-10-05: gate-check against configs/eval_gates.yaml.
# Requires SUMMARY= path to a summary.json from an eval_matrix run.
# Exit 0 = all hard gates passed; exit 1 = hard-gate violation; exit 2 = infra failure.
# Aspirational misses (e.g. gpt-5-mini below v2.2 target) are reported but non-blocking.
# See docs/eval_gates.md for semantics; gate values live only in configs/eval_gates.yaml.
.PHONY: eval-gates-check
eval-gates-check: ## Check summary.json against configs/eval_gates.yaml (EVAL-03 / D-10-05)
	$(POETRY_RUN) python scripts/check_eval_gates.py \
	  $(SUMMARY) \
	  --gates-config configs/eval_gates.yaml

# Phase 11 / D-11-15 / BASE-03: gate-check against committed baseline JSONs (live-key-free CI).
# Reads configs/eval_baselines/*.json and synthesises the summary shape internally.
# Exit 0 = all hard gates passed; exit 1 = hard-gate violation; exit 2 = infra failure.
# Aspirational misses (e.g. gpt-5-mini) are non-blocking. Advisory entries report-only WARN.
.PHONY: eval-gates-check-baselines
eval-gates-check-baselines: ## Check gates against committed baselines (D-11-15; no live keys)
	$(POETRY_RUN) python scripts/check_eval_gates.py \
	  --baselines-mode \
	  --baselines-dir configs/eval_baselines \
	  --gates-config configs/eval_gates.yaml

# Phase 11 / D-11-07 / BASE-01: write baseline JSONs from a completed summary.json.
# Requires SUMMARY= path; overrides RUNS= for --n-requested (default 1, runbook uses 5).
# Exit 0 = all cells written; exit 1 = one or more cells refused (D-10-03/D-10-09);
# exit 2 = infra failure (missing/malformed summary.json).
.PHONY: write-baselines
write-baselines: ## Write baseline JSONs from a completed summary.json (D-11-07; SUMMARY= required, RUNS= run count)
	$(POETRY_RUN) python scripts/write_baselines.py \
	  $(SUMMARY) \
	  --n-requested $(RUNS) \
	  --baselines-dir configs/eval_baselines

# Phase 11 / D-11-09 / BASE-01: snapshot current canonical baselines before Wave 2 regen.
# Creates pre-phase11 copies in configs/eval_baselines/_snapshots/ for audit trail.
.PHONY: snapshot-baselines
snapshot-baselines: ## Snapshot current canonical baselines to _snapshots/ as pre-phase11 (D-11-09)
	cp configs/eval_baselines/omakase_mission_open_ended.json \
	   configs/eval_baselines/_snapshots/omakase_mission_open_ended.pre-phase11.json
	cp configs/eval_baselines/refinement_cheaper.json \
	   configs/eval_baselines/_snapshots/refinement_cheaper.pre-phase11.json
	cp configs/eval_baselines/late_night_closure_cascade.json \
	   configs/eval_baselines/_snapshots/late_night_closure_cascade.pre-phase11.json

# ─── Testing ──────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run the full test suite
	$(POETRY_RUN) pytest tests/ -v --cov=app --cov-report=term-missing

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(POETRY_RUN) pytest tests/unit/ -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(POETRY_RUN) pytest tests/integration/ -v

.PHONY: test-reasoning-conformance
test-reasoning-conformance: ## Run reasoning-state conformance harness (quarantined; not in make test)
	$(POETRY_RUN) pytest -m reasoning_conformance -v

# Cloud SQL connection details for IAM-DB-auth via the proxy.
# Override on the command line if your gcloud identity differs.
CLOUD_SQL_INSTANCE ?= mlops-491820:us-central1:mlops--city-concierge
CLOUD_SQL_DB ?= mlops-city-concierge
CLOUD_SQL_PORT ?= 5433
CLOUD_SQL_USER ?= $(shell gcloud config get-value account 2>/dev/null)

.PHONY: test-integration-cloud
test-integration-cloud: ## Integration tests against Cloud SQL via the proxy + IAM-DB-auth
	@command -v cloud-sql-proxy >/dev/null || { echo "cloud-sql-proxy not installed"; exit 1; }
	@command -v gcloud >/dev/null || { echo "gcloud not installed"; exit 1; }
	@[ -n "$(CLOUD_SQL_USER)" ] || { echo "no active gcloud account; run 'gcloud auth login'"; exit 1; }
	@echo "→ proxy: $(CLOUD_SQL_INSTANCE) on :$(CLOUD_SQL_PORT)"
	@echo "→ user:  $(CLOUD_SQL_USER)"
	@cloud-sql-proxy --port $(CLOUD_SQL_PORT) "$(CLOUD_SQL_INSTANCE)" \
		> /tmp/cloud-sql-proxy.log 2>&1 & \
	PROXY_PID=$$!; \
	trap "kill $$PROXY_PID 2>/dev/null || true" EXIT INT TERM; \
	for i in $$(seq 1 30); do \
		nc -z 127.0.0.1 $(CLOUD_SQL_PORT) && break; sleep 1; \
	done; \
	nc -z 127.0.0.1 $(CLOUD_SQL_PORT) || { echo "proxy failed to start"; cat /tmp/cloud-sql-proxy.log; exit 1; }; \
	PGPASSWORD=$$(gcloud sql generate-login-token) \
	DATABASE_URL="postgresql://$$(printf '%s' '$(CLOUD_SQL_USER)' | sed 's/@/%40/g'):$$PGPASSWORD@127.0.0.1:$(CLOUD_SQL_PORT)/$(CLOUD_SQL_DB)" \
	APP_ENV=integration \
	$(POETRY_RUN) pytest tests/integration/ -v

# ─── Linting / Formatting ─────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run ruff linter
	$(POETRY_RUN) ruff check .

.PHONY: format
format: ## Auto-format code with ruff
	$(POETRY_RUN) ruff format .

.PHONY: typecheck
typecheck: ## Run mypy type checker
	$(POETRY_RUN) mypy app/

# ─── Install ──────────────────────────────────────────────────────────────────
.PHONY: install
install: ## Install production dependencies
	poetry install --only main

.PHONY: install-dev
install-dev: ## Install all dependencies (app + dev tools)
	poetry install

# ─── Clean ────────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove Python cache files and test artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
