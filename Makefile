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

.PHONY: log-mlflow
log-mlflow: ## Log a RAG configuration and sample outputs to MLflow
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
