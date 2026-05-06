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
