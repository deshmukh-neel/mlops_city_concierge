# Copilot Instructions

This file provides guidance to GitHub Copilot when working with code in this repository.

> **Sync notice:** This file is kept in sync with `CLAUDE.md` (Claude Code) and `AGENTS.md` (Cursor). When updating project guidance in any of these files, mirror the changes to the other two.

## Project Overview

City Concierge is a RAG (Retrieval-Augmented Generation) application for city information. It uses FastAPI, PostgreSQL with pgvector for vector similarity search, and OpenAI embeddings. The project is part of USF MSDS 603 (MLOps) and uses a shared MLflow tracking server on GCP (`http://35.223.147.177:5000`).

## Commands

```bash
# Setup
make env                  # Create .env from .env.example
make install-dev          # Install all dependencies (includes prod deps)

# Docker services
make dev                  # Start all services (Postgres + app) with docker compose
make dev-detached         # Start in background
make down                 # Stop all containers
make db-up                # Start only Postgres

# Database
make migrate              # Run Alembic migrations (alembic upgrade head)
make migration MSG="..."  # Create new Alembic migration

# Testing
make test                 # Full suite with coverage (pytest tests/ -v --cov=app)
make test-unit            # Unit tests only (pytest tests/unit/ -v)
make test-integration     # Integration tests only (requires running DB + APP_ENV=integration)
pytest tests/unit/test_ingest.py -v  # Run a single test file
pytest tests/unit/test_ingest.py::TestIngestScript::test_loads_jsonl_records -v  # Single test

# Linting & formatting
make lint                 # ruff check .
make format               # ruff format .
make typecheck            # mypy app/

# Data
make ingest               # Run ingestion pipeline (python scripts/ingest.py)
make log-mlflow          # Log a RAG config + sample outputs to MLflow
python scripts/seed.py    # Generate sample JSONL data
```

## Architecture

- **FastAPI app** served via uvicorn at port 8000 (`app.main:app`) with a startup-loaded RAG chain
- **PostgreSQL 18 + pgvector** stores Google Places metadata in `places_raw` and semantic vectors in `place_embeddings` (Cloud SQL prod instance is `POSTGRES_18`; local docker may still be 16)
- **Retriever stack** uses OpenAI embeddings for query vectors and pgvector HNSW cosine similarity search for source retrieval
- **LangChain RAG chain** supports OpenAI or Gemini chat models, selected from MLflow Model Registry params
- **Scripts**: `scripts/ingest_places_sf.py` loads raw Google Places data, `scripts/embed_places_pgvector.py` refreshes embeddings, and `scripts/log_model_to_mlflow.py` logs experiment runs
- **Alembic** for database migrations (not yet initialized beyond Makefile targets)
- **MLflow** for experiment tracking and model-registry-backed runtime selection (shared GCP server)

## Key Conventions

- Python 3.10+; ruff for linting/formatting (line-length 100, rules: E, F, I, N, UP, B, SIM)
- Dependency management: single `pyproject.toml` with Poetry main dependencies plus a `dev` group
- Tests use pytest with `asyncio_mode = "auto"`; conftest patches env vars so tests never hit real services
- Integration tests are skipped unless `APP_ENV=integration` is set
- Environment variables configured via `.env` (see `.env.example` for all required vars)
- Docker Compose service names: `db` (Postgres), `app` (FastAPI)
