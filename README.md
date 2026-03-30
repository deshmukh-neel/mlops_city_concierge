# City Concierge

Course: USF MSDS 603 - MLOps

A RAG (Retrieval-Augmented Generation) application for city information, powered by FastAPI, PostgreSQL + pgvector, and OpenAI embeddings.

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker & Docker Compose

### Setup

```bash
# 1. Clone the repo and cd into it

# 2. Create your .env file
cp .env.example .env
# Fill in your secrets (POSTGRES_PASSWORD, OPENAI_API_KEY, etc.)

# 3. Install dependencies (see "Installing Dependencies" below)
poetry install

# 4. Start services
docker compose up --build
```

## Installing Dependencies

All dependencies are managed in a single `pyproject.toml` using Poetry dependency groups. Install only what you need for your context:

| Context | Command | What it installs |
|---|---|---|
| **Local dev** | `poetry install` | Everything (app + mlflow + dev tools) |
| **App container** | `poetry install --only main` | Production app deps only |
| **MLflow VM** | `poetry install --only main,mlflow` | App deps + MLflow |
| **CI / testing** | `poetry install --with dev` | App deps + dev/test tools |

### Docker Usage

In your `Dockerfile` for the app container, install only production dependencies:

```dockerfile
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root
```

For a dedicated MLflow container or VM, include the mlflow group:

```dockerfile
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main,mlflow --no-root
```

This keeps container images lean — each environment only pulls in what it actually needs.

## Common Commands

```bash
# Docker services
docker compose up --build       # Start all services (Postgres + app)
docker compose up --build -d    # Start in background
docker compose down             # Stop all containers
docker compose up -d db         # Start only Postgres

# Database
alembic upgrade head                           # Run migrations
alembic revision --autogenerate -m "message"   # Create a migration

# Data
python scripts/seed.py          # Generate sample JSONL data
python scripts/ingest.py        # Run ingestion pipeline

# Testing
pytest tests/ -v                # Full test suite
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests (requires DB + APP_ENV=integration)

# Linting & formatting
ruff check .                    # Lint
ruff format .                   # Format
mypy app/                       # Type check
```

All of the above are also available via `make` — run `make help` to see all targets.

## MLflow Tracking Server

The team shares an MLflow tracking server on GCP.

**Dashboard:** http://35.223.147.177:5000

### Logging Experiments

```python
import mlflow

mlflow.set_tracking_uri("http://35.223.147.177:5000")
mlflow.set_experiment("city-concierge-analysis")

with mlflow.start_run(run_name="Your_Name_Run"):
    mlflow.log_param("model_type", "regression")
    mlflow.log_metric("rmse", 0.123)
```

Or set the URI once per terminal session:

```bash
export MLFLOW_TRACKING_URI="http://35.223.147.177:5000"
```

### Best Practices

- Use `mlflow.set_experiment("city-concierge-analysis")` to keep all team runs in the same view
- Always provide a `run_name` (e.g., `"James_Baseline"`) so we can identify who generated which results
- If the script hangs, check your network/VPN allows outgoing traffic on port 5000
