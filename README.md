# City Concierge

Course: USF MSDS 603 - MLOps

A RAG (Retrieval-Augmented Generation) application for San Francisco restaurant and place recommendations, powered by FastAPI, PostgreSQL + pgvector, LangChain, and MLflow.

## Architecture

```
User Query
    |
    v
FastAPI (/predict)
    |
    v
MLflow Model Registry  -->  Selects LLM config (OpenAI or Gemini)
    |
    v
LangChain RetrievalQA Chain
    |
    +---> PgVectorRetriever (cosine similarity search on Cloud SQL)
    +---> LLM (OpenAI gpt-4o-mini or Gemini 2.5 Flash)
    |
    v
Response + Source Places
```

- **Vector Store**: PostgreSQL 16 + pgvector on Cloud SQL with HNSW cosine index (4,356 SF place embeddings)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dims)
- **LLM**: Configurable via MLflow Model Registry — supports OpenAI and Gemini
- **MLflow**: Shared tracking server at http://35.223.147.177:5000
- **Container Registry**: GCP Artifact Registry (`us-central1-docker.pkg.dev/mlops-491820/ml-repo/city-concierge`)

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker & Docker Compose (optional, for containerized runs)

### Setup

```bash
# 1. Clone the repo and cd into it

# 2. Create your .env file
cp .env.example .env
# Fill in: DATABASE_URL, OPENAI_API_KEY, GEMINI_API_KEY (optional)

# 3. Install dependencies
poetry install

# 4. Run the app
poetry run uvicorn app.main:app --reload
```

The app reads the `production` alias from the MLflow Model Registry at startup and builds the matching RAG chain.

### Installing Dependencies

| Context | Command | What it installs |
|---|---|---|
| **Local dev / CI** | `poetry install` | App + dev/test tools |
| **App container** | `poetry install --only main` | Production deps only |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/root` | Welcome message |
| GET | `/health` | Active model config (provider, model) |
| GET | `/health/db` | Database connectivity check |
| POST | `/predict` | RAG query endpoint |
| GET | `/docs` | Interactive API docs (Swagger) |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"query": "Best tacos in the Mission", "limit": 5}'
```

## Docker

### Build and Run with Docker Compose

```bash
make dev          # Build and start (app + local Postgres)
make dev-detached # Start in background
make down         # Stop all containers
```

Note: The app container reads `DATABASE_URL` from `.env`. Set it to Cloud SQL for production data, or leave it pointing to the local `db` container for local development.

### Build and Run Standalone

```bash
docker build -t city-concierge .
docker run --rm -p 8000:8000 --env-file .env city-concierge
```

### Push to GCP Artifact Registry

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
docker tag city-concierge us-central1-docker.pkg.dev/mlops-491820/ml-repo/city-concierge:latest
docker push us-central1-docker.pkg.dev/mlops-491820/ml-repo/city-concierge:latest
```

## MLflow

**Dashboard:** http://35.223.147.177:5000

### Logging Experiments

```bash
# Log an OpenAI run with default sample queries
poetry run python scripts/log_model_to_mlflow.py

# Log and register a Gemini-backed config
poetry run python scripts/log_model_to_mlflow.py \
  --llm-provider gemini \
  --chat-model gemini-2.5-flash \
  --k 5 \
  --temperature 0.2 \
  --register-model
```

### Setting the Production Model

1. Run experiments with different configs using the script above
2. Compare results in the MLflow UI
3. Register the best run with `--register-model`
4. In the MLflow UI (or via script), assign the `production` alias to the desired model version:

```bash
poetry run python -c "
import mlflow
mlflow.set_tracking_uri('http://35.223.147.177:5000')
client = mlflow.MlflowClient()
client.set_registered_model_alias('city-concierge-rag', 'production', '<VERSION>')
"
```

5. Restart the app — it picks up the new config automatically

## Common Commands

Run `make help` to see all targets. Key ones:

```bash
make install-dev    # Install all dependencies
make test           # Full test suite with coverage
make test-unit      # Unit tests only
make lint           # Ruff linter
make format         # Auto-format code
make ingest-places  # Pull SF Google Places data
make embed-places   # Generate pgvector embeddings
make log-mlflow     # Log a RAG config to MLflow
```
