from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extensions import connection
from pydantic import BaseModel, Field

from .chain import build_rag_chain
from .config import get_settings, resolve_llm_api_key
from .db import get_db
from .db_pool import close_db_pool, init_db_pool

logger = logging.getLogger(__name__)

RAG_UNAVAILABLE_DETAIL = (
    "RAG chain unavailable: MLflow registry could not be reached at startup. "
    "Ensure the MLflow IAP tunnel is open and restart the app."
)

db_connection_dependency = Depends(get_db)


class RecommendationRequest(BaseModel):
    query: str = Field(
        ...,
        description="User's recommendation query (for example, 'Best tacos in the Mission').",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of source places to include in the response.",
    )


class RecommendationSource(BaseModel):
    name: str | None = None
    rating: float | None = None
    address: str | None = None
    similarity: float | None = None


class RecommendationResponse(BaseModel):
    response: str
    sources: list[RecommendationSource]


class ActiveModelConfig(BaseModel):
    llm_provider: str
    chat_model: str
    k: int
    temperature: float = 0.0
    run_id: str | None = None
    model_version: str | None = None


def parse_active_model_config(
    params: dict[str, str], run_id: str, model_version: str
) -> ActiveModelConfig:
    settings = get_settings()
    llm_provider = (params.get("llm_provider") or "openai").lower()
    if llm_provider == "openai":
        default_chat_model = settings.openai_chat_model
    elif llm_provider == "gemini":
        default_chat_model = settings.gemini_chat_model
    else:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    return ActiveModelConfig(
        llm_provider=llm_provider,
        chat_model=params.get("chat_model") or default_chat_model,
        k=int(params.get("k", settings.retriever_k)),
        temperature=float(params.get("temperature", "0.0")),
        run_id=run_id,
        model_version=model_version,
    )


def load_registered_rag_chain() -> tuple[Any, ActiveModelConfig]:
    settings = get_settings()
    tracking_uri = settings.mlflow_tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    try:
        model_version = client.get_model_version_by_alias(
            settings.mlflow_model_name,
            "production",
        )
    except Exception as exc:  # pragma: no cover - exercised via startup tests
        raise RuntimeError(
            "Unable to load the MLflow production alias for "
            f"registered model '{settings.mlflow_model_name}'."
        ) from exc

    run = client.get_run(model_version.run_id)
    config = parse_active_model_config(
        run.data.params,
        run_id=model_version.run_id,
        model_version=str(model_version.version),
    )
    database_url = settings.resolved_database_url
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")
    chain = build_rag_chain(
        connection_string=database_url,
        api_key=resolve_llm_api_key(config.llm_provider),
        llm_provider=config.llm_provider,
        chat_model=config.chat_model,
        k=config.k,
        temperature=config.temperature,
    )
    return chain, config


def serialize_sources(source_documents: list[Any], limit: int) -> list[RecommendationSource]:
    sources: list[RecommendationSource] = []
    for document in source_documents[:limit]:
        metadata = getattr(document, "metadata", {}) or {}
        sources.append(
            RecommendationSource(
                name=metadata.get("name"),
                rating=metadata.get("rating"),
                address=metadata.get("address"),
                similarity=metadata.get("similarity"),
            )
        )
    return sources


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database_url = settings.resolved_database_url
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL or POSTGRES_* database settings.")

    init_db_pool(
        database_url,
        settings.db_pool_min_connections,
        settings.db_pool_max_connections,
    )
    try:
        try:
            rag_chain, model_config = load_registered_rag_chain()
        except Exception:
            logger.warning(
                "Failed to load RAG chain from MLflow registry — app will boot in "
                "degraded mode and RAG endpoints will return 503.",
                exc_info=True,
            )
            rag_chain = None
            model_config = None
        app.state.rag_chain = rag_chain
        app.state.active_model_config = model_config
        yield
    finally:
        close_db_pool()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_origin_regex=r"https://.*\.vercel\.app$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    return app


app = create_app()


@app.get("/root")
def root() -> dict[str, str]:
    return {"message": "Welcome!"}


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    model_config = getattr(request.app.state, "active_model_config", None)
    if model_config is None:
        return {"status": "degraded", "rag_chain": "unavailable"}

    return {
        "status": "ok",
        "llm_provider": model_config.llm_provider,
        "chat_model": model_config.chat_model,
    }


@app.get("/health/db")
def health_db(conn: connection = db_connection_dependency) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        _ = cur.fetchone()
    return {"status": "ok"}


@app.post("/predict", response_model=RecommendationResponse)
def predict(request_body: RecommendationRequest, request: Request) -> RecommendationResponse:
    rag_chain = getattr(request.app.state, "rag_chain", None)
    if rag_chain is None:
        raise HTTPException(status_code=503, detail=RAG_UNAVAILABLE_DETAIL)

    result = rag_chain.invoke({"query": request_body.query})
    response_text = result.get("result") or result.get("response") or ""
    source_documents = result.get("source_documents") or []

    return RecommendationResponse(
        response=response_text,
        sources=serialize_sources(source_documents, request_body.limit),
    )
