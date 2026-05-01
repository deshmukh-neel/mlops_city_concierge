from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from psycopg2.extensions import connection

from .db import get_db
from .schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationSource,
)

router = APIRouter()
db_connection_dependency = Depends(get_db)


def serialize_sources(source_documents: list[Any]) -> list[RecommendationSource]:
    sources: list[RecommendationSource] = []
    for document in source_documents:
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


@router.get("/root")
def root() -> dict[str, str]:
    return {"message": "Welcome!"}


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    model_config = getattr(request.app.state, "active_model_config", None)
    if model_config is None:
        startup_error = getattr(request.app.state, "startup_error", None) or "unknown"
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "error": startup_error},
        )

    return {
        "status": "ok",
        "llm_provider": model_config.llm_provider,
        "chat_model": model_config.chat_model,
    }


@router.get("/health/db")
def health_db(conn: connection = db_connection_dependency) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        _ = cur.fetchone()
    return {"status": "ok"}


@router.post("/predict", response_model=RecommendationResponse)
def predict(request_body: RecommendationRequest, request: Request) -> RecommendationResponse:
    rag_chain = getattr(request.app.state, "rag_chain", None)
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain is not loaded.")

    result = rag_chain.invoke({"query": request_body.query})
    response_text = result.get("result") or result.get("response") or ""
    source_documents = result.get("source_documents") or []

    return RecommendationResponse(
        response=response_text,
        sources=serialize_sources(source_documents),
    )
