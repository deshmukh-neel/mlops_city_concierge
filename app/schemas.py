from __future__ import annotations

from pydantic import BaseModel, Field


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
