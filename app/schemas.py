from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        description="User's recommendation query (for example, 'Best tacos in the Mission').",
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
