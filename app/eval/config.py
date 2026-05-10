"""Typed loader for offline RAG eval queries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_QUERIES_PATH = Path("configs/eval_queries.yaml")


def strip_non_empty(value: object, field_name: str) -> str:
    """Normalize one required string field and reject blanks."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be a non-empty string")
    return stripped


def strip_non_empty_list(values: list[str], field_name: str) -> list[str]:
    """Normalize a list of strings while preserving item-specific error labels."""
    cleaned: list[str] = []
    for index, value in enumerate(values):
        cleaned.append(strip_non_empty(value, f"{field_name}[{index}]"))
    return cleaned


class ExpectedConstraints(BaseModel):
    """Constraints a produced itinerary is expected to satisfy for one query."""

    model_config = ConfigDict(extra="forbid")

    neighborhood: str | None = None
    price_level_max: int | None = Field(default=None, ge=0, le=4)
    min_rating: float | None = Field(default=None, ge=0.0, le=5.0)
    min_user_rating_count: int | None = Field(default=None, ge=0)
    open_at_iso: datetime | None = None
    types_any: list[str] = Field(default_factory=list)
    serves_brunch: bool | None = None
    serves_vegetarian: bool | None = None
    serves_coffee: bool | None = None

    @field_validator("neighborhood", mode="before")
    @classmethod
    def strip_neighborhood(cls, value: object) -> object:
        """Trim optional neighborhood text when present."""
        if value is None:
            return None
        return strip_non_empty(value, "neighborhood")

    @field_validator("types_any")
    @classmethod
    def types_any_non_empty(cls, value: list[str]) -> list[str]:
        """Ensure expected Google place types are not blank strings."""
        return strip_non_empty_list(value, "types_any")

    @field_validator("open_at_iso")
    @classmethod
    def open_at_iso_is_timezone_aware(cls, value: datetime | None) -> datetime | None:
        """Require explicit timezone offsets so eval runs are reproducible."""
        if value is None:
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("open_at_iso must include a timezone offset")
        return value


class ExpectedResults(BaseModel):
    """Flexible result-count expectations for one eval case."""

    model_config = ConfigDict(extra="forbid")

    min_stops: int = Field(ge=0)
    max_stops: int = Field(ge=0)

    @model_validator(mode="after")
    def max_stops_at_least_min_stops(self) -> ExpectedResults:
        """Require a coherent inclusive stop-count range."""
        if self.max_stops < self.min_stops:
            raise ValueError("max_stops must be greater than or equal to min_stops")
        return self


class EvalQuery(BaseModel):
    """One hand-written offline eval case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    query: str
    reference: str
    expected_constraints: ExpectedConstraints = Field(default_factory=ExpectedConstraints)
    expected_results: ExpectedResults | None = None
    expected_walking_budget_m: int | None = Field(default=None, gt=0)
    expects_clarification_or_relaxation: bool = False
    tags: list[str] = Field(default_factory=list)

    @field_validator("id", "query", "reference", mode="before")
    @classmethod
    def strip_required_text(cls, value: object, info) -> str:
        """Trim required text fields and reject empty values."""
        return strip_non_empty(value, info.field_name)

    @field_validator("tags")
    @classmethod
    def tags_non_empty(cls, value: list[str]) -> list[str]:
        """Keep tags normalized for filtering and reporting."""
        return strip_non_empty_list(value, "tags")

    @model_validator(mode="after")
    def normal_cases_have_expected_stops(self) -> EvalQuery:
        """Require a result-count range unless the correct behavior is to relax."""
        if not self.expects_clarification_or_relaxation and self.expected_results is None:
            raise ValueError(
                "expected_results is required unless relaxation/clarification is expected"
            )
        return self


class GeneratedEvalSpec(BaseModel):
    """Future RAGAS-generated test-set settings."""

    model_config = ConfigDict(extra="forbid")

    source_table: Literal["place_embeddings", "place_embeddings_v2"]
    count: int = Field(ge=0)
    seed: int | None = None

    @field_validator("source_table", mode="before")
    @classmethod
    def strip_source_table(cls, value: object) -> str:
        """Normalize the generated-test source table before literal validation."""
        return strip_non_empty(value, "source_table")


class EvalQueriesConfig(BaseModel):
    """Top-level config for offline eval queries."""

    model_config = ConfigDict(extra="forbid")

    hand_written: list[EvalQuery] = Field(min_length=1)
    generated: GeneratedEvalSpec | None = None

    @model_validator(mode="after")
    def ids_are_unique(self) -> EvalQueriesConfig:
        """Prevent ambiguous per-query reporting by requiring unique case ids."""
        ids = [case.id for case in self.hand_written]
        if len(ids) != len(set(ids)):
            raise ValueError("hand_written query ids must be unique")
        return self


def resolve_eval_queries_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root when needed."""
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    return REPO_ROOT / config_path


def load_eval_queries(path: str | Path = DEFAULT_EVAL_QUERIES_PATH) -> EvalQueriesConfig:
    """Load and validate the offline eval query YAML."""
    config_path = resolve_eval_queries_path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError("Eval query config must be a YAML mapping.")
    return EvalQueriesConfig.model_validate(raw)
