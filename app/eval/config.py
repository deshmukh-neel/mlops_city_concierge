"""Typed loader for offline RAG eval queries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_QUERIES_PATH = Path("configs/eval_queries.yaml")
DEFAULT_EVAL_MATRIX_PATH = Path("configs/eval_matrix.yaml")


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
    requested_primary_types: list[str] = Field(
        default_factory=list,
        description=(
            "Per-slot Google primary_type values (Title Case, e.g. 'Sushi Restaurant') "
            "the agent should match on each committed stop. Empty list means no slot "
            "expectations (D-03 abstain)."
        ),
    )
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

    @field_validator("types_any", "requested_primary_types")
    @classmethod
    def strip_non_empty_list(cls, value: list[str], info: ValidationInfo) -> list[str]:
        """Trim expected place-type lists and discard blank entries."""
        field_name = info.field_name or "list"
        cleaned: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"{field_name}[{index}] must be a string")
            stripped = item.strip()
            if stripped:
                cleaned.append(stripped)
        return cleaned

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


class ExpectedRefinement(BaseModel):
    """Refinement-turn expectations (Phase 6 / D-06-08).

    Only set when the eval scenario is a refinement turn. ``target_slot`` is
    the 1-indexed stop the user asks to change in the LAST turn — the index
    convention matches:
      - the user-facing prose ("make stop 2 cheaper"),
      - the ``is_refinement_request`` helper return convention (plan 06-02), and
      - the ``refinement_minimal_edit`` scorer convention (plan 06-03).

    Drives the ``refinement_minimal_edit`` scorer via ``state.scratch``
    population in ``evaluate_multi_turn_case`` under ``threading_mode='prod'``
    only. Default ``None`` on the parent ``EvalQuery`` keeps the field opt-in
    (no impact on the 30 existing legacy cases).

    Phase 7 / D-07-06 extension: ``prior_committed_stops`` entries now carry
    ``primary_type`` per entry (in addition to ``slot`` and ``place_id``) so
    ``refinement_minimal_edit`` can enforce same-category on the target slot.
    The model-facing JSON block in ``build_refinement_prompt_message`` remains
    ``{slot, place_id, arrival_time}`` only — the category rule moved into the
    scorer, not the prompt (HIGH-4 prompt-injection mitigation preserved). See
    ``app/agent/critique/checks.py::refinement_minimal_edit`` for the canonical
    scratch-shape contract.
    """

    model_config = ConfigDict(extra="forbid")

    target_slot: int = Field(ge=1)


class EvalQuery(BaseModel):
    """One hand-written offline eval case.

    Phase 6 additions (plan 06-04):
      - ``threading_mode``: opt-in switch between legacy eval-only message
        threading and prod-equivalent threading (D-06-05). Default ``'legacy'``
        preserves the 30 existing cases.
      - ``expected_refinement``: nested ``ExpectedRefinement`` model that
        carries the target slot for refinement-turn scenarios (D-06-08).
        Default ``None`` keeps non-refinement cases unaffected.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    query: str
    reference: str
    expected_constraints: ExpectedConstraints = Field(default_factory=ExpectedConstraints)
    expected_results: ExpectedResults | None = None
    expected_walking_budget_m: int | None = Field(default=None, gt=0)
    expects_clarification_or_relaxation: bool = False
    tags: list[str] = Field(default_factory=list)
    turns: list[str] | None = None
    # Phase 6 / D-06-05: opt-in threading switch — `legacy` keeps the existing
    # 30 cases on the old eval threading; only `refinement_cheaper` flips to
    # `prod` in plan 06-07. `Literal` enforces membership at validation time;
    # `yaml.safe_load` already produces lowercase strings so no normalization
    # validator is needed.
    threading_mode: Literal["legacy", "prod"] = "legacy"
    # Phase 6 / D-06-08: nested refinement expectations, only meaningful for
    # refinement-turn scenarios. `None` default = backward compat.
    expected_refinement: ExpectedRefinement | None = None
    # Phase 10 / D-10-09: opt-in quarantine flag. When False the scenario still
    # runs as a diagnostic but is excluded from baseline aggregation and merge
    # gates. The late_night_closure_cascade scenario sets this to False because
    # its turn-2 scorers were designed against the full-tool-history threading
    # shape (project_eval_multi_turn_threading_bug); migrating to prod threading
    # redesigns the scenario, which is out of scope for a harness-honesty phase.
    # Default True keeps all 30 legacy cases and omakase_mission_open_ended
    # baseline-eligible; extra="forbid" is satisfied because the field is
    # declared (so YAML keys that set it are accepted, not rejected).
    baseline_eligible: bool = True

    @field_validator("id", "query", "reference", mode="before")
    @classmethod
    def strip_required_text(cls, value: object, info: ValidationInfo) -> str:
        """Trim required text fields and reject empty values."""
        return strip_non_empty(value, info.field_name or "field")

    @field_validator("tags")
    @classmethod
    def tags_non_empty(cls, value: list[str]) -> list[str]:
        """Keep tags normalized for filtering and reporting."""
        return strip_non_empty_list(value, "tags")

    @field_validator("turns")
    @classmethod
    def turns_non_empty_when_present(cls, value: list[str] | None) -> list[str] | None:
        """Multi-turn scripted scenarios (EVAL-03): None means single-turn
        (backward compat for the 29 existing cases). When provided, the list
        must be non-empty and every turn must be a non-blank trimmed string."""
        if value is None:
            return None
        if len(value) == 0:
            raise ValueError("turns must be omitted (None) or contain at least one follow-up turn")
        return strip_non_empty_list(value, "turns")

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


class MatrixEntry(BaseModel):
    """One (provider, model) pair in the cross-provider eval matrix.

    The matrix is anchored to openai/gpt-4o-mini + deepseek/deepseek-chat
    for v2.0 (D-06) but this model does not hard-code those — the YAML
    config carries the concrete pairs.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    # Phase 6 / D-06-10: per-cell env override gateway for plan 06-06's matrix
    # runner. `StrictStr` on values rejects YAML coercion (e.g. unquoted
    # `true`/`false` -> bool -> coerced str) at type-validation time, which
    # closes the coercion-before-validator gap a `mode='after'` validator
    # would otherwise leave open (MEDIUM env-StrictStr fix from REVIEWS.md).
    # Keys are guarded by the `env_keys_must_be_strings` validator below.
    env: dict[StrictStr, StrictStr] | None = None

    @field_validator("provider", "model", mode="before")
    @classmethod
    def strip_required_text(cls, value: object, info: ValidationInfo) -> str:
        """Trim provider/model names and reject blanks (parity with EvalQuery)."""
        return strip_non_empty(value, info.field_name or "field")

    @field_validator("env", mode="before")
    @classmethod
    def env_keys_must_be_strings(cls, value: object) -> object:
        """Reject non-string env keys for symmetry with the StrictStr value guard.

        Without this `mode='before'` check, Pydantic would coerce a numeric
        key like ``{1: "true"}`` into a string via its standard dict-key
        coercion before validation, silently producing ``{"1": "true"}``.
        That would crash downstream when ``subprocess.run(env=...)`` rejects
        the (technically valid str) key for unexpected reasons. Surface the
        type mismatch at config load time, in line with the StrictStr value
        contract.
        """
        if value is None:
            return None
        if not isinstance(value, dict):
            # Pass through; downstream Pydantic typing handles non-dict shapes.
            return value
        for key in value:
            if not isinstance(key, str):
                raise ValueError(
                    f"env keys must be strings; got {type(key).__name__} for key {key!r}"
                )
        return value

    @field_validator("provider", "model", mode="after")
    @classmethod
    def reject_double_dash(cls, value: str, info: ValidationInfo) -> str:
        """Reject `--` in provider/model strings (plan 03-08 / WR-01).

        `scripts/eval_matrix.py` uses `--` as the cell-filename separator
        (`{provider}--{model}--{scenario_id}--run-{n}.json`). A model named
        e.g. `gpt-4--turbo` would silently produce 5 split-parts, the
        filename parser would return None, and the cell would be dropped
        from `summary.json` with no diagnostic. Failing fast at config-load
        time keeps the parser's `--`-as-separator invariant intact.
        """
        if "--" in value:
            raise ValueError(
                f"{info.field_name} value '{value}' contains '--'; '--' is reserved "
                "as the cell-filename separator in scripts/eval_matrix.py"
            )
        return value


class EvalMatrixConfig(BaseModel):
    """Top-level config for the cross-provider eval matrix runner (EVAL-04).

    Mirrors the EvalQueriesConfig shape (extra='forbid', strip_non_empty
    validators, uniqueness check) so the two configs stay structurally
    consistent for callers.
    """

    model_config = ConfigDict(extra="forbid")

    entries: list[MatrixEntry] = Field(min_length=1)
    scenarios: list[str] = Field(min_length=1)

    @field_validator("scenarios")
    @classmethod
    def scenarios_non_empty(cls, value: list[str]) -> list[str]:
        """Scenario ids reference EvalQuery.id — reject blanks at load time."""
        return strip_non_empty_list(value, "scenarios")

    @model_validator(mode="after")
    def entries_are_unique(self) -> EvalMatrixConfig:
        """Same (provider, model) twice would double-bill the same run and
        skew the cross-provider median. Mirror EvalQueriesConfig.ids_are_unique."""
        keys = [(entry.provider, entry.model) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("entries must be unique by (provider, model)")
        return self


def resolve_eval_matrix_path(path: str | Path) -> Path:
    """Resolve a matrix config path relative to the repository root when needed."""
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    return REPO_ROOT / config_path


def load_eval_matrix(path: str | Path = DEFAULT_EVAL_MATRIX_PATH) -> EvalMatrixConfig:
    """Load and validate the cross-provider eval matrix YAML.

    Mirrors load_eval_queries exactly so callers can swap loaders without
    surprise. The matrix YAML file (configs/eval_matrix.yaml) ships in the
    matrix-runner plan (03-05); this loader lands here so downstream plans
    can import it now.
    """
    config_path = resolve_eval_matrix_path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError("Eval matrix config must be a YAML mapping.")
    return EvalMatrixConfig.model_validate(raw)
