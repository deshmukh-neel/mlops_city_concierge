from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.eval.config import (
    DEFAULT_EVAL_MATRIX_PATH,
    DEFAULT_EVAL_QUERIES_PATH,
    REPO_ROOT,
    EvalMatrixConfig,
    EvalQueriesConfig,
    EvalQuery,
    ExpectedRefinement,
    MatrixEntry,
    load_eval_matrix,
    load_eval_queries,
)


def valid_payload() -> dict:
    """Return a minimal valid eval-query payload for schema tests."""
    return {
        "hand_written": [
            {
                "id": "north_beach_italian_dinner",
                "query": "italian dinner in North Beach on May 16 2026 at 7pm",
                "reference": "Recommend an Italian place in North Beach open at 7pm.",
                "expected_constraints": {
                    "neighborhood": "North Beach",
                    "price_level_max": 3,
                    "open_at_iso": "2026-05-16T19:00:00-07:00",
                    "types_any": ["italian_restaurant", "restaurant"],
                    "min_user_rating_count": 50,
                },
                "expected_results": {
                    "min_stops": 1,
                    "max_stops": 3,
                },
                "tags": ["restaurant"],
            }
        ],
        "generated": {
            "source_table": "place_embeddings_v2",
            "count": 50,
            "seed": 42,
        },
    }


def write_config(tmp_path: Path, payload: dict) -> Path:
    """Write a temporary eval-query config and return its path."""
    config_path = tmp_path / "eval_queries.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_load_eval_queries_loads_valid_yaml(tmp_path: Path) -> None:
    """Load a valid YAML file into typed eval-query models."""
    config_path = write_config(tmp_path, valid_payload())

    config = load_eval_queries(config_path)

    assert isinstance(config, EvalQueriesConfig)
    case = config.hand_written[0]
    assert case.id == "north_beach_italian_dinner"
    assert case.expected_results is not None
    assert case.expected_results.min_stops == 1
    assert case.expected_results.max_stops == 3
    assert case.expected_constraints.open_at_iso is not None
    assert case.expected_constraints.open_at_iso.tzinfo is not None
    assert config.generated is not None
    assert config.generated.source_table == "place_embeddings_v2"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("query", "   ", "query"),
        ("reference", "   ", "reference"),
        ("id", "North Beach", "id"),
    ],
)
def test_load_eval_queries_rejects_invalid_required_text(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    """Reject blank or malformed required text fields."""
    payload = valid_payload()
    payload["hand_written"][0][field] = value
    config_path = write_config(tmp_path, payload)

    with pytest.raises(ValidationError, match=match):
        load_eval_queries(config_path)


def test_load_eval_queries_rejects_invalid_expected_results_range(tmp_path: Path) -> None:
    """Reject incoherent expected result ranges."""
    payload = valid_payload()
    payload["hand_written"][0]["expected_results"] = {
        "min_stops": 3,
        "max_stops": 1,
    }
    config_path = write_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="max_stops"):
        load_eval_queries(config_path)


def test_load_eval_queries_requires_expected_results_for_normal_cases(tmp_path: Path) -> None:
    """Require normal eval cases to declare their expected result range."""
    payload = valid_payload()
    payload["hand_written"][0].pop("expected_results")
    config_path = write_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="expected_results"):
        load_eval_queries(config_path)


def test_load_eval_queries_allows_missing_expected_stops_for_relaxation_case(
    tmp_path: Path,
) -> None:
    """Allow known-bad cases to omit expected stops when relaxation is desired."""
    payload = valid_payload()
    case = payload["hand_written"][0]
    case.pop("expected_results")
    case["expects_clarification_or_relaxation"] = True
    config_path = write_config(tmp_path, payload)

    config = load_eval_queries(config_path)

    assert config.hand_written[0].expected_results is None
    assert config.hand_written[0].expects_clarification_or_relaxation is True


def test_load_eval_queries_rejects_naive_open_at_iso(tmp_path: Path) -> None:
    """Reject timestamps without timezone offsets."""
    payload = valid_payload()
    payload["hand_written"][0]["expected_constraints"]["open_at_iso"] = "2026-05-16T19:00:00"
    config_path = write_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="timezone"):
        load_eval_queries(config_path)


def test_load_eval_queries_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Reject duplicate case ids so reports can key by id safely."""
    payload = valid_payload()
    payload["hand_written"].append(dict(payload["hand_written"][0]))
    config_path = write_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="unique"):
        load_eval_queries(config_path)


def test_load_eval_queries_rejects_unknown_generated_source_table(tmp_path: Path) -> None:
    """Reject generated-test source tables outside the app allowlist."""
    payload = valid_payload()
    payload["generated"]["source_table"] = "unknown_embeddings"
    config_path = write_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="source_table"):
        load_eval_queries(config_path)


def test_repo_eval_queries_yaml_is_valid() -> None:
    """Keep the checked-in eval query set aligned with the typed schema.

    Count was 30 pre-Plan 03-05; now 33 (Plan 03-05 / EVAL-06 appended the
    three baseline scenarios for Phase 3-7).
    """
    config = load_eval_queries(REPO_ROOT / DEFAULT_EVAL_QUERIES_PATH)

    assert len(config.hand_written) == 33
    assert [case.id for case in config.hand_written[:5]] == [
        "north_beach_italian_dinner",
        "date_night_dinner_drinks_walkable",
        "mission_vegan_brunch",
        "soma_quiet_late_cafe",
        "impossible_four_am_five_star",
    ]
    assert all(case.query for case in config.hand_written)
    assert all(case.reference for case in config.hand_written)


# ─── EvalQuery.turns multi-turn scripted scenarios (EVAL-03) ───


def _minimal_query_payload(**overrides: object) -> dict:
    """Return the smallest valid EvalQuery payload; tests override fields."""
    payload: dict = {
        "id": "x",
        "query": "q",
        "reference": "r",
        "expected_results": {"min_stops": 1, "max_stops": 1},
    }
    payload.update(overrides)
    return payload


def test_eval_query_turns_defaults_to_none() -> None:
    """Backward compat: existing 29 cases omit `turns` and load unchanged."""
    case = EvalQuery.model_validate(_minimal_query_payload())
    assert case.turns is None


def test_eval_query_turns_accepts_non_empty_list_of_strings() -> None:
    """A multi-turn scenario carries one follow-up message per turn after the
    first. The runner thread `conversation_state` between turns (EVAL-06)."""
    case = EvalQuery.model_validate(_minimal_query_payload(turns=["follow-up 1", "follow-up 2"]))
    assert case.turns == ["follow-up 1", "follow-up 2"]


def test_eval_query_rejects_empty_turns_list() -> None:
    """An explicit empty list is meaningless — either omit `turns` (None) or
    provide at least one follow-up. No in-between state."""
    with pytest.raises(ValidationError, match="turns"):
        EvalQuery.model_validate(_minimal_query_payload(turns=[]))


def test_eval_query_rejects_blank_turn_string() -> None:
    """Blank-string turns would send empty messages to the agent — reject
    them at config load time. Mirrors strip_non_empty_list usage."""
    with pytest.raises(ValidationError, match="turns"):
        EvalQuery.model_validate(_minimal_query_payload(turns=["   "]))


def test_eval_query_turns_strips_whitespace() -> None:
    """Non-blank turns are trimmed (consistent with strip_non_empty_list)."""
    case = EvalQuery.model_validate(_minimal_query_payload(turns=["  hello  ", "world"]))
    assert case.turns == ["hello", "world"]


# ─── MatrixEntry (provider, model) pair (EVAL-04) ───


def test_matrix_entry_round_trips() -> None:
    """The matrix is anchored to openai/gpt-4o-mini + deepseek/deepseek-chat
    (D-06); the model itself doesn't hard-code those — the YAML does."""
    entry = MatrixEntry.model_validate({"provider": "openai", "model": "gpt-4o-mini"})
    assert entry.provider == "openai"
    assert entry.model == "gpt-4o-mini"


@pytest.mark.parametrize("field", ["provider", "model"])
def test_matrix_entry_rejects_blank_required_field(field: str) -> None:
    """provider and model are both required non-blank strings."""
    payload = {"provider": "openai", "model": "gpt-4o-mini", field: "   "}
    with pytest.raises(ValidationError, match=field):
        MatrixEntry.model_validate(payload)


def test_matrix_entry_rejects_extra_keys() -> None:
    """Parity with EvalQuery: typos on unknown fields fail loudly."""
    with pytest.raises(ValidationError):
        MatrixEntry.model_validate({"provider": "openai", "model": "gpt-4o-mini", "extra": "nope"})


# ─── MatrixEntry '--' rejection (plan 03-08 / WR-01) ───


def test_matrix_entry_rejects_double_dash_in_model() -> None:
    """WR-01: scripts/eval_matrix.py uses '--' as the cell-filename separator
    (`{provider}--{model}--{scenario_id}--run-{n}.json`). A model named
    `gpt-4--turbo` would silently produce 5 split-parts, the parser would
    return None, and the cell would be dropped from summary.json with no
    diagnostic. Reject '--' in `model` at validation time."""
    with pytest.raises(ValidationError) as exc_info:
        MatrixEntry.model_validate({"provider": "openai", "model": "gpt-4--turbo"})
    errors = exc_info.value.errors()
    # At least one error must name the `model` field AND mention the '--' contract.
    assert any("model" in err["loc"] for err in errors), (
        f"expected an error on `model`; got loc list: {[err['loc'] for err in errors]}"
    )
    assert "'--' is reserved" in str(exc_info.value), (
        f"expected '--' reservation message in error; got: {exc_info.value}"
    )


def test_matrix_entry_rejects_double_dash_in_provider() -> None:
    """WR-01 (mirror): '--' in provider is equally fatal to the filename
    parser. Reject `open--ai` for the same reason as gpt-4--turbo."""
    with pytest.raises(ValidationError) as exc_info:
        MatrixEntry.model_validate({"provider": "open--ai", "model": "gpt-4o-mini"})
    errors = exc_info.value.errors()
    assert any("provider" in err["loc"] for err in errors), (
        f"expected an error on `provider`; got loc list: {[err['loc'] for err in errors]}"
    )


def test_matrix_entry_accepts_single_dash_anywhere() -> None:
    """A single dash is fine — only the literal `--` separator is reserved.
    `gpt-4o-mini`, `open-ai`, `gpt-3.5-turbo` etc. must all still validate."""
    # Sanity-check the canonical D-06 anchor first (would have caught a
    # regression in the existing entries had we shipped one).
    MatrixEntry.model_validate({"provider": "openai", "model": "gpt-4o-mini"})
    # Single-dash in either field passes:
    MatrixEntry.model_validate({"provider": "open-ai", "model": "gpt-4o-mini"})
    MatrixEntry.model_validate({"provider": "openai", "model": "gpt-3.5-turbo"})


# ─── EvalMatrixConfig top-level loader (EVAL-04) ───


def _valid_matrix_payload(**overrides: object) -> dict:
    """Minimal-but-valid matrix payload for tests."""
    payload: dict = {
        "entries": [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "deepseek", "model": "deepseek-chat"},
        ],
        "scenarios": ["omakase_mission_open_ended"],
    }
    payload.update(overrides)
    return payload


def test_eval_matrix_config_round_trips_minimal_valid_payload() -> None:
    """The two locked anchors (D-06) plus one scenario id is the smallest
    matrix the runner will execute."""
    matrix = EvalMatrixConfig.model_validate(_valid_matrix_payload())
    assert len(matrix.entries) == 2
    assert matrix.entries[0].provider == "openai"
    assert matrix.entries[1].provider == "deepseek"
    assert matrix.scenarios == ["omakase_mission_open_ended"]


def test_eval_matrix_config_rejects_empty_entries() -> None:
    """`entries` is the cross-product dimension — empty would mean 'run nothing'."""
    with pytest.raises(ValidationError, match="entries"):
        EvalMatrixConfig.model_validate(_valid_matrix_payload(entries=[]))


def test_eval_matrix_config_rejects_empty_scenarios() -> None:
    """`scenarios` is the other cross-product dimension — both must be non-empty."""
    with pytest.raises(ValidationError, match="scenarios"):
        EvalMatrixConfig.model_validate(_valid_matrix_payload(scenarios=[]))


def test_eval_matrix_config_rejects_blank_scenario_id() -> None:
    """Scenario ids reference EvalQuery.id — blanks would never resolve."""
    with pytest.raises(ValidationError, match="scenarios"):
        EvalMatrixConfig.model_validate(_valid_matrix_payload(scenarios=["   "]))


def test_eval_matrix_config_rejects_duplicate_entries() -> None:
    """Same (provider, model) twice would double-bill the same run and skew
    the cross-provider median. Mirror EvalQueriesConfig.ids_are_unique."""
    payload = _valid_matrix_payload(
        entries=[
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "openai", "model": "gpt-4o-mini"},
        ]
    )
    with pytest.raises(ValidationError, match="unique"):
        EvalMatrixConfig.model_validate(payload)


def test_eval_matrix_config_rejects_extra_keys() -> None:
    """extra='forbid' parity with EvalQueriesConfig."""
    payload = _valid_matrix_payload(extra_key="nope")
    with pytest.raises(ValidationError):
        EvalMatrixConfig.model_validate(payload)


# ─── load_eval_matrix YAML loader (EVAL-04) ───


def write_matrix_config(tmp_path: Path, payload: dict) -> Path:
    """Write a temporary eval-matrix config and return its path."""
    config_path = tmp_path / "eval_matrix.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_load_eval_matrix_loads_valid_yaml(tmp_path: Path) -> None:
    """Mirrors load_eval_queries: YAML safe_load -> model_validate."""
    config_path = write_matrix_config(tmp_path, _valid_matrix_payload())

    matrix = load_eval_matrix(config_path)

    assert isinstance(matrix, EvalMatrixConfig)
    assert {entry.provider for entry in matrix.entries} == {"openai", "deepseek"}
    assert matrix.scenarios == ["omakase_mission_open_ended"]


def test_load_eval_matrix_rejects_non_mapping_root(tmp_path: Path) -> None:
    """Mirror load_eval_queries: top-level YAML must be a mapping."""
    config_path = tmp_path / "eval_matrix.yaml"
    config_path.write_text("- just a list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_eval_matrix(config_path)


def test_default_eval_matrix_path_is_in_configs() -> None:
    """The default points at configs/eval_matrix.yaml (the matrix-runner
    plan 03-05 ships the actual file)."""
    assert Path("configs/eval_matrix.yaml") == DEFAULT_EVAL_MATRIX_PATH


# ─── Plan 03-05 Task 2: three baseline scenarios for plan 03-07 ──────────────


def test_repo_yaml_includes_omakase_mission_open_ended_case() -> None:
    """EVAL-06: the open-ended omakase case is required for plan 03-07's
    category_compliance baseline (the original failure scenario per the
    5 post-merge runs). Single-turn case (turns=None)."""
    config = load_eval_queries(REPO_ROOT / DEFAULT_EVAL_QUERIES_PATH)
    case = next((c for c in config.hand_written if c.id == "omakase_mission_open_ended"), None)
    assert case is not None, "omakase_mission_open_ended case missing"
    assert case.turns is None, "omakase open-ended is a single-turn case"
    assert case.query  # non-blank


def test_repo_yaml_includes_refinement_cheaper_case() -> None:
    """EVAL-06: refinement-turn cheaper is a multi-turn scenario that
    exercises rationale_stop_alignment (turn-2 rationale bleed) AND the
    EvalQuery.turns wiring landed in plan 03-02."""
    config = load_eval_queries(REPO_ROOT / DEFAULT_EVAL_QUERIES_PATH)
    case = next((c for c in config.hand_written if c.id == "refinement_cheaper"), None)
    assert case is not None, "refinement_cheaper case missing"
    assert case.turns is not None, "refinement_cheaper is multi-turn"
    assert len(case.turns) >= 1, "at least one follow-up turn required"


def test_repo_yaml_includes_late_night_closure_cascade_case() -> None:
    """EVAL-06 (gated on D-07 closure pre-check): late-night closure-cascade
    is a multi-turn scenario for Phase 6 closure-aware swap baseline."""
    config = load_eval_queries(REPO_ROOT / DEFAULT_EVAL_QUERIES_PATH)
    case = next((c for c in config.hand_written if c.id == "late_night_closure_cascade"), None)
    assert case is not None, "late_night_closure_cascade case missing"
    assert case.turns is not None, "late_night_closure_cascade is multi-turn"
    assert len(case.turns) >= 1
    # The closure scenario must have an open_at_iso set so the closure path
    # fires (per the plan's behavior bullets).
    assert case.expected_constraints.open_at_iso is not None, (
        "late_night_closure_cascade needs open_at_iso to trigger closure swap"
    )


def test_repo_yaml_new_baseline_cases_are_appended() -> None:
    """All three new IDs are present and unique; existing 30 cases unchanged.
    Asserts the YAML file is append-only (D-05 / plan 03-07 compatibility)."""
    config = load_eval_queries(REPO_ROOT / DEFAULT_EVAL_QUERIES_PATH)
    ids = {c.id for c in config.hand_written}
    assert {
        "omakase_mission_open_ended",
        "refinement_cheaper",
        "late_night_closure_cascade",
    }.issubset(ids), "all three baseline scenarios must ship"
    # The first 5 ids are the original 5 — append-only invariant.
    assert [c.id for c in config.hand_written[:5]] == [
        "north_beach_italian_dinner",
        "date_night_dinner_drinks_walkable",
        "mission_vegan_brunch",
        "soma_quiet_late_cafe",
        "impossible_four_am_five_star",
    ]


def test_omakase_case_tags_include_category_compliance() -> None:
    """Plan 03-07's baseline run uses tags to filter the eval set; the
    omakase case must self-identify so the category-compliance scorer
    surfaces in the per-tag aggregate."""
    config = load_eval_queries(REPO_ROOT / DEFAULT_EVAL_QUERIES_PATH)
    case = next(c for c in config.hand_written if c.id == "omakase_mission_open_ended")
    assert "category_compliance" in case.tags


# ─── Phase 6 / Plan 06-04: Eval config schema additions ─────────────────────────
#
# Four Pydantic schema additions land in plan 06-04:
#   - EvalQuery.threading_mode: Literal["legacy", "prod"] = "legacy" (D-06-05)
#   - ExpectedRefinement nested model with target_slot: int = Field(ge=1) (D-06-08)
#   - EvalQuery.expected_refinement: ExpectedRefinement | None = None (D-06-08)
#   - MatrixEntry.env: dict[str, StrictStr] | None = None (D-06-10)
#
# Backward-compat invariant: every existing YAML config (eval_queries.yaml +
# eval_matrix.yaml) MUST validate unchanged. The four additions are all opt-in
# (default values preserve current behavior).


def _phase6_query_payload(**overrides: object) -> dict:
    """Smallest valid EvalQuery payload for Phase 6 schema tests."""
    payload: dict = {
        "id": "x",
        "query": "q",
        "reference": "r",
        "expects_clarification_or_relaxation": True,
    }
    payload.update(overrides)
    return payload


class TestPhase6EvalConfigAdditions:
    """Phase 6 / Plan 06-04 schema additions on EvalQuery + MatrixEntry."""

    # --- EvalQuery.threading_mode (D-06-05) ---

    def test_threading_mode_default_is_legacy(self) -> None:
        """Default is 'legacy' so the 30 existing YAML cases keep their behavior
        (opt-in flip per D-06-05 / D-06-06)."""
        case = EvalQuery.model_validate(_phase6_query_payload())
        assert case.threading_mode == "legacy"

    def test_threading_mode_accepts_prod(self) -> None:
        """`refinement_cheaper` flips to 'prod' in plan 06-07 — must validate now."""
        case = EvalQuery.model_validate(_phase6_query_payload(threading_mode="prod"))
        assert case.threading_mode == "prod"

    def test_threading_mode_rejects_invalid_literal(self) -> None:
        """Literal enforces membership at validation time — only 'legacy' or 'prod'."""
        with pytest.raises(ValidationError, match="threading_mode"):
            EvalQuery.model_validate(_phase6_query_payload(threading_mode="invalid"))

    # --- EvalQuery.expected_refinement + ExpectedRefinement (D-06-08) ---

    def test_expected_refinement_default_is_none(self) -> None:
        """`None` default = backward compat for non-refinement scenarios."""
        case = EvalQuery.model_validate(_phase6_query_payload())
        assert case.expected_refinement is None

    def test_expected_refinement_validates_target_slot_ge_1(self) -> None:
        """target_slot is 1-indexed (matches user prose + is_refinement_request +
        refinement_minimal_edit scorer convention). 0 must be rejected."""
        case = EvalQuery.model_validate(
            _phase6_query_payload(expected_refinement={"target_slot": 2})
        )
        assert case.expected_refinement is not None
        assert case.expected_refinement.target_slot == 2

        with pytest.raises(ValidationError, match="target_slot"):
            EvalQuery.model_validate(_phase6_query_payload(expected_refinement={"target_slot": 0}))

    def test_expected_refinement_forbids_extra_fields(self) -> None:
        """`extra='forbid'` on the nested model — typos must fail loudly."""
        with pytest.raises(ValidationError):
            EvalQuery.model_validate(
                _phase6_query_payload(expected_refinement={"target_slot": 2, "extra_field": "x"})
            )

    def test_expected_refinement_round_trips_via_expected_refinement_class(self) -> None:
        """ExpectedRefinement can be instantiated directly (parity with
        ExpectedConstraints / ExpectedResults)."""
        ref = ExpectedRefinement(target_slot=3)
        assert ref.target_slot == 3
        with pytest.raises(ValidationError, match="target_slot"):
            ExpectedRefinement(target_slot=0)

    # --- MatrixEntry.env (D-06-10) ---

    def test_matrix_entry_env_default_is_none(self) -> None:
        """`None` default keeps the existing 2-entry eval_matrix.yaml valid."""
        entry = MatrixEntry.model_validate({"provider": "openai", "model": "gpt-4o-mini"})
        assert entry.env is None

    def test_matrix_entry_env_accepts_string_dict(self) -> None:
        """The plan 06-06 gateway: a single per-cell flag override like
        REFINEMENT_STRUCTURED_PLAN_ENABLED=true (string value)."""
        entry = MatrixEntry.model_validate(
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "env": {"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"},
            }
        )
        assert entry.env == {"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"}

    def test_matrix_entry_env_rejects_non_string_value_bool(self) -> None:
        """StrictStr rejects bool — without it, subprocess.run(env=...) would
        crash on the bool downstream. Catch at config-load time."""
        with pytest.raises(ValidationError):
            MatrixEntry.model_validate(
                {"provider": "openai", "model": "gpt-4o-mini", "env": {"FLAG": True}}
            )

    def test_matrix_entry_env_rejects_non_string_value_int(self) -> None:
        """StrictStr also rejects int (no silent coercion to '1')."""
        with pytest.raises(ValidationError):
            MatrixEntry.model_validate(
                {"provider": "openai", "model": "gpt-4o-mini", "env": {"FLAG": 1}}
            )

    def test_matrix_entry_env_yaml_unquoted_boolean_rejected(self) -> None:
        """MEDIUM env StrictStr regression guard — without StrictStr a `mode='after'`
        validator runs AFTER Pydantic coerces YAML bool to str, silently passing.
        Simulate `yaml.safe_load` of `env: {FLAG: true}` (unquoted) and prove
        EvalMatrixConfig.model_validate rejects it."""
        payload = {
            "entries": [{"provider": "openai", "model": "gpt-4o-mini", "env": {"FLAG": True}}],
            "scenarios": ["omakase_mission_open_ended"],
        }
        with pytest.raises(ValidationError):
            EvalMatrixConfig.model_validate(payload)

    def test_matrix_entry_env_rejects_non_string_key(self) -> None:
        """Keys must also be strings (subprocess env). `mode='before'` validator
        catches non-string keys for symmetry with the StrictStr value guard."""
        with pytest.raises(ValidationError):
            MatrixEntry.model_validate(
                {"provider": "openai", "model": "gpt-4o-mini", "env": {1: "true"}}
            )

    def test_matrix_entry_env_preserves_extra_forbid(self) -> None:
        """`extra='forbid'` on MatrixEntry still applies — unknown top-level
        keys still 422 even after env is added."""
        with pytest.raises(ValidationError):
            MatrixEntry.model_validate(
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "env": {"FLAG": "true"},
                    "extra_key": "nope",
                }
            )

    # --- Backward compat: existing production YAML files still load ---

    def test_existing_eval_queries_yaml_still_loads(self) -> None:
        """Every existing case in configs/eval_queries.yaml must validate
        unchanged after the schema additions land. Zero impact on Phase 3/4
        baselines."""
        config = load_eval_queries()
        assert len(config.hand_written) >= 30

    def test_existing_eval_matrix_yaml_still_loads(self) -> None:
        """The anchored entries in configs/eval_matrix.yaml still validate.

        Phase 11 / D-11-12: 3 new cross-model entries added (flag-OFF — env=None);
        total is now 5 (was 2 at Phase 6 time). All entries remain flag-OFF so the
        `all(env is None)` invariant holds.
        """
        config = load_eval_matrix()
        assert (
            len(config.entries) == 5
        )  # D-11-12: was 2; gpt-5-mini, claude-sonnet-4-6, deepseek-reasoner added
        assert all(entry.env is None for entry in config.entries)

    def test_threading_mode_invariant_after_plan_06_07(self) -> None:
        """After plan 06-07 ships, refinement_cheaper is the ONLY YAML case
        on threading_mode='prod'. Every other hand_written case stays on
        the default 'legacy' threading per D-06-06. Proves the opt-in
        invariant (only refinement_cheaper opts into prod-threading)."""
        config = load_eval_queries()
        prod_cases = [c.id for c in config.hand_written if c.threading_mode == "prod"]
        legacy_cases = [c.id for c in config.hand_written if c.threading_mode == "legacy"]
        assert prod_cases == ["refinement_cheaper"], (
            f"Phase 6 (D-06-06) expects only refinement_cheaper on "
            f"threading_mode='prod'; got prod_cases={prod_cases}"
        )
        # Every other hand-written case stays on legacy.
        other_ids = {c.id for c in config.hand_written if c.id != "refinement_cheaper"}
        assert set(legacy_cases) == other_ids

    def test_expected_refinement_invariant_after_plan_06_07(self) -> None:
        """After plan 06-07 ships, refinement_cheaper is the ONLY YAML
        case carrying expected_refinement (target_slot=2 per D-06-06). All
        other cases keep expected_refinement=None."""
        config = load_eval_queries()
        with_refinement = [c for c in config.hand_written if c.expected_refinement is not None]
        assert len(with_refinement) == 1
        assert with_refinement[0].id == "refinement_cheaper"
        assert with_refinement[0].expected_refinement.target_slot == 2
