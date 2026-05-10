from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.eval.config import (
    DEFAULT_EVAL_QUERIES_PATH,
    REPO_ROOT,
    EvalQueriesConfig,
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
                "expected_stops": 1,
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
    assert case.expected_stops == 1
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


def test_load_eval_queries_rejects_zero_expected_stops(tmp_path: Path) -> None:
    """Reject non-positive expected stop counts."""
    payload = valid_payload()
    payload["hand_written"][0]["expected_stops"] = 0
    config_path = write_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="expected_stops"):
        load_eval_queries(config_path)


def test_load_eval_queries_requires_expected_stops_for_normal_cases(tmp_path: Path) -> None:
    """Require normal eval cases to declare their target stop count."""
    payload = valid_payload()
    payload["hand_written"][0].pop("expected_stops")
    config_path = write_config(tmp_path, payload)

    with pytest.raises(ValidationError, match="expected_stops"):
        load_eval_queries(config_path)


def test_load_eval_queries_allows_missing_expected_stops_for_relaxation_case(
    tmp_path: Path,
) -> None:
    """Allow known-bad cases to omit expected stops when relaxation is desired."""
    payload = valid_payload()
    case = payload["hand_written"][0]
    case.pop("expected_stops")
    case["expects_clarification_or_relaxation"] = True
    config_path = write_config(tmp_path, payload)

    config = load_eval_queries(config_path)

    assert config.hand_written[0].expected_stops is None
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
    """Keep the checked-in eval query set aligned with the typed schema."""
    config = load_eval_queries(REPO_ROOT / DEFAULT_EVAL_QUERIES_PATH)

    assert len(config.hand_written) >= 5
    assert {case.id for case in config.hand_written} == {
        "north_beach_italian_dinner",
        "date_night_dinner_drinks_walkable",
        "mission_vegan_brunch",
        "soma_quiet_late_cafe",
        "impossible_four_am_five_star",
    }
    assert all(case.query for case in config.hand_written)
    assert all(case.reference for case in config.hand_written)
