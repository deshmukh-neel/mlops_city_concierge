from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scripts.log_model_to_mlflow import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    ExperimentConfig,
    RunConfig,
    load_experiment_config,
    log_rag_experiments_from_config,
    main,
)


def _valid_payload() -> dict:
    return {
        "experiment_name": "city-concierge-rag-test",
        "sample_queries": [
            "Best tacos in the Mission",
            "Coffee shops near North Beach",
        ],
        "runs": [
            {
                "llm_provider": "openai",
                "chat_model": "gpt-4o-mini",
                "k": 5,
                "temperature": 0,
            },
            {
                "llm_provider": "gemini",
                "chat_model": "gemini-2.5-flash",
                "k": 7,
                "temperature": 0.2,
            },
        ],
    }


def write_experiment_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "experiments.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_load_experiment_config_loads_valid_yaml(tmp_path: Path) -> None:
    config_path = write_experiment_config(tmp_path, _valid_payload())

    config = load_experiment_config(config_path)

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "city-concierge-rag-test"
    assert config.sample_queries == [
        "Best tacos in the Mission",
        "Coffee shops near North Beach",
    ]
    assert config.runs == [
        RunConfig(llm_provider="openai", chat_model="gpt-4o-mini", k=5, temperature=0.0),
        RunConfig(llm_provider="gemini", chat_model="gemini-2.5-flash", k=7, temperature=0.2),
    ]


def _payload_with_run(**overrides: object) -> dict:
    payload = _valid_payload()
    run = dict(payload["runs"][0])
    run.update(overrides)
    payload["runs"] = [run]
    return payload


def _payload_without_run_field(field: str) -> dict:
    payload = _valid_payload()
    run = dict(payload["runs"][0])
    run.pop(field)
    payload["runs"] = [run]
    return payload


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_payload_with_run(llm_provider="anthropic"), r"llm_provider"),
        (_payload_without_run_field("temperature"), r"(?s)temperature.*[Ff]ield required"),
        (_payload_without_run_field("chat_model"), r"(?s)chat_model.*[Ff]ield required"),
        (_payload_with_run(k=0), r"(?s)k.*greater than 0"),
        (_payload_with_run(k=True), r"k"),
        (_payload_with_run(k="five"), r"k"),
        (_payload_with_run(temperature="hot"), r"temperature"),
        (_payload_with_run(chat_model=""), r"chat_model"),
        (_payload_with_run(chat_model="   "), r"chat_model"),
        ({**_valid_payload(), "sample_queries": "not a list"}, r"sample_queries"),
        ({**_valid_payload(), "sample_queries": []}, r"sample_queries"),
        ({**_valid_payload(), "sample_queries": ["  "]}, r"sample_queries"),
        ({**_valid_payload(), "runs": []}, r"runs"),
        ({**_valid_payload(), "runs": None}, r"runs"),
        ({**_valid_payload(), "experiment_name": ""}, r"experiment_name"),
    ],
)
def test_load_experiment_config_rejects_invalid_payloads(
    tmp_path: Path,
    payload: dict,
    match: str,
) -> None:
    config_path = write_experiment_config(tmp_path, payload)

    with pytest.raises(ValidationError, match=match):
        load_experiment_config(config_path)


def test_load_experiment_config_rejects_top_level_list(tmp_path: Path) -> None:
    config_path = tmp_path / "experiments.yaml"
    config_path.write_text(yaml.safe_dump([{"experiment_name": "x"}]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"YAML mapping"):
        load_experiment_config(config_path)


def test_repo_experiments_yaml_is_valid() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments.yaml")

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name
    assert len(config.runs) >= 1
    assert len(config.sample_queries) >= 1


def test_log_rag_experiments_from_config_logs_each_run(mocker) -> None:
    config = ExperimentConfig.model_validate(_valid_payload())
    log_rag_experiment = mocker.patch(
        "scripts.log_model_to_mlflow.log_rag_experiment",
        side_effect=["run-1", "run-2"],
    )
    start_run = mocker.patch("scripts.log_model_to_mlflow.mlflow.start_run")
    start_run.return_value.__enter__.return_value = mocker.Mock()
    mocker.patch("scripts.log_model_to_mlflow.mlflow.set_tracking_uri")
    mocker.patch("scripts.log_model_to_mlflow.mlflow.set_experiment")
    mocker.patch("scripts.log_model_to_mlflow.mlflow.log_artifact")

    run_ids = log_rag_experiments_from_config(
        config=config,
        tracking_uri="http://mlflow.test",
        model_name="test-model",
    )

    assert run_ids == ["run-1", "run-2"]
    assert log_rag_experiment.call_count == 2
    for call in log_rag_experiment.call_args_list:
        assert call.kwargs["experiment_name"] == "city-concierge-rag-test"
        assert call.kwargs["sample_queries"] == config.sample_queries
        assert call.kwargs["tracking_uri"] == "http://mlflow.test"
        assert call.kwargs["model_name"] == "test-model"
        assert call.kwargs["nested"] is True
    assert log_rag_experiment.call_args_list[0].kwargs["llm_provider"] == "openai"
    assert log_rag_experiment.call_args_list[1].kwargs["llm_provider"] == "gemini"


def test_log_rag_experiments_from_config_applies_overrides(mocker) -> None:
    config = ExperimentConfig.model_validate(_valid_payload())
    log_rag_experiment = mocker.patch(
        "scripts.log_model_to_mlflow.log_rag_experiment",
        side_effect=["run-1", "run-2"],
    )
    start_run = mocker.patch("scripts.log_model_to_mlflow.mlflow.start_run")
    start_run.return_value.__enter__.return_value = mocker.Mock()
    mocker.patch("scripts.log_model_to_mlflow.mlflow.set_tracking_uri")
    mocker.patch("scripts.log_model_to_mlflow.mlflow.set_experiment")

    log_rag_experiments_from_config(
        config=config,
        experiment_name="override-experiment",
        sample_queries=["only query"],
    )

    for call in log_rag_experiment.call_args_list:
        assert call.kwargs["experiment_name"] == "override-experiment"
        assert call.kwargs["sample_queries"] == ["only query"]


@pytest.fixture
def mock_log_rag_experiment(mocker):
    return mocker.patch(
        "scripts.log_model_to_mlflow.log_rag_experiment",
        return_value="run-abc",
    )


@pytest.fixture
def mock_mlflow_batch(mocker):
    start_run = mocker.patch("scripts.log_model_to_mlflow.mlflow.start_run")
    start_run.return_value.__enter__.return_value = mocker.Mock()
    mocker.patch("scripts.log_model_to_mlflow.mlflow.set_tracking_uri")
    mocker.patch("scripts.log_model_to_mlflow.mlflow.set_experiment")
    mocker.patch("scripts.log_model_to_mlflow.mlflow.log_artifact")
    return start_run


def test_main_config_mode_uses_yaml_values(
    tmp_path: Path, mock_log_rag_experiment, mock_mlflow_batch
) -> None:
    config_path = write_experiment_config(tmp_path, _valid_payload())
    mock_log_rag_experiment.side_effect = ["run-1", "run-2"]

    main(["--config", str(config_path)])

    assert mock_log_rag_experiment.call_count == 2
    for call in mock_log_rag_experiment.call_args_list:
        assert call.kwargs["experiment_name"] == "city-concierge-rag-test"
        assert call.kwargs["sample_queries"] == [
            "Best tacos in the Mission",
            "Coffee shops near North Beach",
        ]


def test_main_config_mode_experiment_name_override(
    tmp_path: Path, mock_log_rag_experiment, mock_mlflow_batch
) -> None:
    config_path = write_experiment_config(tmp_path, _valid_payload())
    mock_log_rag_experiment.side_effect = ["run-1", "run-2"]

    main(["--config", str(config_path), "--experiment-name", "override-exp"])

    for call in mock_log_rag_experiment.call_args_list:
        assert call.kwargs["experiment_name"] == "override-exp"


def test_main_config_mode_sample_query_override(
    tmp_path: Path, mock_log_rag_experiment, mock_mlflow_batch
) -> None:
    config_path = write_experiment_config(tmp_path, _valid_payload())
    mock_log_rag_experiment.side_effect = ["run-1", "run-2"]

    main(["--config", str(config_path), "--sample-query", "only this"])

    for call in mock_log_rag_experiment.call_args_list:
        assert call.kwargs["sample_queries"] == ["only this"]


def test_main_ad_hoc_mode_preserved(mock_log_rag_experiment) -> None:
    main(["--llm-provider", "openai", "--k", "5", "--sample-query", "hello"])

    assert mock_log_rag_experiment.call_count == 1
    call = mock_log_rag_experiment.call_args_list[0]
    assert call.kwargs["llm_provider"] == "openai"
    assert call.kwargs["k"] == 5
    assert call.kwargs["sample_queries"] == ["hello"]


def test_main_ad_hoc_mode_loads_default_sample_queries_from_yaml(
    mocker, mock_log_rag_experiment
) -> None:
    mocker.patch(
        "scripts.log_model_to_mlflow._load_default_sample_queries",
        return_value=["yaml-query"],
    )

    main(["--llm-provider", "openai"])

    call = mock_log_rag_experiment.call_args_list[0]
    assert call.kwargs["sample_queries"] == ["yaml-query"]


def test_default_config_path_exists() -> None:
    # Sanity check: the default path the ad-hoc fallback relies on is real.
    assert (REPO_ROOT / DEFAULT_CONFIG_PATH).exists()
