from __future__ import annotations

import pytest
from mlflow.exceptions import MlflowException

from app.bootstrap import load_registered_rag_chain, parse_active_model_config


def test_parse_active_model_config_uses_string_params_from_mlflow() -> None:
    config = parse_active_model_config(
        {"llm_provider": "openai", "chat_model": "gpt-4o-mini", "k": "7", "temperature": "0.3"},
        run_id="run-abc",
        model_version="3",
    )
    assert config.k == 7
    assert config.temperature == 0.3
    assert config.chat_model == "gpt-4o-mini"
    assert config.run_id == "run-abc"
    assert config.model_version == "3"


def test_parse_active_model_config_falls_back_to_defaults_when_missing() -> None:
    config = parse_active_model_config({}, run_id="r", model_version="1")
    assert config.llm_provider == "openai"
    assert config.k >= 1  # falls back to settings.retriever_k
    assert config.temperature == 0.0


def test_parse_active_model_config_handles_empty_string_numerics() -> None:
    config = parse_active_model_config({"k": "", "temperature": ""}, run_id="r", model_version="1")
    # Empty string should be treated as missing, not raise ValueError.
    assert config.k >= 1
    assert config.temperature == 0.0


def test_load_registered_rag_chain_wraps_mlflow_exception(mocker, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://test-mlflow:5000")
    mocker.patch(
        "app.bootstrap.get_settings",
        return_value=mocker.Mock(
            mlflow_tracking_uri="http://test-mlflow:5000",
            mlflow_model_name="city-concierge-rag",
        ),
    )
    mocker.patch("app.bootstrap.mlflow.set_tracking_uri")
    fake_client = mocker.Mock()
    fake_client.get_model_version_by_alias.side_effect = MlflowException("alias not found")
    mocker.patch("app.bootstrap.mlflow.MlflowClient", return_value=fake_client)

    with pytest.raises(RuntimeError, match="Unable to load the MLflow production alias"):
        load_registered_rag_chain()
