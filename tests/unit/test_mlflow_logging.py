from __future__ import annotations

from pathlib import Path

import mlflow

from scripts.log_model_to_mlflow import log_rag_experiment


def test_log_rag_experiment_logs_params_and_artifacts(
    mocker,
    monkeypatch,
    tmp_path: Path,
) -> None:
    tracking_db = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{tracking_db}"
    monkeypatch.chdir(tmp_path)

    fake_chain = mocker.Mock()
    fake_chain.invoke.side_effect = [
        {"result": "Try Taqueria Example.", "source_documents": []},
        {"result": "Try Caffe Trieste.", "source_documents": []},
    ]
    mocker.patch("scripts.log_model_to_mlflow.build_rag_chain", return_value=fake_chain)

    run_id = log_rag_experiment(
        llm_provider="openai",
        chat_model="gpt-4o-mini",
        k=4,
        temperature=0.15,
        experiment_name="test-city-concierge-rag",
        tracking_uri=tracking_uri,
        sample_queries=[
            "Best tacos in the Mission",
            "Coffee shops near North Beach",
        ],
    )

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("test-city-concierge-rag")

    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert run.info.run_id == run_id
    assert run.data.params["llm_provider"] == "openai"
    assert run.data.params["chat_model"] == "gpt-4o-mini"
    assert run.data.params["k"] == "4"
    assert run.data.params["temperature"] == "0.15"
    assert run.data.params["embedding_model"] == "text-embedding-3-small"
    assert run.data.params["vector_store"] == "pgvector"

    config_artifacts = client.list_artifacts(run_id, "config")
    sample_artifacts = client.list_artifacts(run_id, "sample_outputs")
    assert [artifact.path for artifact in config_artifacts] == ["config/rag_config.json"]
    assert sorted(artifact.path for artifact in sample_artifacts) == [
        "sample_outputs/query_1.txt",
        "sample_outputs/query_2.txt",
    ]
