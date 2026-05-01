from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ActiveModelConfig


def test_predict_endpoint_returns_response_and_sources(mocker) -> None:
    fake_chain = mocker.Mock()
    fake_chain.invoke.return_value = {
        "result": "You should try Taqueria Example.",
        "source_documents": [
            SimpleNamespace(
                metadata={
                    "name": "Taqueria Example",
                    "rating": 4.7,
                    "address": "123 Mission St, San Francisco, CA",
                    "similarity": 0.981,
                }
            )
        ],
    }

    mocker.patch(
        "app.main.load_registered_rag_chain",
        return_value=(
            fake_chain,
            ActiveModelConfig(
                llm_provider="openai",
                chat_model="gpt-4o-mini",
                k=5,
                temperature=0.0,
                run_id="run-123",
                model_version="7",
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"query": "Best tacos in the Mission"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "response": "You should try Taqueria Example.",
        "sources": [
            {
                "name": "Taqueria Example",
                "rating": 4.7,
                "address": "123 Mission St, San Francisco, CA",
                "similarity": 0.981,
            }
        ],
    }
    fake_chain.invoke.assert_called_once_with({"query": "Best tacos in the Mission"})


def test_predict_ignores_unknown_request_fields(mocker) -> None:
    fake_chain = mocker.Mock()
    fake_chain.invoke.return_value = {"result": "ok", "source_documents": []}
    mocker.patch(
        "app.main.load_registered_rag_chain",
        return_value=(
            fake_chain,
            ActiveModelConfig(
                llm_provider="openai", chat_model="gpt-4o-mini", k=5, temperature=0.0
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"query": "anything", "limit": 99, "made_up_field": True},
        )

    assert response.status_code == 200
    fake_chain.invoke.assert_called_once_with({"query": "anything"})


def test_health_reports_degraded_when_startup_fails(mocker) -> None:
    mocker.patch(
        "app.main.load_registered_rag_chain",
        side_effect=RuntimeError("MLFLOW_TRACKING_URI is required (set it in .env)."),
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["status"] == "degraded"
    assert "MLFLOW_TRACKING_URI" in body["detail"]["error"]
