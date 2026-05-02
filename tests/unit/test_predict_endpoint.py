from __future__ import annotations

from types import SimpleNamespace

import openai
import psycopg2
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ActiveModelConfig


def _stub_chain_raising(mocker, exc: Exception):
    fake_chain = mocker.Mock()
    fake_chain.invoke.side_effect = exc
    mocker.patch(
        "app.main.load_registered_rag_chain",
        return_value=(
            fake_chain,
            ActiveModelConfig(
                llm_provider="openai", chat_model="gpt-4o-mini", k=5, temperature=0.0
            ),
        ),
    )


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


class _FakeRateLimit(openai.RateLimitError):
    def __init__(self, msg: str = "rate limited") -> None:
        Exception.__init__(self, msg)


def test_predict_returns_502_on_openai_error(mocker) -> None:
    _stub_chain_raising(mocker, openai.OpenAIError("upstream borked"))

    with TestClient(app) as client:
        response = client.post("/predict", json={"query": "anything"})

    assert response.status_code == 502
    assert "Upstream LLM error" in response.json()["detail"]
    assert "request_id=" in response.json()["detail"]


def test_predict_returns_429_on_rate_limit(mocker) -> None:
    _stub_chain_raising(mocker, _FakeRateLimit())

    with TestClient(app) as client:
        response = client.post("/predict", json={"query": "anything"})

    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"].lower()


def test_predict_returns_503_on_db_operational_error(mocker) -> None:
    _stub_chain_raising(mocker, psycopg2.OperationalError("connection timed out"))

    with TestClient(app) as client:
        response = client.post("/predict", json={"query": "anything"})

    assert response.status_code == 503
    assert "Database" in response.json()["detail"]


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
