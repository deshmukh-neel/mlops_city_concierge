from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import ActiveModelConfig, LoadedConfig, app


def _stub_loaded_config(fake_chain) -> LoadedConfig:
    return LoadedConfig(
        chain=fake_chain,
        llm=object(),
        params=ActiveModelConfig(
            llm_provider="openai",
            chat_model="gpt-4o-mini",
            k=5,
            temperature=0.0,
            run_id="run-123",
            model_version="7",
        ),
    )


def test_predict_returns_chain_result_and_sources(mocker) -> None:
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
                    "place_id": "p1",
                    "primary_type": "restaurant",
                }
            )
        ],
    }

    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config(fake_chain))
    mocker.patch("app.main.build_agent_graph", return_value=object())

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"query": "Best tacos in the Mission", "limit": 5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "You should try Taqueria Example."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["name"] == "Taqueria Example"
    assert body["sources"][0]["place_id"] == "p1"
    assert body["sources"][0]["similarity"] == 0.981
    fake_chain.invoke.assert_called_once_with({"query": "Best tacos in the Mission"})


def test_predict_uses_chain_even_when_agent_graph_is_built(mocker) -> None:
    """The agent never sits on the /predict path; /predict is the legacy contract."""
    fake_chain = mocker.Mock()
    fake_chain.invoke.return_value = {"result": "ok", "source_documents": []}
    fake_graph = mocker.Mock()
    fake_graph.ainvoke = mocker.AsyncMock()

    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config(fake_chain))
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"query": "anything", "limit": 5},
        )

    assert response.status_code == 200
    fake_chain.invoke.assert_called_once()
    fake_graph.ainvoke.assert_not_called()


def test_predict_returns_503_when_chain_unavailable(mocker) -> None:
    mocker.patch(
        "app.main.load_registered_rag_chain",
        side_effect=RuntimeError("mlflow unavailable"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"query": "Best tacos in the Mission", "limit": 5},
        )

    assert response.status_code == 503
