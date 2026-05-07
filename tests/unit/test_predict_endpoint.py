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


def test_predict_endpoint_proxies_through_agent(mocker) -> None:
    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        return {
            "messages": [],
            "constraints": {},
            "stops": [
                {
                    "place_id": "p1",
                    "name": "Taqueria Example",
                    "rationale": "Top tacos in the Mission",
                    "source": "google_places",
                    "primary_type": "restaurant",
                    "planned_duration_min": 90,
                    "arrival_time": None,
                }
            ],
            "scratch": {},
            "step_count": 1,
            "done": True,
            "final_reply": "You should try Taqueria Example.",
            "awaiting_stops_count": False,
            "walked_meters_so_far": 0.0,
        }

    fake_graph.ainvoke = _ainvoke

    fake_chain = mocker.Mock()
    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config(fake_chain))
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

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


def test_predict_falls_back_to_chain_when_agent_unavailable(mocker) -> None:
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

    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config(fake_chain))
    mocker.patch("app.main.build_agent_graph", side_effect=RuntimeError("graph build failed"))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"query": "Best tacos in the Mission", "limit": 5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "You should try Taqueria Example."
    assert body["sources"][0]["name"] == "Taqueria Example"


def test_predict_returns_503_when_everything_unavailable(mocker) -> None:
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
