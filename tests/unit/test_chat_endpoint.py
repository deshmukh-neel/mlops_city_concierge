from __future__ import annotations

from typing import Any

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


def _final_state_dict(stops: list[dict[str, Any]] | None = None, reply: str = "Try it.") -> dict:
    return {
        "messages": [],
        "constraints": {},
        "stops": stops or [],
        "scratch": {},
        "step_count": 1,
        "done": True,
        "final_reply": reply,
        "awaiting_stops_count": False,
        "walked_meters_so_far": 0.0,
    }


def test_chat_endpoint_returns_reply_places_raglabel(mocker) -> None:
    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        return _final_state_dict(
            stops=[
                {
                    "place_id": "p1",
                    "name": "Trick Dog",
                    "rationale": "iconic SF cocktail bar",
                    "source": "google_places",
                    "primary_type": "cocktail_bar",
                    "planned_duration_min": 60,
                    "arrival_time": None,
                }
            ],
            reply="Trick Dog at 7pm, ~60 min.",
        )

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "plan me a cocktail night", "history": []},
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"reply", "places", "ragLabel"}
    assert body["reply"] == "Trick Dog at 7pm, ~60 min."
    assert body["ragLabel"] == "openai:gpt-4o-mini"
    assert len(body["places"]) == 1
    expected_place_keys = {
        "place_id",
        "name",
        "address",
        "rating",
        "price_level",
        "primary_type",
        "arrival_time",
        "rationale",
        "booking_url",
    }
    assert set(body["places"][0].keys()) == expected_place_keys
    assert body["places"][0]["place_id"] == "p1"


def test_chat_endpoint_returns_503_when_agent_unavailable(mocker) -> None:
    mocker.patch(
        "app.main.load_registered_rag_chain",
        side_effect=RuntimeError("mlflow unavailable"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "anything", "history": []},
        )

    assert response.status_code == 503
    assert "Agent graph unavailable" in response.json()["detail"]


def test_chat_endpoint_passes_history_to_graph(mocker) -> None:
    fake_graph = mocker.Mock()
    captured: dict[str, Any] = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "actually make it 4 stops",
                "history": [
                    {"role": "user", "content": "plan a date"},
                    {"role": "assistant", "content": "how many stops?"},
                ],
            },
        )

    assert response.status_code == 200
    state = captured["state"]
    assert len(state.messages) == 3
    # roles in order: user, assistant, user
    assert state.messages[0].type == "human"
    assert state.messages[1].type == "ai"
    assert state.messages[2].type == "human"
    assert state.messages[2].content == "actually make it 4 stops"


def test_chat_endpoint_accepts_empty_history(mocker) -> None:
    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 200
