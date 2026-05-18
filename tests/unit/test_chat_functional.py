"""Functional tests that exercise the /chat endpoint with a *real* agent graph.

Only the LLM and the Postgres-backed retrieval functions are stubbed. The
graph, tool wrapping, message handling, and FastAPI plumbing all run for real
so we catch shape mismatches that pure unit tests miss.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.graph import build_agent_graph
from app.main import ActiveModelConfig, LoadedConfig, app
from app.tools.directions import DirectionsLeg, DirectionsResult
from app.tools.retrieval import PlaceHit


class _ScriptedLLM(BaseChatModel):
    scripted: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self.scripted.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedLLM:
        return self


def _stub_loaded_config() -> LoadedConfig:
    return LoadedConfig(
        chain=object(),
        llm=object(),
        params=ActiveModelConfig(
            llm_provider="openai",
            chat_model="gpt-4o-mini",
            k=5,
            temperature=0.0,
            run_id="run-fn",
            model_version="1",
        ),
    )


def test_chat_runs_real_graph_with_tool_call(monkeypatch, mocker) -> None:
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [
            PlaceHit(
                place_id="p1",
                name="Trick Dog",
                source="google_places",
                similarity=0.9,
                latitude=37.77,
                longitude=-122.41,
                rating=4.6,
                price_level="PRICE_LEVEL_MODERATE",
                business_status="OPERATIONAL",
                primary_type="cocktail_bar",
                formatted_address="3010 20th St, San Francisco",
                snippet=None,
            )
        ],
    )

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "id": "call-1",
                    "args": {"query": "cocktail bar"},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "commit_itinerary",
                    "id": "call-2",
                    "args": {
                        "stops": [
                            {
                                "place_id": "p1",
                                "name": "Trick Dog",
                                "rationale": "iconic SF cocktail bar",
                                "source": "google_places",
                                "primary_type": "cocktail_bar",
                            }
                        ]
                    },
                }
            ],
        ),
        AIMessage(content="Try Trick Dog.", tool_calls=[]),
    ]
    fake_llm = _ScriptedLLM(scripted=list(scripted))
    real_graph = build_agent_graph(fake_llm, max_steps=4)

    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)
    # place_id "p1" doesn't exist in places_raw in the test environment; without
    # this patch, a real DB pool (activated by load_dotenv in ingest_places_sf.py
    # during full-suite collection) causes no_hallucinated_place_ids -> 0.0 ->
    # revision loop -> scripted LLM exhausted.
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "cocktail bar in SF"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Try Trick Dog."
    assert body["ragLabel"] == "openai:gpt-4o-mini"
    assert len(body["places"]) == 1
    assert body["places"][0]["place_id"] == "p1"
    assert body["places"][0]["name"] == "Trick Dog"
    assert body["places"][0]["primary_type"] == "cocktail_bar"


def test_commit_itinerary_rejects_ungrounded_place_ids(monkeypatch, mocker) -> None:
    """A place_id not seen via prior tool results must be dropped, not
    silently accepted — that's the anti-hallucination guarantee."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "id": "s1",
                    "args": {"query": "cocktail bar"},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "commit_itinerary",
                    "id": "c1",
                    "args": {
                        "stops": [
                            {
                                "place_id": "hallucinated",
                                "name": "Made Up Bar",
                                "rationale": "the LLM imagined this",
                                "source": "google_places",
                            }
                        ]
                    },
                }
            ],
        ),
        AIMessage(content="No grounded options.", tool_calls=[]),
    ]
    real_graph = build_agent_graph(_ScriptedLLM(scripted=list(scripted)), max_steps=4)

    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["places"] == []


_T0 = datetime(2024, 6, 1, 18, 0, 0, tzinfo=timezone.utc)  # 6pm UTC anchor for tests


def _two_stop_script() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "semantic_search", "id": "s1", "args": {"query": "date"}}],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "commit_itinerary",
                    "id": "c1",
                    "args": {
                        "stops": [
                            {
                                "place_id": "p1",
                                "name": "Bar One",
                                "rationale": "start",
                                "source": "google_places",
                                "primary_type": "cocktail_bar",
                                "arrival_time": _T0.isoformat(),
                                "latitude": 37.770,
                                "longitude": -122.410,
                            },
                            {
                                "place_id": "p2",
                                "name": "Bar Two",
                                "rationale": "next",
                                "source": "google_places",
                                "primary_type": "cocktail_bar",
                                "latitude": 37.780,
                                "longitude": -122.410,
                            },
                        ]
                    },
                }
            ],
        ),
        AIMessage(content="Bar One then Bar Two.", tool_calls=[]),
    ]


def _two_hits(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [
            PlaceHit(
                place_id="p1",
                name="Bar One",
                source="google_places",
                similarity=0.9,
                latitude=37.770,
                longitude=-122.410,
                business_status="OPERATIONAL",
                primary_type="cocktail_bar",
                formatted_address="1 A St",
                snippet=None,
            ),
            PlaceHit(
                place_id="p2",
                name="Bar Two",
                source="google_places",
                similarity=0.9,
                latitude=37.780,
                longitude=-122.410,
                business_status="OPERATIONAL",
                primary_type="cocktail_bar",
                formatted_address="2 B St",
                snippet=None,
            ),
        ],
    )


def test_chat_retimes_arrival_with_real_directions(monkeypatch, mocker) -> None:
    """The committed plan's arrival_time is overwritten by real Directions
    data — not the haversine estimate."""
    _two_hits(monkeypatch)

    async def _slow_directions(stops, mode="walk"):
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=2400, distance_m=3000.0)],
            total_duration_s=2400,
            mode=mode,
            source="google",
        )

    mocker.patch("app.agent.graph.route_legs", _slow_directions)
    mocker.patch("app.agent.graph.temporal_coherence", lambda _s: 1.0)
    mocker.patch("app.agent.revision.itinerary_violations", lambda _s: [])

    real_graph = build_agent_graph(_ScriptedLLM(scripted=_two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        body = client.post(
            "/chat",
            json={"message": "a date with two bars, arrive 6pm"},
        ).json()

    a1 = body["places"][0]["arrival_time"]
    a2 = body["places"][1]["arrival_time"]
    assert a1 is not None and a2 is not None
    delta_min = (datetime.fromisoformat(a2) - datetime.fromisoformat(a1)).total_seconds() / 60
    assert delta_min == 100  # 60 cocktail_bar dwell + 40 real travel (NOT ~2min haversine)


def test_chat_retiming_flips_temporal_pass_to_fail(monkeypatch, mocker) -> None:
    """Slower real travel pushes stop 2 past its closing time: the re-run
    temporal check fails and the reply gains the caveat."""
    _two_hits(monkeypatch)

    async def _slow_directions(stops, mode="walk"):
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=3600, distance_m=4000.0)],
            total_duration_s=3600,
            mode=mode,
            source="google",
        )

    mocker.patch("app.agent.graph.route_legs", _slow_directions)
    # temporal_coherence is only called in retime (critique loop uses itinerary_violations,
    # which is stubbed to []). One call: retime re-check -> fail.
    scores = iter([0.0])
    mocker.patch("app.agent.graph.temporal_coherence", lambda _s: next(scores))
    mocker.patch("app.agent.revision.itinerary_violations", lambda _s: [])

    real_graph = build_agent_graph(_ScriptedLLM(scripted=_two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        body = client.post("/chat", json={"message": "two bars, arrive 6pm"}).json()

    assert "Caveats" in body["reply"]
    assert "temporal_coherence" in body["reply"]
    assert body["reply"].startswith("Bar One then Bar Two.")
    assert body["reply"].count("Caveats:") == 1


def test_chat_directions_failure_keeps_haversine_reply(monkeypatch, mocker) -> None:
    """route_legs internally degrades to fallback -> /chat still 200, no
    spurious caveat, arrival_times come from the haversine fallback."""
    _two_hits(monkeypatch)

    async def _fallback_directions(stops, mode="walk"):
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=120, distance_m=160.0)],
            total_duration_s=120,
            mode=mode,
            source="haversine_fallback",
        )

    mocker.patch("app.agent.graph.route_legs", _fallback_directions)
    mocker.patch("app.agent.graph.temporal_coherence", lambda _s: 1.0)
    mocker.patch("app.agent.revision.itinerary_violations", lambda _s: [])

    real_graph = build_agent_graph(_ScriptedLLM(scripted=_two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        resp = client.post("/chat", json={"message": "two bars"})

    assert resp.status_code == 200
    body = resp.json()
    assert "Caveats" not in body["reply"]
    assert body["reply"] == "Bar One then Bar Two."
