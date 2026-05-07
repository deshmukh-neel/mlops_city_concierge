"""Functional tests that exercise the /chat endpoint with a *real* agent graph.

Only the LLM and the Postgres-backed retrieval functions are stubbed. The
graph, tool wrapping, message handling, and FastAPI plumbing all run for real
so we catch shape mismatches that pure unit tests miss.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.graph import build_agent_graph
from app.main import ActiveModelConfig, LoadedConfig, app
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
