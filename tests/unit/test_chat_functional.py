"""Functional tests that exercise the /chat endpoint with a *real* agent graph.

Only the LLM and the Postgres-backed retrieval functions are stubbed. The
graph, tool wrapping, message handling, and FastAPI plumbing all run for real
so we catch shape mismatches that pure unit tests miss.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agent.graph import build_agent_graph
from app.main import ActiveModelConfig, LoadedConfig, app
from app.tools.directions import DirectionsLeg, DirectionsResult
from app.tools.retrieval import PlaceHit
from tests._helpers.scripted_llm import ScriptedLLM


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
                place_id="ChIJtest_p1_aaaaaaaa",
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
                                "place_id": "ChIJtest_p1_aaaaaaaa",
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
        # Obsolete under finalize-on-commit: the graph ends at the passing
        # commit above, so this narration turn is never reached. Kept only to
        # show the model would have stopped here anyway.
        AIMessage(content="Try Trick Dog.", tool_calls=[]),
    ]
    fake_llm = ScriptedLLM(scripted=list(scripted))
    real_graph = build_agent_graph(fake_llm, max_steps=4)

    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)
    # place_id "ChIJtest_p1_aaaaaaaa" doesn't exist in places_raw in the test environment; without
    # this patch, a real DB pool (activated by load_dotenv in ingest_places_sf.py
    # during full-suite collection) causes no_hallucinated_place_ids -> 0.0 ->
    # revision loop -> scripted LLM exhausted.
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "cocktail bar in SF"})

    assert response.status_code == 200
    body = response.json()
    # Finalize-on-commit: reply is synthesized from the committed stop, not
    # the model's narration turn.
    assert body["reply"].startswith("Here's your itinerary:")
    assert "Trick Dog" in body["reply"]
    assert body["ragLabel"] == "openai:gpt-4o-mini"
    assert len(body["places"]) == 1
    assert body["places"][0]["place_id"] == "ChIJtest_p1_aaaaaaaa"
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
                    "id": "ChIJtest_s1_aaaaaaaa",
                    "args": {"query": "cocktail bar"},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "commit_itinerary",
                    "id": "ChIJtest_c1_aaaaaaaa",
                    "args": {
                        "stops": [
                            {
                                "place_id": "ChIJtest_hallucinated_",
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
    real_graph = build_agent_graph(ScriptedLLM(scripted=list(scripted)), max_steps=4)

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
            tool_calls=[
                {"name": "semantic_search", "id": "ChIJtest_s1_aaaaaaaa", "args": {"query": "date"}}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "commit_itinerary",
                    "id": "ChIJtest_c1_aaaaaaaa",
                    "args": {
                        "stops": [
                            {
                                "place_id": "ChIJtest_p1_aaaaaaaa",
                                "name": "Bar One",
                                "rationale": "start",
                                "source": "google_places",
                                "primary_type": "cocktail_bar",
                                "arrival_time": _T0.isoformat(),
                                "latitude": 37.770,
                                "longitude": -122.410,
                            },
                            {
                                "place_id": "ChIJtest_p2_aaaaaaaa",
                                "name": "Bar Two",
                                "rationale": "ChIJtest_next_aaaaaa",
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
                place_id="ChIJtest_p1_aaaaaaaa",
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
                place_id="ChIJtest_p2_aaaaaaaa",
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
    # Closure detection moved to the swap node; stub it "all open" so retime's
    # arrival-time projection is the only thing under test.
    mocker.patch(
        "app.agent.swap._per_stop_closure_status",
        side_effect=lambda stops: [False] * len(stops),
    )
    mocker.patch("app.agent.revision.itinerary_violations", lambda _s: [])

    real_graph = build_agent_graph(ScriptedLLM(scripted=_two_stop_script()), max_steps=6)
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
    # Closure detection moved to the swap node; "all open" -> no rewrite,
    # reply is the synthesized summary.
    mocker.patch(
        "app.agent.swap._per_stop_closure_status",
        side_effect=lambda stops: [False] * len(stops),
    )
    mocker.patch("app.agent.revision.itinerary_violations", lambda _s: [])

    real_graph = build_agent_graph(ScriptedLLM(scripted=_two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        resp = client.post("/chat", json={"message": "two bars"})

    assert resp.status_code == 200
    body = resp.json()
    assert "Caveats" not in body["reply"]
    # Finalize-on-commit: reply synthesized from stops, no caveat (clean pass).
    assert body["reply"].startswith("Here's your itinerary:")
    assert "Bar One" in body["reply"] and "Bar Two" in body["reply"]


def test_chat_graph_injects_primary_type_family_for_slot(monkeypatch, mocker) -> None:
    """Phase 4 D-04-04 end-to-end: when the state carries
    requested_primary_types and the LLM emits slot_index on a retrieval tool
    call, the graph injects primary_type_family on the way to the underlying
    retrieval — observable in state.scratch.

    This is the production-/chat-layer proof that the graph-layer enforcement
    plumbs through the full FastAPI request stack. Wiring of
    requested_primary_types from the user message lives in Plan 04-06; for
    this functional test we monkeypatch app.main.UserConstraints so the
    constraints arrive pre-populated, isolating the graph-injection contract.
    """
    captured_scratch: dict = {}

    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [
            PlaceHit(
                place_id="ChIJtest_p1_aaaaaaaa",
                name="Sushi Spot",
                source="google_places",
                similarity=0.9,
                latitude=37.77,
                longitude=-122.41,
                rating=4.6,
                price_level="PRICE_LEVEL_MODERATE",
                business_status="OPERATIONAL",
                primary_type="Sushi Restaurant",
                formatted_address="123 Mission St, San Francisco",
                snippet=None,
            )
        ],
    )

    # Inject requested_primary_types into UserConstraints via a thin shim so
    # this graph-layer contract test stays isolated from Plan 04-06's intake
    # pipeline. After 04-06 lands the handler writes `requested_primary_types=
    # extracted_types` unconditionally, so we override (not setdefault) the
    # empty list the intake fallback produces for the dummy `loaded.llm` here.
    from app.agent.state import UserConstraints as _RealUserConstraints

    def _make_constraints(**kwargs):
        kwargs["requested_primary_types"] = ["Sushi Restaurant"]
        return _RealUserConstraints(**kwargs)

    monkeypatch.setattr("app.main.UserConstraints", _make_constraints)

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "id": "call-slot-0",
                    "args": {"query": "omakase", "slot_index": 0},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "commit_itinerary",
                    "id": "call-commit",
                    "args": {
                        "stops": [
                            {
                                "place_id": "ChIJtest_p1_aaaaaaaa",
                                "name": "Sushi Spot",
                                "rationale": "Sushi Restaurant in Mission",
                                "source": "google_places",
                                "primary_type": "Sushi Restaurant",
                            }
                        ]
                    },
                }
            ],
        ),
        AIMessage(content="Try Sushi Spot.", tool_calls=[]),
    ]
    real_graph = build_agent_graph(ScriptedLLM(scripted=list(scripted)), max_steps=4)

    # Capture the final state's scratch to assert injection. We wrap
    # graph.ainvoke so the captured scratch is the post-graph state.
    real_ainvoke = real_graph.ainvoke

    async def _capturing_ainvoke(state, **kw):
        result = await real_ainvoke(state, **kw)
        captured_scratch["scratch"] = (
            result.scratch if hasattr(result, "scratch") else result.get("scratch")
        )
        return result

    real_graph.ainvoke = _capturing_ainvoke  # type: ignore[assignment]

    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)
    # Per project memory full_suite_db_pool_contamination.md: the
    # itinerary_violations call activates the live DB pool in full-suite runs.
    # Stub it out so this functional test is hermetic.
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "plan an omakase night, slot 0"},
        )

    assert response.status_code == 200
    # The graph recorded the post-injection effective_args in scratch.
    semantic_search_entries = (captured_scratch.get("scratch") or {}).get("semantic_search") or []
    assert semantic_search_entries, "graph never recorded a semantic_search call"
    recorded_args = semantic_search_entries[0]["args"]
    assert isinstance(recorded_args.get("filters"), dict), (
        f"filters must round-trip as a dict, got {type(recorded_args.get('filters')).__name__}"
    )
    assert recorded_args["filters"]["primary_type_family"] == "restaurant"
    # The slot_index marker was stripped before reaching the recorded args.
    assert "slot_index" not in recorded_args


def test_chat_intake_pipeline_populates_constraints_end_to_end(monkeypatch, mocker) -> None:
    """Plan 04-06 end-to-end: a slot-structured /chat message flows through
    the hybrid intake pipeline (pre-check + structured-output intake LLM)
    and reaches the real graph with state.constraints.requested_primary_types
    populated before graph.ainvoke fires.

    This is the production-/chat-layer proof that the intake pipeline
    plumbs through the FastAPI request stack. The scripted intake LLM
    returns the per-slot Title-Case primary_types so we can assert exactly
    what the validation layer accepted.
    """
    from typing import Any as _Any

    from app.agent.input_parsing import SlotExtractionResult

    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [
            PlaceHit(
                place_id="ChIJtest_p1_aaaaaaaa",
                name="Sushi Spot",
                source="google_places",
                similarity=0.9,
                latitude=37.77,
                longitude=-122.41,
                rating=4.6,
                price_level="PRICE_LEVEL_MODERATE",
                business_status="OPERATIONAL",
                primary_type="Sushi Restaurant",
                formatted_address="123 Mission St, San Francisco",
                snippet=None,
            )
        ],
    )

    # Build a fake intake LLM that quacks for the production hybrid
    # pipeline: .bind(...).with_structured_output(...).ainvoke(...).
    class _Structured:
        async def ainvoke(self, prompt: str, *args: _Any, **kwargs: _Any):
            return SlotExtractionResult(
                requested_primary_types=[
                    "Sushi Restaurant",
                    "Cocktail Bar",
                    "Dessert Shop",
                ]
            )

    class _Bound:
        def with_structured_output(self, *args: _Any, **kwargs: _Any) -> _Structured:
            return _Structured()

    class _IntakeLLM:
        def bind(self, **kwargs: _Any) -> _Bound:
            return _Bound()

    # Pass a graph script that does a single semantic_search + commit_itinerary.
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "id": "ChIJtest_s1_aaaaaaaa",
                    "args": {"query": "omakase", "slot_index": 0},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "commit_itinerary",
                    "id": "ChIJtest_c1_aaaaaaaa",
                    "args": {
                        "stops": [
                            {
                                "place_id": "ChIJtest_p1_aaaaaaaa",
                                "name": "Sushi Spot",
                                "rationale": "Sushi Restaurant in Mission",
                                "source": "google_places",
                                "primary_type": "Sushi Restaurant",
                            }
                        ]
                    },
                }
            ],
        ),
        AIMessage(content="Try Sushi Spot.", tool_calls=[]),
    ]
    graph_llm = ScriptedLLM(scripted=list(scripted))
    real_graph = build_agent_graph(graph_llm, max_steps=4)

    captured: dict[str, _Any] = {}
    real_ainvoke = real_graph.ainvoke

    async def _capturing_ainvoke(state, **kw):
        captured["state_in"] = state
        return await real_ainvoke(state, **kw)

    real_graph.ainvoke = _capturing_ainvoke  # type: ignore[assignment]

    # The intake pipeline reads app.state.agent_llm (set in lifespan from
    # loaded.llm). Override the loaded config so loaded.llm is the intake
    # fake.
    cfg = _stub_loaded_config()
    cfg.llm = type("ChatOpenAI", (_IntakeLLM,), {})()
    mocker.patch("app.main.load_registered_rag_chain", return_value=cfg)
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "omakase, drinks, dessert in Mission"},
        )

    assert response.status_code == 200
    # The state the graph received carries the populated Title-Case list.
    state_in = captured["state_in"]
    assert state_in.constraints.requested_primary_types == [
        "Sushi Restaurant",
        "Cocktail Bar",
        "Dessert Shop",
    ]


# ─── Phase 6 / 06-01 — ConversationState.committed_stops round-trip ───
#
# Per D-06-01 / D-06-02: ConversationState carries `committed_stops: list[Stop]`
# defaulted to [], and `_build_outbound_state` stamps it from the post-graph
# stops list. The frontend treats `conversation_state` as opaque, so this
# field's only job is to round-trip the prior turn's committed plan into the
# next /chat call (so the refinement injection block in plan 06-05 has
# structured ground truth, and the eval runner in 06-06 has something to
# thread).


class TestConversationStateCommittedStopsRoundTrip:
    """Functional proof that `committed_stops` survives the /chat round-trip.

    Per memory `project_full_suite_db_pool_contamination.md`, every test in
    this class stubs `app.agent.revision.itinerary_violations` so no live DB
    pool is activated during full-suite collection. Every `Stop` fixture
    uses a Google-Place-ID-conforming `place_id` (>= 20 chars, alphanumeric
    + underscore + dash) so it passes the Task-3 model-boundary validator.
    """

    def test_committed_stops_stamped_on_outbound_response(self, monkeypatch, mocker) -> None:
        """The /chat response's conversation_state.committed_stops is non-empty
        and mirrors the committed plan (place_id-equal to the response.places)."""
        monkeypatch.setattr(
            "app.agent.tools._semantic_search",
            lambda **_kw: [
                PlaceHit(
                    place_id="ChIJtest_round_trip_aaaaaaaa",
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
                        "id": "ChIJtest_s1_aaaaaaaa",
                        "args": {"query": "cocktail bar"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "commit_itinerary",
                        "id": "ChIJtest_c1_aaaaaaaa",
                        "args": {
                            "stops": [
                                {
                                    "place_id": "ChIJtest_round_trip_aaaaaaaa",
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
        real_graph = build_agent_graph(ScriptedLLM(scripted=list(scripted)), max_steps=4)

        mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
        mocker.patch("app.main.build_agent_graph", return_value=real_graph)
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "cocktail bar in SF"})

        assert response.status_code == 200
        body = response.json()
        assert body["conversation_state"] is not None
        committed = body["conversation_state"]["committed_stops"]
        assert isinstance(committed, list)
        assert len(committed) == 1
        assert committed[0]["place_id"] == "ChIJtest_round_trip_aaaaaaaa"
        # place_id-equal to the response.places (same source of truth).
        assert [c["place_id"] for c in committed] == [p["place_id"] for p in body["places"]]

    def test_committed_stops_round_trips_through_model_validate(self, monkeypatch, mocker) -> None:
        """The outbound conversation_state dict can be fed back as the next
        ChatRequest body and ConversationState.model_validate decodes
        committed_stops element-by-element (place_id-equal)."""
        from app.main import ChatRequest, ConversationState

        monkeypatch.setattr(
            "app.agent.tools._semantic_search",
            lambda **_kw: [
                PlaceHit(
                    place_id="ChIJtest_round_trip_aaaaaaaa",
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
                        "id": "ChIJtest_s1_aaaaaaaa",
                        "args": {"query": "cocktail bar"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "commit_itinerary",
                        "id": "ChIJtest_c1_aaaaaaaa",
                        "args": {
                            "stops": [
                                {
                                    "place_id": "ChIJtest_round_trip_aaaaaaaa",
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
        real_graph = build_agent_graph(ScriptedLLM(scripted=list(scripted)), max_steps=4)

        mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
        mocker.patch("app.main.build_agent_graph", return_value=real_graph)
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "cocktail bar in SF"})

        assert response.status_code == 200
        outbound = response.json()["conversation_state"]

        # Simulate the frontend echoing conversation_state on the next /chat call.
        req = ChatRequest(message="next turn", conversation_state=outbound)
        assert req.conversation_state is not None
        decoded = ConversationState.model_validate(req.conversation_state)
        outbound_committed = outbound["committed_stops"]
        assert len(decoded.committed_stops) == len(outbound_committed)
        for incoming, sent in zip(decoded.committed_stops, outbound_committed, strict=True):
            assert incoming.place_id == sent["place_id"]

    def test_legacy_conversation_state_payload_without_committed_stops_still_decodes(
        self, monkeypatch, mocker
    ) -> None:
        """A legacy payload that omits `committed_stops` (pre-Phase-6 client)
        still decodes via Pydantic's default_factory=list — no 422, and the
        decoded ConversationState.committed_stops is []."""
        from app.main import ConversationState

        monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])
        scripted = [
            AIMessage(content="No matches.", tool_calls=[]),
        ]
        real_graph = build_agent_graph(ScriptedLLM(scripted=list(scripted)), max_steps=2)

        mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
        mocker.patch("app.main.build_agent_graph", return_value=real_graph)
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

        legacy_payload = {
            "schema_version": 1,
            "closure_context": [],
            "prior_stops": [],
        }
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"message": "anything", "conversation_state": legacy_payload},
            )

        # The handler accepts the legacy payload (no 422).
        assert response.status_code == 200
        # Pydantic decoded the legacy payload with committed_stops defaulted to [].
        decoded = ConversationState.model_validate(legacy_payload)
        assert decoded.committed_stops == []
