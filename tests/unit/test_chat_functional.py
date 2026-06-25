"""Functional tests that exercise the /chat endpoint with a *real* agent graph.

Only the LLM and the Postgres-backed retrieval functions are stubbed. The
graph, tool wrapping, message handling, and FastAPI plumbing all run for real
so we catch shape mismatches that pure unit tests miss.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent_graph
from app.main import ActiveModelConfig, LoadedConfig, app
from app.tools.directions import DirectionsLeg, DirectionsResult
from app.tools.retrieval import PlaceHit
from tests._helpers.scripted_llm import RecordingScriptedLLM, ScriptedLLM


def stub_loaded_config() -> LoadedConfig:
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
        "app.agent.tools.semantic_search_impl",
        lambda **unused_kwargs: [
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

    mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
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
    monkeypatch.setattr("app.agent.tools.semantic_search_impl", lambda **unused_kwargs: [])

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

    mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["places"] == []


T0 = datetime(2024, 6, 1, 18, 0, 0, tzinfo=timezone.utc)  # 6pm UTC anchor for tests


def two_stop_script() -> list[AIMessage]:
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
                                "arrival_time": T0.isoformat(),
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


def two_hits(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.tools.semantic_search_impl",
        lambda **unused_kwargs: [
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
    two_hits(monkeypatch)

    async def slow_directions(stops, mode="walk"):
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=2400, distance_m=3000.0)],
            total_duration_s=2400,
            mode=mode,
            source="google",
        )

    mocker.patch("app.agent.graph.route_legs", slow_directions)
    # Closure detection moved to the swap node; stub it "all open" so retime's
    # arrival-time projection is the only thing under test.
    mocker.patch(
        "app.agent.swap.per_stop_closure_status",
        side_effect=lambda stops: [False] * len(stops),
    )
    mocker.patch("app.agent.revision.itinerary_violations", lambda stop_obj: [])

    real_graph = build_agent_graph(ScriptedLLM(scripted=two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
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
    two_hits(monkeypatch)

    async def fallback_directions(stops, mode="walk"):
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=120, distance_m=160.0)],
            total_duration_s=120,
            mode=mode,
            source="haversine_fallback",
        )

    mocker.patch("app.agent.graph.route_legs", fallback_directions)
    # Closure detection moved to the swap node; "all open" -> no rewrite,
    # reply is the synthesized summary.
    mocker.patch(
        "app.agent.swap.per_stop_closure_status",
        side_effect=lambda stops: [False] * len(stops),
    )
    mocker.patch("app.agent.revision.itinerary_violations", lambda stop_obj: [])

    real_graph = build_agent_graph(ScriptedLLM(scripted=two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
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
        "app.agent.tools.semantic_search_impl",
        lambda **unused_kwargs: [
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
    from app.agent.state import UserConstraints as RealUserConstraints

    def make_constraints(**kwargs):
        kwargs["requested_primary_types"] = ["Sushi Restaurant"]
        return RealUserConstraints(**kwargs)

    monkeypatch.setattr("app.main.UserConstraints", make_constraints)

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

    async def capturing_ainvoke(state, **kw):
        result = await real_ainvoke(state, **kw)
        captured_scratch["scratch"] = (
            result.scratch if hasattr(result, "scratch") else result.get("scratch")
        )
        return result

    real_graph.ainvoke = capturing_ainvoke  # type: ignore[assignment]

    mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
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
    from typing import Any as AnyAlias

    from app.agent.input_parsing import SlotExtractionResult

    monkeypatch.setattr(
        "app.agent.tools.semantic_search_impl",
        lambda **unused_kwargs: [
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
    class Structured:
        async def ainvoke(self, prompt: str, *args: AnyAlias, **kwargs: AnyAlias):
            return SlotExtractionResult(
                requested_primary_types=[
                    "Sushi Restaurant",
                    "Cocktail Bar",
                    "Dessert Shop",
                ]
            )

    class Bound:
        def with_structured_output(self, *args: AnyAlias, **kwargs: AnyAlias) -> Structured:
            return Structured()

    class IntakeLLM:
        def bind(self, **kwargs: AnyAlias) -> Bound:
            return Bound()

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

    captured: dict[str, AnyAlias] = {}
    real_ainvoke = real_graph.ainvoke

    async def capturing_ainvoke(state, **kw):
        captured["state_in"] = state
        return await real_ainvoke(state, **kw)

    real_graph.ainvoke = capturing_ainvoke  # type: ignore[assignment]

    # The intake pipeline reads app.state.agent_llm (set in lifespan from
    # loaded.llm). Override the loaded config so loaded.llm is the intake
    # fake.
    cfg = stub_loaded_config()
    cfg.llm = type("ChatOpenAI", (IntakeLLM,), {})()
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
# defaulted to [], and `build_outbound_state` stamps it from the post-graph
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
            "app.agent.tools.semantic_search_impl",
            lambda **unused_kwargs: [
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

        mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
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
            "app.agent.tools.semantic_search_impl",
            lambda **unused_kwargs: [
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

        mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
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

        monkeypatch.setattr("app.agent.tools.semantic_search_impl", lambda **unused_kwargs: [])
        scripted = [
            AIMessage(content="No matches.", tool_calls=[]),
        ]
        real_graph = build_agent_graph(ScriptedLLM(scripted=list(scripted)), max_steps=2)

        mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
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


# ─── Phase 6 / 06-05 — /chat refinement injection truth table ───
#
# 3 binary dimensions (REFINEMENT_STRUCTURED_PLAN_ENABLED × refinement-regex
# match × committed_stops non-empty) = 8 cases. Only the all-True cell
# (flag ON + regex match + committed_stops non-empty) actually injects the
# structured-plan HumanMessage built by `app.agent.io.build_refinement_prompt_message`.
# The other 7 cells must prove the absence of injection.
#
# A 9th case (Residual-2 fix) asserts that a malformed conversation_state
# (e.g. an inbound stop with a `place_id` that fails the plan-06-01 Task-3
# validator) degrades to empty `committed_stops` and returns 200 — NOT 422 —
# because `app/main.py` catches `ValidationError` and falls back to an empty
# ConversationState, which the three-way guard short-circuits on.


class TestChatRefinementInjection:
    """Phase 6 plan 06-05 Task 2b — full 8-cell truth-table for the /chat
    refinement injection branch.

    Per `project_full_suite_db_pool_contamination.md` every test in this
    class stubs `app.agent.revision.itinerary_violations` so no live DB
    pool activates during full-suite collection.

    Every Stop fixture uses a Google-Place-ID-conforming `place_id` per
    plan 06-01 Task 3 validator (`^[A-Za-z0-9_-]{20,255}$`); the canonical
    value is `"ChIJtest_fixture_id_aaaaaa"` (26 chars).
    """

    # Canonical fixture used by every test in this class.
    CANON_PLACE_ID = "ChIJtest_fixture_id_aaaaaa"
    CANON_PLACE_ID_2 = "ChIJtest_fixture_id_bbbbbb"
    # Phase 7 plan 07-06 (PROMPT-01) — the replacement slot-2 place_id the
    # scripted refinement commit emits. Must match `^[A-Za-z0-9_-]{20,255}$`
    # per the plan-06-01 Task-3 validator (PATTERNS.md "Place_id fixture
    # convention"). 28 chars.
    NEW_SLOT2_PLACE_ID = "ChIJtest_fixture_NEW2_xxxxxx"

    @classmethod
    def committed_stops_payload(cls) -> list[dict]:
        """Build a 3-stop committed_stops payload (matches "stop 2" target_slot
        bounds: 1 <= 2 <= 3 passes the MEDIUM target_slot-bounds guard)."""
        return [
            {
                "place_id": cls.CANON_PLACE_ID,
                "name": "Stop One",
                "rationale": "first",
                "source": "google_places",
            },
            {
                "place_id": cls.CANON_PLACE_ID_2,
                "name": "Stop Two",
                "rationale": "second",
                "source": "google_places",
            },
            {
                "place_id": "ChIJtest_fixture_id_cccccc",
                "name": "Stop Three",
                "rationale": "third",
                "source": "google_places",
            },
        ]

    @staticmethod
    def make_recording_llm() -> RecordingScriptedLLM:
        """A `RecordingScriptedLLM` that ends the graph immediately (no
        tool calls) so we can inspect its `seen[0]` — what the LLM was
        prompted with on its first invocation — without running through
        a full retrieval/commit trajectory."""
        return RecordingScriptedLLM(
            scripted=[
                AIMessage(content="(stub final reply)", tool_calls=[]),
            ]
        )

    @classmethod
    def post_chat(
        cls,
        *,
        mocker,
        monkeypatch,
        message: str,
        conversation_state: dict | None,
        recording_llm: RecordingScriptedLLM,
    ):
        """Drive a single POST /chat round-trip with the supplied scripted LLM."""
        monkeypatch.setattr("app.agent.tools.semantic_search_impl", lambda **unused_kwargs: [])
        real_graph = build_agent_graph(recording_llm, max_steps=2)
        mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
        mocker.patch("app.main.build_agent_graph", return_value=real_graph)
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        with TestClient(app) as client:
            body: dict = {"message": message}
            if conversation_state is not None:
                body["conversation_state"] = conversation_state
            return client.post("/chat", json=body)

    # The unique sentinel for an injected refinement message — the helper's
    # preamble starts with this exact prose, and the prose does NOT appear in
    # the SYSTEM_PROMPT itself, so it is a clean "is this an injection?"
    # signal. (The earlier sentinel `"current_plan"` is ambiguous because
    # the SYSTEM_PROMPT's addendum names the JSON field by name when telling
    # the model how to read the structured plan.)
    INJECTION_SENTINEL = "REFINEMENT TURN"

    @staticmethod
    def human_content_strings_seen(recording_llm) -> list[str]:
        """Every HumanMessage.content across every invocation the LLM saw.

        Filters to HumanMessage so the SYSTEM_PROMPT's prose addendum (which
        mentions `current_plan` to teach the model how to read the
        structured plan) doesn't appear as a false positive."""
        out: list[str] = []
        for messages_list in recording_llm.seen:
            for m in messages_list:
                if isinstance(m, HumanMessage) and isinstance(m.content, str):
                    out.append(m.content)
        return out

    def assert_no_injection(self, recording_llm) -> None:
        """No HumanMessage anywhere in the seen sequence contains the
        helper's preamble sentinel. We assert across the FULL message list
        (not just position 0) because per HIGH-3 the structured plan is no
        longer at index 0 — searching the whole HumanMessage list is
        robust to ordering shifts."""
        contents = self.human_content_strings_seen(recording_llm)
        for content in contents:
            assert self.INJECTION_SENTINEL not in content, (
                f"unexpected structured-plan injection: {content!r}"
            )

    # ─── 8 truth-table cells ────────────────────────────────────────────

    def test_flag_off_refinement_message_committed_stops_present_no_injection(
        self, monkeypatch, mocker
    ) -> None:
        """Cell 1: flag OFF, regex MATCH, committed_stops NON-EMPTY → no inject."""
        monkeypatch.delenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", raising=False)
        recording_llm = self.make_recording_llm()
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="make stop 2 cheaper",
            conversation_state={
                "schema_version": 1,
                "closure_context": [],
                "prior_stops": [],
                "committed_stops": self.committed_stops_payload(),
            },
            recording_llm=recording_llm,
        )
        assert response.status_code == 200
        self.assert_no_injection(recording_llm)

    def test_flag_on_non_refinement_message_committed_stops_present_no_injection(
        self, monkeypatch, mocker
    ) -> None:
        """Cell 2: flag ON, regex NO-MATCH, committed_stops NON-EMPTY → no inject."""
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        recording_llm = self.make_recording_llm()
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="Plan a date night",
            conversation_state={
                "schema_version": 1,
                "closure_context": [],
                "prior_stops": [],
                "committed_stops": self.committed_stops_payload(),
            },
            recording_llm=recording_llm,
        )
        assert response.status_code == 200
        self.assert_no_injection(recording_llm)

    def test_flag_on_refinement_message_committed_stops_empty_no_injection(
        self, monkeypatch, mocker
    ) -> None:
        """Cell 3: flag ON, regex MATCH, committed_stops EMPTY → no inject."""
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        recording_llm = self.make_recording_llm()
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="make stop 2 cheaper",
            conversation_state={
                "schema_version": 1,
                "closure_context": [],
                "prior_stops": [],
                "committed_stops": [],
            },
            recording_llm=recording_llm,
        )
        assert response.status_code == 200
        self.assert_no_injection(recording_llm)

    def test_flag_on_refinement_message_committed_stops_present_injects(
        self, monkeypatch, mocker
    ) -> None:
        """Cell 4: flag ON, regex MATCH, committed_stops NON-EMPTY → INJECT.

        Also pins HIGH-3 adjacency (structured plan is the message
        IMMEDIATELY BEFORE the user's turn-2 HumanMessage in the sequence
        the LLM sees) AND Caveat #5 byte-identity (the injected message
        equals a direct call to `build_refinement_prompt_message` with
        the same committed_stops)."""
        from app.agent.io import build_refinement_prompt_message
        from app.agent.state import Stop

        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        recording_llm = self.make_recording_llm()
        committed_payload = self.committed_stops_payload()
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="make stop 2 cheaper",
            conversation_state={
                "schema_version": 1,
                "closure_context": [],
                "prior_stops": [],
                "committed_stops": committed_payload,
            },
            recording_llm=recording_llm,
        )
        assert response.status_code == 200

        # The first plan() invocation saw the post-injection messages.
        assert recording_llm.seen, "LLM was never invoked"
        first_seen = recording_llm.seen[0]
        # HIGH-3 adjacency: the message IMMEDIATELY BEFORE the user's
        # final HumanMessage (index -2) is the structured-plan injection.
        assert len(first_seen) >= 2, f"expected >= 2 messages, got {len(first_seen)}"
        injected_msg = first_seen[-2]
        user_msg = first_seen[-1]
        assert isinstance(user_msg, HumanMessage)
        assert user_msg.content == "make stop 2 cheaper"
        assert isinstance(injected_msg, HumanMessage)
        assert isinstance(injected_msg.content, str)
        # Both anchors must be present in the actual injected HumanMessage.
        assert self.INJECTION_SENTINEL in injected_msg.content
        assert "current_plan" in injected_msg.content

        # Caveat #5 byte-identity: the /chat-injected message equals a
        # direct call to the shared helper with the same committed_stops.
        expected_stops = [Stop(**s) for s in committed_payload]
        expected_msg = build_refinement_prompt_message(expected_stops)
        assert injected_msg.content == expected_msg.content

    def test_flag_off_non_refinement_first_turn_unchanged(self, monkeypatch, mocker) -> None:
        """Cell 5 (REF-04 parity): flag OFF, regex NO-MATCH, committed_stops
        NON-EMPTY → no inject (the first-turn-style code path is unchanged
        when the flag is off, regardless of payload)."""
        monkeypatch.delenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", raising=False)
        recording_llm = self.make_recording_llm()
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="Plan a date night",
            conversation_state={
                "schema_version": 1,
                "closure_context": [],
                "prior_stops": [],
                "committed_stops": self.committed_stops_payload(),
            },
            recording_llm=recording_llm,
        )
        assert response.status_code == 200
        self.assert_no_injection(recording_llm)

    def test_flag_off_refinement_message_committed_stops_empty_no_injection(
        self, monkeypatch, mocker
    ) -> None:
        """Cell 6: flag OFF, regex MATCH, committed_stops EMPTY → no inject."""
        monkeypatch.delenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", raising=False)
        recording_llm = self.make_recording_llm()
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="make stop 2 cheaper",
            conversation_state=None,
            recording_llm=recording_llm,
        )
        assert response.status_code == 200
        self.assert_no_injection(recording_llm)

    def test_flag_on_non_refinement_message_committed_stops_empty_no_injection(
        self, monkeypatch, mocker
    ) -> None:
        """Cell 7: flag ON, regex NO-MATCH, committed_stops EMPTY → no inject."""
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        recording_llm = self.make_recording_llm()
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="Plan a date night",
            conversation_state=None,
            recording_llm=recording_llm,
        )
        assert response.status_code == 200
        self.assert_no_injection(recording_llm)

    def test_flag_off_non_refinement_message_committed_stops_empty_no_injection(
        self, monkeypatch, mocker
    ) -> None:
        """Cell 8 (turn-1 dominant case): flag OFF, regex NO-MATCH,
        committed_stops EMPTY → no inject."""
        monkeypatch.delenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", raising=False)
        recording_llm = self.make_recording_llm()
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="Plan a date night",
            conversation_state=None,
            recording_llm=recording_llm,
        )
        assert response.status_code == 200
        self.assert_no_injection(recording_llm)

    # ─── Residual-2 fix — malformed conversation_state degrades, not 422 ──

    def test_chat_with_malformed_committed_stops_degrades_gracefully(
        self, monkeypatch, mocker
    ) -> None:
        """Per plan 06-05 Residual-2 fix language: when `conversation_state`
        decodes through the field-level Pydantic validator and one of the
        nested Stop entries fails the plan-06-01 Task-3 `place_id` format
        validator, `app/main.py:660-666` catches `ValidationError` and
        degrades to an empty ConversationState. Because the three-way
        injection guard short-circuits on empty `committed_stops`, the
        structured-plan helper is never built — even though the flag is
        ON and the message matches the refinement regex.

        Asserts:
          (1) HTTP 200 (NOT 422 — the handler caught ValidationError),
          (2) NO message the LLM saw contains `current_plan` (guard short-
              circuited on empty committed_stops after degrade),
          (3) the response body has a non-empty `reply` string (the agent
              ran to completion).
        """
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        recording_llm = self.make_recording_llm()
        malformed_payload = {
            "schema_version": 1,
            "closure_context": [],
            "prior_stops": [],
            # "INVALID short" contains a space → fails the plan-06-01 Task-3
            # `place_id` format validator (`^[A-Za-z0-9_-]{20,255}$`).
            "committed_stops": [
                {
                    "place_id": "INVALID short",
                    "name": "x",
                    "rationale": "r",
                    "source": "google_places",
                }
            ],
        }
        response = self.post_chat(
            mocker=mocker,
            monkeypatch=monkeypatch,
            message="make stop 2 cheaper",
            conversation_state=malformed_payload,
            recording_llm=recording_llm,
        )
        # (1) HTTP 200, not 422.
        assert response.status_code == 200, (
            f"expected 200 degrade-to-empty, got {response.status_code}: {response.text}"
        )
        # (2) Guard short-circuited (empty committed_stops after degrade)
        #     → no `current_plan` anywhere.
        self.assert_no_injection(recording_llm)
        # (3) The reply is a non-empty string (agent ran to completion).
        body = response.json()
        assert isinstance(body.get("reply"), str)
        assert body["reply"], "expected a non-empty reply"

    # ─── Phase 7 / plan 07-06 — PROMPT-01 functional acceptance test ────
    #
    # The Phase 7 prompt rewrite (plan 07-01) deletes the SYSTEM_PROMPT
    # rule that previously told the model to preserve `place_id`s
    # byte-for-byte. PROMPT-01 verifies the user-observable behavior
    # survives that deletion: a /chat refinement turn with
    # REFINEMENT_STRUCTURED_PLAN_ENABLED=on returns a full itinerary where
    # the requested edit is applied (slot 2 place_id differs) AND all
    # other stops are byte-identical to the prior committed plan.
    #
    # Per D-07-11 the runner is `openai/gpt-4o-mini` for determinism =
    # ScriptedLLM. Per PATTERNS.md (and the existing class structure at
    # lines 800-1026) this is the canonical location for /chat functional
    # tests — `tests/integration/` has no chat-endpoint analog.
    #
    # Approach: single-POST refinement turn. The class pattern (every
    # truth-table test above) drives a single POST with pre-populated
    # `committed_stops`, which exercises the same code path as a real
    # turn-1 refinement (handler reads `committed_stops` →
    # `build_refinement_prompt_message` injects → real LangGraph agent
    # runs → `commit_itinerary` tool call → response). The pre-population
    # is the moral equivalent of "turn 0 already happened in a prior
    # conversation".
    #
    # `post_chat` is NOT used here because it hardcodes
    # `semantic_search_impl → []` and `max_steps=2`. PROMPT-01 needs prior-
    # turn place_ids GROUNDED in scratch (the agent's anti-hallucination
    # guard `grounded_place_ids` in `app/agent/commit.py` rejects any
    # `place_id` not seen via a prior tool result) AND enough steps for
    # the agent to issue a semantic_search before `commit_itinerary`.

    def test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied(
        self, monkeypatch, mocker
    ) -> None:
        """PROMPT-01 / D-07-11: a flag-ON refinement turn returns a 3-stop
        itinerary where slots 1 and 3 keep the prior committed `place_id`s
        byte-for-byte and slot 2 carries the new (edit-applied) `place_id`.

        Asserts (per D-07-11 acceptance):
          (1) HTTP 200
          (2) response has same stop count (3) as the prior committed plan
          (3) slot 1 `place_id` == prior slot 1 (byte-equal)
          (4) slot 3 `place_id` == prior slot 3 (byte-equal)
          (5) slot 2 `place_id` == the new `_NEW_SLOT2_PLACE_ID` fixture
          (6) slot 2 `place_id` != prior slot 2 (`_CANON_PLACE_ID_2`) —
              sanity guard that the edit was actually applied, not a no-op.
        """
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")

        # Stub `semantic_search_impl` to return PlaceHits for ALL four place_ids
        # this test references — the three prior committed stops AND the new
        # slot-2 replacement. The agent's `grounded_place_ids` set is built
        # from every PlaceHit observed across scratch entries (commit.py:23-
        # 42), so a single semantic_search whose result list contains all
        # four place_ids grounds the entire post-refinement commit list at
        # once. This keeps the trajectory short (1 search + 1 commit) so
        # `max_steps=4` is plenty.
        monkeypatch.setattr(
            "app.agent.tools.semantic_search_impl",
            lambda **unused_kwargs: [
                PlaceHit(
                    place_id=self.CANON_PLACE_ID,
                    name="Stop One",
                    source="google_places",
                    similarity=0.9,
                    latitude=37.770,
                    longitude=-122.410,
                    business_status="OPERATIONAL",
                    primary_type="cocktail_bar",
                    formatted_address="1 A St, San Francisco",
                    snippet=None,
                ),
                PlaceHit(
                    place_id=self.NEW_SLOT2_PLACE_ID,
                    name="Cheap Stop Two",
                    source="google_places",
                    similarity=0.9,
                    latitude=37.780,
                    longitude=-122.420,
                    business_status="OPERATIONAL",
                    primary_type="cocktail_bar",
                    formatted_address="2 B St, San Francisco",
                    snippet=None,
                ),
                PlaceHit(
                    place_id="ChIJtest_fixture_id_cccccc",
                    name="Stop Three",
                    source="google_places",
                    similarity=0.9,
                    latitude=37.790,
                    longitude=-122.430,
                    business_status="OPERATIONAL",
                    primary_type="cocktail_bar",
                    formatted_address="3 C St, San Francisco",
                    snippet=None,
                ),
            ],
        )

        # Scripted trajectory:
        #   1. semantic_search to ground the four place_ids in scratch
        #   2. commit_itinerary with the post-refinement 3-stop list:
        #      slot 1 = _CANON_PLACE_ID (unchanged), slot 2 = NEW (edited),
        #      slot 3 = "..._cccccc" (unchanged)
        # Per `project_finalize_on_commit_fix`: the graph terminates on a
        # successful commit, so no trailing narration message is needed.
        # A narration AIMessage is included as defense-in-depth (matches
        # the convention used by `test_chat_runs_real_graph_with_tool_call`).
        scripted = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "ChIJtest_s1_promprt_01_aa",
                        "args": {"query": "cheaper cocktail bar SF"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "commit_itinerary",
                        "id": "ChIJtest_c1_prompt_01_aa",
                        "args": {
                            "stops": [
                                {
                                    "place_id": self.CANON_PLACE_ID,
                                    "name": "Stop One",
                                    "rationale": "first stop preserved",
                                    "source": "google_places",
                                    "primary_type": "cocktail_bar",
                                },
                                {
                                    "place_id": self.NEW_SLOT2_PLACE_ID,
                                    "name": "Cheap Stop Two",
                                    "rationale": "cheaper alternative per user edit",
                                    "source": "google_places",
                                    "primary_type": "cocktail_bar",
                                },
                                {
                                    "place_id": "ChIJtest_fixture_id_cccccc",
                                    "name": "Stop Three",
                                    "rationale": "third stop preserved",
                                    "source": "google_places",
                                    "primary_type": "cocktail_bar",
                                },
                            ]
                        },
                    }
                ],
            ),
            AIMessage(content="(stub final reply)", tool_calls=[]),
        ]
        # `max_steps=4` budgets: plan (search) → act (search exec) → plan
        # (commit) → act (commit) → critique → END (finalize-on-commit).
        real_graph = build_agent_graph(ScriptedLLM(scripted=list(scripted)), max_steps=4)

        mocker.patch("app.main.load_registered_rag_chain", return_value=stub_loaded_config())
        mocker.patch("app.main.build_agent_graph", return_value=real_graph)
        # Per `project_full_suite_db_pool_contamination`: itinerary_violations
        # activates the live DB pool in full-suite runs. Stub to [] so the
        # commit path doesn't drive a revision loop (which would exhaust the
        # scripted LLM).
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": "make stop 2 cheaper",
                    "conversation_state": {
                        "schema_version": 1,
                        "closure_context": [],
                        "prior_stops": [],
                        "committed_stops": self.committed_stops_payload(),
                    },
                },
            )

        # (1) HTTP 200.
        assert response.status_code == 200, (
            f"expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        # The /chat response surfaces stops under the `places` key (the
        # frontend-contract name in `ChatResponse.places`, app/main.py:230).
        # PROMPT-01 / D-07-11 calls this "stops" in prose; the wire field
        # is `places`. They are the same list.
        places = data["places"]
        # (2) Same stop count as the prior committed plan.
        assert len(places) == 3, f"expected 3 stops post-refinement, got {len(places)}: {places}"
        # (3) Slot 1 byte-equal to the prior committed plan.
        assert places[0]["place_id"] == self.CANON_PLACE_ID, (
            f"slot 1 place_id changed: expected {self.CANON_PLACE_ID}, got {places[0]['place_id']}"
        )
        # (4) Slot 3 byte-equal to the prior committed plan.
        assert places[2]["place_id"] == "ChIJtest_fixture_id_cccccc", (
            f"slot 3 place_id changed: expected ChIJtest_fixture_id_cccccc, "
            f"got {places[2]['place_id']}"
        )
        # (5) Slot 2 is the new replacement place_id.
        assert places[1]["place_id"] == self.NEW_SLOT2_PLACE_ID, (
            f"slot 2 place_id mismatch: expected {self.NEW_SLOT2_PLACE_ID}, "
            f"got {places[1]['place_id']}"
        )
        # (6) Sanity guard: slot 2 is NOT the prior slot 2 (the edit was
        # actually applied, not a no-op preserving _CANON_PLACE_ID_2).
        assert places[1]["place_id"] != self.CANON_PLACE_ID_2, (
            "slot 2 place_id was not changed by the refinement turn — "
            "the edit was a no-op, violating PROMPT-01"
        )
