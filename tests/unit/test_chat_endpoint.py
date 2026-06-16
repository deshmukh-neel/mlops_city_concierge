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
                    "place_id": "ChIJtest_p1_aaaaaaaa",
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
    assert set(body.keys()) == {"reply", "places", "ragLabel", "conversation_state"}
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
        "latitude",
        "longitude",
        "arrival_time",
        "rationale",
        "booking_url",
        "booking_provider",
    }
    assert set(body["places"][0].keys()) == expected_place_keys
    assert body["places"][0]["place_id"] == "ChIJtest_p1_aaaaaaaa"


def test_chat_endpoint_schedules_query_log_with_captured_slots(mocker) -> None:
    """Proves that chat() schedules log_user_query exactly once with the D-02 captured slots.

    Installs a LOCAL spy via mocker.patch (overrides the autouse conftest no-op),
    so this test can assert the exact kwargs BackgroundTasks passed — without
    touching the real DB pool.  TestClient runs BackgroundTasks synchronously
    before client.post() returns, so the spy is guaranteed called by assertion time.
    """
    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        return _final_state_dict(reply="Sure, here is your plan.")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    # Override the autouse conftest no-op with a real spy for this test.
    spy = mocker.patch("app.main.log_user_query")

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "find me a great taco spot", "history": []},
        )

    assert response.status_code == 200

    # BackgroundTasks runs synchronously under TestClient — spy must have been called.
    spy.assert_called_once_with(
        message="find me a great taco spot",
        # Free-text message → has_slot_structure returns False → extracted_types = []
        requested_primary_types=[],
        # No explicit stop count in message → None
        num_stops=None,
        rag_label="openai:gpt-4o-mini",
        # WR-03: session_id is now threaded from the in-scope trace_id. In the
        # test env LANGFUSE_SECRET_KEY is unset so trace_request yields None
        # (session_id=None), but the value is non-deterministic when a real
        # Langfuse client exists — match with ANY to pin the kwarg, not the value.
        session_id=mocker.ANY,
    )


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
    assert state.constraints.num_stops == 4


def test_chat_endpoint_parses_explicit_stop_count(mocker) -> None:
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
                "message": (
                    "plan a 3-stop omakase date night near Japantown SF: drinks, dinner, dessert"
                )
            },
        )

    assert response.status_code == 200
    assert captured["state"].constraints.num_stops == 3


def test_chat_endpoint_preserves_explicit_stop_count_across_turns(mocker) -> None:
    """Multi-turn guardrail: turn 1 says "3 stops", turn 2 refines without
    naming a count. /chat is stateless — every POST rebuilds ItineraryState
    from scratch — so num_stops must be parsed across req.history + req.message,
    not only the current message, or the deterministic count guardrail is lost
    on every follow-up turn."""
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
                "message": "make the second one cheaper",
                "history": [
                    {"role": "user", "content": "plan a 3-stop omakase date night"},
                    {"role": "assistant", "content": "Here's your itinerary: ..."},
                ],
            },
        )

    assert response.status_code == 200
    # The current message has no stop count; the count must come from history.
    assert captured["state"].constraints.num_stops == 3


def test_chat_endpoint_current_message_count_wins_over_history(mocker) -> None:
    """If the user revises the count mid-conversation, the latest explicit
    count wins (e.g. "actually make it 4")."""
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
                    {"role": "user", "content": "plan a 3-stop omakase date night"},
                    {"role": "assistant", "content": "Here's your itinerary: ..."},
                ],
            },
        )

    assert response.status_code == 200
    assert captured["state"].constraints.num_stops == 4


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


# ─── conversation_state round-trip (Task 14) ─────────────────────────────


def test_chat_endpoint_accepts_conversation_state(mocker) -> None:
    """An inbound conversation_state must hydrate into ItineraryState.closure_context."""
    fake_graph = mocker.Mock()
    captured: dict = {}

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
                "message": "make stop 2 cheaper",
                "history": [
                    {"role": "user", "content": "plan a 3-stop date"},
                    {"role": "assistant", "content": "Here's your itinerary..."},
                ],
                "conversation_state": {
                    "schema_version": 1,
                    "closure_context": [
                        {
                            "schema_version": 1,
                            "place_id": "ChIJtest_closed_aaaa",
                            "place_name": "Mochill",
                            "family": "dessert",
                            "attempted_arrival": "2026-05-19T20:02:00-07:00",
                            "outcome": "auto_swapped",
                            "insert_after_place_id": "ChIJtest_prev_aaaaaa",
                            "insert_before_place_id": None,
                            "stop_index_hint": 2,
                            "proposed_alternative": None,
                            "proposed_distance_m": None,
                        }
                    ],
                    "prior_stops": [],
                },
            },
        )

    assert response.status_code == 200
    state = captured["state"]
    assert len(state.closure_context) == 1
    assert state.closure_context[0].place_id == "ChIJtest_closed_aaaa"
    assert state.closure_context[0].outcome == "auto_swapped"


def test_chat_endpoint_returns_conversation_state(mocker) -> None:
    """Final state's closure_context must be echoed in the response."""
    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        d = _final_state_dict(reply="ok")
        d["closure_context"] = [
            {
                "schema_version": 1,
                "place_id": "ChIJtest_closed_aaaa",
                "place_name": "Mochill",
                "family": "dessert",
                "attempted_arrival": "2026-05-19T20:02:00-07:00",
                "outcome": "auto_swapped",
                "insert_after_place_id": None,
                "insert_before_place_id": None,
                "stop_index_hint": 2,
                "proposed_alternative": None,
                "proposed_distance_m": None,
            }
        ]
        return d

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "plan a date", "history": []},
        )

    assert response.status_code == 200
    body = response.json()
    assert "conversation_state" in body
    cs = body["conversation_state"]
    assert cs["schema_version"] == 1
    assert len(cs["closure_context"]) == 1
    assert cs["closure_context"][0]["place_id"] == "ChIJtest_closed_aaaa"


def test_chat_endpoint_degrades_on_malformed_conversation_state(mocker) -> None:
    """A malformed conversation_state must not 422 — the handler logs and
    falls back to empty state."""
    fake_graph = mocker.Mock()
    captured: dict = {}

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
                "message": "anything",
                "conversation_state": {"schema_version": 1, "closure_context": "not-a-list"},
            },
        )
    # Degrades silently -> 200, no closure_context hydrated.
    assert response.status_code == 200
    assert captured["state"].closure_context == []


def test_chat_endpoint_first_turn_omits_conversation_state(mocker) -> None:
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
    body = response.json()
    # Backend always emits a typed conversation_state, never null
    assert "conversation_state" in body
    assert body["conversation_state"]["schema_version"] == 1
    assert body["conversation_state"]["closure_context"] == []


# ─── Accept / decline / alternative early-return (Task 15) ───────────────


def _pending_state(
    place_id: str = "ChIJtest_closed_aaaa",
    family: str = "dessert",
    proposed_id: str = "ChIJtest_sophies_aaa",
    prior_stop_id: str = "ChIJtest_stop1_aaaaa",
) -> dict:
    return {
        "schema_version": 1,
        "closure_context": [
            {
                "schema_version": 1,
                "place_id": place_id,
                "place_name": "Mochill",
                "family": family,
                "attempted_arrival": "2026-05-19T20:02:00-07:00",
                "outcome": "pending_user_decision",
                "insert_after_place_id": prior_stop_id,
                "insert_before_place_id": None,
                "stop_index_hint": 2,
                "proposed_alternative": {
                    "place_id": proposed_id,
                    "name": "Sophie's Crepes",
                    "rationale": "closest open dessert",
                    "source": "google_places",
                    "latitude": 37.7849,
                    "longitude": -122.4093,
                    "primary_type": "Dessert Shop",
                    "arrival_time": "2026-05-19T20:02:00-07:00",
                    "planned_duration_min": 30,
                },
                "proposed_distance_m": 4800.0,
            }
        ],
        "prior_stops": [
            {
                "place_id": prior_stop_id,
                "name": "Stop 1",
                "rationale": "ChIJtest_anchor_aaaa",
                "source": "google_places",
                "latitude": 37.78,
                "longitude": -122.41,
                "primary_type": "Bar",
                "arrival_time": "2026-05-19T18:00:00-07:00",
                "planned_duration_min": 60,
            },
            {
                "place_id": "ChIJtest_stop2_aaaaa",
                "name": "Stop 2",
                "rationale": "ChIJtest_anchor_aaaa",
                "source": "google_places",
                "latitude": 37.785,
                "longitude": -122.41,
                "primary_type": "Restaurant",
                "arrival_time": "2026-05-19T19:00:00-07:00",
                "planned_duration_min": 90,
            },
        ],
    }


def test_chat_endpoint_accept_path_inserts_proposed_alternative(mocker) -> None:
    """User replies "yes" → proposed_alternative inserted, graph NOT
    invoked, response carries 3 stops + user_accepted_drive outcome."""
    fake_graph = mocker.Mock()
    fake_graph.ainvoke = mocker.AsyncMock(
        side_effect=AssertionError("graph should not run on accept path")
    )
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)
    # Re-validation of proposed_alternative: re-fetch details + re-check open
    from app.tools.retrieval import PlaceDetails

    mocker.patch(
        "app.main.get_details",
        return_value=PlaceDetails(
            place_id="ChIJtest_sophies_aaa",
            name="Sophie's Crepes",
            source="google_places",
            similarity=0.0,
            latitude=37.7849,
            longitude=-122.4093,
            primary_type="Dessert Shop",
            formatted_address="123 Fillmore",
            regular_opening_hours={
                "periods": [{"open": {"day": 2, "hour": 10}, "close": {"day": 2, "hour": 22}}]
            },
        ),
    )
    mocker.patch("app.main._place_is_open_now", return_value=True)
    mocker.patch("app.main._per_stop_closure_status", return_value=[False, False, False])
    mocker.patch("app.main._bounded_retime_after_swap", side_effect=lambda stops: stops)
    mocker.patch("app.main.enrich_stops_with_booking", return_value=None)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "yes",
                "history": [
                    {"role": "user", "content": "plan a date"},
                    {"role": "assistant", "content": "The closest open dessert..."},
                ],
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    place_ids = [p["place_id"] for p in body["places"]]
    assert "ChIJtest_sophies_aaa" in place_ids
    cs = body["conversation_state"]
    outcomes = [c["outcome"] for c in cs["closure_context"]]
    assert "user_accepted_drive" in outcomes


def test_chat_endpoint_decline_path_drops_closed_stop(mocker) -> None:
    """User replies "no" → closed stop marked declined, no graph invocation."""
    fake_graph = mocker.Mock()
    fake_graph.ainvoke = mocker.AsyncMock(
        side_effect=AssertionError("graph should not run on decline path")
    )
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "no thanks",
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    cs = body["conversation_state"]
    outcomes = [c["outcome"] for c in cs["closure_context"]]
    assert "user_declined_dropped" in outcomes


def test_chat_endpoint_alternative_path_falls_through_to_graph(mocker) -> None:
    """User replies "find something cheaper" → graph IS invoked with a
    HumanMessage hint about the declined drive option."""
    fake_graph = mocker.Mock()
    captured: dict = {}

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
                "message": "find something cheaper instead",
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    state = captured["state"]
    # The last human message should reference the closed place name OR the
    # user's free-text guidance.
    last_human = next((m for m in reversed(state.messages) if m.type == "human"), None)
    assert last_human is not None
    combined_humans = " ".join(
        m.content for m in state.messages if m.type == "human" and isinstance(m.content, str)
    )
    assert "Mochill" in combined_humans or "find something cheaper" in combined_humans


# ─── WR-01: early-return paths must NOT schedule a demand-query log ──────
#
# These pin the behavioral contract documented at app/main.py: the
# _try_accept_path / _decline_path early-returns exit before
# background_tasks.add_task(log_user_query, ...) is reached, so closure
# replies are never logged. Both tests install a LOCAL spy that OVERRIDES the
# autouse `_neutralize_query_log` no-op (tests/conftest.py) — the no-op alone
# cannot detect a regression, since it swallows every call. With a real spy,
# `assert_not_called()` fails loudly if a future refactor hoists add_task above
# the early-return branches (which would also log with extracted_types /
# num_stops undefined). Same override pattern as
# test_chat_endpoint_schedules_query_log_with_captured_slots.


def test_chat_endpoint_accept_path_does_not_log(mocker) -> None:
    """Accept early-return exits before add_task → log_user_query NOT scheduled."""
    fake_graph = mocker.Mock()
    fake_graph.ainvoke = mocker.AsyncMock(
        side_effect=AssertionError("graph should not run on accept path")
    )
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)
    # Re-validation of proposed_alternative: re-fetch details + re-check open
    # (same setup as test_chat_endpoint_accept_path_inserts_proposed_alternative,
    # so the accept path early-returns rather than escalating to the graph).
    from app.tools.retrieval import PlaceDetails

    mocker.patch(
        "app.main.get_details",
        return_value=PlaceDetails(
            place_id="ChIJtest_sophies_aaa",
            name="Sophie's Crepes",
            source="google_places",
            similarity=0.0,
            latitude=37.7849,
            longitude=-122.4093,
            primary_type="Dessert Shop",
            formatted_address="123 Fillmore",
            regular_opening_hours={
                "periods": [{"open": {"day": 2, "hour": 10}, "close": {"day": 2, "hour": 22}}]
            },
        ),
    )
    mocker.patch("app.main._place_is_open_now", return_value=True)
    mocker.patch("app.main._per_stop_closure_status", return_value=[False, False, False])
    mocker.patch("app.main._bounded_retime_after_swap", side_effect=lambda stops: stops)
    mocker.patch("app.main.enrich_stops_with_booking", return_value=None)

    # Override the autouse conftest no-op with a real spy for this test.
    spy = mocker.patch("app.main.log_user_query")

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "yes",
                "history": [
                    {"role": "user", "content": "plan a date"},
                    {"role": "assistant", "content": "The closest open dessert..."},
                ],
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    # Accept is an early return — the demand-query log must never be scheduled.
    spy.assert_not_called()


def test_chat_endpoint_decline_path_does_not_log(mocker) -> None:
    """Decline early-return exits before add_task → log_user_query NOT scheduled."""
    fake_graph = mocker.Mock()
    fake_graph.ainvoke = mocker.AsyncMock(
        side_effect=AssertionError("graph should not run on decline path")
    )
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    # Override the autouse conftest no-op with a real spy for this test.
    spy = mocker.patch("app.main.log_user_query")

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "no thanks",
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    # Decline is an early return — the demand-query log must never be scheduled.
    spy.assert_not_called()


# ─── Phase 4 hybrid intake pipeline (D-04-01..D-04-03) ─────────────────


def _make_intake_llm(extraction_result, observed: dict[str, Any] | None = None) -> Any:
    """Fake LLM with .bind/.with_structured_output/.ainvoke that returns the
    provided SlotExtractionResult (or raises if it's an Exception)."""
    observed = observed if observed is not None else {}
    observed.setdefault("bind_calls", [])
    observed.setdefault("wso_call_count", 0)
    observed.setdefault("ainvoke_call_count", 0)

    class _Structured:
        async def ainvoke(self, prompt: str, *args: Any, **kwargs: Any):
            observed["ainvoke_call_count"] += 1
            if isinstance(extraction_result, Exception):
                raise extraction_result
            return extraction_result

    class _Bound:
        def with_structured_output(self, *args: Any, **kwargs: Any) -> _Structured:
            observed["wso_call_count"] += 1
            return _Structured()

    class _LLM:
        def bind(self, **kwargs: Any) -> _Bound:
            observed["bind_calls"].append(dict(kwargs))
            return _Bound()

    return type("ChatOpenAI", (_LLM,), {})()


def test_chat_free_text_skips_intake(mocker) -> None:
    """Free-text /chat → has_slot_structure=False; no bind, no
    with_structured_output, no LLM ainvoke. Zero-latency-tax invariant."""
    observed: dict[str, Any] = {}
    fake_llm = _make_intake_llm(Exception("should never be invoked on free-text"), observed)
    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    cfg = _stub_loaded_config(mocker.Mock())
    cfg.llm = fake_llm
    mocker.patch("app.main.load_registered_rag_chain", return_value=cfg)
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "find me good tacos"},
        )

    assert response.status_code == 200
    assert observed.get("bind_calls") == []
    assert observed.get("wso_call_count") == 0
    assert observed.get("ainvoke_call_count") == 0


def test_chat_slot_structured_triggers_intake(mocker) -> None:
    """Slot-structured /chat → intake fires once; the validated list reaches
    state.constraints.requested_primary_types before graph.ainvoke."""
    observed: dict[str, Any] = {}
    from app.agent.input_parsing import SlotExtractionResult

    fake_llm = _make_intake_llm(
        SlotExtractionResult(
            requested_primary_types=["Sushi Restaurant", "Cocktail Bar", "Dessert Shop"]
        ),
        observed,
    )
    fake_graph = mocker.Mock()
    captured: dict[str, Any] = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    cfg = _stub_loaded_config(mocker.Mock())
    cfg.llm = fake_llm
    mocker.patch("app.main.load_registered_rag_chain", return_value=cfg)
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "omakase, drinks, dessert in Mission"},
        )

    assert response.status_code == 200
    assert observed["bind_calls"], "intake bind never invoked on slot-structured input"
    assert observed["ainvoke_call_count"] == 1
    state = captured["state"]
    assert state.constraints.requested_primary_types == [
        "Sushi Restaurant",
        "Cocktail Bar",
        "Dessert Shop",
    ]


def test_chat_intake_exception_fails_open(mocker) -> None:
    """Intake LLM raises → handler logs warning and falls back to
    requested_primary_types=[]; the request still 200s (D-04-03 fail-open)."""
    observed: dict[str, Any] = {}
    fake_llm = _make_intake_llm(RuntimeError("intake provider down"), observed)
    fake_graph = mocker.Mock()
    captured: dict[str, Any] = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    cfg = _stub_loaded_config(mocker.Mock())
    cfg.llm = fake_llm
    mocker.patch("app.main.load_registered_rag_chain", return_value=cfg)
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "dinner, drinks, dessert"},
        )

    assert response.status_code == 200
    assert observed["ainvoke_call_count"] == 1  # tried once before raising
    assert captured["state"].constraints.requested_primary_types == []


def test_chat_endpoint_accept_escalates_when_proposed_alternative_missing(mocker) -> None:
    """Accept path with proposed_alternative no longer in DB → graph runs
    (not early-return) with the closure_context preserved."""
    fake_graph = mocker.Mock()
    captured: dict = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict(reply="ok")

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain", return_value=_stub_loaded_config(mocker.Mock())
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)
    mocker.patch("app.main.get_details", return_value=None)  # gone from DB

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "yes",
                "conversation_state": _pending_state(),
            },
        )

    assert response.status_code == 200
    # Graph WAS invoked -> not an early return
    assert "state" in captured
