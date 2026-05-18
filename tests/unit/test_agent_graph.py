"""Integration of the graph with a fake LLM that emits a scripted sequence
of tool calls. Verifies plan->act->critique loops correctly and terminates."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import psycopg2
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.commit import commit_stops, enrich_stops_with_booking
from app.agent.graph import _prune_for_llm, build_agent_graph
from app.agent.state import ItineraryState, Stop, UserConstraints
from app.tools.booking import BookingProposal
from app.tools.directions import DirectionsLeg, DirectionsResult
from app.tools.retrieval import PlaceDetails, PlaceHit


def _details_for(place_id: str, name: str | None = None) -> PlaceDetails:
    """Minimal PlaceDetails fixture for booking-enrichment tests. Booking
    construction itself is mocked via propose_booking_from_details; only the
    place_id needs to be load-bearing here."""
    return PlaceDetails(
        place_id=place_id,
        name=name or f"Place {place_id}",
        source="google_places",
        similarity=0.0,
    )


def _rich_details_for(place_id: str) -> PlaceDetails:
    """PlaceDetails carrying address/rating/price for card-field tests."""
    return PlaceDetails(
        place_id=place_id,
        name=f"Place {place_id}",
        source="google_places",
        similarity=0.0,
        formatted_address=f"{place_id} Main St, San Francisco",
        rating=4.4,
        price_level="PRICE_LEVEL_MODERATE",
    )


def _patch_details_many_for(place_ids: list[str]):
    """Patch get_details_many to return a minimal PlaceDetails for each id.
    Returns the patcher so tests can also assert call_count if they care."""
    return patch(
        "app.agent.commit.get_details_many",
        return_value={pid: _details_for(pid) for pid in place_ids},
    )


class _ScriptedLLM(BaseChatModel):
    """Test double that returns scripted AIMessages in order."""

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
        if not self.scripted:
            raise RuntimeError("scripted responses exhausted")
        msg = self.scripted.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedLLM:
        return self


def _make_fake(scripted: list[AIMessage]) -> _ScriptedLLM:
    return _ScriptedLLM(scripted=list(scripted))


async def test_graph_terminates_on_no_tool_call() -> None:
    fake = _make_fake([AIMessage(content="here is your plan", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
    assert out["final_reply"] == "here is your plan"


async def test_graph_executes_tool_and_continues(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **kw: [
            PlaceHit(
                place_id="p1",
                name="X",
                source="google_places",
                similarity=0.9,
                latitude=None,
                longitude=None,
                rating=4.5,
                price_level="PRICE_LEVEL_MODERATE",
                business_status="OPERATIONAL",
                primary_type="restaurant",
                formatted_address="123 Main",
                snippet=None,
            )
        ],
    )
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "1",
                        "args": {"query": "italian", "k": 3},
                    }
                ],
            ),
            AIMessage(content="found one place", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="italian please")]))
    assert out["step_count"] == 1
    assert "semantic_search" in out["scratch"]
    assert out["scratch"]["semantic_search"][0]["args"] == {"query": "italian", "k": 3}
    assert out["done"] is True


async def test_graph_respects_max_steps(monkeypatch) -> None:
    looping = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "id": str(i),
                    "args": {"query": "x"},
                }
            ],
        )
        for i in range(20)
    ]
    fake = _make_fake(looping)
    graph = build_agent_graph(fake, max_steps=3)
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="x")]))
    assert out["step_count"] == 3
    assert out["done"] is True
    assert out["final_reply"]


async def test_graph_handles_unknown_tool_name(monkeypatch) -> None:
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "not_a_real_tool",
                        "id": "1",
                        "args": {},
                    }
                ],
            ),
            AIMessage(content="recovered", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    assert out["done"] is True
    assert out["final_reply"] == "recovered"


async def test_graph_records_tool_exception_in_scratch(monkeypatch) -> None:
    def _boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.agent.tools._semantic_search", _boom)

    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "1",
                        "args": {"query": "italian"},
                    }
                ],
            ),
            AIMessage(content="apologies, retrieval failed", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))
    scratch = out["scratch"]["semantic_search"][0]
    assert "error" in scratch["result"]
    assert "db down" in scratch["result"]["error"]
    assert out["done"] is True

    # The exception must also surface to the LLM via the ToolMessage content,
    # not just to scratch — otherwise the model has no way to react.
    from langchain_core.messages import ToolMessage as _ToolMessage

    tool_messages = [m for m in out["messages"] if isinstance(m, _ToolMessage)]
    assert tool_messages, "act() must append a ToolMessage even on tool failure"
    assert "db down" in tool_messages[-1].content


async def test_plan_does_not_double_insert_system_prompt(monkeypatch) -> None:
    """If the caller already supplied a SystemMessage, plan() must not stack a
    second one on top of it."""
    from langchain_core.messages import SystemMessage

    fake = _make_fake([AIMessage(content="hi", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(
        ItineraryState(
            messages=[
                SystemMessage(content="custom system prompt"),
                HumanMessage(content="hello"),
            ]
        )
    )
    system_messages = [m for m in out["messages"] if isinstance(m, SystemMessage)]
    assert len(system_messages) == 1
    assert system_messages[0].content == "custom system prompt"


async def test_act_handles_parallel_tool_calls(monkeypatch) -> None:
    """Modern OpenAI/Gemini fan out multiple tool calls in one AIMessage. Both
    must run, both ToolMessages append, and step_count increments by 1."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])
    monkeypatch.setattr("app.agent.tools._nearby", lambda **_kw: [])

    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "semantic_search", "id": "a", "args": {"query": "x"}},
                    {"name": "nearby", "id": "b", "args": {"place_id": "p1"}},
                ],
            ),
            AIMessage(content="done", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))

    from langchain_core.messages import ToolMessage as _ToolMessage

    tool_messages = [m for m in out["messages"] if isinstance(m, _ToolMessage)]
    assert {tm.tool_call_id for tm in tool_messages} == {"a", "b"}
    assert out["step_count"] == 1
    assert "semantic_search" in out["scratch"]
    assert "nearby" in out["scratch"]


def test_prune_for_llm_keeps_short_history_intact() -> None:
    msgs: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "semantic_search", "id": "1", "args": {"query": "x"}}],
        ),
        ToolMessage(content="[]", tool_call_id="1"),
        AIMessage(content="done", tool_calls=[]),
    ]
    assert _prune_for_llm(msgs) == msgs


def test_prune_for_llm_drops_oldest_tool_exchanges() -> None:
    """With more than 2 tool-issuing AIMessages, the older ones lose their
    tool_calls and their ToolMessages are dropped — but the rest survives."""
    msgs: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(
            content="searching",
            tool_calls=[{"name": "semantic_search", "id": "a", "args": {"q": "1"}}],
        ),
        ToolMessage(content="r1", tool_call_id="a"),
        AIMessage(
            content="searching again",
            tool_calls=[{"name": "semantic_search", "id": "b", "args": {"q": "2"}}],
        ),
        ToolMessage(content="r2", tool_call_id="b"),
        AIMessage(
            content="searching once more",
            tool_calls=[{"name": "semantic_search", "id": "c", "args": {"q": "3"}}],
        ),
        ToolMessage(content="r3", tool_call_id="c"),
        AIMessage(content="done", tool_calls=[]),
    ]
    pruned = _prune_for_llm(msgs)
    tool_messages = [m for m in pruned if isinstance(m, ToolMessage)]
    # The oldest ToolMessage ("r1") is dropped.
    assert {tm.content for tm in tool_messages} == {"r2", "r3"}
    # The oldest AIMessage that issued tool_calls keeps its content but loses
    # the tool_calls so the LLM doesn't see an unanswered call.
    ai_with_tools = [m for m in pruned if isinstance(m, AIMessage) and m.tool_calls]
    assert len(ai_with_tools) == 2  # the two most recent


def _state_with_grounded(place_ids: list[str], party_size: int = 2) -> ItineraryState:
    """Build a state where the given place_ids appear in scratch, so
    commit_stops considers them grounded."""
    hits = [
        PlaceHit(place_id=pid, name=f"Place {pid}", source="google_places", similarity=0.9)
        for pid in place_ids
    ]
    return ItineraryState(
        scratch={"semantic_search": [{"args": {}, "result": hits, "step": 0, "id": "s1"}]},
        constraints=UserConstraints(party_size=party_size, when=datetime(2026, 5, 7, 19, 0)),
    )


def test_commit_stops_enriches_with_booking() -> None:
    """Auto-enrichment must stamp booking_url + booking_provider on every
    committed stop, without the LLM calling a tool. Now batched: one
    get_details_many fetches details for all stops in a single round-trip,
    then per-stop URL construction is pure."""
    state = _state_with_grounded(["p1", "p2"])
    raw_stops = [
        {"place_id": "p1", "name": "Place p1", "rationale": "first", "source": "google_places"},
        {"place_id": "p2", "name": "Place p2", "rationale": "second", "source": "google_places"},
    ]

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id,
            provider="resy",
            booking_url=f"https://resy.com/{details.place_id}?seats={party_size}",
        )

    with (
        _patch_details_many_for(["p1", "p2"]) as mock_get,
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        committed, payload = commit_stops(state, raw_stops)

    assert payload["committed"] == ["p1", "p2"]
    assert all(s.booking_url is not None for s in committed)
    assert all(s.booking_provider == "resy" for s in committed)
    assert "p1" in committed[0].booking_url
    assert "seats=2" in committed[0].booking_url
    # The whole point of this refactor: ONE DB call for the whole commit, not N.
    assert mock_get.call_count == 1


def test_commit_stops_skips_enrichment_for_place_id_missing_from_db() -> None:
    """If get_details_many returns no row for a committed place_id (race
    condition: deletion between scratch grounding and enrichment, or a stale
    id), the stop is committed without a booking link instead of crashing.
    Same recoverable semantic the old ValueError("unknown place_id") had."""
    state = _state_with_grounded(["p1"])
    raw_stops = [
        {"place_id": "p1", "name": "Place p1", "rationale": "x", "source": "google_places"},
    ]
    with patch("app.agent.commit.get_details_many", return_value={}):
        committed, payload = commit_stops(state, raw_stops)

    assert payload["committed"] == ["p1"]
    assert committed[0].booking_url is None
    assert committed[0].booking_provider is None


def test_commit_stops_enrichment_swallows_psycopg_db_blip() -> None:
    """A transient DB error during the batched read shouldn't kill the whole
    commit — the user still gets the planned stops, just without booking
    links. The error is caught at the single point of DB contact."""
    state = _state_with_grounded(["p1"])
    raw_stops = [
        {"place_id": "p1", "name": "Place p1", "rationale": "x", "source": "google_places"},
    ]
    with patch(
        "app.agent.commit.get_details_many",
        side_effect=psycopg2.OperationalError("connection blip"),
    ):
        committed, payload = commit_stops(state, raw_stops)

    assert payload["committed"] == ["p1"]
    assert committed[0].booking_url is None
    assert committed[0].booking_provider is None


def test_commit_stops_enrichment_propagates_programmer_errors() -> None:
    """Bugs in URL construction (TypeError, AttributeError, etc.) must NOT
    be silently swallowed — that's how regressions ship to prod undetected.
    Only the documented recoverable cases (missing-from-DB, DB blip) are caught."""
    state = _state_with_grounded(["p1"])
    raw_stops = [
        {"place_id": "p1", "name": "Place p1", "rationale": "x", "source": "google_places"},
    ]
    with (
        _patch_details_many_for(["p1"]),
        patch(
            "app.agent.commit.propose_booking_from_details",
            side_effect=TypeError("propose_booking_from_details() got an unexpected kwarg"),
        ),
        pytest.raises(TypeError),
    ):
        commit_stops(state, raw_stops)


def test_commit_stops_re_commit_is_idempotent() -> None:
    """The W3 critique loop drives a second commit_itinerary call when the
    first plan fails a check. Re-committing the same place_id must produce the
    same booking link — same inputs, same output. Locks the contract so a
    future 'skip if already enriched' optimization can't silently break it."""
    state = _state_with_grounded(["p1"])
    raw_stops = [
        {"place_id": "p1", "name": "Place p1", "rationale": "first", "source": "google_places"},
    ]

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id,
            provider="resy",
            booking_url=f"https://resy.com/{details.place_id}?seats={party_size}",
        )

    with (
        _patch_details_many_for(["p1"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        first_committed, _ = commit_stops(state, raw_stops)
        second_committed, _ = commit_stops(state, raw_stops)

    assert first_committed[0].booking_url == second_committed[0].booking_url
    assert first_committed[0].booking_provider == second_committed[0].booking_provider


def test_commit_stops_per_stop_independence_when_one_id_missing() -> None:
    """A 2-stop commit where one place_id is missing from the DB result must
    still ship the other stop's booking link. The for-loop in
    enrich_stops_with_booking skips per-stop on missing details rather than
    bailing on the whole commit."""
    state = _state_with_grounded(["p1", "p2"])
    raw_stops = [
        {"place_id": "p1", "name": "Place p1", "rationale": "first", "source": "google_places"},
        {"place_id": "p2", "name": "Place p2", "rationale": "second", "source": "google_places"},
    ]

    # get_details_many returns details for p2 but NOT p1 — same shape as the
    # DB filtering p1 out (e.g. the row was deleted mid-commit).
    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id,
            provider="opentable",
            booking_url=f"https://opentable.com/{details.place_id}",
        )

    with (
        patch(
            "app.agent.commit.get_details_many",
            return_value={"p2": _details_for("p2")},
        ),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        committed, _ = commit_stops(state, raw_stops)

    # p1 missing from DB → skipped, no booking fields populated.
    assert committed[0].place_id == "p1"
    assert committed[0].booking_url is None
    assert committed[0].booking_provider is None
    # p2 was independent of p1 and succeeded.
    assert committed[1].place_id == "p2"
    assert committed[1].booking_url == "https://opentable.com/p2"
    assert committed[1].booking_provider == "opentable"


def test_enrich_stops_with_booking_mutates_in_place() -> None:
    """Direct coverage of the public helper. Future constraint-edit flows
    will call enrich_stops_with_booking on already-built Stop objects without
    going through commit_stops; the in-place-mutation contract is what makes
    that re-enrichment cheap."""
    stops = [
        Stop(place_id="p1", name="A", rationale="r", source="google_places"),
        Stop(place_id="p2", name="B", rationale="r", source="google_places"),
    ]
    state = ItineraryState(
        constraints=UserConstraints(party_size=4, when=datetime(2026, 5, 7, 19, 0)),
    )

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(
            place_id=details.place_id,
            provider="tock",
            booking_url=f"https://exploretock.com/{details.place_id}?size={party_size}",
        )

    original_ids = [id(s) for s in stops]
    with (
        _patch_details_many_for(["p1", "p2"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    # Same Stop objects (mutated, not replaced).
    assert [id(s) for s in stops] == original_ids
    # Both got party_size=4 from constraints (no per-stop arrival_time).
    assert all(s.booking_provider == "tock" for s in stops)
    assert all("size=4" in (s.booking_url or "") for s in stops)


def test_enrich_populates_card_fields_from_details() -> None:
    """address/rating/price_level flow from PlaceDetails onto Stop."""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        return BookingProposal(place_id=details.place_id, provider="tock", booking_url="https://x")

    with (
        patch(
            "app.agent.commit.get_details_many",
            return_value={"p1": _rich_details_for("p1")},
        ),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert stops[0].address == "p1 Main St, San Francisco"
    assert stops[0].rating == 4.4
    assert stops[0].price_level == 2


def test_enrich_card_fields_set_even_when_no_booking_time() -> None:
    """Regression guard: the no-time path skips booking but must NOT skip
    address/rating/price. These do not depend on `when`."""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(constraints=UserConstraints(party_size=2))  # when=None

    with (
        patch(
            "app.agent.commit.get_details_many",
            return_value={"p1": _rich_details_for("p1")},
        ),
        patch("app.agent.commit.propose_booking_from_details") as mock_build,
    ):
        enrich_stops_with_booking(stops, state)

    mock_build.assert_not_called()  # no booking without a time
    assert stops[0].booking_url is None
    assert stops[0].address == "p1 Main St, San Francisco"
    assert stops[0].rating == 4.4
    assert stops[0].price_level == 2


def test_enrich_card_fields_none_when_details_missing() -> None:
    """place_id missing from DB at enrichment time -> fields stay None
    (same degradation as booking links)."""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    with patch("app.agent.commit.get_details_many", return_value={}):
        enrich_stops_with_booking(stops, state)

    assert stops[0].address is None
    assert stops[0].rating is None
    assert stops[0].price_level is None


def test_enrich_skipped_when_no_time_anywhere() -> None:
    """Neither stop.arrival_time nor constraints.when is set → enrichment must
    skip the stop, NOT inject datetime.now(). The wall-clock fallback would
    embed a timestamp in the URL that's meaningless to the user and breaks
    re-commit idempotency (same inputs, different URL each call)."""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(constraints=UserConstraints(party_size=2))  # when=None

    with (
        _patch_details_many_for(["p1"]),
        patch("app.agent.commit.propose_booking_from_details") as mock_build,
    ):
        enrich_stops_with_booking(stops, state)

    # Crucially, the URL builder was NOT called — the no-time path skips entirely.
    mock_build.assert_not_called()
    assert stops[0].booking_url is None
    assert stops[0].booking_provider is None


def test_enrich_idempotent_when_no_time_set() -> None:
    """The no-time skip preserves re-commit idempotency: same inputs in,
    same (absent) booking link out, both calls."""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(constraints=UserConstraints(party_size=2))

    with (
        _patch_details_many_for(["p1"]),
        patch("app.agent.commit.propose_booking_from_details") as mock_build,
    ):
        enrich_stops_with_booking(stops, state)
        first_url = stops[0].booking_url
        enrich_stops_with_booking(stops, state)
        second_url = stops[0].booking_url

    assert first_url is None
    assert second_url is None
    assert mock_build.call_count == 0


def test_enrich_uses_constraints_when_when_arrival_time_missing() -> None:
    """If a stop has no arrival_time but constraints.when IS set, the constraint
    fills in. (Counterpoint to the skip test above — make sure we don't skip
    too aggressively.)"""
    stops = [Stop(place_id="p1", name="A", rationale="r", source="google_places")]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    captured: list[datetime] = []

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        captured.append(when)
        return BookingProposal(place_id=details.place_id, provider="resy", booking_url="https://x")

    with (
        _patch_details_many_for(["p1"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert captured == [datetime(2026, 5, 7, 19, 0)]


@pytest.mark.parametrize("party_size_in,expected_in_call", [(None, 2), (0, 2), (4, 4), (1, 1)])
def test_enrich_party_size_defaulting(party_size_in: int | None, expected_in_call: int) -> None:
    """party_size None or 0 → defaults to 2 (you can't book a table for 0).
    Other positive values pass through unchanged. Locks the `or 2` semantics
    so a future refactor doesn't accidentally allow 0-party bookings or change
    the default."""
    stops = [
        Stop(
            place_id="p1",
            name="A",
            rationale="r",
            source="google_places",
            arrival_time=datetime(2026, 5, 7, 19, 0),
        )
    ]
    state = ItineraryState(constraints=UserConstraints(party_size=party_size_in))

    captured: list[int] = []

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        captured.append(party_size)
        return BookingProposal(place_id=details.place_id, provider="resy", booking_url="https://x")

    with (
        _patch_details_many_for(["p1"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert captured == [expected_in_call]


def test_enrich_stops_with_booking_uses_arrival_time_per_stop() -> None:
    """When stops have different arrival_times, each stop's URL must reflect
    its OWN time, not constraints.when. Otherwise a 3-stop itinerary's late
    stops get a 'date' for the first stop's time slot."""
    stops = [
        Stop(
            place_id="p1",
            name="dinner",
            rationale="r",
            source="google_places",
            arrival_time=datetime(2026, 5, 7, 19, 0),
        ),
        Stop(
            place_id="p2",
            name="drinks",
            rationale="r",
            source="google_places",
            arrival_time=datetime(2026, 5, 7, 21, 30),
        ),
    ]
    state = ItineraryState(
        constraints=UserConstraints(party_size=2, when=datetime(2026, 5, 7, 19, 0)),
    )

    captured: list[tuple[str, datetime]] = []

    def fake_build(details: PlaceDetails, when: datetime, party_size: int) -> BookingProposal:
        captured.append((details.place_id, when))
        return BookingProposal(
            place_id=details.place_id, provider="resy", booking_url="https://resy.com/x"
        )

    with (
        _patch_details_many_for(["p1", "p2"]),
        patch("app.agent.commit.propose_booking_from_details", side_effect=fake_build),
    ):
        enrich_stops_with_booking(stops, state)

    assert captured == [
        ("p1", datetime(2026, 5, 7, 19, 0)),
        ("p2", datetime(2026, 5, 7, 21, 30)),
    ]


def _committed_state(n_stops: int, *, with_coords: bool) -> ItineraryState:
    base = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = []
    for i in range(n_stops):
        stops.append(
            Stop(
                place_id=f"p{i}",
                name=f"P{i}",
                source="google_places",
                rationale="",
                arrival_time=base if i == 0 else None,
                planned_duration_min=60,
                latitude=37.77 + i * 0.01 if with_coords else None,
                longitude=-122.41 if with_coords else None,
            )
        )
    return ItineraryState(stops=stops, done=True, final_reply="Here is your plan.")


async def test_retime_node_present_and_routed(monkeypatch) -> None:
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    assert "retime" in graph.get_graph().nodes


async def test_retime_at_most_one_directions_call(monkeypatch, mocker) -> None:
    calls = {"n": 0}

    async def _counting_route_legs(stops, mode="walk"):
        calls["n"] += 1
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=600, distance_m=800.0)] * (len(stops) - 1),
            total_duration_s=600 * (len(stops) - 1),
            mode=mode,
            source="google",
        )

    mocker.patch("app.agent.graph.route_legs", _counting_route_legs)
    # The node calls temporal_coherence (imported into graph.py in Step 3),
    # NOT itinerary_violations. Patch the symbol the node actually uses.
    mocker.patch("app.agent.graph.temporal_coherence", lambda _s: 1.0)
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    await graph.ainvoke(_committed_state(3, with_coords=True))
    assert calls["n"] == 1


async def test_retime_passthrough_when_not_routable(monkeypatch, mocker) -> None:
    route = mocker.patch("app.agent.graph.route_legs")
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(_committed_state(1, with_coords=True))
    route.assert_not_called()
    assert out["stops"][0].arrival_time == datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)


async def test_retime_passthrough_when_coordless(monkeypatch, mocker) -> None:
    route = mocker.patch("app.agent.graph.route_legs")
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    await graph.ainvoke(_committed_state(3, with_coords=False))
    route.assert_not_called()
