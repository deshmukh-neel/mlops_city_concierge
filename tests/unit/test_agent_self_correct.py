"""Tests for W3 self-correction.

Per-step deterministic critique behavior is exercised end-to-end through the
graph with a scripted LLM. Per-itinerary critique is exercised via the
checks module and via a graph test that monkeypatches itinerary_violations
(real itinerary checks hit the DB; we validate them separately).
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.critique import vibe
from app.agent.critique.checks import (
    geographic_coherence,
    walking_budget_respected,
)
from app.agent.graph import (
    LOW_SIMILARITY_THRESHOLD,
    MAX_REVISIONS_PER_REASON,
    build_agent_graph,
)
from app.agent.state import ItineraryState, Stop, UserConstraints
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
        if not self.scripted:
            raise RuntimeError("scripted responses exhausted")
        msg = self.scripted.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedLLM:
        return self


def _make_fake(scripted: list[AIMessage]) -> _ScriptedLLM:
    return _ScriptedLLM(scripted=list(scripted))


def _hit(
    place_id: str = "p1",
    similarity: float = 0.9,
    business_status: str = "OPERATIONAL",
) -> PlaceHit:
    return PlaceHit(
        place_id=place_id,
        name=place_id.upper(),
        source="google_places",
        similarity=similarity,
        latitude=37.78,
        longitude=-122.41,
        rating=4.5,
        price_level="PRICE_LEVEL_MODERATE",
        business_status=business_status,
        primary_type="restaurant",
        formatted_address="123 Main",
        snippet=None,
    )


# -------- Per-step deterministic critique -------------------------------------


async def test_empty_results_emits_drop_filter_hint(monkeypatch) -> None:
    """An empty tool result triggers a drop_filter hint, and the next plan
    cycle gets a chance to retry."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "1",
                        "args": {
                            "query": "x",
                            "filters": {"price_level_max": 1, "neighborhood": "Mission"},
                        },
                    }
                ],
            ),
            AIMessage(content="couldn't find anything", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="x")]))

    hints = out["revision_hints"]
    assert any(h.reason == "empty_results" for h in hints)
    drop = next(h for h in hints if h.reason == "empty_results")
    # Priority order in _most_restrictive_filter — open_at would win if set.
    assert drop.suggested_action == "drop_filter"
    assert drop.target == {"filter": "price_level_max"}
    assert out["revision_counts"]["empty_results"] == 1
    assert out["done"] is True


async def test_all_closed_emits_broaden_query_hint(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [_hit(business_status="CLOSED_PERMANENTLY")],
    )
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "semantic_search", "id": "1", "args": {"query": "x"}},
                ],
            ),
            AIMessage(content="all closed", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="x")]))

    hints = out["revision_hints"]
    assert any(h.reason == "all_closed" for h in hints)
    h = next(h for h in hints if h.reason == "all_closed")
    assert h.suggested_action == "broaden_query"


async def test_low_similarity_emits_broaden_query_hint(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [_hit(similarity=LOW_SIMILARITY_THRESHOLD - 0.1)],
    )
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "semantic_search", "id": "1", "args": {"query": "obscure"}},
                ],
            ),
            AIMessage(content="weak match", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="x")]))

    hints = out["revision_hints"]
    assert any(h.reason == "low_similarity" for h in hints)


async def test_tool_error_emits_try_different_tool_hint(monkeypatch) -> None:
    def _boom(**_kw):
        raise RuntimeError("db down")

    monkeypatch.setattr("app.agent.tools._semantic_search", _boom)
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "semantic_search", "id": "1", "args": {"query": "x"}},
                ],
            ),
            AIMessage(content="apologies", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="x")]))

    hints = out["revision_hints"]
    assert any(h.reason == "tool_error" for h in hints)
    h = next(h for h in hints if h.reason == "tool_error")
    assert h.target == {"tool": "semantic_search"}


async def test_healthy_result_emits_no_hint(monkeypatch) -> None:
    """A normal high-similarity OPERATIONAL result must not emit a hint —
    we don't want to thrash on good data."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [_hit()])
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "semantic_search", "id": "1", "args": {"query": "x"}},
                ],
            ),
            AIMessage(content="found one", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="x")]))
    assert out["revision_hints"] == []


async def test_retry_budget_caps_hints_per_reason(monkeypatch) -> None:
    """After MAX_REVISIONS_PER_REASON empty_results hints, no further hint is
    emitted for the same reason — the agent ships rather than loops."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])

    def _empty_call(i: int) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "semantic_search", "id": str(i), "args": {"query": "x"}},
            ],
        )

    fake = _make_fake(
        [
            _empty_call(1),
            _empty_call(2),
            _empty_call(3),
            AIMessage(content="giving up cleanly", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=6)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="x")]))

    counts = out["revision_counts"]
    assert counts.get("empty_results", 0) == MAX_REVISIONS_PER_REASON
    # Three empty calls but only MAX_REVISIONS_PER_REASON hints stored.
    empty_hints = [h for h in out["revision_hints"] if h.reason == "empty_results"]
    assert len(empty_hints) == MAX_REVISIONS_PER_REASON


# -------- Per-itinerary critique (graph-level, with checks monkeypatched) -----


async def test_itinerary_violation_triggers_revision(monkeypatch) -> None:
    """When itinerary_violations() reports a failure on a finalizing turn with
    committed stops, critique injects a revision HumanMessage and routes back
    to plan."""
    # Stand in for the DB-backed check.
    monkeypatch.setattr(
        "app.agent.graph.itinerary_violations",
        lambda _state: ["geographic_coherence"],
    )
    # Pre-seed state with stops to trigger the per-itinerary path.
    state = ItineraryState(
        messages=[HumanMessage(content="plan it")],
        stops=[
            Stop(place_id="p1", name="A", source="google_places", rationale=""),
            Stop(place_id="p2", name="B", source="google_places", rationale=""),
        ],
    )
    fake = _make_fake(
        [
            # First turn: LLM tries to finalize without a tool call.
            AIMessage(content="here's the plan", tool_calls=[]),
            # After critique injects [critique:itinerary], LLM produces a new
            # finalizing message — this time the violations check is replaced
            # below to return [].
            AIMessage(content="revised plan", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)

    # Toggle violations off after the first invocation so the revision
    # actually clears the gate.
    calls = {"n": 0}

    def _violations(_state):
        calls["n"] += 1
        return ["geographic_coherence"] if calls["n"] == 1 else []

    monkeypatch.setattr("app.agent.graph.itinerary_violations", _violations)

    out = await graph.ainvoke(state)

    hints = out["revision_hints"]
    assert any(h.reason == "geographic_incoherence" for h in hints)
    assert out["revision_counts"]["geographic_coherence"] == 1
    assert out["done"] is True
    # Ships the revised final reply (not a caveat string).
    assert "Caveats:" not in (out["final_reply"] or "")


async def test_itinerary_violation_ships_with_caveats_after_exhaustion(monkeypatch) -> None:
    """If retries are exhausted on a violation, the final reply gets a
    caveat suffix instead of looping forever."""
    monkeypatch.setattr(
        "app.agent.graph.itinerary_violations",
        lambda _state: ["geographic_coherence"],
    )

    state = ItineraryState(
        messages=[HumanMessage(content="plan it")],
        stops=[
            Stop(place_id="p1", name="A", source="google_places", rationale=""),
            Stop(place_id="p2", name="B", source="google_places", rationale=""),
        ],
        revision_counts={"geographic_coherence": MAX_REVISIONS_PER_REASON},
    )
    fake = _make_fake([AIMessage(content="my best plan", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(state)

    assert out["done"] is True
    assert "Caveats:" in out["final_reply"]
    assert "geographic_coherence" in out["final_reply"]


# -------- Pure check functions (no DB) ----------------------------------------


def test_geographic_coherence_perfect_when_close() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            Stop(
                place_id="p1",
                name="A",
                source="google_places",
                rationale="",
                latitude=37.78,
                longitude=-122.41,
            ),
            Stop(
                place_id="p2",
                name="B",
                source="google_places",
                rationale="",
                latitude=37.781,
                longitude=-122.411,
            ),
        ],
    )
    assert geographic_coherence(state) == 1.0


def test_walking_budget_respected_fails_when_over() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=100),
        stops=[
            Stop(
                place_id="p1",
                name="A",
                source="google_places",
                rationale="",
                latitude=37.78,
                longitude=-122.41,
            ),
            Stop(
                place_id="p2",
                name="B",
                source="google_places",
                rationale="",
                latitude=37.80,  # ~2km away
                longitude=-122.41,
            ),
        ],
    )
    assert walking_budget_respected(state) == 0.0


# -------- Vibe module ---------------------------------------------------------


def test_vibe_check_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv(vibe.VIBE_ENV_VAR, raising=False)
    state = ItineraryState(
        stops=[
            Stop(place_id="p1", name="A", source="google_places", rationale=""),
            Stop(place_id="p2", name="B", source="google_places", rationale=""),
        ],
    )
    # Even with a non-None judge, disabled env var short-circuits to None.

    class _Judge:
        def invoke(self, _msgs):  # pragma: no cover - never called
            raise AssertionError("judge_llm should not be invoked when disabled")

    assert vibe.vibe_check(state, _Judge()) is None


def test_vibe_check_returns_none_for_single_stop(monkeypatch) -> None:
    monkeypatch.setenv(vibe.VIBE_ENV_VAR, "true")

    class _Judge:
        def invoke(self, _msgs):  # pragma: no cover
            raise AssertionError("judge_llm should not be invoked for 1 stop")

    state = ItineraryState(
        stops=[Stop(place_id="p1", name="A", source="google_places", rationale="")],
    )
    assert vibe.vibe_check(state, _Judge()) is None


def test_vibe_check_parses_judge_score(monkeypatch) -> None:
    monkeypatch.setenv(vibe.VIBE_ENV_VAR, "true")

    class _Judge:
        def invoke(self, _msgs):
            return AIMessage(content='{"score": 4.2, "rationale": "great match"}')

    state = ItineraryState(
        messages=[HumanMessage(content="date night")],
        stops=[
            Stop(place_id="p1", name="A", source="google_places", rationale=""),
            Stop(place_id="p2", name="B", source="google_places", rationale=""),
        ],
    )
    assert vibe.vibe_check(state, _Judge()) == 4.2


def test_vibe_check_returns_none_on_unparseable_json(monkeypatch) -> None:
    """If the judge returns malformed JSON, we fail open rather than crash
    a real user request."""
    monkeypatch.setenv(vibe.VIBE_ENV_VAR, "true")

    class _Judge:
        def invoke(self, _msgs):
            return AIMessage(content="not json at all")

    state = ItineraryState(
        stops=[
            Stop(place_id="p1", name="A", source="google_places", rationale=""),
            Stop(place_id="p2", name="B", source="google_places", rationale=""),
        ],
    )
    assert vibe.vibe_check(state, _Judge()) is None
