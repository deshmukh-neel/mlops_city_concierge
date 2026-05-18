"""Tests for W3 self-correction.

Per-step deterministic critique behavior is exercised end-to-end through the
graph with a scripted LLM. Per-itinerary critique is exercised via the
checks module and via a graph test that monkeypatches itinerary_violations
(real itinerary checks hit the DB; we validate them separately).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.critique import vibe
from app.agent.critique.checks import (
    geographic_coherence,
    walking_budget_respected,
)
from app.agent.graph import build_agent_graph
from app.agent.revision import LOW_SIMILARITY_THRESHOLD, MAX_REVISIONS_PER_REASON
from app.agent.state import ItineraryState, UserConstraints
from tests.conftest import make_hit, make_stop


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


_hit = make_hit  # backwards-compat alias for the existing tests in this file


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
        "app.agent.revision.itinerary_violations",
        lambda _state: ["geographic_coherence"],
    )
    # Pre-seed state with stops to trigger the per-itinerary path.
    state = ItineraryState(
        messages=[HumanMessage(content="plan it")],
        stops=[
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
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

    monkeypatch.setattr("app.agent.revision.itinerary_violations", _violations)

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
        "app.agent.revision.itinerary_violations",
        lambda _state: ["geographic_coherence"],
    )

    state = ItineraryState(
        messages=[HumanMessage(content="plan it")],
        stops=[
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
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
            make_stop("p1", name="A", latitude=37.78, longitude=-122.41),
            make_stop("p2", name="B", latitude=37.781, longitude=-122.411),
        ],
    )
    assert geographic_coherence(state) == 1.0


def test_walking_budget_respected_fails_when_over() -> None:
    state = ItineraryState(
        constraints=UserConstraints(walking_budget_m=100),
        stops=[
            make_stop("p1", name="A", latitude=37.78, longitude=-122.41),
            make_stop("p2", name="B", latitude=37.80, longitude=-122.41),  # ~2km away
        ],
    )
    assert walking_budget_respected(state) == 0.0


# -------- Vibe module ---------------------------------------------------------


def test_vibe_check_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv(vibe.VIBE_ENV_VAR, raising=False)
    state = ItineraryState(
        stops=[
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
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
        stops=[make_stop("p1", name="A")],
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
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
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
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
        ],
    )
    assert vibe.vibe_check(state, _Judge()) is None


# -------- Edge cases ----------------------------------------------------------


async def test_finalizing_with_no_stops_just_finalizes(monkeypatch) -> None:
    """A clarifying-question response (LLM finalized without committing any
    stops) must not run itinerary violations — there's nothing to check."""
    called = {"violations": False}

    def _violations(_state):
        called["violations"] = True
        return []

    monkeypatch.setattr("app.agent.revision.itinerary_violations", _violations)
    fake = _make_fake([AIMessage(content="how many stops?", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="plan it")]))

    assert called["violations"] is False
    assert out["done"] is True
    assert out["final_reply"] == "how many stops?"


async def test_multiple_violations_picks_first_actionable(monkeypatch) -> None:
    """When several checks fail, critique acts on the first one with retries
    left — not all of them at once."""
    monkeypatch.setattr(
        "app.agent.revision.itinerary_violations",
        lambda _state: ["temporal_coherence", "geographic_coherence"],
    )
    state = ItineraryState(
        messages=[HumanMessage(content="plan")],
        stops=[
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
        ],
        # Temporal exhausted; geographic has budget left.
        revision_counts={"temporal_coherence": MAX_REVISIONS_PER_REASON},
    )
    fake = _make_fake(
        [
            AIMessage(content="initial plan", tool_calls=[]),
            AIMessage(content="revised plan", tool_calls=[]),
        ]
    )
    # Toggle the second pass to clean.
    n = {"i": 0}

    def _violations(_state):
        n["i"] += 1
        return ["temporal_coherence", "geographic_coherence"] if n["i"] == 1 else []

    monkeypatch.setattr("app.agent.revision.itinerary_violations", _violations)
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(state)

    # We should have skipped temporal (exhausted) and acted on geographic.
    assert any(h.reason == "geographic_incoherence" for h in out["revision_hints"])
    assert all(h.reason != "temporal_incoherence" for h in out["revision_hints"])
    assert out["done"] is True


async def test_max_steps_short_circuits_critique(monkeypatch) -> None:
    """Even mid-revision, max_steps preempts further work — the user gets
    the best plan so far, with the truncation message if no final exists."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])
    looping = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "semantic_search", "id": str(i), "args": {"query": "x"}},
            ],
        )
        for i in range(20)
    ]
    fake = _make_fake(looping)
    graph = build_agent_graph(fake, max_steps=2)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))

    assert out["done"] is True
    assert out["step_count"] == 2
    assert "step limit" in (out["final_reply"] or "")


async def test_revision_counts_persist_across_turns(monkeypatch) -> None:
    """Two empty searches in a row both bump the same counter — we don't
    reset it between turns."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "semantic_search", "id": "1", "args": {"query": "x"}}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "semantic_search", "id": "2", "args": {"query": "y"}}],
            ),
            AIMessage(content="giving up", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=6)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))

    assert out["revision_counts"]["empty_results"] == 2


async def test_critique_message_is_visible_to_plan(monkeypatch) -> None:
    """The HumanMessage that critique injects ends up in state.messages so
    it shows up in tracing and so the next plan call sees it. Sanity check
    that the [critique:step] prefix is present."""
    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **_kw: [])
    fake = _make_fake(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "semantic_search", "id": "1", "args": {"query": "x"}}],
            ),
            AIMessage(content="ok", tool_calls=[]),
        ]
    )
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(ItineraryState(messages=[HumanMessage(content="hi")]))

    critique_msgs = [
        m
        for m in out["messages"]
        if isinstance(m, HumanMessage) and "[critique:step]" in (m.content or "")
    ]
    assert len(critique_msgs) == 1
    assert "empty_results" in critique_msgs[0].content


async def test_diagnose_handles_no_scratch_entry() -> None:
    """If somehow critique fires without any scratch entry (defensive),
    don't crash and don't emit a hint."""
    from app.agent.revision import _diagnose_last_tool_result

    state = ItineraryState(messages=[HumanMessage(content="hi")])
    assert _diagnose_last_tool_result(state) is None


async def test_diagnose_walks_every_tool_call_in_round() -> None:
    """When the LLM issues multiple tool_calls in one AIMessage, the
    diagnostic must see all of them and return the first hint in tool_call
    order — not just the last one."""
    from langchain_core.messages import ToolMessage

    from app.agent.revision import _diagnose_last_tool_result

    state = ItineraryState(
        messages=[
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "semantic_search", "id": "a", "args": {"query": "x"}},
                    {"name": "nearby", "id": "b", "args": {"place_id": "p1"}},
                ],
            ),
            ToolMessage(content="<unused>", tool_call_id="a"),
            ToolMessage(content="<unused>", tool_call_id="b"),
        ],
        scratch={
            # First call returned [] (bad); second returned a healthy hit.
            "semantic_search": [
                {"args": {"query": "x"}, "result": [], "step": 0, "id": "a"},
            ],
            "nearby": [
                {
                    "args": {"place_id": "p1"},
                    "result": [_hit()],
                    "step": 0,
                    "id": "b",
                },
            ],
        },
    )
    hint = _diagnose_last_tool_result(state)
    assert hint is not None
    # Tool-call order: semantic_search came first; its empty result wins.
    assert hint.reason == "empty_results"


def test_final_with_caveats_includes_each_violation_name() -> None:
    """Caveats reply must lead with the agent's own body and list every
    violation name verbatim — that's what the user-facing UI relies on."""
    from app.agent.revision import _final_with_caveats

    out = _final_with_caveats("my plan", ["v1", "v2"])
    assert out.startswith("my plan")
    assert "v1" in out
    assert "v2" in out
    assert "Caveats:" in out


def test_final_with_caveats_handles_empty_body() -> None:
    """If the LLM somehow ended up with no content, the caveat must still
    render coherently rather than NoneType-error."""
    from app.agent.revision import _final_with_caveats

    out = _final_with_caveats("", ["v1"])
    assert "Caveats:" in out
    assert "v1" in out


@pytest.mark.parametrize(
    ("check", "expected_reason", "expected_action"),
    [
        ("geographic_coherence", "geographic_incoherence", "tighten_radius"),
        ("temporal_coherence", "temporal_incoherence", "shift_arrival_time"),
        (
            "walking_budget_respected",
            "walking_budget_exceeded",
            "rebalance_walking_budget",
        ),
        ("no_hallucinated_place_ids", "hallucinated_place_id", "swap_stop"),
        ("unknown_check_name", "constraint_unmet_in_final", "swap_stop"),
    ],
)
def test_hint_for_violation_maps_each_check(
    check: str, expected_reason: str, expected_action: str
) -> None:
    """Every check name supported by itinerary_violations() must map to a
    structured RevisionHint. The catch-all (anything unmapped) must produce
    a `constraint_unmet_in_final` hint rather than crash."""
    from app.agent.revision import _hint_for_violation

    state = ItineraryState(stops=[make_stop("p1", name="A"), make_stop("p2", name="B")])
    hint = _hint_for_violation(check, state)
    assert hint.reason == expected_reason
    assert hint.suggested_action == expected_action


async def test_diagnose_pairs_by_tool_call_id() -> None:
    """Two calls of the same tool in one round must each match their own
    scratch entry by id, not blur into max(step)."""
    from langchain_core.messages import ToolMessage

    from app.agent.revision import _diagnose_last_tool_result

    state = ItineraryState(
        messages=[
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "nearby", "id": "first", "args": {"place_id": "p1"}},
                    {"name": "nearby", "id": "second", "args": {"place_id": "p2"}},
                ],
            ),
            ToolMessage(content="<unused>", tool_call_id="first"),
            ToolMessage(content="<unused>", tool_call_id="second"),
        ],
        scratch={
            "nearby": [
                # First (id="first") was bad — empty.
                {"args": {"place_id": "p1"}, "result": [], "step": 0, "id": "first"},
                # Second (id="second") was healthy.
                {
                    "args": {"place_id": "p2"},
                    "result": [_hit()],
                    "step": 0,
                    "id": "second",
                },
            ],
        },
    )
    hint = _diagnose_last_tool_result(state)
    assert hint is not None
    assert hint.reason == "empty_results"


# -------- Vibe pass wired into the graph --------------------------------------


class _StubJudge:
    """Minimal judge LLM: returns a scripted JSON score string."""

    def __init__(self, score: float) -> None:
        self._score = score
        self.calls = 0

    def invoke(self, _msgs):  # pragma: no cover - exact wire format unused
        self.calls += 1
        return AIMessage(content=f'{{"score": {self._score}, "rationale": "stub"}}')


async def test_vibe_pass_injects_revision_when_below_threshold(monkeypatch) -> None:
    """Deterministic checks pass, vibe scores below threshold -> hint and
    re-plan, not finalize."""
    monkeypatch.setattr("app.agent.revision.itinerary_violations", lambda _state: [])
    monkeypatch.setenv(vibe.VIBE_ENV_VAR, "true")

    judge = _StubJudge(score=2.0)  # below VIBE_THRESHOLD=3.0
    state = ItineraryState(
        messages=[HumanMessage(content="date night")],
        stops=[
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
        ],
    )
    fake = _make_fake(
        [
            AIMessage(content="initial plan", tool_calls=[]),
            AIMessage(content="revised plan", tool_calls=[]),
        ]
    )
    # After the first vibe call, raise the score so the second pass clears.
    second_judge = _StubJudge(score=4.5)
    judges = [judge, second_judge]

    def _fake_vibe_check(_state, _judge):
        return judges.pop(0)._score if judges else 4.5

    monkeypatch.setattr("app.agent.revision.vibe.vibe_check", _fake_vibe_check)

    graph = build_agent_graph(fake, max_steps=4, judge_llm=judge)
    out = await graph.ainvoke(state)

    assert any(h.reason == "vibe_mismatch" for h in out["revision_hints"])
    assert out["revision_counts"]["vibe_mismatch"] == 1
    assert out["done"] is True
    assert out["final_reply"] == "revised plan"


async def test_vibe_pass_skips_when_judge_none(monkeypatch) -> None:
    """No judge wired (env disabled, or make_judge returned None) -> finalize
    without a vibe pass even if vibe_check would normally fire."""
    monkeypatch.setattr("app.agent.revision.itinerary_violations", lambda _state: [])

    state = ItineraryState(
        messages=[HumanMessage(content="date night")],
        stops=[
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
        ],
    )
    fake = _make_fake([AIMessage(content="my plan", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4, judge_llm=None)
    # Ensure no env-var path constructs a judge.
    monkeypatch.delenv(vibe.VIBE_ENV_VAR, raising=False)

    out = await graph.ainvoke(state)

    assert out["done"] is True
    assert all(h.reason != "vibe_mismatch" for h in out["revision_hints"])


async def test_vibe_pass_respects_retry_budget(monkeypatch) -> None:
    """Once vibe_mismatch hits MAX_REVISIONS_PER_REASON, finalize with the
    current plan rather than loop forever."""
    monkeypatch.setattr("app.agent.revision.itinerary_violations", lambda _state: [])
    monkeypatch.setattr("app.agent.revision.vibe.vibe_check", lambda *_a, **_k: 1.0)

    state = ItineraryState(
        messages=[HumanMessage(content="date night")],
        stops=[
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
        ],
        revision_counts={"vibe_mismatch": MAX_REVISIONS_PER_REASON},
    )
    fake = _make_fake([AIMessage(content="ship it", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4, judge_llm=_StubJudge(1.0))
    out = await graph.ainvoke(state)

    assert out["done"] is True
    assert out["final_reply"] == "ship it"


async def test_vibe_pass_skipped_when_violations_present(monkeypatch) -> None:
    """If deterministic checks fail, we never burn a vibe-judge call — the
    deterministic revision takes precedence."""
    monkeypatch.setattr(
        "app.agent.revision.itinerary_violations",
        lambda _state: ["geographic_coherence"],
    )
    judge_calls = {"n": 0}

    def _vibe(*_a, **_k):
        judge_calls["n"] += 1
        return 1.0

    monkeypatch.setattr("app.agent.revision.vibe.vibe_check", _vibe)

    state = ItineraryState(
        messages=[HumanMessage(content="hi")],
        stops=[
            make_stop("p1", name="A"),
            make_stop("p2", name="B"),
        ],
        revision_counts={"geographic_coherence": MAX_REVISIONS_PER_REASON},  # exhausted
    )
    fake = _make_fake([AIMessage(content="best effort", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4, judge_llm=_StubJudge(1.0))
    out = await graph.ainvoke(state)

    assert judge_calls["n"] == 0
    assert "Caveats:" in (out["final_reply"] or "")


def test_make_judge_returns_none_without_creds(monkeypatch) -> None:
    """If neither OPENAI nor GEMINI key is set, make_judge logs and returns
    None rather than raising."""
    monkeypatch.setattr("app.agent.critique.vibe.get_settings", lambda: _NoCreds())
    monkeypatch.delenv(vibe.JUDGE_PROVIDER_ENV_VAR, raising=False)
    monkeypatch.delenv(vibe.JUDGE_MODEL_ENV_VAR, raising=False)
    assert vibe.make_judge() is None


class _NoCreds:
    openai_api_key = ""
    gemini_api_key = ""


def test_make_judge_uses_factory_for_configured_provider(monkeypatch, mocker) -> None:
    """make_judge delegates construction to the central llm_factory and
    passes the env-configured provider + model at temperature 0.0."""
    monkeypatch.setenv(vibe.JUDGE_PROVIDER_ENV_VAR, "deepseek")
    monkeypatch.setenv(vibe.JUDGE_MODEL_ENV_VAR, "deepseek-chat")

    class _HasDeepseek:
        openai_api_key = ""
        gemini_api_key = ""
        deepseek_api_key = "ds-key"
        moonshot_api_key = ""

    monkeypatch.setattr("app.agent.critique.vibe.get_settings", lambda: _HasDeepseek())
    factory = mocker.patch("app.agent.critique.vibe.build_chat_model", return_value="judge-llm")

    out = vibe.make_judge()

    assert out == "judge-llm"
    factory.assert_called_once_with("deepseek", "deepseek-chat", temperature=0.0)
