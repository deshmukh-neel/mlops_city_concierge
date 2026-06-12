"""Tests for DEC-03 changes to app/agent/revision.py.

Plan 13-03:
  - LOW_SIMILARITY_THRESHOLD is env-overridable via LOW_SIMILARITY_THRESHOLD_OVERRIDE
    (code default stays 0.55; D-13-07).
  - With VIABILITY_CONTRACT_ENABLED=1 AND all_slots_viable(state)==True,
    _diagnose_last_tool_result returns None (low_similarity suppressed).
  - With VIABILITY_CONTRACT_ENABLED=1 AND all_slots_viable(state)==False,
    low_similarity still fires.
  - With VIABILITY_CONTRACT_ENABLED unset (flag off), behavior is byte-identical
    to current (T-13-03-02 mitigation).
"""

from __future__ import annotations

import importlib

import pytest

from app.agent.state import ItineraryState, UserConstraints
from app.tools.retrieval import PlaceHit

# ── Helpers ──────────────────────────────────────────────────────────────────

_THRESHOLD_DEFAULT = 0.55


def _hit(
    similarity: float,
    primary_type: str = "Sushi Restaurant",
    place_id: str = "pid1",
) -> PlaceHit:
    """Build a minimal semantic_search PlaceHit (matches real tool output type)."""
    return PlaceHit(
        place_id=place_id,
        name=f"Place {place_id}",
        primary_type=primary_type,
        business_status="OPERATIONAL",
        similarity=similarity,
        source="place_embeddings",
    )


def _state_with_search_hits(
    hits: list[PlaceHit],
    requested_types: list[str] | None = None,
    query: str = "sushi",
) -> ItineraryState:
    """Build an ItineraryState with one semantic_search scratch entry."""
    constraints = UserConstraints(
        requested_primary_types=requested_types or [],
        num_stops=1,
    )
    return ItineraryState(
        scratch={
            "semantic_search": [
                {
                    "step": 0,
                    "args": {"query": query},
                    "result": hits,
                    "id": "tc0",
                }
            ]
        },
        messages=[],
        constraints=constraints,
    )


# ── Threshold env-override tests ─────────────────────────────────────────────


def test_threshold_default_when_override_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOW_SIMILARITY_THRESHOLD resolves to 0.55 when override is unset.

    D-13-07: code default stays 0.55; the override only lowers it.
    """
    monkeypatch.delenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", raising=False)
    import app.agent.revision as rev

    importlib.reload(rev)
    assert pytest.approx(0.55) == rev.LOW_SIMILARITY_THRESHOLD


def test_threshold_override_applied_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOW_SIMILARITY_THRESHOLD resolves to the override float when set (e.g. 0.45).

    D-13-07: the experiment knob allows testing values below 0.55 in the A1 arm.
    """
    monkeypatch.setenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", "0.45")
    import app.agent.revision as rev

    importlib.reload(rev)
    assert pytest.approx(0.45) == rev.LOW_SIMILARITY_THRESHOLD


def test_threshold_falls_back_on_empty_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty string override falls back to 0.55 (T-13-03-01 mitigation).

    `float("" or "0.55")` must not raise.
    """
    monkeypatch.setenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", "")
    import app.agent.revision as rev

    importlib.reload(rev)
    assert pytest.approx(0.55) == rev.LOW_SIMILARITY_THRESHOLD


# ── Flag-off: byte-identical behavior ────────────────────────────────────────


def test_flag_off_low_similarity_fires_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """With VIABILITY_CONTRACT_ENABLED unset, a below-threshold semantic_search
    state yields a low_similarity hint — flag-off behavior unchanged (T-13-03-02).
    """
    monkeypatch.delenv("VIABILITY_CONTRACT_ENABLED", raising=False)
    monkeypatch.delenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", raising=False)

    import app.agent.revision as rev

    importlib.reload(rev)

    # Reload viability to pick up the fresh threshold value
    import app.agent.viability as via

    importlib.reload(via)

    # below-threshold hit
    state = _state_with_search_hits(
        [_hit(similarity=0.30, primary_type="Sushi Restaurant", place_id="pid1")],
        requested_types=["Sushi Restaurant"],
    )
    # Need at least one issuing AIMessage + ToolMessage to trigger diagnosis
    from langchain_core.messages import AIMessage, ToolMessage

    state.messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "semantic_search", "args": {"query": "sushi"}, "id": "tc0"}],
        ),
        ToolMessage(content="[]", tool_call_id="tc0"),
    ]

    hint = rev._diagnose_last_tool_result(state)
    assert hint is not None
    assert hint.reason == "low_similarity"


# ── Flag-on: suppression when all slots viable ───────────────────────────────


def test_flag_on_all_viable_low_similarity_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With VIABILITY_CONTRACT_ENABLED=1 and all_slots_viable==True,
    _diagnose_last_tool_result returns None (low_similarity suppressed).

    This is the core DEC-03 scoping behavior: once every requested stop has a
    viable candidate, low_similarity stops firing so the model can commit.
    """
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "1")
    monkeypatch.delenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", raising=False)

    import app.agent.revision as rev

    importlib.reload(rev)

    import app.agent.viability as via

    importlib.reload(via)

    threshold = rev.LOW_SIMILARITY_THRESHOLD  # 0.55 (default)

    # Build a state where:
    # - all_slots_viable is True (above-threshold hit for the requested type)
    # - but the MOST RECENT search result is below-threshold (would normally fire low_similarity)
    from langchain_core.messages import AIMessage, ToolMessage

    state = ItineraryState(
        scratch={
            "semantic_search": [
                # Step 0: above-threshold viable hit (makes all_slots_viable=True)
                {
                    "step": 0,
                    "args": {"query": "sushi"},
                    "result": [
                        _hit(
                            similarity=threshold + 0.1,
                            primary_type="Sushi Restaurant",
                            place_id="pid1",
                        )
                    ],
                    "id": "tc0",
                },
                # Step 1: below-threshold (the "last" search — would normally emit low_similarity)
                {
                    "step": 1,
                    "args": {"query": "sushi restaurant"},
                    "result": [
                        _hit(
                            similarity=0.30,
                            primary_type="Sushi Restaurant",
                            place_id="pid2",
                        )
                    ],
                    "id": "tc1",
                },
            ]
        },
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "args": {"query": "sushi restaurant"},
                        "id": "tc1",
                    }
                ],
            ),
            ToolMessage(content="[]", tool_call_id="tc1"),
        ],
        constraints=UserConstraints(
            requested_primary_types=["Sushi Restaurant"],
            num_stops=1,
        ),
    )

    hint = rev._diagnose_last_tool_result(state)
    assert hint is None, f"Expected None (suppressed) but got hint with reason={hint.reason!r}"


def test_flag_on_not_all_viable_low_similarity_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With VIABILITY_CONTRACT_ENABLED=1 but all_slots_viable==False,
    low_similarity still fires (pre-candidate steps keep the hint).
    """
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "1")
    monkeypatch.delenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", raising=False)

    import app.agent.revision as rev

    importlib.reload(rev)

    import app.agent.viability as via

    importlib.reload(via)

    # No viable candidates at all — all_slots_viable is False
    from langchain_core.messages import AIMessage, ToolMessage

    state = _state_with_search_hits(
        [_hit(similarity=0.30, primary_type="Sushi Restaurant", place_id="pid1")],
        requested_types=["Sushi Restaurant"],
    )
    state.messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "semantic_search", "args": {"query": "sushi"}, "id": "tc0"}],
        ),
        ToolMessage(content="[]", tool_call_id="tc0"),
    ]

    hint = rev._diagnose_last_tool_result(state)
    assert hint is not None
    assert hint.reason == "low_similarity"
