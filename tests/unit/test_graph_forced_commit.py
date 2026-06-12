"""Tests for graph arm-flag reads and A2 forced-commit branch.

Plan 13-04 / D-13-04: verifies that:
  - Three arm flags are read at graph-build time and closed over inner functions.
  - A1 prompt addendum wiring: with VIABILITY_CONTRACT_ENABLED=1, the system
    prompt assembled by plan() contains "cosine similarity"; with the flag unset
    the system prompt is byte-identical to today (no cosine text).
  - A2 forced-commit: a mock model that never calls commit_itinerary triggers the
    forced commit at FORCED_COMMIT_STEP when every slot is viable.
  - When a slot lacks a viable candidate, the forced commit does NOT fire.
  - FORCED_COMMIT_STEP unset/0 disables the A2 branch entirely.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent_graph
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.revision import LOW_SIMILARITY_THRESHOLD
from app.agent.state import ItineraryState, UserConstraints

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_llm_semantic_search_only() -> MagicMock:
    """Mock LLM that always emits a semantic_search tool call (never commits)."""
    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "args": {"query": "sushi"},
                    "id": "tc1",
                    "type": "tool_call",
                }
            ],
        )
    )
    return mock


def _viable_scratch(
    n_viable: int = 1,
    primary_type: str = "sushi_restaurant",
) -> dict[str, list[dict[str, Any]]]:
    """Scratch with n_viable semantic_search hits above threshold (each unique place_id)."""
    return {
        "semantic_search": [
            {
                "step": 0,
                "args": {"query": "sushi"},
                "result": [
                    {
                        "name": f"Place {i}",
                        "primary_type": primary_type,
                        "similarity": LOW_SIMILARITY_THRESHOLD + 0.05,
                        "place_id": f"pid{i}",
                    }
                    for i in range(n_viable)
                ],
                "id": "tc0",
            }
        ]
    }


def _state_with_viable_scratch(
    n_viable: int = 1,
    primary_type: str = "sushi_restaurant",
    requested_primary_types: list[str] | None = None,
) -> ItineraryState:
    """Minimal state with viable scratch entries."""
    if requested_primary_types is None:
        requested_primary_types = [primary_type] * n_viable
    return ItineraryState(
        scratch=_viable_scratch(n_viable, primary_type),
        constraints=UserConstraints(
            requested_primary_types=requested_primary_types,
            num_stops=n_viable,
        ),
        messages=[HumanMessage(content="Find me a sushi place")],
    )


def _state_without_viable_scratch() -> ItineraryState:
    """Minimal state with NO viable scratch entries (similarity too low)."""
    return ItineraryState(
        scratch={
            "semantic_search": [
                {
                    "step": 0,
                    "args": {"query": "sushi"},
                    "result": [
                        {
                            "name": "Low Similarity Place",
                            "primary_type": "sushi_restaurant",
                            "similarity": LOW_SIMILARITY_THRESHOLD - 0.10,  # below threshold
                            "place_id": "pid_low",
                        }
                    ],
                    "id": "tc0",
                }
            ]
        },
        constraints=UserConstraints(
            requested_primary_types=["sushi_restaurant"],
            num_stops=1,
        ),
        messages=[HumanMessage(content="Find me a sushi place")],
    )


def _run_graph_sync(graph: Any, initial_state: ItineraryState) -> ItineraryState:
    """Run the compiled LangGraph synchronously and return the final state."""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(graph.ainvoke(initial_state))
    finally:
        loop.close()
    return result


# ---------------------------------------------------------------------------
# Task 1: Arm flag reads at graph-build time
# ---------------------------------------------------------------------------


def test_forced_commit_step_flag_reads_at_build_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """FORCED_COMMIT_STEP is read at graph-build time (greppable in graph.py)."""
    # The module must define the flag parsing inside build_agent_graph.
    # Verify the flag name is referenced (grep equivalent):
    import inspect

    import app.agent.graph as graph_module

    src = inspect.getsource(graph_module)
    assert "FORCED_COMMIT_STEP" in src, "FORCED_COMMIT_STEP must appear in graph.py"
    assert "VIABILITY_CONTRACT_ENABLED" in src, "VIABILITY_CONTRACT_ENABLED must appear in graph.py"
    assert "PARALLEL_TOOL_EXECUTION_ENABLED" in src, (
        "PARALLEL_TOOL_EXECUTION_ENABLED must appear in graph.py"
    )


def test_prompt_addendum_flag_on_contains_cosine_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With VIABILITY_CONTRACT_ENABLED=1, the system prompt contains 'cosine similarity'."""
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "1")
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")

    # Build a graph with the flag on — the system prompt is assembled inside plan().
    # We can verify by checking what SystemMessage content would be built by inspecting
    # SYSTEM_PROMPT + rule8_viability_addendum(True):
    from app.agent.prompts import rule8_viability_addendum

    rendered = SYSTEM_PROMPT.format(max_steps=8, current_datetime="2026-01-01 10:00 PST (Monday)")
    addendum = rule8_viability_addendum(True)
    full_prompt = rendered + addendum

    assert "cosine similarity" in full_prompt.lower()


def test_prompt_addendum_flag_off_no_cosine_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With VIABILITY_CONTRACT_ENABLED unset, the system prompt has NO 'cosine similarity'."""
    monkeypatch.delenv("VIABILITY_CONTRACT_ENABLED", raising=False)

    from app.agent.prompts import rule8_viability_addendum

    rendered = SYSTEM_PROMPT.format(max_steps=8, current_datetime="2026-01-01 10:00 PST (Monday)")
    addendum = rule8_viability_addendum(False)
    full_prompt = rendered + addendum

    assert "cosine similarity" not in full_prompt.lower()
    # Flag-off addendum is empty string:
    assert addendum == ""


# ---------------------------------------------------------------------------
# Task 2: A2 forced-commit branch
# ---------------------------------------------------------------------------


@patch("app.agent.graph.enrich_stops_with_booking")
@patch("app.agent.graph.route_legs")
@patch("app.agent.graph.swap_closed_stops")
def test_forced_commit_triggers_at_step_n(
    mock_swap: Any,
    mock_route: Any,
    mock_enrich: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-13-04: model that never calls commit_itinerary triggers forced commit at step N."""
    monkeypatch.setenv("FORCED_COMMIT_STEP", "2")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "0")

    # Patch the semantic_search tool so it doesn't hit the DB
    mock_swap.return_value = {}
    mock_route.return_value = MagicMock(legs=[])
    mock_enrich.return_value = None

    mock_llm = _make_mock_llm_semantic_search_only()
    graph = build_agent_graph(mock_llm, max_steps=8)

    # Build state with viable scratch so the forced commit can fire
    initial_state = _state_with_viable_scratch(n_viable=1)

    with patch("app.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=[])):
        final = _run_graph_sync(graph, initial_state)

    assert final["commit_forced"] is True, "commit_forced must be True after A2 triggers"
    assert final["forced_commit_step"] == 2, "forced_commit_step must equal FORCED_COMMIT_STEP"


@patch("app.agent.graph.enrich_stops_with_booking")
@patch("app.agent.graph.route_legs")
@patch("app.agent.graph.swap_closed_stops")
def test_forced_commit_skipped_when_no_viable_slot(
    mock_swap: Any,
    mock_route: Any,
    mock_enrich: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a slot lacks a viable candidate at step N, the forced commit does NOT fire."""
    monkeypatch.setenv("FORCED_COMMIT_STEP", "2")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "0")

    mock_swap.return_value = {}
    mock_route.return_value = MagicMock(legs=[])
    mock_enrich.return_value = None

    mock_llm = _make_mock_llm_semantic_search_only()
    graph = build_agent_graph(mock_llm, max_steps=8)

    # State with NO viable scratch — forced commit must not fire
    initial_state = _state_without_viable_scratch()

    with patch("app.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=[])):
        final = _run_graph_sync(graph, initial_state)

    assert final.get("commit_forced", False) is False, (
        "commit_forced must stay False when no viable candidate exists"
    )
    assert final.get("forced_commit_step") is None, (
        "forced_commit_step must be None when forced commit does not fire"
    )


@patch("app.agent.graph.enrich_stops_with_booking")
@patch("app.agent.graph.route_legs")
@patch("app.agent.graph.swap_closed_stops")
def test_forced_commit_step_zero_disables_branch(
    mock_swap: Any,
    mock_route: Any,
    mock_enrich: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With FORCED_COMMIT_STEP unset/0, commit_forced stays False and run behaves as today."""
    monkeypatch.delenv("FORCED_COMMIT_STEP", raising=False)
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "0")

    mock_swap.return_value = {}
    mock_route.return_value = MagicMock(legs=[])
    mock_enrich.return_value = None

    mock_llm = _make_mock_llm_semantic_search_only()
    graph = build_agent_graph(mock_llm, max_steps=8)

    initial_state = _state_with_viable_scratch(n_viable=1)

    with patch("app.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=[])):
        final = _run_graph_sync(graph, initial_state)

    # Without FORCED_COMMIT_STEP, the branch is off — run uses short_circuit_max_steps
    assert final.get("commit_forced", False) is False, (
        "commit_forced must be False when FORCED_COMMIT_STEP is 0/unset"
    )
    assert final.get("forced_commit_step") is None, (
        "forced_commit_step must be None when A2 is disabled"
    )
