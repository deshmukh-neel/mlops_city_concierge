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

Testing strategy: critique() is a sync closure inside build_agent_graph; it is
not directly accessible outside. Tests that need to verify critique() behavior
start the graph with a pre-seeded state (step_count at or above the threshold)
and a terminal last-message (ToolMessage after the last tool call) so the
graph immediately routes plan -> critique without hitting max_steps first, or
with step_count >= max_steps to hit short_circuit immediately. The mock LLM
uses AsyncMock for ainvoke to avoid "MagicMock can't be used in await".
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
from app.agent.state import ItineraryState, Stop, UserConstraints

# ---------------------------------------------------------------------------
# Valid Google Place ID for Stop construction (20+ chars, alphanumeric + _ + -)
# ---------------------------------------------------------------------------
_VALID_PLACE_ID = "ChIJxxx_sushi_test_0001"
_VALID_PLACE_ID_2 = "ChIJxxx_ramen_test_0002"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_llm_semantic_search_only() -> MagicMock:
    """Mock LLM that always emits a semantic_search tool call (never commits).

    ainvoke is an AsyncMock so the graph's `await llm_with_tools.ainvoke(...)` succeeds.
    """
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
    base_place_id: str = _VALID_PLACE_ID,
) -> dict[str, list[dict[str, Any]]]:
    """Scratch with n_viable semantic_search hits above threshold (each unique place_id)."""
    results = []
    for i in range(n_viable):
        # Build a unique 20+ char place_id by padding with a suffix
        pid = base_place_id[:-1] + str(i)
        results.append(
            {
                "name": f"Place {i}",
                "primary_type": primary_type,
                "similarity": LOW_SIMILARITY_THRESHOLD + 0.05,
                "place_id": pid,
            }
        )
    return {
        "semantic_search": [
            {
                "step": 0,
                "args": {"query": "sushi"},
                "result": results,
                "id": "tc0",
            }
        ]
    }


def _state_with_viable_scratch(
    n_viable: int = 1,
    primary_type: str = "sushi_restaurant",
    requested_primary_types: list[str] | None = None,
    step_count: int = 0,
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
        step_count=step_count,
    )


def _state_without_viable_scratch(step_count: int = 0) -> ItineraryState:
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
                            "place_id": _VALID_PLACE_ID,
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
        step_count=step_count,
    )


def _make_committed_stop() -> Stop:
    """Minimal valid Stop for use in commit_stops mock return values."""
    return Stop(
        place_id=_VALID_PLACE_ID,
        name="Sushi Place",
        primary_type="sushi_restaurant",
        rationale="Good sushi",
        source="google_places",
    )


def _run_graph_sync(graph: Any, initial_state: ItineraryState) -> dict[str, Any]:
    """Run the compiled LangGraph synchronously and return the final state dict."""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(graph.ainvoke(initial_state))
    finally:
        loop.close()
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Task 1: Arm flag reads at graph-build time
# ---------------------------------------------------------------------------


def test_forced_commit_step_flag_reads_at_build_time() -> None:
    """FORCED_COMMIT_STEP is read at graph-build time (greppable in graph.py)."""
    import inspect

    import app.agent.graph as graph_module

    src = inspect.getsource(graph_module)
    assert "FORCED_COMMIT_STEP" in src, "FORCED_COMMIT_STEP must appear in graph.py"
    assert "VIABILITY_CONTRACT_ENABLED" in src, "VIABILITY_CONTRACT_ENABLED must appear in graph.py"
    assert "PARALLEL_TOOL_EXECUTION_ENABLED" in src, (
        "PARALLEL_TOOL_EXECUTION_ENABLED must appear in graph.py"
    )


def test_prompt_addendum_flag_on_contains_cosine_similarity() -> None:
    """With VIABILITY_CONTRACT_ENABLED=1, the system prompt contains 'cosine similarity'."""
    from app.agent.prompts import rule8_viability_addendum

    rendered = SYSTEM_PROMPT.format(max_steps=8, current_datetime="2026-01-01 10:00 PST (Monday)")
    addendum = rule8_viability_addendum(True)
    full_prompt = rendered + addendum

    assert "cosine similarity" in full_prompt.lower()


def test_prompt_addendum_flag_off_no_cosine_similarity() -> None:
    """With VIABILITY_CONTRACT_ENABLED unset, the system prompt has NO 'cosine similarity'."""
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


@patch("app.agent.commit.get_details_many")
def test_forced_commit_triggers_at_step_n(
    mock_details: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-13-04: model that never calls commit_itinerary triggers forced commit at step N.

    Strategy: build a state with step_count already at FORCED_COMMIT_STEP (2) and
    the last message a ToolMessage (so the graph enters critique immediately after
    plan generates a tool call). With commit_stops and all_slots_viable mocked,
    the A2 branch fires when step_count >= _forced_commit_step AND all slots viable.
    The mock LLM uses AsyncMock so plan() can await it.
    """
    monkeypatch.setenv("FORCED_COMMIT_STEP", "2")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "0")

    mock_details.return_value = {}

    import app.agent.graph as graph_module

    committed_stop = _make_committed_stop()

    with (
        patch.object(graph_module, "all_slots_viable", return_value=True),
        patch.object(
            graph_module,
            "best_viable_candidate_per_slot",
            return_value=[
                {
                    "place_id": _VALID_PLACE_ID,
                    "name": "Sushi Place",
                    "primary_type": "sushi_restaurant",
                    "similarity": LOW_SIMILARITY_THRESHOLD + 0.05,
                    "rationale": "Good sushi",
                    "source": "google_places",
                }
            ],
        ),
        patch.object(
            graph_module,
            "commit_stops",
            return_value=(
                [committed_stop],
                {"committed": [_VALID_PLACE_ID], "rejected": []},
            ),
        ),
        # Mock critique_final_with_stops to return done=True (avoids DB connection)
        patch.object(
            graph_module,
            "critique_final_with_stops",
            return_value={"done": True, "final_reply": "Here is your sushi itinerary."},
        ),
        patch.object(
            graph_module,
            "route_legs",
            new=AsyncMock(return_value=MagicMock(legs=[])),
        ),
        patch.object(graph_module, "swap_closed_stops", new=AsyncMock(return_value={})),
    ):
        # Build graph after patches are applied so critique() closure captures patched fns
        mock_llm = _make_mock_llm_semantic_search_only()
        graph = build_agent_graph(mock_llm, max_steps=8)

        # State at step_count=1 — after act() increments to 2, critique fires at
        # step_count 2 >= FORCED_COMMIT_STEP=2.
        state = _state_with_viable_scratch(n_viable=1, step_count=1)

        with patch("app.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=[])):
            final = _run_graph_sync(graph, state)

    assert final.get("commit_forced") is True, "commit_forced must be True after A2 triggers"
    assert final.get("forced_commit_step") is not None, "forced_commit_step must be set"
    assert final.get("forced_commit_step") >= 2, (
        f"forced_commit_step must be >= FORCED_COMMIT_STEP=2, got {final.get('forced_commit_step')}"
    )
    # A forced commit produces real committed stops (non-empty)
    assert final.get("stops"), "A2 forced commit must produce committed stops"


def test_forced_commit_skipped_when_no_viable_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a slot lacks a viable candidate at step N, the forced commit does NOT fire.

    Strategy: mock all_slots_viable=False at step >= FORCED_COMMIT_STEP.
    Start the graph at step_count >= max_steps so it immediately hits
    short_circuit_max_steps (skipping plan). commit_forced must remain False.
    """
    monkeypatch.setenv("FORCED_COMMIT_STEP", "2")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "0")

    import app.agent.graph as graph_module

    with (
        patch.object(graph_module, "all_slots_viable", return_value=False),
        patch.object(
            graph_module,
            "route_legs",
            new=AsyncMock(return_value=MagicMock(legs=[])),
        ),
        patch.object(graph_module, "swap_closed_stops", new=AsyncMock(return_value={})),
    ):
        mock_llm = _make_mock_llm_semantic_search_only()
        graph = build_agent_graph(mock_llm, max_steps=8)

        # State at step_count=8 (>= max_steps=8) — critique fires immediately.
        # The A2 branch fires when step_count >= _forced_commit_step AND viable.
        # Since all_slots_viable=False, A2 skips; then step_count >= max_steps fires.
        state = _state_without_viable_scratch(step_count=8)

        final = _run_graph_sync(graph, state)

    # No viable candidate → forced commit skips → short_circuit_max_steps finalizes
    assert final.get("commit_forced", False) is False, (
        "commit_forced must stay False when no viable candidate"
    )
    assert final.get("forced_commit_step") is None, (
        "forced_commit_step must be None when forced commit does not fire"
    )


def test_forced_commit_step_zero_disables_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With FORCED_COMMIT_STEP unset/0, commit_forced stays False and run behaves as today.

    Strategy: start with step_count >= max_steps so short_circuit_max_steps fires.
    Even though all_slots_viable=True, the A2 branch is disabled (_forced_commit_step=0).
    """
    monkeypatch.delenv("FORCED_COMMIT_STEP", raising=False)
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "0")

    import app.agent.graph as graph_module

    with (
        patch.object(graph_module, "all_slots_viable", return_value=True),
        patch.object(
            graph_module,
            "route_legs",
            new=AsyncMock(return_value=MagicMock(legs=[])),
        ),
        patch.object(graph_module, "swap_closed_stops", new=AsyncMock(return_value={})),
    ):
        mock_llm = _make_mock_llm_semantic_search_only()
        graph = build_agent_graph(mock_llm, max_steps=8)

        # State at step_count=8 (>= max_steps=8) — A2 is off (flag=0), so
        # the branch is never entered even though all_slots_viable=True.
        state = _state_with_viable_scratch(n_viable=1, step_count=8)

        final = _run_graph_sync(graph, state)

    # FORCED_COMMIT_STEP=0 → A2 branch disabled → short_circuit_max_steps runs
    assert final.get("commit_forced", False) is False, (
        "commit_forced must be False when FORCED_COMMIT_STEP is 0/unset"
    )
    assert final.get("forced_commit_step") is None, (
        "forced_commit_step must be None when A2 is disabled"
    )
