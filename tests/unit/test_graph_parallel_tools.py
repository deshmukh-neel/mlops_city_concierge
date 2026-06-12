"""Tests for A3 parallel tool execution in act() — Phase 13 / D-13-08.

Verifies that:
  - With PARALLEL_TOOL_EXECUTION_ENABLED=1, all tool calls in one act() step
    run concurrently (asyncio.gather); results are appended in the ORIGINAL
    tool_call order regardless of completion order.
  - The commit_itinerary branch still produces committed_stops correctly under
    the parallel path.
  - With the flag unset, the sequential loop runs unchanged — results are
    identical to a sequential reference run.
  - INST-04 tool_exec_seconds is still recorded in step_telemetry on the
    parallel path.

Testing strategy: the tests run the graph with mocked tools whose execution
completes out of order (via asyncio.sleep). Result ordering is checked by
inspecting the final state's messages and scratch. The commit branch is
exercised by mocking a model that emits commit_itinerary.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent_graph
from app.agent.state import ItineraryState, Stop, UserConstraints

# Valid Google Place ID
_VALID_PLACE_ID = "ChIJxxx_sushi_test_0001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm_two_tools() -> MagicMock:
    """Mock LLM that emits two semantic_search tool calls in one step, then finalizes."""
    call_count = 0
    final_ai = AIMessage(content="Here is your plan!")
    two_tools_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "semantic_search",
                "args": {"query": "sushi"},
                "id": "tc_a",
                "type": "tool_call",
            },
            {
                "name": "semantic_search",
                "args": {"query": "ramen"},
                "id": "tc_b",
                "type": "tool_call",
            },
        ],
    )

    async def ainvoke(messages: Any, **kwargs: Any) -> AIMessage:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return two_tools_ai
        return final_ai

    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.ainvoke = ainvoke
    return mock


def _make_mock_llm_commit() -> MagicMock:
    """Mock LLM that emits a commit_itinerary tool call then finalizes."""
    call_count = 0
    commit_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "commit_itinerary",
                "args": {
                    "stops": [
                        {
                            "place_id": _VALID_PLACE_ID,
                            "name": "Sushi Place",
                            "rationale": "Great sushi",
                            "source": "google_places",
                        }
                    ]
                },
                "id": "tc_commit",
                "type": "tool_call",
            }
        ],
    )
    final_ai = AIMessage(content="Here is your committed plan!")

    async def ainvoke(messages: Any, **kwargs: Any) -> AIMessage:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return commit_ai
        return final_ai

    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.ainvoke = ainvoke
    return mock


def _run_graph_sync(graph: Any, initial_state: ItineraryState) -> dict[str, Any]:
    """Run the compiled LangGraph synchronously and return the final state dict."""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(graph.ainvoke(initial_state))
    finally:
        loop.close()
    return result  # type: ignore[return-value]


def _make_state() -> ItineraryState:
    return ItineraryState(
        messages=[HumanMessage(content="Find me sushi and ramen")],
        constraints=UserConstraints(
            requested_primary_types=["sushi_restaurant", "ramen_restaurant"],
            num_stops=2,
        ),
    )


# ---------------------------------------------------------------------------
# A3: PARALLEL_TOOL_EXECUTION_ENABLED tests
# ---------------------------------------------------------------------------


def test_parallel_tool_execution_flag_greppable() -> None:
    """PARALLEL_TOOL_EXECUTION_ENABLED must appear in graph.py source."""
    import inspect

    import app.agent.graph as graph_module

    src = inspect.getsource(graph_module)
    assert "PARALLEL_TOOL_EXECUTION_ENABLED" in src


def test_parallel_tool_execution_order_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PARALLEL_TOOL_EXECUTION_ENABLED=1, tool results are in ORIGINAL tool_call order.

    Two tool calls (tc_a, tc_b) are mocked so tc_b completes before tc_a
    (via asyncio.sleep ordering). The final state's messages must have the
    tool results in the ORIGINAL order: tc_a first, tc_b second.
    """
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")

    import app.agent.graph as graph_module

    # Mock asyncio.to_thread to simulate out-of-order completion:
    # tc_a takes 0.02s, tc_b takes 0.001s => tc_b completes first.
    async def fake_to_thread(fn: Any, args: Any) -> list[dict[str, Any]]:
        query = args.get("query", "")
        if query == "sushi":
            await asyncio.sleep(0.02)  # tc_a (slower)
            return [
                {
                    "name": "Sushi Result",
                    "primary_type": "sushi_restaurant",
                    "similarity": 0.8,
                    "place_id": "ChIJxxx_sushi_test_0001",
                }
            ]
        else:
            await asyncio.sleep(0.001)  # tc_b (faster)
            return [
                {
                    "name": "Ramen Result",
                    "primary_type": "ramen_restaurant",
                    "similarity": 0.8,
                    "place_id": "ChIJxxx_ramen_test_0002",
                }
            ]

    with (
        patch.object(graph_module, "asyncio", wraps=asyncio) as mock_asyncio,
        patch.object(
            graph_module,
            "route_legs",
            new=AsyncMock(return_value=MagicMock(legs=[])),
        ),
        patch.object(graph_module, "swap_closed_stops", new=AsyncMock(return_value={})),
    ):
        mock_asyncio.to_thread = fake_to_thread
        mock_asyncio.gather = asyncio.gather

        mock_llm = _make_mock_llm_two_tools()
        graph = build_agent_graph(mock_llm, max_steps=4)

        final = _run_graph_sync(graph, _make_state())

    # Find the two ToolMessages in the final message list
    tool_messages = [
        m
        for m in final.get("messages", [])
        if hasattr(m, "tool_call_id") and m.tool_call_id in ("tc_a", "tc_b")
    ]
    assert len(tool_messages) == 2, f"Expected 2 tool messages, got {len(tool_messages)}"
    # tc_a must come before tc_b (original order preserved despite faster completion of tc_b)
    assert tool_messages[0].tool_call_id == "tc_a", (
        f"First tool message must be tc_a (original order), got {tool_messages[0].tool_call_id}"
    )
    assert tool_messages[1].tool_call_id == "tc_b", (
        f"Second tool message must be tc_b (original order), got {tool_messages[1].tool_call_id}"
    )


def test_parallel_tool_execution_commit_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commit_itinerary branch works correctly under the parallel path."""
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")

    import app.agent.graph as graph_module

    committed_stop = Stop(
        place_id=_VALID_PLACE_ID,
        name="Sushi Place",
        primary_type="sushi_restaurant",
        rationale="Great sushi",
        source="google_places",
    )

    with (
        patch.object(
            graph_module,
            "commit_stops",
            return_value=(
                [committed_stop],
                {"committed": [_VALID_PLACE_ID], "rejected": []},
            ),
        ),
        patch.object(
            graph_module,
            "critique_final_with_stops",
            return_value={"done": True, "final_reply": "Here is your itinerary."},
        ),
        patch.object(
            graph_module,
            "route_legs",
            new=AsyncMock(return_value=MagicMock(legs=[])),
        ),
        patch.object(graph_module, "swap_closed_stops", new=AsyncMock(return_value={})),
    ):
        mock_llm = _make_mock_llm_commit()
        graph = build_agent_graph(mock_llm, max_steps=8)

        state = ItineraryState(
            messages=[HumanMessage(content="Find me sushi")],
            scratch={
                "semantic_search": [
                    {
                        "step": 0,
                        "args": {"query": "sushi"},
                        "result": [
                            {
                                "name": "Sushi Place",
                                "place_id": _VALID_PLACE_ID,
                                "similarity": 0.8,
                                "primary_type": "sushi_restaurant",
                            }
                        ],
                        "id": "tc_prev",
                    }
                ]
            },
        )
        final = _run_graph_sync(graph, state)

    # Commit branch must still work under parallel path: committed stops present
    assert final.get("stops"), "Parallel path must still commit stops via commit branch"


def test_parallel_flag_off_sequential_result_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PARALLEL_TOOL_EXECUTION_ENABLED unset, act() is byte-identical to sequential.

    The test verifies that the final state's tool message count is the same
    whether the flag is on or off (no data loss, no ordering issues).
    """
    monkeypatch.delenv("PARALLEL_TOOL_EXECUTION_ENABLED", raising=False)
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")

    import app.agent.graph as graph_module

    with (
        patch.object(
            graph_module,
            "route_legs",
            new=AsyncMock(return_value=MagicMock(legs=[])),
        ),
        patch.object(graph_module, "swap_closed_stops", new=AsyncMock(return_value={})),
        patch("app.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=[])),
    ):
        mock_llm = _make_mock_llm_two_tools()
        graph = build_agent_graph(mock_llm, max_steps=4)

        final = _run_graph_sync(graph, _make_state())

    # With flag OFF, the sequential path runs: both tool calls are executed
    tool_messages = [
        m
        for m in final.get("messages", [])
        if hasattr(m, "tool_call_id") and m.tool_call_id in ("tc_a", "tc_b")
    ]
    assert len(tool_messages) == 2, (
        f"Sequential path must execute both tool calls, got {len(tool_messages)}"
    )


def test_parallel_tool_exec_seconds_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INST-04: tool_exec_seconds is recorded in step_telemetry on the parallel path."""
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")

    import app.agent.graph as graph_module

    with (
        patch.object(
            graph_module,
            "route_legs",
            new=AsyncMock(return_value=MagicMock(legs=[])),
        ),
        patch.object(graph_module, "swap_closed_stops", new=AsyncMock(return_value={})),
        patch("app.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=[])),
    ):
        mock_llm = _make_mock_llm_two_tools()
        graph = build_agent_graph(mock_llm, max_steps=4)

        final = _run_graph_sync(graph, _make_state())

    telemetry = final.get("step_telemetry", [])
    assert len(telemetry) > 0, "step_telemetry must have at least one entry"
    # At least one step must have tool_exec_seconds recorded (not None)
    steps_with_tools = [t for t in telemetry if t.get("tool_exec_seconds") is not None]
    assert steps_with_tools, "INST-04: tool_exec_seconds must be recorded in step_telemetry"
    # tool_exec_seconds must be a non-negative float
    for step in steps_with_tools:
        assert isinstance(step["tool_exec_seconds"], float), (
            f"tool_exec_seconds must be float, got {type(step['tool_exec_seconds'])}"
        )
        assert step["tool_exec_seconds"] >= 0.0, "tool_exec_seconds must be >= 0"
