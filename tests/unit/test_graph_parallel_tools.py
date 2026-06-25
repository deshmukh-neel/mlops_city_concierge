"""Tests for A3 parallel tool execution in act() — Phase 13 / D-13-08.

Verifies that:
  - With PARALLEL_TOOL_EXECUTION_ENABLED=1, all tool calls in one act() step
    run concurrently (asyncio.gather); results are appended in the ORIGINAL
    tool_call order regardless of completion order.
  - The commit_itinerary branch still produces committed_stops correctly under
    the parallel path.
  - With the flag unset, the sequential loop runs unchanged.
  - INST-04 tool_exec_seconds is still recorded in step_telemetry on the
    parallel path.

Testing strategy: tests use mock LLMs that commit in the first tool call (so
the graph terminates quickly) or a model that finalizes after two tool calls.
The commit_branch and order-stability tests rely on fully mocked act-level
helpers (commit_stops, critique_final_with_stops) so they terminate in one
plan->act->critique cycle. The flag-off test verifies sequential behavior by
checking the final state's tool-message count.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent_graph
from app.agent.revision import LOW_SIMILARITY_THRESHOLD
from app.agent.state import ItineraryState, Stop, UserConstraints

# Valid Google Place ID (>= 20 chars, alphanumeric + _ + -)
VALID_PLACE_ID = "ChIJxxx_sushi_test_0001"
VALID_PLACE_ID_2 = "ChIJxxx_ramen_test_0002"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_llm_two_tools_then_commit() -> MagicMock:
    """Mock LLM: step 0 emits two search tool calls, step 1 commits, step 2 finalizes."""
    call_count = 0
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
    commit_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "commit_itinerary",
                "args": {
                    "stops": [
                        {
                            "place_id": VALID_PLACE_ID,
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
    final_ai = AIMessage(content="Here is your plan!")

    async def ainvoke(messages: Any, **kwargs: Any) -> AIMessage:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return two_tools_ai
        if call_count == 2:
            return commit_ai
        return final_ai

    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.ainvoke = ainvoke
    return mock


def make_mock_llm_commit_only() -> MagicMock:
    """Mock LLM that emits a commit_itinerary tool call then finalizes immediately."""
    call_count = 0
    commit_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "commit_itinerary",
                "args": {
                    "stops": [
                        {
                            "place_id": VALID_PLACE_ID,
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


def run_graph_sync(graph: Any, initial_state: ItineraryState) -> dict[str, Any]:
    """Run the compiled LangGraph synchronously and return the final state dict."""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(graph.ainvoke(initial_state))
    finally:
        loop.close()
    return result  # type: ignore[return-value]


def make_state_with_scratch() -> ItineraryState:
    """State with existing viable scratch so commit can be grounded."""
    return ItineraryState(
        messages=[HumanMessage(content="Find me sushi and ramen")],
        scratch={
            "semantic_search": [
                {
                    "step": 0,
                    "args": {"query": "sushi"},
                    "result": [
                        {
                            "name": "Sushi Place",
                            "place_id": VALID_PLACE_ID,
                            "similarity": LOW_SIMILARITY_THRESHOLD + 0.05,
                            "primary_type": "sushi_restaurant",
                        },
                        {
                            "name": "Ramen Place",
                            "place_id": VALID_PLACE_ID_2,
                            "similarity": LOW_SIMILARITY_THRESHOLD + 0.05,
                            "primary_type": "ramen_restaurant",
                        },
                    ],
                    "id": "tc_prev",
                }
            ]
        },
        constraints=UserConstraints(
            requested_primary_types=["sushi_restaurant", "ramen_restaurant"],
            num_stops=2,
        ),
    )


def make_committed_stop() -> Stop:
    return Stop(
        place_id=VALID_PLACE_ID,
        name="Sushi Place",
        primary_type="sushi_restaurant",
        rationale="Great sushi",
        source="google_places",
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
    # Also check asyncio.gather is referenced (A3 implementation marker)
    assert "asyncio.gather" in src, "asyncio.gather must appear in graph.py (A3 implementation)"


def test_parallel_tool_execution_commit_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commit_itinerary branch works correctly under the parallel path.

    Uses a mock LLM that immediately commits so the graph terminates in
    one plan->act->critique cycle. Verifies that committed stops are set.
    """
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")

    import app.agent.graph as graph_module

    committed_stop = make_committed_stop()

    with (
        patch.object(
            graph_module,
            "commit_stops",
            return_value=(
                [committed_stop],
                {"committed": [VALID_PLACE_ID], "rejected": []},
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
        mock_llm = make_mock_llm_commit_only()
        graph = build_agent_graph(mock_llm, max_steps=8)

        final = run_graph_sync(graph, make_state_with_scratch())

    # Commit branch must still work under parallel path
    assert final.get("stops"), "Parallel path must still commit stops via commit branch"
    assert final.get("stops")[0].place_id == VALID_PLACE_ID


def test_parallel_tool_execution_order_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PARALLEL_TOOL_EXECUTION_ENABLED=1, tool results are in ORIGINAL order.

    Two tool calls (tc_a: sushi, tc_b: ramen) complete in parallel. Even if
    tc_b finishes before tc_a, the ToolMessages must be in ORIGINAL order:
    tc_a first, tc_b second.

    Strategy: run one plan->act->critique cycle. The mock LLM emits two tools
    then commits. commit_stops and critique_final_with_stops are mocked to
    terminate immediately. asyncio.gather is used inside act() — results are
    ordered by input index regardless of completion order.
    """
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")

    import app.agent.graph as graph_module

    committed_stop = make_committed_stop()

    # Simulate out-of-order async completion using real asyncio.sleep
    call_order: list[str] = []

    async def fake_to_thread(fn: Any, args: Any) -> list[dict[str, Any]]:
        query = args.get("query", "")
        if query == "sushi":
            await asyncio.sleep(0.02)  # tc_a is slower
            call_order.append("tc_a")
            return [
                {
                    "name": "Sushi Result",
                    "primary_type": "sushi_restaurant",
                    "similarity": 0.8,
                    "place_id": VALID_PLACE_ID,
                }
            ]
        else:
            await asyncio.sleep(0.001)  # tc_b is faster
            call_order.append("tc_b")
            return [
                {
                    "name": "Ramen Result",
                    "primary_type": "ramen_restaurant",
                    "similarity": 0.8,
                    "place_id": VALID_PLACE_ID_2,
                }
            ]

    with (
        patch.object(
            graph_module,
            "commit_stops",
            return_value=(
                [committed_stop],
                {"committed": [VALID_PLACE_ID], "rejected": []},
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
        # Patch asyncio.to_thread inside graph.py for the search tool calls
        patch.object(asyncio, "to_thread", side_effect=fake_to_thread),
    ):
        mock_llm = make_mock_llm_two_tools_then_commit()
        graph = build_agent_graph(mock_llm, max_steps=8)

        final = run_graph_sync(graph, make_state_with_scratch())

    # tc_b completed before tc_a (verified by _call_order), but tool messages
    # must be in ORIGINAL tool_call order (tc_a first, tc_b second)
    assert "tc_b" in call_order and "tc_a" in call_order, "Both tool calls must have completed"
    tc_b_pos = call_order.index("tc_b")
    tc_a_pos = call_order.index("tc_a")
    assert tc_b_pos < tc_a_pos, "tc_b must complete before tc_a (out-of-order simulation)"

    # Find the two search ToolMessages in the final message list
    tool_messages = [
        m
        for m in final.get("messages", [])
        if hasattr(m, "tool_call_id") and m.tool_call_id in ("tc_a", "tc_b")
    ]
    assert len(tool_messages) == 2, f"Expected 2 tool messages, got {len(tool_messages)}"
    # tc_a must come before tc_b (original input order preserved despite faster tc_b)
    assert tool_messages[0].tool_call_id == "tc_a", (
        f"First must be tc_a (original order), got {tool_messages[0].tool_call_id}"
    )
    assert tool_messages[1].tool_call_id == "tc_b", (
        f"Second must be tc_b (original order), got {tool_messages[1].tool_call_id}"
    )


def test_parallel_flag_off_sequential_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PARALLEL_TOOL_EXECUTION_ENABLED unset, act() runs sequentially.

    Verifies that with the flag off, both tool calls are still executed and
    their ToolMessages appear in the final state.
    """
    monkeypatch.delenv("PARALLEL_TOOL_EXECUTION_ENABLED", raising=False)
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")

    import app.agent.graph as graph_module

    committed_stop = make_committed_stop()

    with (
        patch.object(
            graph_module,
            "commit_stops",
            return_value=(
                [committed_stop],
                {"committed": [VALID_PLACE_ID], "rejected": []},
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
        patch("app.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=[])),
    ):
        mock_llm = make_mock_llm_two_tools_then_commit()
        graph = build_agent_graph(mock_llm, max_steps=8)

        final = run_graph_sync(graph, make_state_with_scratch())

    # Sequential path: both tool calls from the first plan step are in messages
    tool_messages = [
        m
        for m in final.get("messages", [])
        if hasattr(m, "tool_call_id") and m.tool_call_id in ("tc_a", "tc_b")
    ]
    assert len(tool_messages) == 2, (
        f"Sequential path must execute both tool calls, got {len(tool_messages)}"
    )
    # Sequential path preserves order as well
    assert tool_messages[0].tool_call_id == "tc_a"
    assert tool_messages[1].tool_call_id == "tc_b"


def test_parallel_tool_exec_seconds_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INST-04: tool_exec_seconds is recorded in step_telemetry on the parallel path."""
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")

    import app.agent.graph as graph_module

    committed_stop = make_committed_stop()

    with (
        patch.object(
            graph_module,
            "commit_stops",
            return_value=(
                [committed_stop],
                {"committed": [VALID_PLACE_ID], "rejected": []},
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
        patch("app.agent.graph.asyncio.to_thread", new=AsyncMock(return_value=[])),
    ):
        mock_llm = make_mock_llm_commit_only()
        graph = build_agent_graph(mock_llm, max_steps=8)

        final = run_graph_sync(graph, make_state_with_scratch())

    telemetry = final.get("step_telemetry", [])
    assert len(telemetry) > 0, "step_telemetry must have at least one entry"
    # INST-04: at least one step must have tool_exec_seconds as a non-negative float
    steps_with_tools = [t for t in telemetry if t.get("tool_exec_seconds") is not None]
    assert steps_with_tools, "INST-04: tool_exec_seconds must be recorded in step_telemetry"
    for step in steps_with_tools:
        assert isinstance(step["tool_exec_seconds"], float), (
            f"tool_exec_seconds must be float, got {type(step['tool_exec_seconds'])}"
        )
        assert step["tool_exec_seconds"] >= 0.0, "tool_exec_seconds must be non-negative"
