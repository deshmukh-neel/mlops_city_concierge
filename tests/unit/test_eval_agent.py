from __future__ import annotations

import inspect
import json
from argparse import Namespace
from dataclasses import asdict

import pytest

from app.agent.state import ItineraryState
from app.eval.config import EvalQuery, ExpectedConstraints
from scripts.eval_agent import (
    DETERMINISTIC_CHECKS,
    ActualEvalResult,
    CheckResult,
    DeterministicEvalResult,
    EvalRunReport,
    ExpectedEvalResult,
    QueryEvalResult,
    aggregate_results,
    answer_place_names_from_state,
    answer_retrieved_place_coverage,
    contexts_from_state,
    count_tool_calls,
    evaluate_multi_turn_case,
    expected_results_label,
    percentile,
    rate,
    report_has_errors,
    report_has_violations,
    resolve_chat_model,
    retrieved_place_names_from_state,
    revision_reasons_from_state,
    score_expected_results,
    selected_cases,
    state_from_graph_output,
    tool_errors_from_state,
    tool_names_from_state,
    validate_args,
    violations_for_case,
    violations_from_checks,
)


def eval_case(**overrides: object) -> EvalQuery:
    """Build a minimal EvalQuery for eval-agent unit tests."""
    payload = {
        "id": "case_one",
        "query": "coffee in soma",
        "reference": "Recommend a cafe in SOMA.",
        "expected_results": {
            "min_stops": 1,
            "max_stops": 3,
        },
    }
    payload.update(overrides)
    return EvalQuery.model_validate(payload)


def test_expected_constraints_requested_primary_types_defaults_empty() -> None:
    """Category-slot expectations are optional for backward-compatible eval cases."""
    assert ExpectedConstraints().requested_primary_types == []


def test_expected_constraints_accepts_requested_primary_types() -> None:
    """Per-slot Google primary_type expectations parse as a list of strings."""
    constraints = ExpectedConstraints(requested_primary_types=["Sushi Restaurant"])

    assert constraints.requested_primary_types == ["Sushi Restaurant"]


def test_expected_constraints_strips_blank_requested_primary_type_entries() -> None:
    """The existing list validator should clean requested_primary_types too."""
    constraints = ExpectedConstraints(requested_primary_types=["", "Sushi Restaurant"])

    assert constraints.requested_primary_types == ["Sushi Restaurant"]


def test_expected_constraints_keeps_single_shared_list_validator() -> None:
    """ADVISORY 6: do not duplicate the ExpectedConstraints list validator."""
    source = inspect.getsource(ExpectedConstraints)

    assert source.count("def _strip_non_empty_list") == 1


def query_result(**overrides: object) -> QueryEvalResult:
    """Build a QueryEvalResult with passing deterministic checks.

    Mirrors the shape of DETERMINISTIC_CHECKS in scripts/eval_agent.py — any
    scorer added to that dict must also appear here, otherwise
    aggregate_results raises KeyError when iterating per-scorer means.
    """
    checks = {
        "category_compliance": CheckResult(score=1.0, threshold=1.0, passed=True),
        "constraints_satisfied": CheckResult(score=1.0, threshold=0.8, passed=True),
        "geographic_coherence": CheckResult(score=1.0, threshold=1.0, passed=True),
        "no_hallucinated_place_ids": CheckResult(score=1.0, threshold=1.0, passed=True),
        "rationale_stop_alignment": CheckResult(score=1.0, threshold=1.0, passed=True),
        "temporal_coherence": CheckResult(score=1.0, threshold=1.0, passed=True),
        "walking_budget_respected": CheckResult(score=1.0, threshold=1.0, passed=True),
    }
    payload = {
        "id": "case_one",
        "question": "coffee in soma",
        "answer": "Try Example Cafe.",
        "contexts": ["name: Example Cafe | snippet: good coffee"],
        "reference": "Recommend a cafe in SOMA.",
        "tags": ["cafe"],
        "expected": ExpectedEvalResult(
            min_stops=1,
            max_stops=3,
            expects_clarification_or_relaxation=False,
        ),
        "actual": ActualEvalResult(
            result_count=1,
            committed_stop_count=1,
            place_ids=["place-1"],
            place_names=["Example Cafe"],
            sources=["google_places"],
            answer_place_names=["Example Cafe"],
        ),
        "deterministic": DeterministicEvalResult(
            expected_results_met=True,
            checks=checks,
            violations=[],
            tool_errors=[],
            first_tool_error=None,
            tool_calls=2,
            tool_names=["semantic_search"],
            revision_hints=0,
            revision_reasons=[],
        ),
        "final_reply": "Try Example Cafe.",
        "latency_seconds": 1.0,
    }
    payload.update(overrides)
    return QueryEvalResult(**payload)


def test_selected_cases_returns_all_cases_without_limit() -> None:
    """Return all hand-written cases when no quick-run limit is provided."""
    cases = [eval_case(id="case_one"), eval_case(id="case_two")]

    assert selected_cases(cases, None) == cases


def test_selected_cases_rejects_non_positive_limit() -> None:
    """Reject max-query values that would silently run no evals."""
    with pytest.raises(ValueError, match="max-queries"):
        selected_cases([eval_case()], 0)


def test_validate_args_rejects_non_positive_max_steps() -> None:
    """Reject max-step values that would prevent the graph from planning."""
    args = Namespace(max_steps=0, max_queries=None, temperature=0.0)

    with pytest.raises(ValueError, match="max-steps"):
        validate_args(args)


def test_validate_args_rejects_non_positive_max_queries() -> None:
    """IN-01: surface --max-queries violations at validate_args rather than
    letting them bubble up through selected_cases as a generic ValueError
    traceback in main()."""
    args = Namespace(max_steps=8, max_queries=0, temperature=0.0)

    with pytest.raises(ValueError, match="max-queries"):
        validate_args(args)


def test_validate_args_accepts_omitted_max_queries() -> None:
    """IN-01: max_queries=None means 'run all cases' and must not trip the
    positive-int check (argparse default for --max-queries is None)."""
    args = Namespace(max_steps=8, max_queries=None, temperature=0.0)

    validate_args(args)  # must not raise


@pytest.mark.parametrize("bad_temp", [-0.1, 2.1, 5.0])
def test_validate_args_rejects_out_of_range_temperature(bad_temp: float) -> None:
    """IN-01: argparse accepts any float for --temperature; reject anything
    outside [0.0, 2.0] (the LLM provider's accepted range) at validate_args
    so an operator sees an actionable CLI error not a vendor-API 400."""
    args = Namespace(max_steps=8, max_queries=None, temperature=bad_temp)

    with pytest.raises(ValueError, match="temperature"):
        validate_args(args)


def test_validate_args_accepts_temperature_at_boundaries() -> None:
    """IN-01: 0.0 and 2.0 are inclusive endpoints of the accepted range."""
    for ok_temp in (0.0, 1.0, 2.0):
        args = Namespace(max_steps=8, max_queries=None, temperature=ok_temp)
        validate_args(args)  # must not raise


def test_state_from_graph_output_accepts_model_and_dict() -> None:
    """Normalize both LangGraph dict output and direct state objects."""
    state = ItineraryState()

    assert state_from_graph_output(state) is state
    assert isinstance(state_from_graph_output(state.model_dump()), ItineraryState)


def test_state_from_graph_output_rejects_unknown_type() -> None:
    """Raise clearly when LangGraph returns an unexpected shape."""
    with pytest.raises(TypeError, match="Unexpected graph output"):
        state_from_graph_output(["not", "a", "state"])


def test_count_tool_calls_counts_list_entries_only() -> None:
    """Count captured tool calls while ignoring non-list scratch values."""
    state = ItineraryState(
        scratch={
            "semantic_search": [{"id": "a"}, {"id": "b"}],
            "debug": {"not": "a tool-call list"},
        }
    )

    assert count_tool_calls(state) == 2


def test_tool_names_from_state_returns_non_empty_tool_keys() -> None:
    """Return scratchpad tool names with at least one captured entry."""
    state = ItineraryState(
        scratch={
            "semantic_search": [{"id": "a"}],
            "nearby": [],
            "debug": {"ignored": True},
        }
    )

    assert tool_names_from_state(state) == ["semantic_search"]


def test_tool_errors_from_state_extracts_tool_error_payloads() -> None:
    """Extract readable tool errors from scratchpad entries."""
    state = ItineraryState(
        scratch={
            "semantic_search": [
                {"result": {"error": "permission denied"}},
                {"result": [{"place_id": "p1"}]},
            ],
            "debug": {"ignored": True},
        }
    )

    assert tool_errors_from_state(state) == ["semantic_search: permission denied"]


def test_contexts_from_state_extracts_retrieved_snippets() -> None:
    """Extract RAGAS-ready contexts from retrieval tool results."""
    state = ItineraryState(
        scratch={
            "semantic_search": [
                {
                    "result": [
                        {
                            "name": "Example Cafe",
                            "primary_type": "cafe",
                            "formatted_address": "123 Main",
                            "rating": 4.5,
                            "snippet": "Quiet cafe with late hours.",
                        }
                    ]
                }
            ],
        }
    )

    assert contexts_from_state(state) == [
        "name: Example Cafe | type: cafe | address: 123 Main | "
        "rating: 4.5 | snippet: Quiet cafe with late hours."
    ]


def test_retrieved_place_names_from_state_returns_unique_names() -> None:
    """Extract unique place names from retrieval tool results."""
    state = ItineraryState(
        scratch={
            "semantic_search": [
                {"result": [{"name": "Example Cafe"}, {"name": "Example Cafe"}]},
                {"result": [{"name": "Second Cafe"}]},
            ],
        }
    )

    assert retrieved_place_names_from_state(state) == ["Example Cafe", "Second Cafe"]


def test_answer_place_names_from_state_matches_names_in_reply() -> None:
    """Count user-visible options when no structured stops are committed."""
    state = ItineraryState(
        final_reply="Try Example Cafe, or Second Cafe if you want more space.",
        scratch={
            "semantic_search": [
                {"result": [{"name": "Example Cafe"}, {"name": "Second Cafe"}]},
            ],
        },
    )

    assert answer_place_names_from_state(state) == ["Example Cafe", "Second Cafe"]


def test_revision_reasons_from_state_returns_hint_reasons() -> None:
    """Expose critique reasons for easier eval debugging."""
    state = ItineraryState(
        revision_hints=[
            {
                "reason": "empty_results",
                "detail": "nothing matched",
                "suggested_action": "drop_filter",
            },
            {
                "reason": "low_similarity",
                "detail": "weak match",
                "suggested_action": "broaden_query",
            },
        ]
    )

    assert revision_reasons_from_state(state) == ["empty_results", "low_similarity"]


def test_score_expected_results_handles_relaxation_cases() -> None:
    """Skip stop-count scoring when the eval case expects relaxation."""
    case = eval_case(expected_results=None, expects_clarification_or_relaxation=True)
    actual = ActualEvalResult(0, 0, [], [], [], [])

    assert score_expected_results(case, actual) is None


def test_score_expected_results_accepts_counts_inside_range() -> None:
    """Pass flexible result-count expectations when result count is in range."""
    actual = ActualEvalResult(
        result_count=2,
        committed_stop_count=0,
        place_ids=[],
        place_names=["P1", "P2"],
        sources=[],
        answer_place_names=["P1", "P2"],
    )
    case = eval_case()

    assert score_expected_results(case, actual) is True


def test_rate_handles_empty_denominators() -> None:
    """Return a stable zero rate for empty eval suites."""
    assert rate(3, 0) == 0.0
    assert rate(1, 4) == 0.25


def test_percentile_uses_nearest_rank() -> None:
    """Summarize latency distributions without extra dependencies."""
    values = [1.0, 5.0, 2.0, 10.0]

    assert percentile([], 50) == 0.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 2.0
    assert percentile(values, 95) == 10.0
    assert percentile(values, 100) == 10.0


def test_answer_retrieved_place_coverage_scores_grounded_answer_names() -> None:
    """Score how many produced options are visible in retrieved contexts."""
    assert answer_retrieved_place_coverage(query_result()) == 1.0
    assert (
        answer_retrieved_place_coverage(
            query_result(
                actual=ActualEvalResult(
                    result_count=2,
                    committed_stop_count=0,
                    place_ids=[],
                    place_names=["P1", "P2"],
                    sources=[],
                    answer_place_names=["P1"],
                )
            )
        )
        == 0.5
    )
    assert (
        answer_retrieved_place_coverage(
            query_result(
                actual=ActualEvalResult(
                    result_count=0,
                    committed_stop_count=0,
                    place_ids=[],
                    place_names=[],
                    sources=[],
                    answer_place_names=[],
                )
            )
        )
        is None
    )


def test_expected_results_label_formats_range() -> None:
    """Format expected result ranges for eval-only system context."""
    case = eval_case()

    assert expected_results_label(case.expected_results) == "1 to 3 results"


def test_violations_from_checks_includes_errors_and_low_scores() -> None:
    """Treat failed checks and errored checks as report violations."""
    checks = {
        "ok": CheckResult(score=1.0, threshold=1.0, passed=True),
        "low": CheckResult(score=0.0, threshold=1.0, passed=False),
        "error": CheckResult(score=None, threshold=1.0, passed=False, error="db down"),
    }

    assert violations_from_checks(checks) == ["low", "error"]


def test_violations_for_case_includes_expected_results_mismatch() -> None:
    """Treat a result-count mismatch as an eval violation."""
    checks = {"ok": CheckResult(score=1.0, threshold=1.0, passed=True)}

    assert violations_for_case(False, checks) == ["expected_results"]


def test_aggregate_results_flattens_mean_metrics() -> None:
    """Aggregate per-query checks into flat numeric metrics."""
    result_one = query_result()
    result_two = query_result(
        actual=ActualEvalResult(
            result_count=4,
            committed_stop_count=4,
            place_ids=["p1", "p2", "p3", "p4"],
            place_names=["P1", "P2", "P3", "P4"],
            sources=["google_places"],
            answer_place_names=[],
        ),
        deterministic=DeterministicEvalResult(
            expected_results_met=False,
            checks={
                **result_one.deterministic.checks,
                "constraints_satisfied": CheckResult(score=0.5, threshold=0.8, passed=False),
            },
            violations=["constraints_satisfied"],
            tool_errors=[],
            first_tool_error=None,
            tool_calls=4,
            tool_names=["semantic_search"],
            revision_hints=1,
            revision_reasons=["low_similarity"],
        ),
    )

    aggregate = aggregate_results([result_one, result_two])

    assert aggregate["query_count"] == 2
    assert aggregate["queries_with_violations"] == 1
    assert aggregate["deterministic_pass_rate"] == 0.5
    assert aggregate["deterministic_violation_rate"] == 0.5
    assert aggregate["expected_results_mismatch_count"] == 1
    assert aggregate["expected_results_mismatch_rate"] == 0.5
    assert aggregate["tool_error_count"] == 0
    assert aggregate["queries_with_tool_errors"] == 0
    assert aggregate["tool_error_rate"] == 0.0
    assert aggregate["tool_success_rate"] == 1.0
    assert aggregate["expected_results_match_rate"] == 0.5
    assert aggregate["results_mean"] == 2.5
    assert aggregate["committed_stops_mean"] == 2.5
    assert aggregate["committed_itinerary_rate"] == 1.0
    assert aggregate["contexts_mean"] == 1.0
    assert aggregate["context_presence_rate"] == 1.0
    assert aggregate["answer_retrieved_place_coverage_mean"] == 0.5
    assert aggregate["answer_retrieved_place_coverage_count"] == 2
    assert aggregate["tool_calls_mean"] == 3.0
    assert aggregate["revision_hints_mean"] == 0.5
    assert aggregate["latency_total_seconds"] == 2.0
    assert aggregate["latency_mean_seconds"] == 1.0
    assert aggregate["latency_p50_seconds"] == 1.0
    assert aggregate["latency_p95_seconds"] == 1.0
    assert aggregate["latency_max_seconds"] == 1.0
    assert aggregate["constraints_satisfied_mean"] == 0.75
    assert aggregate["geographic_coherence_mean"] == 1.0


def test_report_has_errors_reads_aggregate_error_count() -> None:
    """Surface deterministic-check exceptions as a failing process condition."""
    report = EvalRunReport(
        eval_queries_path="configs/eval_queries.yaml",
        llm_provider="openai",
        chat_model="gpt-4o-mini",
        query_count=1,
        aggregate={"check_error_count": 1},
        queries=[query_result()],
    )

    assert report_has_errors(report) is True


def test_report_has_violations_reads_aggregate_violation_count() -> None:
    """Surface expected-behavior failures as a failing process condition."""
    report = EvalRunReport(
        eval_queries_path="configs/eval_queries.yaml",
        llm_provider="openai",
        chat_model="gpt-4o-mini",
        query_count=1,
        aggregate={"queries_with_violations": 1},
        queries=[query_result()],
    )

    assert report_has_violations(report) is True


def test_resolve_chat_model_uses_cli_value_when_present() -> None:
    """Prefer explicit CLI model names over provider defaults."""
    assert resolve_chat_model("openai", " custom-model ") == "custom-model"


# --- Plan 03-03 wiring: both new scorers must surface in eval reports -------


def test_deterministic_checks_registers_category_and_rationale_scorers() -> None:
    """Plan 03-03: category_compliance + rationale_stop_alignment ship in
    DETERMINISTIC_CHECKS so aggregate_results emits per-scorer mean metrics."""
    assert "category_compliance" in DETERMINISTIC_CHECKS
    assert "rationale_stop_alignment" in DETERMINISTIC_CHECKS


def test_aggregate_results_emits_new_scorer_mean_keys() -> None:
    """The aggregate dict must carry {scorer}_mean for both new scorers so
    baseline JSON has a stable field for Phase 4-6 merge-gate diffing."""
    aggregate = aggregate_results([query_result()])
    assert "category_compliance_mean" in aggregate
    assert "rationale_stop_alignment_mean" in aggregate
    # Passing-fixture builder uses score=1.0 for both -> mean is 1.0.
    assert aggregate["category_compliance_mean"] == 1.0
    assert aggregate["rationale_stop_alignment_mean"] == 1.0


def test_query_result_serializes_to_json_via_asdict() -> None:
    """EVAL-08 / P1 regression guard: the eval report is dataclass-based and
    its wire format is json.dumps(asdict(result)). Any new field on
    QueryEvalResult (or any sub-dataclass) that smuggles a non-json-safe
    object — e.g. a Pydantic model, a datetime, a set — will crash this
    serialization at report-write time. PR #94 fixed an AIMessage.tool_calls
    args[...] Pydantic regression at the agent layer; this test pins the
    same contract at the eval-report layer for the new scorer surface."""
    payload = asdict(query_result())
    # If this raises, the dataclass surface picked up a non-json-safe value.
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["deterministic"]["checks"]["category_compliance"]["score"] == 1.0
    assert decoded["deterministic"]["checks"]["rationale_stop_alignment"]["score"] == 1.0


# --- Plan 03-04: multi-turn runner (EVAL-06) -------------------------------


def test_evaluate_multi_turn_case_is_async_helper() -> None:
    """Plan 03-04 / EVAL-06: scripts/eval_agent.py exposes an async helper
    `evaluate_multi_turn_case(graph, case)` that runs len(case.turns)+1
    invocations against a shared graph. This test fails at import time on
    the pre-03-04 codebase (the symbol does not exist) — the strongest form
    of RED — and turns green once the helper lands."""
    assert inspect.iscoroutinefunction(evaluate_multi_turn_case)


# --- Plan 03-04 Task 2: multi-turn behavior tests (EVAL-06 + EVAL-08) ------
# The five tests below are the canonical behavior contract for the multi-turn
# runner. They use a RecordingScriptedLLM (shared helper) that doubles as a sniffer for the
# messages each plan() step actually saw, so threading + json-safety can be
# asserted without subclassing langgraph internals.
#
# Project-memory invariants this section honors:
#   - `project_full_suite_db_pool_contamination`: build all test states with
#     stops=[] AND mock `app.agent.revision.itinerary_violations` to [] so no
#     DB-touching scorer fires on hallucinated place_ids during full-suite
#     runs. Without this, a live DB pool leaks via load_dotenv at collection
#     time and the scripted LLM gets exhausted by the revision loop.
#   - `project_aimessage_tool_call_args_json_safe`: PR #94 commit be541a3
#     fixed an AIMessage.tool_calls[i]["args"] Pydantic-in-filters bug that
#     crashed the NEXT plan() step on json.dumps. The EVAL-08 test below
#     locks that contract on the multi-turn message-threading boundary
#     (turn N's AIMessages are re-injected into turn N+1's input state).


from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from app.agent.graph import build_agent_graph  # noqa: E402
from scripts.eval_agent import evaluate_case  # noqa: E402
from tests._helpers.scripted_llm import RecordingScriptedLLM  # noqa: E402


def _finalize_msg(content: str) -> AIMessage:
    """Trajectory shorthand: a no-tool-calls AIMessage that finalizes a turn.

    With stops=[] + constraints.num_stops=None, this routes
    plan -> critique -> finalize_as_is -> done -> retime no-op -> swap no-op
    -> END, so each turn ends in a single plan() invocation. Keeps the
    scripted list tiny (one message per turn) and the test fast."""
    return AIMessage(content=content, tool_calls=[])


@pytest.mark.asyncio
async def test_evaluate_case_single_turn_unchanged(mocker) -> None:
    """Backward-compat regression guard: EvalQuery.turns=None must run
    through evaluate_case via the pre-03-04 single-turn code path. We assert
    on observable contract — exactly one plan() invocation, no synthetic
    `multi_turn_runner` tool error, and the final_reply is the scripted
    AIMessage's content — rather than byte-comparing JSON, which is the
    same guarantee surfaced differently."""
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RecordingScriptedLLM(scripted=[_finalize_msg("turn1 reply")])
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=None)
    result = await evaluate_case(graph, case)

    assert result.final_reply == "turn1 reply"
    # Single-turn path = exactly one plan() invocation.
    assert len(llm.seen) == 1
    # Single-turn path must NOT inject the multi_turn_runner synthetic error.
    assert all("multi_turn_runner" not in err for err in result.deterministic.tool_errors)


@pytest.mark.asyncio
async def test_evaluate_multi_turn_threads_messages(mocker) -> None:
    """EVAL-06: turn N+1's input state must contain turn N's HumanMessage so
    the agent sees prior conversation. Asserts on what the LLM ACTUALLY SAW
    on turn 2 (via RecordingScriptedLLM.seen[1]) — the strongest threading
    proof. If we only checked result.final_reply we'd miss a regression that
    nukes the prior turn's messages."""
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RecordingScriptedLLM(
        scripted=[_finalize_msg("turn1 reply"), _finalize_msg("turn2 reply")],
    )
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(query="coffee in soma", turns=["make stop 2 cheaper"])
    result = await evaluate_case(graph, case)

    # Final reply comes from the LAST turn's state, not the first.
    assert result.final_reply == "turn2 reply"
    # Two plan() invocations: one per turn.
    assert len(llm.seen) == 2
    # Turn 2's input messages must include BOTH HumanMessages.
    turn_two_human_contents = [m.content for m in llm.seen[1] if isinstance(m, HumanMessage)]
    assert "coffee in soma" in turn_two_human_contents
    assert "make stop 2 cheaper" in turn_two_human_contents
    # WR-06: the eval_context SystemMessage must survive into turn 2 with its
    # substring intact ("Expected open time:" is part of the EVAL_CONTEXT_TEMPLATE).
    # Asserting "any SystemMessage present" is too weak — a future refactor of
    # add_messages that strips system messages, or a `state.model_copy(update=...)`
    # rewrite that drops the eval context, would silently regress multi-turn cases.
    turn_two_systems = [m.content for m in llm.seen[1] if isinstance(m, SystemMessage)]
    assert any("Expected open time:" in c for c in turn_two_systems), (
        "eval_context SystemMessage substring must survive into turn 2 — "
        "this pins the WR-06 invariant against future add_messages refactors."
    )


@pytest.mark.asyncio
async def test_multi_turn_latency_sums(mocker) -> None:
    """EVAL-06 / aggregate-latency contract: latency_seconds is the SUM of
    per-turn elapsed time, not the max or the last.

    Replaces ONLY scripts.eval_agent's reference to the `time` module with
    a fake whose monotonic() pops deterministic floats from a controlled
    list. asyncio's event loop holds its own reference to the real `time`
    module, so this surgical patch isolates the helper's clock from the
    test's asyncio plumbing — without it, asyncio drains the side_effect
    iterator and StopIteration crashes the loop. Each ainvoke consumes
    exactly two monotonic() calls (start, end); a 2-turn run = 4 calls
    => total_latency = (1.0 - 0.0) + (3.0 - 1.0) = 3.0."""
    import types

    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    ticks = iter([0.0, 1.0, 1.0, 3.0])
    fake_time = types.SimpleNamespace(monotonic=lambda: next(ticks))
    mocker.patch("scripts.eval_agent.time", fake_time)
    llm = RecordingScriptedLLM(
        scripted=[_finalize_msg("turn1"), _finalize_msg("turn2")],
    )
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=["refine"])
    result = await evaluate_case(graph, case)

    assert result.latency_seconds == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_multi_turn_intermediate_failure_captured(mocker) -> None:
    """EVAL-06 fail-open contract: if any turn raises, the helper records a
    synthetic `multi_turn_runner` tool error and returns a partial
    QueryEvalResult instead of crashing the whole eval run (mirrors the
    existing fail-open pattern in evaluate_cases).

    We force turn 2 to raise by scripting only one AIMessage — the second
    invocation pops from an empty list and raises IndexError. The plan()
    coroutine propagates it; our helper's try/except catches it."""
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RecordingScriptedLLM(scripted=[_finalize_msg("turn1 reply")])
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=["this turn will explode"])
    result = await evaluate_case(graph, case)

    # The run did NOT crash — we have a result.
    assert isinstance(result, QueryEvalResult)
    # Turn 1's reply survives in the partial state's final_reply.
    assert result.final_reply == "turn1 reply"
    # The synthetic multi_turn_runner error is in tool_errors with the
    # failing turn's index threaded through the message.
    assert any(
        "multi_turn_runner" in err and "turn 1" in err for err in result.deterministic.tool_errors
    )
    # Two plan() invocations were attempted (the second is the one that
    # raised). The recorder shows turn 1 succeeded.
    assert len(llm.seen) >= 1


@pytest.mark.asyncio
async def test_multi_turn_tool_calls_are_json_safe(mocker) -> None:
    """EVAL-08 / P1 regression guard (PR #94 commit be541a3): every
    AIMessage.tool_calls[i]["args"] observed during multi-turn execution
    must remain json.dumps-safe, AND the final QueryEvalResult dataclass
    wire shape must serialize cleanly via asdict() -> json.dumps().

    PR #94 fixed an AIMessage.tool_calls args[...] Pydantic-in-filters bug
    that crashed the NEXT plan() step on the agent's json.dumps. The
    multi-turn helper re-injects turn N's AIMessages into turn N+1's input
    state, so the same regression class applies at this boundary. If a
    future change smuggles a Pydantic model, a datetime, or any other
    non-json-safe object into args, this test fails at the asdict-level
    json.dumps OR at the per-tool-call args walk."""
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RecordingScriptedLLM(
        scripted=[_finalize_msg("turn1 reply"), _finalize_msg("turn2 reply")],
    )
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=["refine please"])
    result = await evaluate_case(graph, case)

    # Wire-contract: the dataclass result must json.dumps without raising.
    # If a Pydantic model or datetime ever leaks into a QueryEvalResult
    # field via the multi-turn helper's state construction, asdict ->
    # json.dumps catches it here at the eval-report wire boundary.
    json.dumps(asdict(result))

    # Walk every AIMessage the LLM saw across all turns and assert
    # json.dumps over each tool_call's args. With our finalize-only
    # script this loop is vacuous (no tool_calls emitted), but it locks
    # in the contract shape so a future test that scripts a tool-calling
    # trajectory inherits the same guarantee for free.
    for messages_for_turn in llm.seen:
        for msg in messages_for_turn:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    # be541a3: never mutate tc["args"]; the next plan()
                    # step re-serializes the whole AIMessage for the API.
                    json.dumps(tc["args"])


# ─── Plan 03-05 Task 1: --llm-provider scripted + --scenario-ids ─────────────


def test_parse_args_accepts_scripted_provider() -> None:
    """EVAL-09 / P4: scripted is a first-class --llm-provider choice."""
    from scripts.eval_agent import parse_args

    args = parse_args(["--llm-provider", "scripted"])
    assert args.llm_provider == "scripted"


def test_parse_args_accepts_scenario_ids_flag() -> None:
    """--scenario-ids is comma-separated; default None means 'run all cases'."""
    from scripts.eval_agent import parse_args

    args = parse_args(["--llm-provider", "openai", "--scenario-ids", "case_one,case_two"])
    # Either a list or a comma string is acceptable as long as filter_cases
    # below honors it. The spec says "comma-separated list".
    assert args.scenario_ids == ["case_one", "case_two"]


def test_parse_args_scenario_ids_defaults_to_none() -> None:
    """Forward-compatible: omitting --scenario-ids must keep existing CLI
    usage (no filter) working exactly as before."""
    from scripts.eval_agent import parse_args

    args = parse_args(["--llm-provider", "openai"])
    assert args.scenario_ids is None


def test_parse_args_scenario_ids_strips_blanks() -> None:
    """Comma-trim semantics: ' a , , b ' -> ['a', 'b']."""
    from scripts.eval_agent import parse_args

    args = parse_args(["--llm-provider", "scripted", "--scenario-ids", " a , , b "])
    assert args.scenario_ids == ["a", "b"]


def test_resolve_chat_model_scripted_needs_no_env_vars(monkeypatch) -> None:
    """resolve_chat_model('scripted', None) MUST NOT call get_settings or
    read env vars — the CI matrix run sets no keys."""
    for key in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    # If scripted leaked into a real settings lookup, this would crash.
    model = resolve_chat_model("scripted", None)
    assert isinstance(model, str)
    assert model  # non-empty sentinel


def test_resolve_chat_model_scripted_passes_through_cli_value() -> None:
    """A user-supplied --chat-model is still honored for scripted (it's a
    label in the report; the actual model isn't used)."""
    assert resolve_chat_model("scripted", "my-label") == "my-label"


def test_build_eval_llm_scripted_returns_basechatmodel(monkeypatch) -> None:
    """build_eval_llm('scripted', ...) returns a usable BaseChatModel with
    no env-var dependencies."""
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    from langchain_core.language_models import BaseChatModel

    from scripts.eval_agent import build_eval_llm

    llm = build_eval_llm("scripted", "placeholder", 0.0)
    assert isinstance(llm, BaseChatModel)


def test_selected_cases_filters_by_scenario_ids() -> None:
    """When scenario_ids is provided, selected_cases keeps only matching IDs."""
    from scripts.eval_agent import selected_cases

    cases = [
        eval_case(id="case_one"),
        eval_case(id="case_two"),
        eval_case(id="case_three"),
    ]
    filtered = selected_cases(cases, None, scenario_ids=["case_two"])
    assert [c.id for c in filtered] == ["case_two"]


def test_selected_cases_scenario_ids_preserves_yaml_order() -> None:
    """Filter preserves YAML order even when --scenario-ids lists them
    differently — deterministic baselines depend on stable order."""
    from scripts.eval_agent import selected_cases

    cases = [eval_case(id="a"), eval_case(id="b"), eval_case(id="c")]
    filtered = selected_cases(cases, None, scenario_ids=["c", "a"])
    assert [c.id for c in filtered] == ["a", "c"]


def test_selected_cases_scenario_ids_returns_empty_when_no_match() -> None:
    """Unknown scenario id returns an empty list (rather than crashing) so
    the matrix runner sees zero queries and writes a clean empty report."""
    from scripts.eval_agent import selected_cases

    cases = [eval_case(id="a"), eval_case(id="b")]
    filtered = selected_cases(cases, None, scenario_ids=["nonexistent"])
    assert filtered == []


def test_selected_cases_scenario_ids_takes_precedence_with_max_queries() -> None:
    """When both --max-queries N and --scenario-ids are provided, scenario
    filter is applied; max_queries N then slices the filtered list."""
    from scripts.eval_agent import selected_cases

    cases = [
        eval_case(id="a"),
        eval_case(id="b"),
        eval_case(id="c"),
        eval_case(id="d"),
    ]
    filtered = selected_cases(cases, max_queries=2, scenario_ids=["a", "c", "d"])
    assert [c.id for c in filtered] == ["a", "c"]


def test_selected_cases_backward_compat_without_scenario_ids() -> None:
    """Existing call sites that don't pass scenario_ids continue to work."""
    from scripts.eval_agent import selected_cases

    cases = [eval_case(id="a"), eval_case(id="b")]
    # No scenario_ids arg at all — pre-03-05 signature is preserved.
    assert selected_cases(cases, None) == cases
