from __future__ import annotations

from argparse import Namespace

import pytest

from app.agent.state import ItineraryState
from app.eval.config import EvalQuery
from scripts.eval_agent import (
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


def query_result(**overrides: object) -> QueryEvalResult:
    """Build a QueryEvalResult with passing deterministic checks."""
    checks = {
        "constraints_satisfied": CheckResult(score=1.0, threshold=0.8, passed=True),
        "geographic_coherence": CheckResult(score=1.0, threshold=1.0, passed=True),
        "temporal_coherence": CheckResult(score=1.0, threshold=1.0, passed=True),
        "walking_budget_respected": CheckResult(score=1.0, threshold=1.0, passed=True),
        "no_hallucinated_place_ids": CheckResult(score=1.0, threshold=1.0, passed=True),
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
    args = Namespace(max_steps=0)

    with pytest.raises(ValueError, match="max-steps"):
        validate_args(args)


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
