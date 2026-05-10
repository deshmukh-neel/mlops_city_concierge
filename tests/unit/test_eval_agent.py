from __future__ import annotations

from argparse import Namespace

import pytest

from app.agent.state import ItineraryState
from app.eval.config import EvalQuery
from scripts.eval_agent import (
    CheckResult,
    EvalRunReport,
    QueryEvalResult,
    aggregate_results,
    count_tool_calls,
    report_has_errors,
    report_has_violations,
    resolve_chat_model,
    score_expected_stops,
    selected_cases,
    state_from_graph_output,
    tool_errors_from_state,
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
        "expected_stops": 1,
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
        "case_id": "case_one",
        "query": "coffee in soma",
        "tags": ["cafe"],
        "expected_stops": 1,
        "stops_count": 1,
        "expected_stops_met": True,
        "checks": checks,
        "violations": [],
        "tool_errors": [],
        "tool_calls": 2,
        "revision_hints": 0,
        "final_reply": "Try Example Cafe.",
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


def test_score_expected_stops_handles_relaxation_cases() -> None:
    """Skip stop-count scoring when the eval case expects relaxation."""
    state = ItineraryState()
    case = eval_case(expected_stops=None, expects_clarification_or_relaxation=True)

    assert score_expected_stops(case, state) is None


def test_violations_from_checks_includes_errors_and_low_scores() -> None:
    """Treat failed checks and errored checks as report violations."""
    checks = {
        "ok": CheckResult(score=1.0, threshold=1.0, passed=True),
        "low": CheckResult(score=0.0, threshold=1.0, passed=False),
        "error": CheckResult(score=None, threshold=1.0, passed=False, error="db down"),
    }

    assert violations_from_checks(checks) == ["low", "error"]


def test_violations_for_case_includes_expected_stops_mismatch() -> None:
    """Treat a stop-count mismatch as an eval violation."""
    checks = {"ok": CheckResult(score=1.0, threshold=1.0, passed=True)}

    assert violations_for_case(False, checks) == ["expected_stops"]


def test_aggregate_results_flattens_mean_metrics() -> None:
    """Aggregate per-query checks into flat numeric metrics."""
    result_one = query_result()
    result_two = query_result(
        expected_stops_met=False,
        stops_count=2,
        tool_calls=4,
        revision_hints=1,
        violations=["constraints_satisfied"],
        checks={
            **result_one.checks,
            "constraints_satisfied": CheckResult(score=0.5, threshold=0.8, passed=False),
        },
    )

    aggregate = aggregate_results([result_one, result_two])

    assert aggregate["query_count"] == 2
    assert aggregate["queries_with_violations"] == 1
    assert aggregate["expected_stops_mismatch_count"] == 1
    assert aggregate["tool_error_count"] == 0
    assert aggregate["expected_stops_match_rate"] == 0.5
    assert aggregate["stops_mean"] == 1.5
    assert aggregate["tool_calls_mean"] == 3.0
    assert aggregate["revision_hints_mean"] == 0.5
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
