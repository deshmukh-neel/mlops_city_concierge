from __future__ import annotations

import inspect
import json
from argparse import Namespace
from dataclasses import asdict

import pytest

from app.agent.state import ItineraryState, UserConstraints
from app.eval.config import EvalQuery, ExpectedConstraints, load_eval_queries
from app.tools.filters import family_of
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
    constraints_for_case,
    contexts_from_state,
    count_tool_calls,
    evaluate_multi_turn_case,
    expected_results_label,
    first_commit_call_step_from_state,
    make_error_record,
    percentile,
    query_result_from_state,
    rate,
    report_has_errors,
    report_has_violations,
    resolve_chat_model,
    retrieved_place_names_from_state,
    revision_reasons_from_state,
    rule8_met_per_step_from_state,
    score_checks,
    score_expected_results,
    selected_cases,
    state_from_graph_output,
    tool_errors_from_state,
    tool_names_from_state,
    validate_args,
    viable_candidates_per_step_from_state,
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

    assert source.count("def strip_non_empty_list") == 1


class TestConstraintsForCaseNumStops:
    """Phase-6 root-cause regression: ``constraints_for_case`` MUST pass
    ``num_stops`` so the eval prod-threading branch mirrors ``/chat``'s
    constraint extraction. Without this, queries that say "3 stops" in
    prose cause the model to ask "how many stops?" instead of committing
    (the SystemMessage that previously suppressed that question is dropped
    by the N-1 fix in plan 06-06).
    """

    def test_extracts_num_stops_from_query_text(self) -> None:
        # The refinement_cheaper query body says "3 stops" in prose.
        case = eval_case(query="Plan a date night dinner-then-drinks in Hayes Valley, 3 stops")
        constraints = constraints_for_case(case)
        assert constraints.num_stops == 3

    def test_falls_back_to_yaml_min_max_when_text_silent(self) -> None:
        case = eval_case(
            query="show me cool spots in soma",  # no count in prose
            expected_results={"min_stops": 4, "max_stops": 4},
        )
        constraints = constraints_for_case(case)
        assert constraints.num_stops == 4

    def test_does_not_invent_count_when_range_ambiguous(self) -> None:
        # min != max means the YAML range is a range, not a target — don't
        # fabricate a single count from it.
        case = eval_case(
            query="show me cool spots in soma",
            expected_results={"min_stops": 1, "max_stops": 5},
        )
        constraints = constraints_for_case(case)
        assert constraints.num_stops is None

    def test_text_extraction_wins_over_yaml(self) -> None:
        # If text says 3 but YAML says 5/5, the prose is authoritative — that
        # matches `/chat`'s behavior of trusting the user-spoken count.
        case = eval_case(
            query="something something 3 stops",
            expected_results={"min_stops": 5, "max_stops": 5},
        )
        constraints = constraints_for_case(case)
        assert constraints.num_stops == 3

    def test_requested_primary_types_still_set(self) -> None:
        # Regression: num_stops fix must not break the existing
        # requested_primary_types pass-through.
        case = eval_case(
            query="dinner-then-drinks-then-dessert, 3 stops",
            expected_constraints={
                "requested_primary_types": ["Restaurant", "Cocktail Bar", "Dessert Shop"],
            },
        )
        constraints = constraints_for_case(case)
        assert constraints.requested_primary_types == [
            "Restaurant",
            "Cocktail Bar",
            "Dessert Shop",
        ]
        assert constraints.num_stops == 3


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        (
            "omakase_mission_open_ended",
            ["Sushi Restaurant", "Cocktail Bar", "Dessert Shop"],
        ),
        (
            "refinement_cheaper",
            ["Restaurant", "Cocktail Bar", "Dessert Shop"],
        ),
    ],
)
def test_eval_queries_target_cases_declare_requested_primary_types(
    case_id: str,
    expected: list[str],
) -> None:
    """Live eval YAML should carry authoritative per-slot category expectations."""
    cases = {case.id: case for case in load_eval_queries("configs/eval_queries.yaml").hand_written}
    requested = cases[case_id].expected_constraints.requested_primary_types

    assert requested == expected
    assert all(family_of(value) is not None for value in requested)


def test_eval_queries_late_night_closure_cascade_has_no_requested_primary_types() -> None:
    """D-04-12: late-night closure remains tracked but ungated for Phase 4."""
    cases = {case.id: case for case in load_eval_queries("configs/eval_queries.yaml").hand_written}

    assert cases["late_night_closure_cascade"].expected_constraints.requested_primary_types == []


def query_result(**overrides: object) -> QueryEvalResult:
    """Build a QueryEvalResult with passing deterministic checks.

    Mirrors the shape of DETERMINISTIC_CHECKS in scripts/eval_agent.py — any
    scorer added to that dict must also appear here, otherwise
    aggregate_results raises KeyError when iterating per-scorer means.
    """
    checks = {
        "category_compliance": CheckResult(score=1.0, threshold=1.0, passed=True),
        "category_compliance_strict": CheckResult(score=1.0, threshold=1.0, passed=True),
        "constraints_satisfied": CheckResult(score=1.0, threshold=0.8, passed=True),
        "geographic_coherence": CheckResult(score=1.0, threshold=1.0, passed=True),
        "no_hallucinated_place_ids": CheckResult(score=1.0, threshold=1.0, passed=True),
        "rationale_stop_alignment": CheckResult(score=1.0, threshold=1.0, passed=True),
        "refinement_minimal_edit": CheckResult(score=1.0, threshold=1.0, passed=True),
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
            first_commit_call_step=None,
            first_commit_mention_step=None,
            viable_candidates_per_step=[],
            rule8_met_per_step=[],
            rule8_met_but_kept_searching_steps=[],
            step_telemetry=[],
            viability_threshold=0.55,
            # D-13-04/05 Plan 13-01 safe defaults for fixtures
            commit_forced=False,
            forced_commit_step=None,
            arm_flags={},
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
                {"result": [{"place_id": "ChIJtest_p1_aaaaaaaa"}]},
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
            place_ids=["ChIJtest_p1_aaaaaaaa", "ChIJtest_p2_aaaaaaaa", "p3", "p4"],
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
            first_commit_call_step=None,
            first_commit_mention_step=None,
            viable_candidates_per_step=[],
            rule8_met_per_step=[],
            rule8_met_but_kept_searching_steps=[],
            step_telemetry=[],
            viability_threshold=0.55,
            # D-13-04/05 Plan 13-01 safe defaults for fixtures
            commit_forced=False,
            forced_commit_step=None,
            arm_flags={},
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


from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from app.agent.graph import build_agent_graph  # noqa: E402
from scripts.eval_agent import evaluate_case  # noqa: E402
from tests._helpers.scripted_llm import RecordingScriptedLLM  # noqa: E402


class RaisingChatModel(BaseChatModel):
    """Test stub that raises on every _generate call.

    Simulates infra failures: 429 quota, DB-down, 400 config errors.
    Used for the 21-14-30Z replay acceptance tests (D-10-04, EVAL-01).
    """

    raise_exc: type[Exception] = Exception
    raise_msg: str = "simulated infra failure"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.raise_exc(self.raise_msg)

    @property
    def _llm_type(self) -> str:
        return "raising"

    def bind_tools(self, tools, **kwargs):
        return self


def finalize_msg(content: str) -> AIMessage:
    """Trajectory shorthand: a no-tool-calls AIMessage that finalizes a turn.

    With stops=[] + constraints.num_stops=None, this routes
    plan -> critique -> finalize_as_is -> done -> retime no-op -> swap no-op
    -> END, so each turn ends in a single plan() invocation. Keeps the
    scripted list tiny (one message per turn) and the test fast."""
    return AIMessage(content=content, tool_calls=[])


class CapturingGraph:
    """Small eval-agent test double that records every invoked state."""

    def __init__(self) -> None:
        self.states: list[ItineraryState] = []

    async def ainvoke(self, state: ItineraryState) -> ItineraryState:
        self.states.append(state)
        return state.model_copy(update={"final_reply": "captured"})


@pytest.mark.asyncio
async def test_evaluate_case_single_turn_unchanged(mocker) -> None:
    """Backward-compat regression guard: EvalQuery.turns=None must run
    through evaluate_case via the pre-03-04 single-turn code path. We assert
    on observable contract — exactly one plan() invocation, no synthetic
    `multi_turn_runner` tool error, and the final_reply is the scripted
    AIMessage's content — rather than byte-comparing JSON, which is the
    same guarantee surfaced differently."""
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RecordingScriptedLLM(scripted=[finalize_msg("turn1 reply")])
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=None)
    result = await evaluate_case(graph, case)

    assert result.final_reply == "turn1 reply"
    # Single-turn path = exactly one plan() invocation.
    assert len(llm.seen) == 1
    # Single-turn path must NOT inject the multi_turn_runner synthetic error.
    assert all("multi_turn_runner" not in err for err in result.deterministic.tool_errors)


@pytest.mark.asyncio
async def test_evaluate_case_passes_requested_primary_types_to_state() -> None:
    """Eval bypasses prod intake and copies YAML slot expectations into state."""
    graph = CapturingGraph()
    case = eval_case(
        expected_constraints={"requested_primary_types": ["Sushi Restaurant"]},
    )

    await evaluate_case(graph, case)

    assert graph.states[0].constraints.requested_primary_types == ["Sushi Restaurant"]


@pytest.mark.asyncio
async def test_evaluate_case_defaults_empty_requested_primary_types() -> None:
    """Cases without slot expectations keep the UserConstraints default."""
    graph = CapturingGraph()

    await evaluate_case(graph, eval_case())

    assert graph.states[0].constraints.requested_primary_types == []


@pytest.mark.asyncio
async def test_evaluate_multi_turn_passes_requested_primary_types_to_every_turn() -> None:
    """Multi-turn eval rebuilds constraints on every graph invocation."""
    graph = CapturingGraph()
    case = eval_case(
        expected_constraints={"requested_primary_types": ["Restaurant", "Cocktail Bar"]},
        turns=["make stop 2 cheaper"],
    )

    await evaluate_case(graph, case)

    assert [state.constraints.requested_primary_types for state in graph.states] == [
        ["Restaurant", "Cocktail Bar"],
        ["Restaurant", "Cocktail Bar"],
    ]


@pytest.mark.asyncio
async def test_evaluate_multi_turn_threads_messages(mocker) -> None:
    """EVAL-06: turn N+1's input state must contain turn N's HumanMessage so
    the agent sees prior conversation. Asserts on what the LLM ACTUALLY SAW
    on turn 2 (via RecordingScriptedLLM.seen[1]) — the strongest threading
    proof. If we only checked result.final_reply we'd miss a regression that
    nukes the prior turn's messages."""
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RecordingScriptedLLM(
        scripted=[finalize_msg("turn1 reply"), finalize_msg("turn2 reply")],
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
        scripted=[finalize_msg("turn1"), finalize_msg("turn2")],
    )
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=["refine"])
    result = await evaluate_case(graph, case)

    assert result.latency_seconds == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_multi_turn_intermediate_failure_captured(mocker) -> None:
    """D-10-02 error-status contract: if any turn raises, the helper returns an
    ERROR-status QueryEvalResult rather than a partial scored result. Scorers
    are NOT invoked on the failed run.

    We force turn 2 to raise by scripting only one AIMessage — the second
    invocation pops from an empty list and raises IndexError. The plan()
    coroutine propagates it; our helper's try/except catches it and returns
    an error record (replacing the old partial-state fail-open path).
    """
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RecordingScriptedLLM(scripted=[finalize_msg("turn1 reply")])
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=["this turn will explode"])
    result = await evaluate_case(graph, case)

    # The run did NOT crash — we have a result.
    assert isinstance(result, QueryEvalResult)
    # D-10-02: result is an ERROR record (turn 1 raised IndexError).
    assert result.status == "error"
    assert result.error is not None
    assert result.error["stage"] == "turnN"  # index=1 → "turnN"
    assert result.error["type"] == "IndexError"
    # Scorers must NOT have been called — all check scores should be None.
    for check_result in result.deterministic.checks.values():
        assert check_result.score is None
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
        scripted=[finalize_msg("turn1 reply"), finalize_msg("turn2 reply")],
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


# ============================================================================
# EVAL-01 / Plan 10-01 Task 2: partial_state scoring REMOVED; error records on
# exception in both threading branches (D-10-02)
# ============================================================================


def test_no_partial_state_scoring_in_eval_agent() -> None:
    """D-10-02: The partial_state SCORING path must be fully removed from
    eval_agent.py — no call to query_result_from_state on a partial_state
    inside an except block. Comments mentioning the old pattern are fine.
    """
    import ast
    from pathlib import Path

    source = Path("scripts/eval_agent.py").read_text(encoding="utf-8")
    # AST parse first to ensure no syntax errors crept in.
    tree = ast.parse(source)

    # Walk all except handlers and verify none call query_result_from_state.
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            handler_src = ast.get_source_segment(source, node) or ""
            assert "query_result_from_state" not in handler_src, (
                "D-10-02: query_result_from_state must NOT be called inside an "
                "except handler — use make_error_record instead."
            )


def test_make_error_record_called_in_prod_threading_except() -> None:
    """D-10-02: run_prod_threading except clause must call make_error_record,
    not query_result_from_state on partial state.
    """
    import ast
    from pathlib import Path

    source = Path("scripts/eval_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find run_prod_threading function and check its except handlers.
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_prod_threading":
            source_segment = ast.get_source_segment(source, node) or ""
            assert "make_error_record" in source_segment, (
                "run_prod_threading except clause must call make_error_record"
            )
            break
    else:
        pytest.fail("run_prod_threading not found in eval_agent.py")


def test_make_error_record_called_in_legacy_threading_except() -> None:
    """D-10-02: run_legacy_threading except clause must call make_error_record."""
    import ast
    from pathlib import Path

    source = Path("scripts/eval_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_legacy_threading":
            source_segment = ast.get_source_segment(source, node) or ""
            assert "make_error_record" in source_segment, (
                "run_legacy_threading except clause must call make_error_record"
            )
            break
    else:
        pytest.fail("run_legacy_threading not found in eval_agent.py")


@pytest.mark.asyncio
async def test_legacy_threading_turn0_exception_produces_error_record(mocker) -> None:
    """D-10-02 / EVAL-01: A turn-0 exception in run_legacy_threading returns
    a QueryEvalResult with status='error' and error.stage='turn0'.
    Scorers are NOT invoked on the failed run.
    """
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RaisingChatModel(raise_exc=RuntimeError, raise_msg="quota exceeded")
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=["make it cheaper"])
    result = await evaluate_case(graph, case)

    assert result.status == "error"
    assert result.error is not None
    assert result.error["stage"] == "turn0"
    assert result.error["type"] == "RuntimeError"
    # Scorers must NOT have been called — all check scores should be None.
    for check_result in result.deterministic.checks.values():
        assert check_result.score is None, f"Scorer ran on a failed turn: {check_result}"


# ============================================================================
# EVAL-01 / Plan 10-01: ERROR-status record schema + make_error_record builder
# ============================================================================


def test_query_eval_result_has_status_field_with_default_ok() -> None:
    """D-10-01: QueryEvalResult must have a `status` field defaulting to 'ok'.

    This is the discriminator field that aggregate_results uses to filter out
    errored runs. Existing scored-row construction (no status arg) must still
    work without change.
    """
    import dataclasses

    from scripts.eval_agent import QueryEvalResult

    field_names = {f.name for f in dataclasses.fields(QueryEvalResult)}
    assert "status" in field_names, "QueryEvalResult must have a 'status' field"

    # Default must be "ok" so existing scored rows keep status="ok" unchanged.
    result = query_result()  # built by the existing helper — no status arg
    assert result.status == "ok"


def test_make_error_record_builds_error_schema_record() -> None:
    """D-10-01: make_error_record returns a QueryEvalResult with status='error'
    and an error dict with keys {stage, type, message}.
    """
    from scripts.eval_agent import make_error_record

    case = eval_case()
    exc = RuntimeError("db connection failed")
    record = make_error_record(case, "turn0", exc)

    assert record.status == "error"
    assert isinstance(record.error, dict)
    assert record.error["stage"] == "turn0"
    assert record.error["type"] == "RuntimeError"
    assert "db connection failed" in record.error["message"]


def test_make_error_record_truncates_message_to_500_chars() -> None:
    """D-10-01: error.message is truncated to 500 chars per the schema."""
    from scripts.eval_agent import make_error_record

    long_msg = "x" * 1000
    exc = ValueError(long_msg)
    record = make_error_record(eval_case(), "turnN", exc)

    assert len(record.error["message"]) <= 500


def test_make_error_record_stage_values_are_valid() -> None:
    """D-10-01: make_error_record accepts stage in {setup, turn0, turnN}."""
    from scripts.eval_agent import make_error_record

    valid_stages = {"setup", "turn0", "turnN"}
    case = eval_case()
    exc = Exception("oops")

    for stage in valid_stages:
        record = make_error_record(case, stage, exc)
        assert record.error["stage"] == stage


def test_make_error_record_carries_no_scored_checks() -> None:
    """D-10-01: error records carry no scored check data — scorers NEVER run
    on failed turns. The deterministic.checks dict may be empty or have
    None scores, but must not contain real scorer scores.
    """
    from scripts.eval_agent import make_error_record

    record = make_error_record(eval_case(), "turn0", Exception("fail"))

    # Error record's deterministic.checks must all have score=None (or be empty).
    for check_result in record.deterministic.checks.values():
        assert check_result.score is None, (
            f"Error record should not carry scored checks; got score={check_result.score}"
        )


def test_make_error_record_type_is_exception_class_name() -> None:
    """D-10-01: error.type is type(exc).__name__, not the string representation."""
    from scripts.eval_agent import make_error_record

    class CustomError(Exception):
        pass

    exc = CustomError("custom failure")
    record = make_error_record(eval_case(), "setup", exc)

    assert record.error["type"] == "CustomError"


def test_query_eval_result_with_error_serializes_via_asdict() -> None:
    """Error records must serialize cleanly via asdict() -> json.dumps()
    (same contract as test_query_result_serializes_to_json_via_asdict).
    """
    import json
    from dataclasses import asdict

    from scripts.eval_agent import make_error_record

    record = make_error_record(eval_case(), "turn0", Exception("quota exceeded"))
    payload = asdict(record)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["status"] == "error"
    assert decoded["error"]["stage"] == "turn0"


# ============================================================================
# EVAL-01 / Plan 10-01 Task 3: aggregate_results filters on status=="ok";
# n_scored / n_errored / errors[] added; 21-14-30Z replay acceptance test
# ============================================================================


def test_aggregate_results_filters_scored_only() -> None:
    """D-10-03: aggregate_results must compute scorer means over ONLY results
    with status=='ok'. An all-error result list yields n_scored==0 and scorer
    means of None (not 1.0 from Branch-1 abstain, and — per D-11-03/CR-01 —
    not the former mean([]) == 0.0 that conflated "no signal" with "zero score").
    """
    from scripts.eval_agent import aggregate_results, make_error_record

    error_record = make_error_record(eval_case(), "turn0", Exception("quota"))
    aggregate = aggregate_results([error_record])

    assert aggregate["n_scored"] == 0
    assert aggregate["n_errored"] == 1
    # Scorer means must be None (zero non-None scores → no signal), NOT 1.0
    # fail-open and NOT the former mean([]) == 0.0.
    assert aggregate["refinement_minimal_edit_mean"] is None
    assert aggregate["category_compliance_mean"] is None


def test_aggregate_results_n_scored_and_n_errored_counts() -> None:
    """D-10-03: aggregate dict gains n_scored, n_errored, errors list."""
    from scripts.eval_agent import aggregate_results, make_error_record

    ok_record = query_result()
    error_record = make_error_record(eval_case(id="fail_case"), "turnN", ValueError("db down"))
    aggregate = aggregate_results([ok_record, error_record])

    assert aggregate["n_scored"] == 1
    assert aggregate["n_errored"] == 1
    assert isinstance(aggregate["errors"], list)
    assert len(aggregate["errors"]) == 1
    err = aggregate["errors"][0]
    assert err["stage"] == "turnN"
    assert err["type"] == "ValueError"


def test_aggregate_results_ok_only_excludes_errored_from_means() -> None:
    """D-10-03: scored means only include status=='ok' results."""
    from scripts.eval_agent import aggregate_results, make_error_record

    ok_record = query_result()  # all scores=1.0
    error_record = make_error_record(eval_case(), "turn0", Exception("fail"))
    aggregate = aggregate_results([ok_record, error_record])

    # Only the ok_record contributes to refinement_minimal_edit_mean.
    assert aggregate["refinement_minimal_edit_mean"] == 1.0
    assert aggregate["n_scored"] == 1
    assert aggregate["n_errored"] == 1


def test_report_has_errors_detects_n_errored() -> None:
    """D-10-03: report_has_errors returns True when n_errored > 0."""
    from scripts.eval_agent import make_error_record

    error_record = make_error_record(eval_case(), "turn0", Exception("fail"))
    report = EvalRunReport(
        eval_queries_path="configs/eval_queries.yaml",
        llm_provider="openai",
        chat_model="gpt-4o-mini",
        query_count=1,
        aggregate={"check_error_count": 0, "n_errored": 1},
        queries=[error_record],
    )
    assert report_has_errors(report) is True


@pytest.mark.asyncio
async def test_21_14_30z_replay_turn0_exception_produces_error_record(mocker) -> None:
    """D-10-04: 21-14-30Z replay — turn-0 LLM exception must produce a
    status='error' record. The former fail-open outcome (Branch-1 abstain 1.0)
    must NOT appear on the all-error report.
    """
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    llm = RaisingChatModel(raise_exc=RuntimeError, raise_msg="429 quota exceeded")
    graph = build_agent_graph(llm, max_steps=4)

    case = eval_case(turns=["this should error on turn 0"])
    result = await evaluate_case(graph, case)

    assert result.status == "error"
    assert result.error["stage"] == "turn0"
    assert result.error["type"] == "RuntimeError"
    # Former fail-open outcome: Branch-1 abstain returned 1.0 on partial state.
    # Now scorers must NOT be called; all scores are None.
    for check_result in result.deterministic.checks.values():
        assert check_result.score is None


@pytest.mark.asyncio
async def test_21_14_30z_replay_all_error_report_has_zero_scored(mocker) -> None:
    """D-10-04: 21-14-30Z replay — an all-error run produces n_scored==0
    and refinement_minimal_edit_mean != 1.0. This directly proves the
    Branch-1-abstain-1.0 fail-open is gone.
    """
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
    from scripts.eval_agent import aggregate_results, make_error_record

    # Simulate the 21-14-30Z scenario: all three cases errored.
    error_records = [
        make_error_record(eval_case(id="case_a"), "turn0", RuntimeError("429")),
        make_error_record(eval_case(id="case_b"), "turn0", RuntimeError("429")),
        make_error_record(eval_case(id="case_c"), "turnN", RuntimeError("db down")),
    ]
    aggregate = aggregate_results(error_records)

    # D-10-04: n_scored must be 0.
    assert aggregate["n_scored"] == 0
    assert aggregate["n_errored"] == 3
    # Former fail-open: refinement_minimal_edit_mean was 1.0 due to Branch-1
    # abstain returning 1.0 on partial state. Now it must be None — zero
    # non-None scores publish None per D-11-03/CR-01, never a fabricated 1.0
    # (or the former mean([]) == 0.0).
    assert aggregate["refinement_minimal_edit_mean"] != 1.0
    assert aggregate["refinement_minimal_edit_mean"] is None


@pytest.mark.asyncio
async def test_21_14_30z_replay_turn_n_exception_produces_error_record(mocker) -> None:
    """D-10-04: 21-14-30Z replay — turn-N (N>=1) exception produces status='error'
    with error.stage='turnN'.
    """
    mocker.patch("app.agent.revision.itinerary_violations", return_value=[])

    # Turn 0 succeeds (scripted), turn 1 raises.
    from tests._helpers.scripted_llm import ScriptedLLM

    # Turn 0: scripted to produce a final reply; turn 1 will raise via exhaustion.
    llm_turn0 = ScriptedLLM(scripted=[finalize_msg("turn0 reply")])
    graph = build_agent_graph(llm_turn0, max_steps=4)

    case = eval_case(turns=["this turn should error"])
    result = await evaluate_case(graph, case)

    assert result.status == "error"
    assert result.error["stage"] == "turnN"
    assert result.error["type"] == "IndexError"  # ScriptedLLM exhaustion


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


# --- Plan 06-03 HIGH-1 fix: refinement_minimal_edit dual-registration -------


class TestDeterministicChecksRegistration:
    """HIGH-1 regression guard (06-REVIEWS.md): the merge-gate scorer
    `refinement_minimal_edit` must be registered in BOTH
    `app/agent/critique/checks.py:CRITIQUE_THRESHOLDS` (tested in
    tests/unit/test_critique_checks.py) AND
    `scripts/eval_agent.py:DETERMINISTIC_CHECKS` (tested here).

    Without dual registration, the scorer auto-runs in the revision-loop
    critique but is silently absent from every per-cell baseline JSON
    output — meaning the merge gate cannot diff refinement_minimal_edit
    pass-rates across PRs. This class asserts both registrations agree
    on the same callable identity (not just presence)."""

    def test_refinement_minimal_edit_registered_in_deterministic_checks(self) -> None:
        """HIGH-1 fix: the scorer must appear as a key in DETERMINISTIC_CHECKS
        AND the registered callable must be identical to the imported one
        (catches the silent-rename / shadow-by-mock failure mode)."""
        from app.agent.critique.checks import refinement_minimal_edit

        assert "refinement_minimal_edit" in DETERMINISTIC_CHECKS
        # Identity, not just equality — same callable, not a wrapper or stub.
        assert DETERMINISTIC_CHECKS["refinement_minimal_edit"] is refinement_minimal_edit


# --- Plan 06-06: evaluate_multi_turn_case prod-threading branch ------------
#
# These tests pin the Phase 6 plan 06-06 contract for `threading_mode='prod'`:
#   - N-1 fix: prod branch DOES NOT inject SystemMessage(eval_context); the
#     graph's plan() node prepends SYSTEM_PROMPT naturally on the first
#     invocation when no SystemMessage is present (graph.py:264), matching
#     /chat's prompt chain exactly.
#   - N-2 fix: prod branch ALWAYS sets state.scratch['refinement_context'] =
#     True for refinement scenarios. Paired with plan 06-03's Branch 2
#     fail-loud, this closes the false-pass path where turn 0 commits empty.
#   - HIGH-3 + Caveat #5: shared `build_refinement_prompt_message` helper
#     produces byte-identical messages between /chat and the prod branch.
#   - NEW HIGH-B: prod branch reads REFINEMENT_STRUCTURED_PLAN_ENABLED INSIDE
#     `run_prod_threading` per OVR-05; injection is gated on the flag.
#   - NEW HIGH-C: helper call is gated on `if prior_committed_stops:` so
#     empty-prior never raises ValueError; scratch keys still written.
#
# Per `project_full_suite_db_pool_contamination.md` every test patches
# `app.agent.revision.itinerary_violations` to [] so no live DB pool fires.


class TestEvaluateMultiTurnProdThreading:
    """Phase 6 plan 06-06 Task 1 — prod-threading branch contract.

    Every test in this class follows the project full-suite DB-pool
    contamination guard: `app.agent.revision.itinerary_violations` is
    mocked to [] so the revision loop cannot leak a live DB pool.

    Every Stop fixture uses a Google-Place-ID-conforming `place_id`
    per plan 06-01 Task 3 validator (`^[A-Za-z0-9_-]{20,255}$`).
    """

    PID_1 = "ChIJ_test_id_aaaaaaaa_a"
    PID_2 = "ChIJ_test_id_bbbbbbbb_b"
    PID_3 = "ChIJ_test_id_cccccccc_c"

    @classmethod
    def prod_case(
        cls,
        *,
        target_slot: int = 2,
        turns: list[str] | None = None,
        query: str = "Plan a date night with sushi, cocktails, and dessert",
    ) -> EvalQuery:
        """Build a prod-mode refinement EvalQuery with `expected_refinement`."""
        return eval_case(
            id="refinement_cheaper",
            query=query,
            turns=turns if turns is not None else ["make stop 2 cheaper"],
            threading_mode="prod",
            expected_refinement={"target_slot": target_slot},
        )

    @classmethod
    def commit_turn_message(cls, place_ids: list[str], reply: str = "committed") -> AIMessage:
        """Build a scripted AIMessage that emits a commit_itinerary tool call
        AND finalizes — but for these tests we use a simpler shape: a
        finalize-only AIMessage paired with an explicit state override at
        the graph layer. We don't fully exercise commit; instead we set
        state.stops directly via the graph layer or use a CapturingGraph.
        """
        return AIMessage(content=reply, tool_calls=[])

    @classmethod
    def stops_for_pids(cls, pids: list[str]) -> list:
        """Build fixture Stops for the prod-threading scratch contract tests.

        Phase 7 / D-07-06 / D-07-07: every Stop gets ``primary_type="Cafe"`` so
        the eval runner's turn-0 happy-path can populate the new
        ``prior_committed_stops[i]["primary_type"]`` scratch field (per plan
        07-02) with a deterministic, assertable value. The existing call sites
        across this class (all using the no-keyword form) pick up the
        deterministic value automatically; no test below needs to override it.
        """
        from tests.conftest import make_stop

        return [
            make_stop(place_id=pid, name=f"Stop {i + 1}", primary_type="Cafe")
            for i, pid in enumerate(pids)
        ]

    class ProdCapturingGraph:
        """Graph double for the prod-branch tests that captures every
        invocation's input state AND lets the test script per-turn stops
        and final_reply on the returned state.

        Each turn pops the next entry from `script` which is a tuple
        (stops_to_set, final_reply_to_set). If the script is exhausted,
        an IndexError is raised — surfaces a mis-count loudly.
        """

        def __init__(self, script: list[tuple[list, str]]) -> None:
            self.script = list(script)
            self.invocations: list = []
            self.seen_messages: list[list] = []

        async def ainvoke(self, state) -> object:
            self.invocations.append(state)
            self.seen_messages.append(list(state.messages))
            stops, reply = self.script.pop(0)
            return state.model_copy(update={"stops": list(stops), "final_reply": reply})

    @pytest.mark.asyncio
    async def test_legacy_mode_threads_messages_unchanged(self, mocker) -> None:
        """Legacy branch byte-identical to pre-Phase-6 — SystemMessage(eval_context)
        survives into turn 2 just like `test_evaluate_multi_turn_threads_messages`.
        Regression guard against accidental N-1-style fixes to the legacy path.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        llm = RecordingScriptedLLM(
            scripted=[finalize_msg("turn1 reply"), finalize_msg("turn2 reply")],
        )
        graph = build_agent_graph(llm, max_steps=4)
        case = eval_case(
            query="coffee in soma",
            turns=["make stop 2 cheaper"],
            threading_mode="legacy",
        )
        result = await evaluate_multi_turn_case(graph, case)
        assert result.final_reply == "turn2 reply"
        # Legacy STILL injects SystemMessage(eval_context) — substring survives.
        turn_two_systems = [m.content for m in llm.seen[1] if isinstance(m, SystemMessage)]
        assert any("Expected open time:" in c for c in turn_two_systems)

    @pytest.mark.asyncio
    async def test_prod_mode_turn_0_messages_do_not_contain_system_message(
        self, mocker, monkeypatch
    ) -> None:
        """N-1 fix: prod branch does NOT inject SystemMessage(eval_context) on turn 0.
        graph.plan() prepends SYSTEM_PROMPT naturally per graph.py:264 — the only
        SystemMessage seen by the LLM is the SYSTEM_PROMPT itself, which must NOT
        contain the eval-context substring (e.g. 'Expected open time:').
        """
        from app.agent.prompts import SYSTEM_PROMPT  # noqa: F401  (anchor for the assertion)

        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        llm = RecordingScriptedLLM(
            scripted=[finalize_msg("turn0 reply"), finalize_msg("turn1 reply")],
        )
        graph = build_agent_graph(llm, max_steps=4)
        case = self.prod_case()
        await evaluate_multi_turn_case(graph, case)

        # On turn 0 graph.plan() prepended SYSTEM_PROMPT — but NO SystemMessage
        # carrying the eval_context body. The eval-context substring must be
        # absent from every SystemMessage seen on turn 0.
        turn_zero_systems = [m.content for m in llm.seen[0] if isinstance(m, SystemMessage)]
        for sysmsg in turn_zero_systems:
            assert "Expected open time:" not in sysmsg
            assert "Expected result range:" not in sysmsg

    @pytest.mark.asyncio
    async def test_prod_mode_injects_structured_plan_on_turn_1(self, mocker, monkeypatch) -> None:
        """Turn 1 of a prod refinement scenario must include the structured-plan
        HumanMessage when the env-var flag is ON and turn 0 committed >= 1 stop.

        Phase 7 / D-07-03 + D-07-04: the task-only preamble dropped the
        ``byte-for-byte`` behavioral prescription (it moved into
        ``refinement_minimal_edit``). Assertions here pin the NEW preamble's
        task-description contract instead: the ``REFINEMENT TURN`` sentinel,
        the ``commit_itinerary`` output-channel callout, and the JSON-block
        field names (``slot``, ``place_id``, ``arrival_time``) — none of which
        appear on the D-07-04 forbidden-phrase list.

        Phase 7 / D-07-06 extension: also pins that turn-0 prior_committed_stops
        scratch entries now carry ``primary_type`` per entry per plan 07-02.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        # We use the prod-branch's helper run_prod_threading directly so we can
        # script turn-0 commits via a _ProdCapturingGraph; the real graph cannot
        # script committed stops without a full retrieval/commit trajectory.
        from scripts.eval_agent import run_prod_threading

        stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        graph = self.ProdCapturingGraph(script=[(stops, "turn0 reply"), ([], "turn1 reply")])
        case = self.prod_case()

        result, final_state = await run_prod_threading(graph, case)

        # Turn 1's messages contain a HumanMessage with structured-plan content.
        turn_one_humans = [m for m in graph.seen_messages[1] if isinstance(m, HumanMessage)]
        contents = [m.content for m in turn_one_humans if isinstance(m.content, str)]
        assert any("current_plan" in c for c in contents)
        # Phase 7 task-only preamble assertions (D-07-03): the new preamble
        # carries the REFINEMENT TURN sentinel + commit_itinerary output-channel
        # callout + the JSON-block field names. None of these are on the
        # D-07-04 forbidden-phrase list.
        assert any("REFINEMENT TURN" in c for c in contents), (
            "Phase 7 sentinel 'REFINEMENT TURN' missing from structured-plan HumanMessage"
        )
        assert any("commit_itinerary" in c for c in contents), (
            "Phase 7 output-channel callout 'commit_itinerary' missing from preamble"
        )
        # JSON-block field names from build_refinement_prompt_message (io.py:123-132).
        assert any("slot" in c for c in contents)
        assert any("place_id" in c for c in contents)
        assert any("arrival_time" in c for c in contents)
        # Result is the expected dataclass.
        assert isinstance(result, QueryEvalResult)

        # Phase 7 / D-07-06 scratch-payload extension (plan 07-02): turn-0
        # prior_committed_stops entries carry primary_type per entry. The
        # stops_for_pids helper sets primary_type='Cafe' on every Stop, so
        # the scratch echoes that value into each entry.
        prior = final_state.scratch["prior_committed_stops"]
        assert all("primary_type" in entry for entry in prior), (
            "Phase 7 / D-07-06: primary_type missing from one or more scratch entries"
        )
        assert all(entry["primary_type"] == "Cafe" for entry in prior), (
            "Phase 7 / D-07-06: scratch primary_type does not match Stop.primary_type "
            "from the eval runner's turn-0 commit"
        )

    @pytest.mark.asyncio
    async def test_prod_mode_message_byte_identical_to_chat_helper_output(
        self, mocker, monkeypatch
    ) -> None:
        """Caveat #5 invariant: prod branch's structured-plan HumanMessage content
        equals the SHARED helper's direct output for the same committed_stops.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        from app.agent.io import build_refinement_prompt_message
        from scripts.eval_agent import run_prod_threading

        stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        graph = self.ProdCapturingGraph(script=[(stops, "turn0 reply"), ([], "turn1 reply")])
        case = self.prod_case()

        await run_prod_threading(graph, case)

        # Direct call to the helper with the same committed_stops.
        expected_msg = build_refinement_prompt_message(stops)
        # Find the structured-plan HumanMessage on turn 1's seen sequence.
        turn_one_humans = [m for m in graph.seen_messages[1] if isinstance(m, HumanMessage)]
        structured_human_contents = [
            m.content
            for m in turn_one_humans
            if isinstance(m.content, str) and "current_plan" in m.content
        ]
        assert structured_human_contents, "no structured-plan HumanMessage on turn 1"
        assert structured_human_contents[0] == expected_msg.content

    @pytest.mark.asyncio
    async def test_prod_mode_message_sequence_matches_chat_invocation(
        self, mocker, monkeypatch
    ) -> None:
        """HIGH-3 + N-1: the full message sequence the LLM sees on turn 1 of a
        prod-mode case must match the sequence /chat builds on a refinement
        request. Specifically: positions and contents agree at the
        synthesized-history → structured-plan → user-turn boundary.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")

        from app.agent.io import build_refinement_prompt_message
        from scripts.eval_agent import run_prod_threading

        stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        graph = self.ProdCapturingGraph(
            script=[(stops, "turn0 assistant reply"), ([], "turn1 reply")]
        )
        case = self.prod_case(query="Plan a date night", turns=["make stop 2 cheaper"])
        await run_prod_threading(graph, case)

        # On turn 1 the prod branch builds [*history(user+assistant turn 0),
        # structured-plan HumanMessage, user-turn-1 HumanMessage].
        # There must be NO SystemMessage at position 0 in messages_in
        # (graph.plan() will prepend SYSTEM_PROMPT; the ProdCapturingGraph
        # does NOT do that prepend, so we look at the raw state.messages
        # that the runner constructed).
        turn_one_msgs = graph.seen_messages[1]
        # First element must be a HumanMessage (history user turn 0).
        assert isinstance(turn_one_msgs[0], HumanMessage)
        assert turn_one_msgs[0].content == "Plan a date night"
        # Second element: AIMessage (history assistant turn 0).
        assert isinstance(turn_one_msgs[1], AIMessage)
        assert turn_one_msgs[1].content == "turn0 assistant reply"
        # Then the structured-plan HumanMessage.
        expected_struct = build_refinement_prompt_message(stops)
        assert isinstance(turn_one_msgs[2], HumanMessage)
        assert turn_one_msgs[2].content == expected_struct.content
        # Last element: the user's turn-1 message.
        assert isinstance(turn_one_msgs[-1], HumanMessage)
        assert turn_one_msgs[-1].content == "make stop 2 cheaper"
        # NO SystemMessage anywhere in the prod-branch-constructed messages.
        assert not any(isinstance(m, SystemMessage) for m in turn_one_msgs)

    @pytest.mark.asyncio
    async def test_prod_mode_always_sets_refinement_context_true(self, mocker, monkeypatch) -> None:
        """N-2 regression guard: refinement_context is True regardless of
        whether turn 0 commits empty or non-empty stops.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        from scripts.eval_agent import run_prod_threading

        # Non-empty commit.
        stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        graph_a = self.ProdCapturingGraph(script=[(stops, "ok"), ([], "ok2")])
        _, state_a = await run_prod_threading(graph_a, self.prod_case())
        assert state_a.scratch["refinement_context"] is True

        # Empty commit.
        graph_b = self.ProdCapturingGraph(script=[([], "empty turn 0"), ([], "ok")])
        _, state_b = await run_prod_threading(graph_b, self.prod_case())
        assert state_b.scratch["refinement_context"] is True

    @pytest.mark.asyncio
    async def test_prod_mode_turn_0_empty_commit_sets_empty_prior_and_target(
        self, mocker, monkeypatch
    ) -> None:
        """N-2 fix: empty commit on turn 0 still writes the three scratch keys
        (refinement_context=True, prior_committed_stops=[], refinement_target_slot).
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        from scripts.eval_agent import run_prod_threading

        graph = self.ProdCapturingGraph(script=[([], "empty"), ([], "ok")])
        case = self.prod_case(target_slot=2)
        _, final_state = await run_prod_threading(graph, case)
        assert final_state.scratch["prior_committed_stops"] == []
        assert final_state.scratch["refinement_target_slot"] == 2
        assert final_state.scratch["refinement_context"] is True

    @pytest.mark.asyncio
    async def test_prod_mode_empty_prior_does_not_raise_and_scorer_returns_0_0(
        self, mocker, monkeypatch
    ) -> None:
        """NEW HIGH-C: helper call site gated on `if prior_committed_stops:`.
        When turn 0 commits empty, the helper is NOT called; no ValueError;
        the scorer (Branch 2) returns 0.0.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        from scripts.eval_agent import run_prod_threading

        graph = self.ProdCapturingGraph(script=[([], "empty"), ([], "ok")])
        case = self.prod_case()
        result, _ = await run_prod_threading(graph, case)

        # No structured-plan content was emitted on turn 1 (helper not called).
        turn_one_msgs = graph.seen_messages[1]
        for m in turn_one_msgs:
            if isinstance(m, HumanMessage) and isinstance(m.content, str):
                assert "current_plan" not in m.content
        # Scorer returns 0.0 via Branch 2 fail-loud (refinement_context=True,
        # prior_committed_stops=[]).
        assert result.deterministic.checks["refinement_minimal_edit"].score == 0.0

    @pytest.mark.asyncio
    async def test_prod_mode_skips_injection_when_flag_off(self, mocker, monkeypatch) -> None:
        """NEW HIGH-B: when REFINEMENT_STRUCTURED_PLAN_ENABLED is unset,
        the structured-plan HumanMessage is NOT emitted even though turn 0
        committed 3 stops. The scratch keys are still written (gate only
        controls emission, not scratch).
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.delenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", raising=False)
        from scripts.eval_agent import run_prod_threading

        stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        graph = self.ProdCapturingGraph(script=[(stops, "ok"), ([], "ok2")])
        case = self.prod_case()
        _, final_state = await run_prod_threading(graph, case)

        # No structured-plan content on turn 1 (flag was off).
        turn_one_msgs = graph.seen_messages[1]
        for m in turn_one_msgs:
            if isinstance(m, HumanMessage) and isinstance(m.content, str):
                assert "current_plan" not in m.content
        # Scratch keys still written.
        assert final_state.scratch["refinement_context"] is True
        assert final_state.scratch["prior_committed_stops"]  # non-empty
        assert len(final_state.scratch["prior_committed_stops"]) == 3

    @pytest.mark.asyncio
    async def test_prod_mode_turn_0_empty_commit_scorer_returns_0_0_end_to_end(
        self, mocker, monkeypatch
    ) -> None:
        """N-2 fix end-to-end: through the public evaluate_multi_turn_case API,
        an empty turn-0 commit on a prod refinement scenario yields
        refinement_minimal_edit score == 0.0 (Branch 2 fail-loud).
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")

        graph = self.ProdCapturingGraph(script=[([], "empty"), ([], "ok")])
        case = self.prod_case()
        result = await evaluate_multi_turn_case(graph, case)
        assert result.deterministic.checks["refinement_minimal_edit"].score == 0.0

    @pytest.mark.asyncio
    async def test_prod_mode_populates_state_scratch_for_scorer(self, mocker, monkeypatch) -> None:
        """Caveat #6 + N-2: FINAL returned state carries ALL THREE scratch keys
        — refinement_context=True, prior_committed_stops (list[dict]), and
        refinement_target_slot (int).

        Phase 7 / D-07-06 extension (plan 07-02 / PROMPT-03): each
        ``prior_committed_stops`` entry now also carries a ``primary_type``
        field sourced from ``Stop.primary_type``. The ``stops_for_pids``
        helper sets ``primary_type='Cafe'`` on every fixture stop so the
        assertion below is deterministic.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        from scripts.eval_agent import run_prod_threading

        stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        graph = self.ProdCapturingGraph(script=[(stops, "ok"), (stops, "ok2")])
        case = self.prod_case(target_slot=2)
        _, final_state = await run_prod_threading(graph, case)

        assert final_state.scratch["refinement_context"] is True
        assert final_state.scratch["refinement_target_slot"] == 2
        prior = final_state.scratch["prior_committed_stops"]
        assert isinstance(prior, list)
        assert len(prior) == 3
        assert prior[0]["slot"] == 1
        assert prior[0]["place_id"] == self.PID_1
        assert prior[1]["slot"] == 2
        assert prior[1]["place_id"] == self.PID_2
        # Phase 7 / D-07-06: primary_type lives on every scratch entry now.
        assert "primary_type" in prior[0], (
            "Phase 7 / D-07-06: primary_type missing from scratch entry 0"
        )
        assert prior[0]["primary_type"] == "Cafe"
        assert "primary_type" in prior[1]
        assert prior[1]["primary_type"] == "Cafe"
        assert "primary_type" in prior[2]
        assert prior[2]["primary_type"] == "Cafe"

    @pytest.mark.asyncio
    async def test_prod_mode_raises_on_missing_expected_refinement(
        self, mocker, monkeypatch
    ) -> None:
        """Defensive: prod-mode case without `expected_refinement` raises."""
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        from scripts.eval_agent import run_prod_threading

        graph = self.ProdCapturingGraph(script=[([], "ok")])
        case = eval_case(
            id="bad_prod_case",
            query="x",
            turns=["y"],
            threading_mode="prod",
            # no expected_refinement
        )
        with pytest.raises(ValueError, match="requires expected_refinement"):
            await run_prod_threading(graph, case)

    @pytest.mark.asyncio
    async def test_prod_mode_preserves_fail_open_on_exception(self, mocker, monkeypatch) -> None:
        """D-10-02 error-status contract in prod branch: a per-turn exception
        returns an ERROR-status QueryEvalResult (not a partial scored record).
        Replaces the old 'synthetic multi_turn_runner scratch entry' behavior.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")

        # Script only ONE turn — second invocation pops from empty list.
        stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        graph = self.ProdCapturingGraph(script=[(stops, "turn0 reply")])
        case = self.prod_case()
        result = await evaluate_multi_turn_case(graph, case)

        # D-10-02: exception on turn 1 produces an ERROR record.
        assert result.status == "error"
        assert result.error is not None
        assert result.error["stage"] == "turnN"  # index=1 → "turnN"
        # Scorers were NOT invoked.
        for check_result in result.deterministic.checks.values():
            assert check_result.score is None

    @pytest.mark.asyncio
    async def test_prod_mode_synthesized_history_includes_user_and_assistant(
        self, mocker, monkeypatch
    ) -> None:
        """MEDIUM: synthesized text history for turn 1 must contain BOTH the
        prior turn's user HumanMessage AND the prior turn's assistant
        AIMessage (via state.final_reply). Not just assistant-only.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")
        from scripts.eval_agent import run_prod_threading

        stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        graph = self.ProdCapturingGraph(script=[(stops, "turn0 assistant prose"), ([], "ok")])
        case = self.prod_case(query="initial query")
        await run_prod_threading(graph, case)

        turn_one_msgs = graph.seen_messages[1]
        human_contents = [m.content for m in turn_one_msgs if isinstance(m, HumanMessage)]
        ai_contents = [m.content for m in turn_one_msgs if isinstance(m, AIMessage)]
        # Prior user message threaded through.
        assert "initial query" in human_contents
        # Prior assistant message (state.final_reply) threaded through.
        assert "turn0 assistant prose" in ai_contents

    @pytest.mark.asyncio
    async def test_prod_mode_refinement_minimal_edit_scores_1_0_end_to_end(
        self, mocker, monkeypatch
    ) -> None:
        """WARNING-4 + MEDIUM-2: a happy-path prod refinement that preserves
        non-target slots byte-equal must score refinement_minimal_edit == 1.0.
        """
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "true")

        # Turn 0: commit stops 1, 2, 3 with pids A, B, C.
        # Turn 1: commit stops 1, 2', 3 with pids A, NEWB, C (slot 2 swapped).
        turn_0_stops = self.stops_for_pids([self.PID_1, self.PID_2, self.PID_3])
        new_b = "ChIJ_test_id_zzzzzzzz_z"
        turn_1_stops = self.stops_for_pids([self.PID_1, new_b, self.PID_3])
        graph = self.ProdCapturingGraph(
            script=[(turn_0_stops, "turn0 reply"), (turn_1_stops, "turn1 reply")]
        )
        case = self.prod_case(target_slot=2)
        result = await evaluate_multi_turn_case(graph, case)

        assert result.deterministic.checks["refinement_minimal_edit"].score == 1.0


# ─── CR-02: constraints_for_case must not crash on clarification cases ──────


class TestConstraintsForCaseClarificationGuard:
    """CR-02 (EVAL-01 harness trustworthiness): constraints_for_case must
    not raise AttributeError when case.expected_results is None.

    Five hand_written cases have expected_results=None by design
    (expects_clarification_or_relaxation=True): impossible_four_am_five_star,
    impossible_cheap_michelin, impossible_north_beach_sushi_4am,
    overconstrained_walkable_three_neighborhoods, closed_monday_brunch.

    Without the None-guard the default full-suite run aborts with
    AttributeError on the first clarification case encountered.
    """

    def test_no_crash_on_known_clarification_case(self) -> None:
        """Focused regression: constraints_for_case must return UserConstraints
        (not raise) for impossible_four_am_five_star (expected_results=None)."""
        from app.agent.state import UserConstraints

        cases = {c.id: c for c in load_eval_queries("configs/eval_queries.yaml").hand_written}
        case = cases["impossible_four_am_five_star"]
        assert case.expected_results is None, (
            "test pre-condition: impossible_four_am_five_star must have expected_results=None"
        )
        result = constraints_for_case(case)
        assert isinstance(result, UserConstraints)
        assert result.num_stops is None, (
            "clarification case with no text-extracted stops must yield num_stops=None"
        )

    def test_no_crash_over_all_hand_written_cases(self) -> None:
        """Regression: constraints_for_case must not raise for ANY case in
        configs/eval_queries.yaml — including all 5 clarification cases that
        have expected_results=None.
        """
        from app.agent.state import UserConstraints

        cases = load_eval_queries("configs/eval_queries.yaml").hand_written
        assert len(cases) > 0, "hand_written cases must be non-empty"
        for case in cases:
            result = constraints_for_case(case)
            assert isinstance(result, UserConstraints), (
                f"constraints_for_case({case.id!r}) must return UserConstraints, got {type(result)}"
            )


# ============================================================================
# Phase 11 / Plan 11-01 Task 1: Phantom-key exclusion (WR-08 / D-11-05)
# + Single-turn error capture (WR-06 / D-11-06)
# ============================================================================


class TestPhantomKeyExclusion:
    """WR-08 / D-11-05: prior_committed_stops and prior_stops_obj scratch keys
    must be excluded from tool-call counting and tool-name listing.

    These keys are injected by the prod-threading branch as part of
    conversation-state serialization — they are NOT tool invocations. Before
    the fix, count_tool_calls and tool_names_from_state would count them as
    real tool calls, inflating tool_calls_mean in baselines.
    """

    def test_count_tool_calls_excludes_prior_committed_stops(self) -> None:
        """WR-08: prior_committed_stops list must not be counted as tool calls."""
        state = ItineraryState(
            scratch={
                "semantic_search": [{"id": "a"}, {"id": "b"}],
                "prior_committed_stops": [{"place_id": "ChIJtest_p1_aaaaaaaa"}],
            }
        )
        # Only the 2 semantic_search entries should be counted.
        assert count_tool_calls(state) == 2

    def test_count_tool_calls_excludes_prior_stops_obj(self) -> None:
        """WR-08: prior_stops_obj list must not be counted as tool calls."""
        state = ItineraryState(
            scratch={
                "nearby": [{"id": "x"}],
                "prior_stops_obj": [{"stop": "data"}],
            }
        )
        assert count_tool_calls(state) == 1

    def test_count_tool_calls_excludes_both_phantom_keys(self) -> None:
        """WR-08: both phantom scratch keys present simultaneously are excluded."""
        state = ItineraryState(
            scratch={
                "semantic_search": [{"id": "a"}],
                "prior_committed_stops": [{"place_id": "ChIJtest_p1_aaaaaaaa"}],
                "prior_stops_obj": [{"stop": "data"}],
            }
        )
        assert count_tool_calls(state) == 1

    def test_tool_names_excludes_prior_committed_stops(self) -> None:
        """WR-08: prior_committed_stops must not appear in tool_names_from_state."""
        state = ItineraryState(
            scratch={
                "semantic_search": [{"id": "a"}],
                "prior_committed_stops": [{"place_id": "ChIJtest_p1_aaaaaaaa"}],
            }
        )
        names = tool_names_from_state(state)
        assert "prior_committed_stops" not in names
        assert "semantic_search" in names

    def test_tool_names_excludes_prior_stops_obj(self) -> None:
        """WR-08: prior_stops_obj must not appear in tool_names_from_state."""
        state = ItineraryState(
            scratch={
                "nearby": [{"id": "x"}],
                "prior_stops_obj": [{"stop": "data"}],
            }
        )
        names = tool_names_from_state(state)
        assert "prior_stops_obj" not in names
        assert "nearby" in names

    def test_non_tool_scratch_keys_constant_exists(self) -> None:
        """WR-08: NON_TOOL_SCRATCH_KEYS constant must be defined in eval_agent."""
        from scripts.eval_agent import NON_TOOL_SCRATCH_KEYS

        assert "prior_committed_stops" in NON_TOOL_SCRATCH_KEYS
        assert "prior_stops_obj" in NON_TOOL_SCRATCH_KEYS


class TestSingleTurnErrorCapture:
    """WR-06 / D-11-06: A single-turn eval case (case.turns is None or empty)
    where graph.ainvoke raises must return an error record, not crash the run.

    Before the fix, exceptions during the single-turn ainvoke propagated to
    the evaluate_cases loop, aborting the whole eval run. Now they are caught
    and returned as make_error_record(case, "turn0", exc).
    """

    @pytest.mark.asyncio
    async def test_single_turn_ainvoke_exception_produces_error_record(self, mocker) -> None:
        """WR-06: single-turn ainvoke exception returns status='error' record."""
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        llm = RaisingChatModel(raise_exc=RuntimeError, raise_msg="quota exceeded")
        graph = build_agent_graph(llm, max_steps=4)

        case = eval_case(turns=None)  # single-turn path
        result = await evaluate_case(graph, case)

        assert isinstance(result, QueryEvalResult)
        assert result.status == "error"
        assert result.error is not None
        assert result.error["stage"] == "turn0"
        assert result.error["type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_single_turn_error_does_not_invoke_scorers(self, mocker) -> None:
        """WR-06: scorers must NOT be called when single-turn ainvoke raises."""
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        llm = RaisingChatModel(raise_exc=ValueError, raise_msg="db down")
        graph = build_agent_graph(llm, max_steps=4)

        case = eval_case(turns=None)
        result = await evaluate_case(graph, case)

        # All check scores must be None — scorers were not invoked.
        for check_result in result.deterministic.checks.values():
            assert check_result.score is None, (
                f"Scorer must not run on single-turn error; got score={check_result.score}"
            )

    @pytest.mark.asyncio
    async def test_single_turn_success_still_returns_scored_result(self, mocker) -> None:
        """WR-06: normal single-turn (no exception) still returns a scored result."""
        mocker.patch("app.agent.revision.itinerary_violations", return_value=[])
        llm = RecordingScriptedLLM(scripted=[finalize_msg("cafe reply")])
        graph = build_agent_graph(llm, max_steps=4)

        case = eval_case(turns=None)
        result = await evaluate_case(graph, case)

        # A successful run must NOT be an error record.
        assert result.status == "ok"
        assert result.error is None


# ============================================================================
# Phase 11 / Plan 11-01 Task 2: Zero-n derived-rate guards (WR-09 / D-11-04)
# + 0/1/2 exit-code contract (WR-07 / D-11-16)
# ============================================================================


class TestZeroNDerivedRateGuards:
    """WR-09 / D-11-04: All five derived-rate fields in aggregate_results must
    publish None (not 1.0) when n_scored == 0 (all-errored cell).

    Before the fix, rate(0, 0) returned 0.0 and the derived rates were
    computed from that, producing deterministic_pass_rate=1.0 on an
    all-errored cell — a measurement error that would be permanently baked
    into regenerated baselines if not fixed.
    """

    def test_all_errored_cell_deterministic_pass_rate_is_none(self) -> None:
        """D-11-04: deterministic_pass_rate is None when n_scored == 0."""
        from scripts.eval_agent import aggregate_results, make_error_record

        error_record = make_error_record(eval_case(), "turn0", Exception("quota"))
        agg = aggregate_results([error_record])

        assert agg["n_scored"] == 0
        assert agg["deterministic_pass_rate"] is None, (
            "deterministic_pass_rate must be None on all-errored cell, not 1.0"
        )

    def test_all_errored_cell_deterministic_violation_rate_is_none(self) -> None:
        """D-11-04: deterministic_violation_rate is None when n_scored == 0."""
        from scripts.eval_agent import aggregate_results, make_error_record

        error_record = make_error_record(eval_case(), "turn0", Exception("quota"))
        agg = aggregate_results([error_record])

        assert agg["deterministic_violation_rate"] is None

    def test_all_errored_cell_expected_results_mismatch_rate_is_none(self) -> None:
        """D-11-04: expected_results_mismatch_rate is None when n_scored == 0."""
        from scripts.eval_agent import aggregate_results, make_error_record

        error_record = make_error_record(eval_case(), "turnN", Exception("db down"))
        agg = aggregate_results([error_record])

        assert agg["expected_results_mismatch_rate"] is None

    def test_all_errored_cell_tool_error_rate_is_none(self) -> None:
        """D-11-04: tool_error_rate is None when n_scored == 0."""
        from scripts.eval_agent import aggregate_results, make_error_record

        error_record = make_error_record(eval_case(), "turn0", Exception("quota"))
        agg = aggregate_results([error_record])

        assert agg["tool_error_rate"] is None

    def test_all_errored_cell_tool_success_rate_is_none(self) -> None:
        """D-11-04: tool_success_rate is None when n_scored == 0."""
        from scripts.eval_agent import aggregate_results, make_error_record

        error_record = make_error_record(eval_case(), "turn0", Exception("quota"))
        agg = aggregate_results([error_record])

        assert agg["tool_success_rate"] is None

    def test_all_errored_cell_committed_itinerary_rate_is_none(self) -> None:
        """WR-01: committed_itinerary_rate — the hard-gate metric — must be
        None on an all-errored cell, never the fabricated mean([]) == 0.0
        that would read as a hard decisiveness regression of the anchor."""
        from scripts.eval_agent import aggregate_results, make_error_record

        error_record = make_error_record(eval_case(), "turn0", Exception("quota"))
        agg = aggregate_results([error_record])

        assert agg["committed_itinerary_rate"] is None, (
            "an infra-dead cell must publish None, not 0.0, for the hard-gate metric"
        )

    def test_all_errored_cell_all_derived_rates_are_none(self) -> None:
        """D-11-04 + WR-01: all six derived rates are None on an all-errored cell."""
        from scripts.eval_agent import aggregate_results, make_error_record

        records = [
            make_error_record(eval_case(id="a"), "turn0", Exception("quota")),
            make_error_record(eval_case(id="b"), "turn0", Exception("quota")),
        ]
        agg = aggregate_results(records)

        assert agg["deterministic_pass_rate"] is None
        assert agg["deterministic_violation_rate"] is None
        assert agg["expected_results_mismatch_rate"] is None
        assert agg["tool_error_rate"] is None
        assert agg["tool_success_rate"] is None
        assert agg["committed_itinerary_rate"] is None

    def test_normal_scored_cell_still_returns_float_rates(self) -> None:
        """D-11-04: non-regression — float derived rates still work when n_scored > 0."""
        agg = aggregate_results([query_result()])

        assert isinstance(agg["deterministic_pass_rate"], float)
        assert isinstance(agg["deterministic_violation_rate"], float)
        assert isinstance(agg["expected_results_mismatch_rate"], float)
        assert isinstance(agg["tool_error_rate"], float)
        assert isinstance(agg["tool_success_rate"], float)
        assert isinstance(agg["committed_itinerary_rate"], float)


class TestZeroStopAbstainPipeline:
    """CR-01 / D-11-03: the category_compliance None-abstain must survive the
    REAL eval pipeline — score_checks → aggregate_results → main() exit code —
    not just the scorer function in isolation.

    Before the fix, score_checks coerced with ``float(check(state))``;
    ``float(None)`` raised TypeError, the blanket except converted the abstain
    into a scorer ERROR (check_error_count > 0) and a violation, and
    report_has_errors forced exit 2 ("infra failure — rerun needed") for what
    is a pure model-behavior outcome. Every zero-stop run tripped it.
    """

    def zero_stop_state(self) -> ItineraryState:
        return ItineraryState(
            constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
            stops=[],
        )

    def test_score_checks_abstain_is_not_error_and_not_violation(self) -> None:
        """The None abstain must not surface as a scorer error or a violation."""
        checks = score_checks(self.zero_stop_state())

        result = checks["category_compliance"]
        assert result.score is None, "zero-stop abstain must keep score=None"
        assert result.error is None, "abstain must NOT be recorded as a scorer error"
        assert result.passed is True, "abstain must NOT count as a failed check"
        assert "category_compliance" not in violations_from_checks(checks)

    def test_zero_stop_aggregate_reports_no_check_error_and_none_mean(self) -> None:
        """Real pipeline: zero-stop state → score_checks → aggregate_results."""
        result = query_result_from_state(eval_case(), self.zero_stop_state())
        agg = aggregate_results([result])

        assert agg["check_error_count"] == 0, (
            "abstain must not inflate check_error_count (it forces exit 2)"
        )
        assert agg["category_compliance_mean"] is None, (
            "a cell with zero non-None scores must publish None, not mean([]) == 0.0 — "
            "'no signal' must never read as 'zero compliance'"
        )

    def test_mixed_cell_mean_excludes_abstained_run(self) -> None:
        """One abstained run + one populated run → mean over the populated run only."""
        zero_stop = query_result_from_state(eval_case(), self.zero_stop_state())
        populated = query_result()  # category_compliance score 1.0

        agg = aggregate_results([zero_stop, populated])

        assert agg["category_compliance_mean"] == 1.0, (
            "the abstained (None) run must be excluded from the mean entirely"
        )

    def test_zero_stop_run_exits_1_not_2(self, mocker) -> None:
        """Full path to the exit code: a zero-stop run is a model-behavior
        outcome (exit 1 via the expected_results violation), never an infra
        failure (exit 2) — the D-11-16 contract."""
        from scripts.eval_agent import main

        result = query_result_from_state(eval_case(), self.zero_stop_state())
        report = EvalRunReport(
            eval_queries_path="configs/eval_queries.yaml",
            llm_provider="openai",
            chat_model="gpt-4o-mini",
            query_count=1,
            aggregate=aggregate_results([result]),
            queries=[result],
        )
        mocker.patch("scripts.eval_agent.asyncio.run", side_effect=lambda coro: report)

        rc = main(["--llm-provider", "openai"])
        assert rc == 1, f"zero-stop run must exit 1 (model behavior), not 2 (infra); got {rc}"


class TestExitCodeContract:
    """WR-07 / D-11-16: main() must return 0/1/2 per the three-way contract.

    Before the fix, main() returned 1 for BOTH infra failures (build_report
    exception) AND violations. CI cannot distinguish a rerun-needed infra
    failure (2) from a model-behavior miss (1) that is expected in exploratory
    runs.

    Exit codes:
        0 = clean (no infra failures, no violations)
        1 = model-behavior violations (non-blocking; rerun not needed)
        2 = infra failure (build_report raised; rerun needed)
    """

    def make_report(self, *, n_errored: int = 0, violations: int = 0) -> EvalRunReport:
        """Build a synthetic EvalRunReport for exit-code testing."""
        return EvalRunReport(
            eval_queries_path="configs/eval_queries.yaml",
            llm_provider="openai",
            chat_model="gpt-4o-mini",
            query_count=1,
            aggregate={
                "check_error_count": 0,
                "n_errored": n_errored,
                "queries_with_violations": violations,
            },
            queries=[query_result()],
        )

    def test_main_returns_2_when_report_has_errors(self, mocker) -> None:
        """D-11-16: main() returns 2 when report_has_errors is True (n_errored > 0)."""
        from scripts.eval_agent import main

        report = self.make_report(n_errored=1, violations=0)
        mocker.patch("scripts.eval_agent.build_report", return_value=report)
        mocker.patch("scripts.eval_agent.asyncio.run", side_effect=lambda coro: report)

        rc = main(["--llm-provider", "openai"])
        assert rc == 2, f"Expected exit 2 for infra errors, got {rc}"

    def test_main_returns_1_when_only_violations(self, mocker) -> None:
        """D-11-16: main() returns 1 when only model-behavior violations present."""
        from scripts.eval_agent import main

        report = self.make_report(n_errored=0, violations=1)
        mocker.patch("scripts.eval_agent.asyncio.run", side_effect=lambda coro: report)

        rc = main(["--llm-provider", "openai"])
        assert rc == 1, f"Expected exit 1 for violations, got {rc}"

    def test_main_returns_0_when_clean(self, mocker) -> None:
        """D-11-16: main() returns 0 when no infra errors and no violations."""
        from scripts.eval_agent import main

        report = self.make_report(n_errored=0, violations=0)
        mocker.patch("scripts.eval_agent.asyncio.run", side_effect=lambda coro: report)

        rc = main(["--llm-provider", "openai"])
        assert rc == 0, f"Expected exit 0 for clean run, got {rc}"

    def test_main_returns_2_when_build_report_raises(self, mocker) -> None:
        """D-11-16: main() returns 2 (not 1) when build_report raises."""
        from scripts.eval_agent import main

        mocker.patch(
            "scripts.eval_agent.asyncio.run",
            side_effect=RuntimeError("embedding quota exceeded"),
        )

        rc = main(["--llm-provider", "openai"])
        assert rc == 2, (
            "build_report exception must map to exit 2 (infra failure), not 1 (violations)"
        )


# ---------------------------------------------------------------------------
# Phase 12 Plan 02: INST-01/02/03 derived-field helper tests
# ---------------------------------------------------------------------------

from app.agent.revision import LOW_SIMILARITY_THRESHOLD  # noqa: E402 — test import


def state_with_commit_at_step(step: int) -> ItineraryState:
    """Minimal ItineraryState that has a commit_itinerary scratch entry at ``step``."""
    return ItineraryState(
        scratch={"commit_itinerary": [{"step": step, "args": {}, "result": {}, "id": "tc1"}]}
    )


def state_with_search_hits(
    hits: list[dict],
    step: int = 0,
    tool: str = "semantic_search",
) -> ItineraryState:
    """Build a state with one retrieval scratch entry at the given step."""
    return ItineraryState(scratch={tool: [{"step": step, "result": hits}]})


class TestFirstCommitCallStepFromState:
    """Tests for first_commit_call_step_from_state (INST-01 / D-12-03)."""

    def test_returns_step_index_for_single_commit(self) -> None:
        state = state_with_commit_at_step(3)
        assert first_commit_call_step_from_state(state) == 3

    def test_returns_min_when_multiple_commit_entries(self) -> None:
        state = ItineraryState(
            scratch={
                "commit_itinerary": [
                    {"step": 5, "args": {}, "result": {}, "id": "tc2"},
                    {"step": 2, "args": {}, "result": {}, "id": "tc1"},
                ]
            }
        )
        assert first_commit_call_step_from_state(state) == 2

    def test_returns_none_on_empty_state(self) -> None:
        assert first_commit_call_step_from_state(ItineraryState()) is None

    def test_returns_none_when_commit_itinerary_is_non_list(self) -> None:
        state = ItineraryState(scratch={"commit_itinerary": "not-a-list"})
        assert first_commit_call_step_from_state(state) is None

    def test_returns_none_when_entries_have_no_step_key(self) -> None:
        state = ItineraryState(scratch={"commit_itinerary": [{"args": {}}]})
        assert first_commit_call_step_from_state(state) is None

    def test_returns_none_when_step_is_none(self) -> None:
        """WR-08: step=None must not crash min() — docstring promises None."""
        state = ItineraryState(scratch={"commit_itinerary": [{"step": None, "args": {}}]})
        assert first_commit_call_step_from_state(state) is None

    def test_skips_non_int_steps_in_mixed_entries(self) -> None:
        """WR-08: mixed [2, "3"] must return 2, not raise TypeError."""
        state = ItineraryState(
            scratch={
                "commit_itinerary": [
                    {"step": "3", "args": {}},
                    {"step": 2, "args": {}},
                ]
            }
        )
        assert first_commit_call_step_from_state(state) == 2

    def test_bool_step_is_not_a_valid_step(self) -> None:
        """WR-08: bool is a subclass of int — reject it explicitly."""
        state = ItineraryState(scratch={"commit_itinerary": [{"step": True, "args": {}}]})
        assert first_commit_call_step_from_state(state) is None

    def test_output_is_json_safe(self) -> None:
        state = state_with_commit_at_step(1)
        result = first_commit_call_step_from_state(state)
        assert json.dumps(result) is not None  # type: ignore[arg-type]


class TestViableCandidatesPerStepFromState:
    """Tests for viable_candidates_per_step_from_state (INST-02 / D-12-04)."""

    def high_sim_hit(self, primary_type: str) -> dict:
        return {"similarity": LOW_SIMILARITY_THRESHOLD + 0.1, "primary_type": primary_type}

    def low_sim_hit(self, primary_type: str) -> dict:
        return {"similarity": 0.0, "primary_type": primary_type}

    def test_counts_hits_above_threshold_matching_type(self) -> None:
        hits = [self.high_sim_hit("cafe"), self.high_sim_hit("cafe")]
        state = state_with_search_hits(hits, step=0)
        result = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, ["cafe"])
        assert result == [2]

    def test_nearby_entries_are_not_scanned(self) -> None:
        """WR-01: viability is semantic-search-only. The nearby tool's SQL
        hardcodes 0.0 AS similarity (app/tools/retrieval.py), so no nearby hit
        could ever clear the threshold — the helper does not read the source it
        can never count. Even a high-similarity hit under the 'nearby' key is
        ignored (documented limitation: nearby-driven flows undercount)."""
        hits = [self.high_sim_hit("cafe")]
        state = state_with_search_hits(hits, step=0, tool="nearby")
        result = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, ["cafe"])
        assert result == [], "nearby scratch entries must not be scanned for viability"

    def test_wrong_type_excluded(self) -> None:
        hits = [self.high_sim_hit("restaurant")]
        state = state_with_search_hits(hits, step=0)
        result = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, ["cafe"])
        assert result == [0]

    def test_per_step_not_cumulative(self) -> None:
        """An entry at step 0 and another at step 1 yield two independent counts."""
        state = ItineraryState(
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [self.high_sim_hit("cafe")]},
                    {"step": 1, "result": [self.high_sim_hit("cafe"), self.high_sim_hit("cafe")]},
                ]
            }
        )
        result = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, ["cafe"])
        # Step 0 has 1 hit, step 1 has 2 hits — NOT cumulative [1, 3]
        assert result == [1, 2]

    def test_empty_requested_types_counts_on_cosine_only(self) -> None:
        hits = [self.high_sim_hit("cafe"), self.low_sim_hit("bar")]
        state = state_with_search_hits(hits, step=0)
        # No type constraint: only cosine matters
        result = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, [])
        assert result == [1]  # only the high-similarity hit counts

    def test_empty_state_returns_empty_list(self) -> None:
        result = viable_candidates_per_step_from_state(
            ItineraryState(), LOW_SIMILARITY_THRESHOLD, []
        )
        assert result == []

    def test_output_is_json_safe(self) -> None:
        hits = [self.high_sim_hit("cafe")]
        state = state_with_search_hits(hits, step=0)
        result = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, ["cafe"])
        assert json.dumps(result) is not None


class TestRule8MetPerStepFromState:
    """Tests for rule8_met_per_step_from_state (INST-03 / D-12-05)."""

    def high_sim_hit(self, primary_type: str) -> dict:
        return {"similarity": LOW_SIMILARITY_THRESHOLD + 0.1, "primary_type": primary_type}

    def test_flips_true_only_once_both_types_covered_cumulatively(self) -> None:
        """Cumulative: False while only one type covered, True once both covered."""
        state = ItineraryState(
            scratch={
                "semantic_search": [
                    # Step 0: only covers 'cafe'
                    {"step": 0, "result": [self.high_sim_hit("cafe")]},
                    # Step 1: covers 'bar' — now both covered cumulatively
                    {"step": 1, "result": [self.high_sim_hit("bar")]},
                ]
            }
        )
        requested = ["cafe", "bar"]
        viable = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, requested)
        result = rule8_met_per_step_from_state(state, viable, requested, LOW_SIMILARITY_THRESHOLD)
        assert result == [False, True], f"expected [False, True], got {result}"

    def test_stays_false_when_only_one_of_two_types_covered(self) -> None:
        state = ItineraryState(
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [self.high_sim_hit("cafe")]},
                    {"step": 1, "result": [self.high_sim_hit("cafe")]},
                ]
            }
        )
        requested = ["cafe", "bar"]
        viable = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, requested)
        result = rule8_met_per_step_from_state(state, viable, requested, LOW_SIMILARITY_THRESHOLD)
        assert result == [False, False]

    def test_empty_requested_types_fallback_uses_cumulative_count(self) -> None:
        """With no requested_types, True once cumulative count >= 1."""
        state = ItineraryState(
            scratch={
                "semantic_search": [
                    {"step": 0, "result": []},  # no viable hits at step 0
                    {"step": 1, "result": [self.high_sim_hit("cafe")]},
                ]
            }
        )
        viable = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, [])
        result = rule8_met_per_step_from_state(state, viable, [], LOW_SIMILARITY_THRESHOLD)
        # Step 0: cumulative=0 → False; Step 1: cumulative=1 → True
        assert result == [False, True]

    def test_empty_state_returns_empty(self) -> None:
        result = rule8_met_per_step_from_state(
            ItineraryState(), [], ["cafe"], LOW_SIMILARITY_THRESHOLD
        )
        assert result == []

    def test_duplicate_requested_types_need_distinct_place_ids(self) -> None:
        """WR-02: ['restaurant', 'restaurant', 'bar'] means two DISTINCT
        restaurant stops — one viable restaurant must not mark both covered."""

        def hit(ptype: str, pid: str) -> dict:
            return {
                "similarity": LOW_SIMILARITY_THRESHOLD + 0.1,
                "primary_type": ptype,
                "place_id": pid,
            }

        requested = ["restaurant", "restaurant", "bar"]
        one_restaurant = ItineraryState(
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [hit("restaurant", "r1"), hit("bar", "b1")]},
                ]
            }
        )
        viable = viable_candidates_per_step_from_state(
            one_restaurant, LOW_SIMILARITY_THRESHOLD, requested
        )
        result = rule8_met_per_step_from_state(
            one_restaurant, viable, requested, LOW_SIMILARITY_THRESHOLD
        )
        assert result == [False], (
            "one viable restaurant must NOT cover two requested restaurant slots"
        )

        two_restaurants = ItineraryState(
            scratch={
                "semantic_search": [
                    {
                        "step": 0,
                        "result": [
                            hit("restaurant", "r1"),
                            hit("restaurant", "r2"),
                            hit("bar", "b1"),
                        ],
                    },
                ]
            }
        )
        viable = viable_candidates_per_step_from_state(
            two_restaurants, LOW_SIMILARITY_THRESHOLD, requested
        )
        result = rule8_met_per_step_from_state(
            two_restaurants, viable, requested, LOW_SIMILARITY_THRESHOLD
        )
        assert result == [True], "two distinct viable restaurants + a bar cover all three slots"

    def test_same_place_id_at_one_step_counts_once_for_typed_coverage(self) -> None:
        """WR-02: the same venue returned twice is still ONE distinct candidate."""

        def hit(pid: str) -> dict:
            return {
                "similarity": LOW_SIMILARITY_THRESHOLD + 0.1,
                "primary_type": "restaurant",
                "place_id": pid,
            }

        requested = ["restaurant", "restaurant"]
        state = ItineraryState(
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [hit("r1"), hit("r1")]},
                ]
            }
        )
        viable = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, requested)
        result = rule8_met_per_step_from_state(state, viable, requested, LOW_SIMILARITY_THRESHOLD)
        assert result == [False]

    def test_no_types_fallback_dedupes_repeated_venue_across_steps(self) -> None:
        """WR-02: with num_stops=3, the SAME place_id returned at steps 0, 1, 2
        must count as ONE viable candidate, not three."""

        def hit(pid: str) -> dict:
            return {"similarity": LOW_SIMILARITY_THRESHOLD + 0.1, "place_id": pid}

        same_venue = ItineraryState(
            constraints=UserConstraints(num_stops=3),
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [hit("p1")]},
                    {"step": 1, "result": [hit("p1")]},
                    {"step": 2, "result": [hit("p1")]},
                ]
            },
        )
        viable = viable_candidates_per_step_from_state(same_venue, LOW_SIMILARITY_THRESHOLD, [])
        result = rule8_met_per_step_from_state(same_venue, viable, [], LOW_SIMILARITY_THRESHOLD)
        assert result == [False, False, False], (
            f"one venue repeated 3x must not satisfy a 3-stop request, got {result}"
        )

        distinct_venues = ItineraryState(
            constraints=UserConstraints(num_stops=3),
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [hit("p1")]},
                    {"step": 1, "result": [hit("p2")]},
                    {"step": 2, "result": [hit("p3")]},
                ]
            },
        )
        viable = viable_candidates_per_step_from_state(
            distinct_venues, LOW_SIMILARITY_THRESHOLD, []
        )
        result = rule8_met_per_step_from_state(
            distinct_venues, viable, [], LOW_SIMILARITY_THRESHOLD
        )
        assert result == [False, False, True]

    def test_output_is_json_safe(self) -> None:
        state = ItineraryState(
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [self.high_sim_hit("cafe")]},
                ]
            }
        )
        viable = viable_candidates_per_step_from_state(state, LOW_SIMILARITY_THRESHOLD, ["cafe"])
        result = rule8_met_per_step_from_state(state, viable, ["cafe"], LOW_SIMILARITY_THRESHOLD)
        assert json.dumps(result) is not None

    def test_caller_threshold_is_honored_not_module_constant(self) -> None:
        """WR-09: the threshold is a caller parameter — a hit below the module
        constant but above the caller's threshold must count."""
        hit = {"similarity": 0.40, "primary_type": "cafe", "place_id": "p1"}
        state = ItineraryState(scratch={"semantic_search": [{"step": 0, "result": [hit]}]})
        viable_low = viable_candidates_per_step_from_state(state, 0.3, ["cafe"])
        assert rule8_met_per_step_from_state(state, viable_low, ["cafe"], 0.3) == [True]
        viable_high = viable_candidates_per_step_from_state(
            state, LOW_SIMILARITY_THRESHOLD, ["cafe"]
        )
        assert rule8_met_per_step_from_state(
            state, viable_high, ["cafe"], LOW_SIMILARITY_THRESHOLD
        ) == [False]


class TestKeptSearchingDerivation:
    """WR-09: query_result_from_state's rule8_met_but_kept_searching_steps —
    steps where rule 8 was met but the model did NOT commit (commit step
    excluded)."""

    def hit(self, pid: str) -> dict:
        return {"similarity": LOW_SIMILARITY_THRESHOLD + 0.1, "place_id": pid}

    def test_commit_step_excluded_from_kept_searching(self) -> None:
        """Rule 8 met at steps 0 and 1; commit at step 1 → only step 0 is a
        decisiveness-gap step."""
        state = ItineraryState(
            constraints=UserConstraints(num_stops=1),
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [self.hit("p1")]},
                    {"step": 1, "result": [self.hit("p2")]},
                ],
                "commit_itinerary": [{"step": 1, "args": {}, "result": {}, "id": "tc1"}],
            },
        )
        result = query_result_from_state(eval_case(), state)
        assert result.deterministic.rule8_met_per_step == [True, True]
        assert result.deterministic.rule8_met_but_kept_searching_steps == [0], (
            "the commit step (1) must be excluded; only step 0 is kept-searching"
        )
        assert result.deterministic.first_commit_call_step == 1

    def test_no_commit_means_every_met_step_is_kept_searching(self) -> None:
        state = ItineraryState(
            constraints=UserConstraints(num_stops=1),
            scratch={
                "semantic_search": [
                    {"step": 0, "result": [self.hit("p1")]},
                    {"step": 1, "result": [self.hit("p2")]},
                ],
            },
        )
        result = query_result_from_state(eval_case(), state)
        assert result.deterministic.rule8_met_but_kept_searching_steps == [0, 1]
        assert result.deterministic.first_commit_call_step is None

    def test_report_threshold_field_matches_module_constant(self) -> None:
        """The self-describing viability_threshold field binds both per-step
        metrics via the same threaded local (WR-09)."""
        result = query_result_from_state(eval_case(), ItineraryState())
        assert result.deterministic.viability_threshold == LOW_SIMILARITY_THRESHOLD


# ── Plan 13-01 / D-13-04 / D-13-05: arm_flags + forced-commit telemetry ─────


class TestArmFlagsAndForcedCommitTelemetry:
    """Run-JSON self-description for Phase 13 experiment arms (D-13-04, D-13-05).

    Task 3 of Plan 13-01: DeterministicEvalResult gains commit_forced,
    forced_commit_step, and arm_flags fields; make_error_record carries safe
    defaults; no new scorers are added.
    """

    def test_deterministic_eval_result_has_arm_flags_field(self) -> None:
        """DeterministicEvalResult dataclass has an arm_flags field."""
        assert hasattr(DeterministicEvalResult, "__dataclass_fields__")
        assert "arm_flags" in DeterministicEvalResult.__dataclass_fields__

    def test_deterministic_eval_result_has_commit_forced_field(self) -> None:
        """DeterministicEvalResult dataclass has a commit_forced field."""
        assert "commit_forced" in DeterministicEvalResult.__dataclass_fields__

    def test_deterministic_eval_result_has_forced_commit_step_field(self) -> None:
        """DeterministicEvalResult dataclass has a forced_commit_step field."""
        assert "forced_commit_step" in DeterministicEvalResult.__dataclass_fields__

    def test_arm_flags_all_off_when_no_env_vars_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With all arm env vars unset, arm_flags reflects 'all off'."""
        monkeypatch.delenv("VIABILITY_CONTRACT_ENABLED", raising=False)
        monkeypatch.delenv("FORCED_COMMIT_STEP", raising=False)
        monkeypatch.delenv("PARALLEL_TOOL_EXECUTION_ENABLED", raising=False)
        monkeypatch.delenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", raising=False)
        # Phase-14 replay keys (also off by default)
        monkeypatch.delenv("REPLAY_MULTI_MESSAGE_ENABLED", raising=False)
        monkeypatch.delenv("REPLAY_CONTENTBLOCKS_ENABLED", raising=False)

        result = query_result_from_state(eval_case(), ItineraryState())

        assert result.deterministic.arm_flags == {
            # Phase-13 DEC arm keys
            "viability_contract": False,
            "forced_commit_step": 0,
            "parallel_tool": False,
            "viability_threshold_override": None,
            # Phase-14 REPLAY arm keys
            "replay_multi_message": False,
            "replay_content_blocks": False,
        }

    def test_arm_flags_reflects_env_vars_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """arm_flags reflects the env at run time when flags are set."""
        monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "1")
        monkeypatch.setenv("FORCED_COMMIT_STEP", "6")
        monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "true")
        monkeypatch.setenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", "0.50")
        # Phase-14 replay keys left unset (verifies Phase-13 keys still present alongside new ones)
        monkeypatch.delenv("REPLAY_MULTI_MESSAGE_ENABLED", raising=False)
        monkeypatch.delenv("REPLAY_CONTENTBLOCKS_ENABLED", raising=False)

        result = query_result_from_state(eval_case(), ItineraryState())

        assert result.deterministic.arm_flags == {
            # Phase-13 DEC arm keys
            "viability_contract": True,
            "forced_commit_step": 6,
            "parallel_tool": True,
            "viability_threshold_override": "0.50",
            # Phase-14 REPLAY arm keys (unset → False)
            "replay_multi_message": False,
            "replay_content_blocks": False,
        }

    def test_commit_forced_default_false_when_state_not_forced(self) -> None:
        """commit_forced is False when state.commit_forced is False (default path)."""
        state = ItineraryState()
        assert state.commit_forced is False
        result = query_result_from_state(eval_case(), state)
        assert result.deterministic.commit_forced is False

    def test_commit_forced_true_when_state_carries_it(self) -> None:
        """commit_forced is True when state carries it (D-13-04)."""
        state = ItineraryState(commit_forced=True, forced_commit_step=4)
        result = query_result_from_state(eval_case(), state)
        assert result.deterministic.commit_forced is True
        assert result.deterministic.forced_commit_step == 4

    def test_forced_commit_step_none_when_not_forced(self) -> None:
        """forced_commit_step is None when state carries no forced commit."""
        result = query_result_from_state(eval_case(), ItineraryState())
        assert result.deterministic.forced_commit_step is None

    def test_error_record_has_safe_defaults_for_new_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """make_error_record carries commit_forced=False, forced_commit_step=None, arm_flags={}."""
        monkeypatch.delenv("VIABILITY_CONTRACT_ENABLED", raising=False)
        monkeypatch.delenv("FORCED_COMMIT_STEP", raising=False)
        monkeypatch.delenv("PARALLEL_TOOL_EXECUTION_ENABLED", raising=False)
        monkeypatch.delenv("LOW_SIMILARITY_THRESHOLD_OVERRIDE", raising=False)
        from dataclasses import asdict

        case = eval_case()
        record = make_error_record(case, "setup", RuntimeError("boom"))
        det = record.deterministic

        assert det.commit_forced is False
        assert det.forced_commit_step is None
        assert det.arm_flags == {}
        # Must serialize without raising
        asdict(record)

    def test_no_new_scorer_registered_for_arm_fields(self) -> None:
        """arm_flags / commit_forced / forced_commit_step must NOT be in DETERMINISTIC_CHECKS
        (D-13-04 anti-scope: these are telemetry fields, not scorers)."""
        assert "arm_flags" not in DETERMINISTIC_CHECKS
        assert "commit_forced" not in DETERMINISTIC_CHECKS
        assert "forced_commit_step" not in DETERMINISTIC_CHECKS
