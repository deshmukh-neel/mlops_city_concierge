#!/usr/bin/env python3
"""Run deterministic offline evals for the City Concierge agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent.critique.checks import (  # noqa: E402
    CRITIQUE_THRESHOLDS,
    constraints_satisfied,
    geographic_coherence,
    no_hallucinated_place_ids,
    temporal_coherence,
    walking_budget_respected,
)
from app.agent.graph import build_agent_graph  # noqa: E402
from app.agent.state import ItineraryState  # noqa: E402
from app.config import get_settings, resolve_llm_api_key  # noqa: E402
from app.eval.config import (  # noqa: E402
    DEFAULT_EVAL_QUERIES_PATH,
    EvalQuery,
    ExpectedResults,
    load_eval_queries,
)

LlmProvider = Literal["openai", "gemini"]
CheckFunction = Callable[[ItineraryState], float]

DETERMINISTIC_CHECKS: dict[str, CheckFunction] = {
    "constraints_satisfied": constraints_satisfied,
    "geographic_coherence": geographic_coherence,
    "temporal_coherence": temporal_coherence,
    "walking_budget_respected": walking_budget_respected,
    "no_hallucinated_place_ids": no_hallucinated_place_ids,
}

EVAL_CONTEXT_TEMPLATE = """Offline eval context:
- Interpret all San Francisco date/time requests in America/Los_Angeles.
- If a case's expected open time is provided below, use that exact ISO timestamp
  when setting tool filters.
- If an expected result range is provided below, answer with a number of
  committed stops inside that range and do not ask how many stops the user
  wants.
- These instructions provide deterministic eval context; the user's query below
  is the behavior being evaluated.

Expected open time: {open_at}
Expected result range: {expected_results}
"""


@dataclass
class CheckResult:
    """A single deterministic check result for one eval case."""

    score: float | None
    threshold: float
    passed: bool
    error: str | None = None


@dataclass
class ExpectedEvalResult:
    """Expected behavior metadata for one eval record."""

    min_stops: int | None
    max_stops: int | None
    expects_clarification_or_relaxation: bool


@dataclass
class ActualEvalResult:
    """Actual structured output committed by the agent."""

    result_count: int
    committed_stop_count: int
    place_ids: list[str]
    place_names: list[str]
    sources: list[str]
    answer_place_names: list[str]


@dataclass
class DeterministicEvalResult:
    """Deterministic eval diagnostics for one RAGAS-compatible record."""

    expected_results_met: bool | None
    checks: dict[str, CheckResult]
    violations: list[str]
    tool_errors: list[str]
    first_tool_error: str | None
    tool_calls: int
    tool_names: list[str]
    revision_hints: int
    revision_reasons: list[str]


@dataclass
class QueryEvalResult:
    """RAGAS-compatible per-query eval output plus deterministic diagnostics."""

    id: str
    question: str
    answer: str
    contexts: list[str]
    reference: str
    tags: list[str]
    expected: ExpectedEvalResult
    actual: ActualEvalResult
    deterministic: DeterministicEvalResult
    final_reply: str


@dataclass
class EvalRunReport:
    """Serializable report for one offline eval run."""

    eval_queries_path: str
    llm_provider: str
    chat_model: str
    query_count: int
    aggregate: dict[str, float | int]
    queries: list[QueryEvalResult]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the deterministic eval runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-queries",
        default=str(DEFAULT_EVAL_QUERIES_PATH),
        help="YAML eval query config to run.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Candidate LLM provider to evaluate.",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Candidate chat model. Defaults to the configured model for the provider.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit the number of hand-written eval cases for a quick smoke run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the JSON report. The report is always printed to stdout.",
    )
    return parser.parse_args(argv)


def resolve_chat_model(provider: LlmProvider, chat_model: str | None) -> str:
    """Resolve the candidate chat model from CLI input or environment settings."""
    if chat_model and chat_model.strip():
        return chat_model.strip()
    settings = get_settings()
    if provider == "openai":
        return settings.openai_chat_model
    return settings.gemini_chat_model


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument relationships that argparse cannot express cleanly."""
    if args.max_steps < 1:
        raise ValueError("--max-steps must be greater than zero")


def build_eval_llm(provider: LlmProvider, chat_model: str, temperature: float) -> BaseChatModel:
    """Construct the candidate chat model used to drive the agent graph."""
    api_key = resolve_llm_api_key(provider)
    if provider == "openai":
        return ChatOpenAI(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=temperature,
        )
    return ChatGoogleGenerativeAI(
        model=chat_model,
        google_api_key=SecretStr(api_key),
        temperature=temperature,
    )


def selected_cases(cases: list[EvalQuery], max_queries: int | None) -> list[EvalQuery]:
    """Return the hand-written eval cases selected for this run."""
    if max_queries is None:
        return cases
    if max_queries < 1:
        raise ValueError("--max-queries must be greater than zero when provided")
    return cases[:max_queries]


def state_from_graph_output(raw: Any) -> ItineraryState:
    """Normalize LangGraph output into an ItineraryState model."""
    if isinstance(raw, ItineraryState):
        return raw
    if isinstance(raw, dict):
        return ItineraryState(**raw)
    raise TypeError(f"Unexpected graph output type: {type(raw).__name__}")


def count_tool_calls(state: ItineraryState) -> int:
    """Count captured tool calls from the agent scratchpad."""
    return sum(len(entries) for entries in state.scratch.values() if isinstance(entries, list))


def tool_names_from_state(state: ItineraryState) -> list[str]:
    """Return tool names that appear in scratchpad insertion order."""
    names: list[str] = []
    for tool_name, entries in state.scratch.items():
        if isinstance(entries, list) and entries:
            names.append(tool_name)
    return names


def value_from_hit(hit: Any, field_name: str) -> Any:
    """Read one field from a pydantic model, dict, or generic object."""
    if isinstance(hit, dict):
        return hit.get(field_name)
    return getattr(hit, field_name, None)


def context_from_hit(hit: Any) -> str | None:
    """Format one retrieved place hit as concise context for RAGAS."""
    name = value_from_hit(hit, "name")
    snippet = value_from_hit(hit, "snippet")
    address = value_from_hit(hit, "formatted_address")
    primary_type = value_from_hit(hit, "primary_type")
    rating = value_from_hit(hit, "rating")
    if not name and not snippet:
        return None
    parts = [
        f"name: {name}" if name else None,
        f"type: {primary_type}" if primary_type else None,
        f"address: {address}" if address else None,
        f"rating: {rating}" if rating is not None else None,
        f"snippet: {snippet}" if snippet else None,
    ]
    return " | ".join(part for part in parts if part)


def contexts_from_state(state: ItineraryState) -> list[str]:
    """Extract retrieved text contexts from semantic and nearby tool results."""
    contexts: list[str] = []
    seen: set[str] = set()
    for tool_name in ("semantic_search", "nearby"):
        entries = state.scratch.get(tool_name)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            result = entry.get("result")
            if not isinstance(result, list):
                continue
            for hit in result:
                context = context_from_hit(hit)
                if context and context not in seen:
                    contexts.append(context)
                    seen.add(context)
    return contexts


def tool_errors_from_state(state: ItineraryState) -> list[str]:
    """Extract tool error messages captured in the agent scratchpad."""
    errors: list[str] = []
    for tool_name, entries in state.scratch.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            result = entry.get("result")
            if isinstance(result, dict) and result.get("error"):
                errors.append(f"{tool_name}: {result['error']}")
    return errors


def revision_reasons_from_state(state: ItineraryState) -> list[str]:
    """Return critique revision reasons in the order they were emitted."""
    return [hint.reason for hint in state.revision_hints]


def expected_eval_result(case: EvalQuery) -> ExpectedEvalResult:
    """Build the expected block for the eval report."""
    expected = case.expected_results
    return ExpectedEvalResult(
        min_stops=expected.min_stops if expected else None,
        max_stops=expected.max_stops if expected else None,
        expects_clarification_or_relaxation=case.expects_clarification_or_relaxation,
    )


def retrieved_place_names_from_state(state: ItineraryState) -> list[str]:
    """Return unique place names from retrieval contexts in scratch order."""
    names: list[str] = []
    seen: set[str] = set()
    for tool_name in ("semantic_search", "nearby"):
        entries = state.scratch.get(tool_name)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            result = entry.get("result")
            if not isinstance(result, list):
                continue
            for hit in result:
                name = value_from_hit(hit, "name")
                if isinstance(name, str) and name and name not in seen:
                    names.append(name)
                    seen.add(name)
    return names


def answer_place_names_from_state(state: ItineraryState) -> list[str]:
    """Return retrieved place names that appear in the final answer text."""
    answer = (state.final_reply or "").lower()
    if not answer:
        return []
    return [name for name in retrieved_place_names_from_state(state) if name.lower() in answer]


def actual_eval_result(state: ItineraryState) -> ActualEvalResult:
    """Build the actual structured-output block for the eval report."""
    committed_names = [stop.name for stop in state.stops]
    answer_names = answer_place_names_from_state(state)
    result_count = len(committed_names) if committed_names else len(answer_names)
    return ActualEvalResult(
        result_count=result_count,
        committed_stop_count=len(state.stops),
        place_ids=[stop.place_id for stop in state.stops],
        place_names=committed_names if committed_names else answer_names,
        sources=[stop.source for stop in state.stops],
        answer_place_names=answer_names,
    )


def expected_results_label(expected: ExpectedResults | None) -> str:
    """Format the expected result range for eval-only system context."""
    if expected is None:
        return "not specified"
    return f"{expected.min_stops} to {expected.max_stops} results"


def score_expected_results(case: EvalQuery, actual: ActualEvalResult) -> bool | None:
    """Return whether the produced stop count is inside the expected range."""
    expected = case.expected_results
    if expected is None:
        return None
    return expected.min_stops <= actual.result_count <= expected.max_stops


def score_checks(state: ItineraryState) -> dict[str, CheckResult]:
    """Run deterministic W3 itinerary checks and capture per-check errors."""
    results: dict[str, CheckResult] = {}
    for name, check in DETERMINISTIC_CHECKS.items():
        threshold = CRITIQUE_THRESHOLDS[name]
        try:
            score = float(check(state))
        except Exception as exc:  # noqa: BLE001
            results[name] = CheckResult(
                score=None,
                threshold=threshold,
                passed=False,
                error=str(exc),
            )
            continue
        results[name] = CheckResult(
            score=score,
            threshold=threshold,
            passed=score >= threshold,
        )
    return results


def violations_from_checks(checks: dict[str, CheckResult]) -> list[str]:
    """Return check names that failed or could not be scored."""
    return [name for name, result in checks.items() if not result.passed]


def violations_for_case(
    expected_results_met: bool | None,
    checks: dict[str, CheckResult],
) -> list[str]:
    """Return all eval violations for a case, including non-check expectations."""
    violations = violations_from_checks(checks)
    if expected_results_met is False:
        violations.append("expected_results")
    return violations


def query_result_from_state(case: EvalQuery, state: ItineraryState) -> QueryEvalResult:
    """Build the per-query report row from a final ItineraryState."""
    checks = score_checks(state)
    actual = actual_eval_result(state)
    expected_results_met = score_expected_results(case, actual)
    tool_errors = tool_errors_from_state(state)
    return QueryEvalResult(
        id=case.id,
        question=case.query,
        answer=state.final_reply or "",
        contexts=contexts_from_state(state),
        reference=case.reference,
        tags=case.tags,
        expected=expected_eval_result(case),
        actual=actual,
        deterministic=DeterministicEvalResult(
            expected_results_met=expected_results_met,
            checks=checks,
            violations=violations_for_case(expected_results_met, checks),
            tool_errors=tool_errors,
            first_tool_error=tool_errors[0] if tool_errors else None,
            tool_calls=count_tool_calls(state),
            tool_names=tool_names_from_state(state),
            revision_hints=len(state.revision_hints),
            revision_reasons=revision_reasons_from_state(state),
        ),
        final_reply=state.final_reply or "",
    )


async def evaluate_case(graph: Any, case: EvalQuery) -> QueryEvalResult:
    """Run the agent graph for one eval case and score the final state."""
    open_at = case.expected_constraints.open_at_iso
    eval_context = EVAL_CONTEXT_TEMPLATE.format(
        open_at=open_at.isoformat() if open_at is not None else "not specified",
        expected_results=expected_results_label(case.expected_results),
    )
    raw = await graph.ainvoke(
        ItineraryState(
            messages=[
                SystemMessage(content=eval_context),
                HumanMessage(content=case.query),
            ]
        )
    )
    state = state_from_graph_output(raw)
    return query_result_from_state(case, state)


async def evaluate_cases(
    cases: Sequence[EvalQuery],
    llm: BaseChatModel,
    max_steps: int,
) -> list[QueryEvalResult]:
    """Run the deterministic eval suite sequentially against one candidate LLM."""
    graph = build_agent_graph(llm, max_steps=max_steps)
    results: list[QueryEvalResult] = []
    for case in cases:
        results.append(await evaluate_case(graph, case))
    return results


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty input."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def rate(count: int, total: int) -> float:
    """Return a stable rate for aggregate eval metrics."""
    if total == 0:
        return 0.0
    return count / total


def answer_retrieved_place_coverage(result: QueryEvalResult) -> float | None:
    """Return how many produced place names are visibly grounded in retrieved results."""
    if result.actual.result_count == 0:
        return None
    return min(len(result.actual.answer_place_names) / result.actual.result_count, 1.0)


def aggregate_results(results: Sequence[QueryEvalResult]) -> dict[str, float | int]:
    """Aggregate per-query deterministic eval results into flat metrics."""
    query_count = len(results)
    queries_with_violations = sum(1 for result in results if result.deterministic.violations)
    expected_results_mismatch_count = sum(
        1 for result in results if result.deterministic.expected_results_met is False
    )
    queries_with_tool_errors = sum(1 for result in results if result.deterministic.tool_errors)
    answer_coverage_scores = [
        score
        for result in results
        if (score := answer_retrieved_place_coverage(result)) is not None
    ]
    aggregate: dict[str, float | int] = {
        "query_count": query_count,
        "queries_with_violations": queries_with_violations,
        "deterministic_pass_rate": 1.0 - rate(queries_with_violations, query_count),
        "deterministic_violation_rate": rate(queries_with_violations, query_count),
        "expected_results_mismatch_count": expected_results_mismatch_count,
        "expected_results_mismatch_rate": rate(expected_results_mismatch_count, query_count),
        "tool_error_count": sum(len(result.deterministic.tool_errors) for result in results),
        "queries_with_tool_errors": queries_with_tool_errors,
        "tool_error_rate": rate(queries_with_tool_errors, query_count),
        "tool_success_rate": 1.0 - rate(queries_with_tool_errors, query_count),
        "check_error_count": sum(
            1
            for result in results
            for check in result.deterministic.checks.values()
            if check.error is not None
        ),
        "expected_results_match_rate": mean(
            [
                1.0 if result.deterministic.expected_results_met else 0.0
                for result in results
                if result.deterministic.expected_results_met is not None
            ]
        ),
        "results_mean": mean([float(result.actual.result_count) for result in results]),
        "committed_stops_mean": mean(
            [float(result.actual.committed_stop_count) for result in results]
        ),
        "committed_itinerary_rate": mean(
            [1.0 if result.actual.committed_stop_count > 0 else 0.0 for result in results]
        ),
        "contexts_mean": mean([float(len(result.contexts)) for result in results]),
        "context_presence_rate": mean([1.0 if result.contexts else 0.0 for result in results]),
        "answer_retrieved_place_coverage_mean": mean(answer_coverage_scores),
        "answer_retrieved_place_coverage_count": len(answer_coverage_scores),
        "tool_calls_mean": mean([float(result.deterministic.tool_calls) for result in results]),
        "revision_hints_mean": mean(
            [float(result.deterministic.revision_hints) for result in results]
        ),
    }
    for name in DETERMINISTIC_CHECKS:
        scores: list[float] = []
        for result in results:
            score = result.deterministic.checks[name].score
            if score is not None:
                scores.append(score)
        aggregate[f"{name}_mean"] = mean(scores)
    return aggregate


def report_to_dict(report: EvalRunReport) -> dict[str, Any]:
    """Convert a dataclass report to plain JSON-serializable containers."""
    return asdict(report)


def report_has_errors(report: EvalRunReport) -> bool:
    """Return True when any deterministic check raised an exception."""
    return int(report.aggregate.get("check_error_count", 0)) > 0


def report_has_violations(report: EvalRunReport) -> bool:
    """Return True when any eval case failed an expected behavior."""
    return int(report.aggregate.get("queries_with_violations", 0)) > 0


async def build_report(args: argparse.Namespace) -> EvalRunReport:
    """Load eval cases, run the candidate agent, and build the JSON report."""
    validate_args(args)
    provider = args.llm_provider
    chat_model = resolve_chat_model(provider, args.chat_model)
    eval_config = load_eval_queries(args.eval_queries)
    cases = selected_cases(eval_config.hand_written, args.max_queries)
    llm = build_eval_llm(provider, chat_model, args.temperature)
    results = await evaluate_cases(cases, llm, max_steps=args.max_steps)
    return EvalRunReport(
        eval_queries_path=str(args.eval_queries),
        llm_provider=provider,
        chat_model=chat_model,
        query_count=len(results),
        aggregate=aggregate_results(results),
        queries=results,
    )


def write_report(path: str | Path, report: EvalRunReport) -> None:
    """Write the JSON report to disk with stable formatting."""
    output_path = Path(path)
    output_path.write_text(
        json.dumps(report_to_dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run deterministic evals from the command line."""
    args = parse_args(argv)
    try:
        report = asyncio.run(build_report(args))
    except Exception as exc:  # noqa: BLE001
        print(f"eval_agent failed: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report_to_dict(report), indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        write_report(args.output, report)
    return 1 if report_has_errors(report) or report_has_violations(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
