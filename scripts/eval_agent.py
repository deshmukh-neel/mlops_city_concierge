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
from app.eval.config import DEFAULT_EVAL_QUERIES_PATH, EvalQuery, load_eval_queries  # noqa: E402

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
- If the expected stop count is 1, answer with exactly one stop and do not ask
  how many stops the user wants.
- These instructions provide deterministic eval context; the user's query below
  is the behavior being evaluated.

Expected open time: {open_at}
Expected stop count: {expected_stops}
"""


@dataclass
class CheckResult:
    """A single deterministic check result for one eval case."""

    score: float | None
    threshold: float
    passed: bool
    error: str | None = None


@dataclass
class QueryEvalResult:
    """Per-query deterministic eval output."""

    case_id: str
    query: str
    tags: list[str]
    expected_stops: int | None
    stops_count: int
    expected_stops_met: bool | None
    checks: dict[str, CheckResult]
    violations: list[str]
    tool_errors: list[str]
    tool_calls: int
    revision_hints: int
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


def score_expected_stops(case: EvalQuery, state: ItineraryState) -> bool | None:
    """Return whether the produced stop count matched the case expectation."""
    if case.expected_stops is None:
        return None
    return len(state.stops) == case.expected_stops


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
    expected_stops_met: bool | None,
    checks: dict[str, CheckResult],
) -> list[str]:
    """Return all eval violations for a case, including non-check expectations."""
    violations = violations_from_checks(checks)
    if expected_stops_met is False:
        violations.append("expected_stops")
    return violations


def query_result_from_state(case: EvalQuery, state: ItineraryState) -> QueryEvalResult:
    """Build the per-query report row from a final ItineraryState."""
    checks = score_checks(state)
    expected_stops_met = score_expected_stops(case, state)
    tool_errors = tool_errors_from_state(state)
    return QueryEvalResult(
        case_id=case.id,
        query=case.query,
        tags=case.tags,
        expected_stops=case.expected_stops,
        stops_count=len(state.stops),
        expected_stops_met=expected_stops_met,
        checks=checks,
        violations=violations_for_case(expected_stops_met, checks),
        tool_errors=tool_errors,
        tool_calls=count_tool_calls(state),
        revision_hints=len(state.revision_hints),
        final_reply=state.final_reply or "",
    )


async def evaluate_case(graph: Any, case: EvalQuery) -> QueryEvalResult:
    """Run the agent graph for one eval case and score the final state."""
    open_at = case.expected_constraints.open_at_iso
    eval_context = EVAL_CONTEXT_TEMPLATE.format(
        open_at=open_at.isoformat() if open_at is not None else "not specified",
        expected_stops=case.expected_stops if case.expected_stops is not None else "not specified",
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


def aggregate_results(results: Sequence[QueryEvalResult]) -> dict[str, float | int]:
    """Aggregate per-query deterministic eval results into flat metrics."""
    query_count = len(results)
    aggregate: dict[str, float | int] = {
        "query_count": query_count,
        "queries_with_violations": sum(1 for result in results if result.violations),
        "expected_stops_mismatch_count": sum(
            1 for result in results if result.expected_stops_met is False
        ),
        "tool_error_count": sum(len(result.tool_errors) for result in results),
        "check_error_count": sum(
            1
            for result in results
            for check in result.checks.values()
            if check.error is not None
        ),
        "expected_stops_match_rate": mean(
            [
                1.0 if result.expected_stops_met else 0.0
                for result in results
                if result.expected_stops_met is not None
            ]
        ),
        "stops_mean": mean([float(result.stops_count) for result in results]),
        "tool_calls_mean": mean([float(result.tool_calls) for result in results]),
        "revision_hints_mean": mean([float(result.revision_hints) for result in results]),
    }
    for name in DETERMINISTIC_CHECKS:
        scores: list[float] = []
        for result in results:
            score = result.checks[name].score
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
