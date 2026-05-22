#!/usr/bin/env python3
"""Run deterministic offline evals for the City Concierge agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.critique.checks import (
    CRITIQUE_THRESHOLDS,
    category_compliance,
    category_compliance_strict,
    constraints_satisfied,
    geographic_coherence,
    no_hallucinated_place_ids,
    rationale_stop_alignment,
    temporal_coherence,
    walking_budget_respected,
)
from app.agent.graph import build_agent_graph
from app.agent.state import ItineraryState, UserConstraints
from app.config import get_settings
from app.eval.config import (
    DEFAULT_EVAL_QUERIES_PATH,
    EvalQuery,
    ExpectedResults,
    load_eval_queries,
)

LlmProvider = Literal["openai", "gemini", "deepseek", "kimi", "scripted"]
CheckFunction = Callable[[ItineraryState], float]

DETERMINISTIC_CHECKS: dict[str, CheckFunction] = {
    "category_compliance": category_compliance,
    "category_compliance_strict": category_compliance_strict,
    "constraints_satisfied": constraints_satisfied,
    "geographic_coherence": geographic_coherence,
    "no_hallucinated_place_ids": no_hallucinated_place_ids,
    "rationale_stop_alignment": rationale_stop_alignment,
    "temporal_coherence": temporal_coherence,
    "walking_budget_respected": walking_budget_respected,
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
    latency_seconds: float


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
        choices=["openai", "gemini", "deepseek", "kimi", "scripted"],
        default="openai",
        help=(
            "Candidate LLM provider to evaluate. 'scripted' is the CI-safe "
            "deterministic no-network branch (EVAL-09 / P4); it needs no "
            "API key and emits canned messages."
        ),
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
        "--scenario-ids",
        type=_parse_scenario_ids,
        default=None,
        help=(
            "Comma-separated EvalQuery.id list to filter cases (default: run "
            "all hand_written cases). The matrix runner (scripts/eval_matrix.py) "
            "shells out one cell per scenario_id via this flag."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the JSON report. The report is always printed to stdout.",
    )
    return parser.parse_args(argv)


def _parse_scenario_ids(value: str) -> list[str]:
    """Parse the --scenario-ids comma-separated flag into a list of IDs.

    Empty entries and whitespace-only entries are dropped so callers can
    pass ' a , , b ' without surprises. An empty string yields an empty
    list (rather than None); the CLI default of None is honored by argparse
    only when the flag is omitted entirely.
    """
    return [token.strip() for token in value.split(",") if token.strip()]


def resolve_chat_model(provider: LlmProvider, chat_model: str | None) -> str:
    """Resolve the candidate chat model from CLI input or environment settings.

    'scripted' is the no-network deterministic provider (EVAL-09 / P4): it
    never reads env vars or calls get_settings (which can crash without
    OPENAI_API_KEY etc.). The chat_model is purely an informational label
    in the eval report — when omitted, we use a stable sentinel.
    """
    if chat_model and chat_model.strip():
        return chat_model.strip()
    if provider == "scripted":
        # No env-var lookups; chat_model is a label, not a real model name.
        return "scripted-default"
    settings = get_settings()
    if provider == "openai":
        return settings.openai_chat_model
    if provider == "gemini":
        return settings.gemini_chat_model
    env_var = {"deepseek": "DEEPSEEK_MODEL", "kimi": "MOONSHOT_MODEL"}[provider]
    model = os.getenv(env_var)
    if not model:
        raise ValueError(f"No chat model for {provider}: pass --chat-model or set {env_var}")
    return model


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument relationships that argparse cannot express cleanly."""
    if args.max_steps < 1:
        raise ValueError("--max-steps must be greater than zero")
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("--max-queries must be greater than zero when provided")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError(f"--temperature must be in [0.0, 2.0]; got {args.temperature}")


def build_eval_llm(provider: LlmProvider, chat_model: str, temperature: float) -> BaseChatModel:
    """Construct the candidate chat model used to drive the agent graph."""
    from app.llm_factory import build_chat_model

    return build_chat_model(provider, chat_model, temperature=temperature)


def selected_cases(
    cases: list[EvalQuery],
    max_queries: int | None,
    scenario_ids: list[str] | None = None,
) -> list[EvalQuery]:
    """Return the hand-written eval cases selected for this run.

    Filter precedence (Plan 03-05 / EVAL-09):
      1. If `scenario_ids` is provided, drop every case whose `id` is not in
         the list. YAML order is preserved; unknown IDs are silently
         dropped (so the matrix runner sees an empty list and writes a
         clean empty report rather than crashing).
      2. After scenario filtering, `max_queries` slices the head of the
         remaining list.

    Backward compat: omitting `scenario_ids` (the default) leaves the
    pre-03-05 max_queries-only behavior unchanged.
    """
    selected = cases
    if scenario_ids is not None:
        wanted = set(scenario_ids)
        selected = [case for case in selected if case.id in wanted]
    if max_queries is None:
        return selected
    if max_queries < 1:
        raise ValueError("--max-queries must be greater than zero when provided")
    return selected[:max_queries]


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


def query_result_from_state(
    case: EvalQuery,
    state: ItineraryState,
    latency_seconds: float = 0.0,
) -> QueryEvalResult:
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
        latency_seconds=latency_seconds,
    )


def _eval_context_for(case: EvalQuery) -> str:
    """Render the offline-eval SystemMessage body for one case."""
    open_at = case.expected_constraints.open_at_iso
    return EVAL_CONTEXT_TEMPLATE.format(
        open_at=open_at.isoformat() if open_at is not None else "not specified",
        expected_results=expected_results_label(case.expected_results),
    )


def _constraints_for_case(case: EvalQuery) -> UserConstraints:
    """Build deterministic eval constraints from checked-in YAML metadata."""
    requested_primary_types = list(case.expected_constraints.requested_primary_types)
    return UserConstraints(requested_primary_types=requested_primary_types)


async def evaluate_case(graph: Any, case: EvalQuery) -> QueryEvalResult:
    """Run the agent graph for one eval case and score the final state.

    Branches on ``case.turns`` (added in plan 03-02 / EVAL-03): ``None`` or
    ``[]`` runs the case as a single turn exactly as before plan 03-04
    (byte-equivalent JSON for the 29 existing single-turn cases); a
    non-empty list delegates to :func:`evaluate_multi_turn_case` which
    threads ``conversation_state`` across ``len(turns) + 1`` invocations
    (EVAL-06 / P2).
    """
    if case.turns:
        return await evaluate_multi_turn_case(graph, case)
    eval_context = _eval_context_for(case)
    start_time = time.monotonic()
    try:
        raw = await graph.ainvoke(
            ItineraryState(
                messages=[
                    SystemMessage(content=eval_context),
                    HumanMessage(content=case.query),
                ],
                constraints=_constraints_for_case(case),
            )
        )
    finally:
        latency_seconds = time.monotonic() - start_time
    state = state_from_graph_output(raw)
    return query_result_from_state(case, state, latency_seconds=latency_seconds)


async def evaluate_multi_turn_case(graph: Any, case: EvalQuery) -> QueryEvalResult:
    """Drive a multi-turn eval case against a shared agent graph (EVAL-06).

    Turn 1 = ``case.query``; each entry in ``case.turns`` is fed back to the
    same graph instance with the prior turn's message history threaded in,
    mirroring how the frontend round-trips ``conversation_state`` via the
    opaque ``/chat`` payload. The final reported QueryEvalResult is built
    from the LAST turn's state (where every confirmed v2.0 refinement bug
    surfaces — turn 2+, not turn 1) and ``latency_seconds`` is the SUM of
    per-turn latencies.

    Fail-open semantics mirror ``evaluate_cases``: if any turn raises, the
    helper records a synthetic ``multi_turn_runner`` entry in
    ``state.scratch`` (surfaced via :func:`tool_errors_from_state`) and
    returns the partial QueryEvalResult instead of crashing the whole run.
    The first failing turn's prior state is what gets reported.
    """
    eval_context = _eval_context_for(case)
    all_turns: list[str] = [case.query, *(case.turns or [])]
    state: ItineraryState | None = None
    total_latency = 0.0
    for index, turn_text in enumerate(all_turns):
        if index == 0:
            messages_in: list[Any] = [
                SystemMessage(content=eval_context),
                HumanMessage(content=turn_text),
            ]
        else:
            # WR-06: explicitly strip any prior SystemMessage from
            # state.messages and re-inject a fresh eval_context per turn so
            # a future refactor of add_messages (or a state.model_copy(update=...)
            # rewrite) cannot silently drop the eval-only context mid-
            # conversation. The remaining message list still threads through
            # so the frontend's stateless /chat shape is preserved. We build
            # a fresh ItineraryState below so step_count, scratch, and
            # revision_counts reset per turn — only `messages` threads
            # through, as documented in the plan.
            assert state is not None
            messages_in = [
                SystemMessage(content=eval_context),
                *[m for m in state.messages if not isinstance(m, SystemMessage)],
                HumanMessage(content=turn_text),
            ]
        start_time = time.monotonic()
        try:
            raw = await graph.ainvoke(
                ItineraryState(
                    messages=messages_in,
                    constraints=_constraints_for_case(case),
                )
            )
        except Exception as exc:  # noqa: BLE001
            total_latency += time.monotonic() - start_time
            # Surface the failure as a synthetic tool error on whichever
            # state we last have a handle on; do NOT short-circuit the
            # whole eval run (mirror the existing fail-open pattern in
            # evaluate_cases). raw is unbound on this branch, so we report
            # against the prior turn's state (or a fresh one if turn 0
            # raised — first-turn failures still get a JSON row, not a
            # bubbled exception).
            # WR-05: deep-copy the prior turn's state before mutating its
            # scratch dict, so a future debug hook that keeps per-turn state
            # snapshots does not see the synthetic error injected backwards
            # into turn N-1's diagnostics. The error logically belongs to
            # turn N's partial state; the copy makes that explicit.
            partial_state = (
                state.model_copy(deep=True)
                if state is not None
                else ItineraryState(
                    messages=messages_in,
                    constraints=_constraints_for_case(case),
                )
            )
            partial_state.scratch.setdefault("multi_turn_runner", []).append(
                {
                    "args": {"turn_index": index, "turn": turn_text},
                    "result": {"error": f"turn {index} raised: {exc}"},
                    "step": index,
                    "id": f"multi_turn_runner_{index}",
                }
            )
            return query_result_from_state(case, partial_state, latency_seconds=total_latency)
        total_latency += time.monotonic() - start_time
        state = state_from_graph_output(raw)
    assert state is not None
    return query_result_from_state(case, state, latency_seconds=total_latency)


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


def percentile(values: Sequence[float], p: float) -> float:
    """Return the nearest-rank percentile, or 0.0 for empty inputs."""
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    index = max(0, math.ceil((p / 100) * len(ordered)) - 1)
    return ordered[index]


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
    latencies = [float(result.latency_seconds) for result in results]
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
        "latency_total_seconds": sum(latencies),
        "latency_mean_seconds": mean(latencies),
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "latency_max_seconds": max(latencies) if latencies else 0.0,
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
    cases = selected_cases(
        eval_config.hand_written,
        args.max_queries,
        scenario_ids=getattr(args, "scenario_ids", None),
    )
    llm = build_eval_llm(provider, chat_model, args.temperature)
    start_time = time.monotonic()
    results = await evaluate_cases(cases, llm, max_steps=args.max_steps)
    total_runtime_seconds = time.monotonic() - start_time
    aggregate = aggregate_results(results)
    aggregate["total_runtime_seconds"] = total_runtime_seconds
    return EvalRunReport(
        eval_queries_path=str(args.eval_queries),
        llm_provider=provider,
        chat_model=chat_model,
        query_count=len(results),
        aggregate=aggregate,
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
