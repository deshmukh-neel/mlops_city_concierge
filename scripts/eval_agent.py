#!/usr/bin/env python3
"""Run deterministic offline evals for the City Concierge agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import types
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
    refinement_minimal_edit,
    temporal_coherence,
    walking_budget_respected,
)
from app.agent.graph import build_agent_graph
from app.agent.input_parsing import explicit_num_stops_from_text
from app.agent.io import build_refinement_prompt_message, messages_from_history
from app.agent.revision import LOW_SIMILARITY_THRESHOLD  # D-12-04: import, never hardcode 0.55
from app.agent.state import ItineraryState, Stop, UserConstraints
from app.config import get_settings
from app.eval.config import (
    DEFAULT_EVAL_QUERIES_PATH,
    EvalQuery,
    ExpectedResults,
    load_eval_queries,
)

_log = logging.getLogger(__name__)

# Keep this Literal in sync with `app.llm_factory.SUPPORTED_PROVIDERS`. Drift
# between the two has bitten this script twice (argparse `choices` in 09-03 /
# `resolve_chat_model` env_var dict in 09 / CR-01 + IN-01). The runtime
# enforcement comes from argparse `choices=list(SUPPORTED_PROVIDERS)`; this
# alias exists so static type-checkers see the same canonical set.
LlmProvider = Literal["openai", "gemini", "deepseek", "kimi", "anthropic", "scripted"]
# D-11-03 / CR-01: checks may return None to ABSTAIN (no signal — e.g. the
# zero-stop category_compliance abstain). score_checks treats None as
# "excluded from aggregation", never as a scorer error or a violation.
CheckFunction = Callable[[ItineraryState], float | None]

DETERMINISTIC_CHECKS: dict[str, CheckFunction] = {
    "category_compliance": category_compliance,
    "category_compliance_strict": category_compliance_strict,
    "constraints_satisfied": constraints_satisfied,
    "geographic_coherence": geographic_coherence,
    "no_hallucinated_place_ids": no_hallucinated_place_ids,
    "rationale_stop_alignment": rationale_stop_alignment,
    "refinement_minimal_edit": refinement_minimal_edit,
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
    # INST-01 / D-12-03: primary commit-step signal (objective, cross-provider-comparable)
    first_commit_call_step: (
        int | None
    )  # step index of first commit_itinerary tool call; null if never
    # INST-01 / D-12-03: secondary heuristic (visible reasoning text); null when reasoning is opaque
    first_commit_mention_step: int | None  # null for encrypted reasoning, Gemini signatures
    # INST-02 / D-12-04: per-step viable-candidate counts (one int per plan step, non-cumulative)
    viable_candidates_per_step: list[int]
    # INST-03 / D-12-05: per-step boolean — every stop had >=1 viable candidate cumulatively
    rule8_met_per_step: list[bool]
    rule8_met_but_kept_searching_steps: list[int]  # step indices where met but model kept searching
    # INST-04 / D-12-01: raw per-step telemetry forwarded from ItineraryState verbatim
    step_telemetry: list[dict[str, Any]]
    # D-12-04: threshold value used for viability judgments (self-describing for Phase 13 diffs)
    viability_threshold: float


@dataclass
class QueryEvalResult:
    """RAGAS-compatible per-query eval output plus deterministic diagnostics.

    The `status` field is the discriminator used by aggregate_results:
      - "ok"    — completed run; scored fields are populated.
      - "error" — infra/config exception; scorers were NOT invoked.
                  `error` dict carries {stage, type, message} per D-10-01.
    """

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
    # D-10-01: status discriminator — default "ok" so all pre-existing
    # scored rows remain status="ok" without any callsite change.
    status: str = "ok"
    # D-10-01: populated only on status="error" runs; None on scored runs.
    error: dict[str, str] | None = None


@dataclass
class EvalRunReport:
    """Serializable report for one offline eval run."""

    eval_queries_path: str
    llm_provider: str
    chat_model: str
    query_count: int
    aggregate: dict[str, float | int]
    queries: list[QueryEvalResult]


def make_error_record(case: EvalQuery, stage: str, exc: BaseException) -> QueryEvalResult:
    """Build a D-10-01-shaped error record for one failed eval run.

    Returns a QueryEvalResult with status="error" and an error dict carrying
    {stage, type, message}. Scorers are NEVER invoked — all check scores are
    None and the deterministic block is empty. Serializes cleanly via asdict().

    Args:
        case:  The eval case that was running when the exception occurred.
        stage: One of {"setup", "turn0", "turnN"} per D-10-01.
        exc:   The exception that caused the run to fail.
    """
    error_dict: dict[str, str] = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc)[:500],
    }
    # All check entries carry score=None so the aggregate filter (status=="ok")
    # correctly skips this record — no scorer output leaks into means.
    empty_checks: dict[str, CheckResult] = {
        name: CheckResult(score=None, threshold=0.0, passed=False) for name in DETERMINISTIC_CHECKS
    }
    return QueryEvalResult(
        id=case.id,
        question=case.query,
        answer="",
        contexts=[],
        reference=case.reference,
        tags=case.tags,
        expected=ExpectedEvalResult(
            min_stops=case.expected_results.min_stops if case.expected_results else None,
            max_stops=case.expected_results.max_stops if case.expected_results else None,
            expects_clarification_or_relaxation=False,
        ),
        actual=ActualEvalResult(
            result_count=0,
            committed_stop_count=0,
            place_ids=[],
            place_names=[],
            sources=[],
            answer_place_names=[],
        ),
        deterministic=DeterministicEvalResult(
            expected_results_met=None,
            checks=empty_checks,
            violations=[],
            tool_errors=[],
            first_tool_error=None,
            tool_calls=0,
            tool_names=[],
            revision_hints=0,
            revision_reasons=[],
            # INST safe defaults for error records (D-12-03/04/05)
            first_commit_call_step=None,
            first_commit_mention_step=None,
            viable_candidates_per_step=[],
            rule8_met_per_step=[],
            rule8_met_but_kept_searching_steps=[],
            step_telemetry=[],
            viability_threshold=LOW_SIMILARITY_THRESHOLD,
        ),
        final_reply="",
        latency_seconds=0.0,
        status="error",
        error=error_dict,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the deterministic eval runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-queries",
        default=str(DEFAULT_EVAL_QUERIES_PATH),
        help="YAML eval query config to run.",
    )
    # Drive --llm-provider choices from the factory's SUPPORTED_PROVIDERS tuple
    # so new providers added there (PROV-03 added "anthropic"; PROV-04 will add
    # nothing — gemini was already supported) auto-propagate to the eval runner
    # without a second edit-site. Previously this list was hardcoded and PROV-03
    # discovered the drift at the matrix-runner level (anthropic cells all
    # failed with `invalid choice: 'anthropic'`).
    from app.llm_factory import SUPPORTED_PROVIDERS

    parser.add_argument(
        "--llm-provider",
        choices=list(SUPPORTED_PROVIDERS),
        default="openai",
        help=(
            "Candidate LLM provider to evaluate. Choices come from "
            "app.llm_factory.SUPPORTED_PROVIDERS. 'scripted' is the CI-safe "
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
    # PROV-03 (Phase 9) added "anthropic" to SUPPORTED_PROVIDERS. The previous
    # hardcoded {"deepseek": ..., "kimi": ...} dict raised KeyError on every
    # other provider — masked in CI because the matrix runner always passes
    # `--chat-model claude-sonnet-4-6` explicitly. A direct invocation
    # `python scripts/eval_agent.py --llm-provider anthropic` (the pattern used
    # by the 09-03 live probes) crashed with a non-actionable traceback.
    # Use `.get()` so unknown providers fall through to the user-friendly
    # ValueError on the next line rather than KeyError.
    env_var_map = {
        "deepseek": "DEEPSEEK_MODEL",
        "kimi": "MOONSHOT_MODEL",
        "anthropic": "ANTHROPIC_MODEL",
    }
    env_var = env_var_map.get(provider)
    if env_var is None:
        raise ValueError(f"No chat model resolver for {provider}: pass --chat-model explicitly")
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


# WR-08 / D-11-05: prod-threading injects these scratch keys as conversation-state
# serialization helpers — they are NOT tool invocations. Exclude them from all
# tool-call counting and tool-name listing so they don't inflate tool_calls_mean
# in committed baselines.
_NON_TOOL_SCRATCH_KEYS = frozenset({"prior_committed_stops", "prior_stops_obj"})


def count_tool_calls(state: ItineraryState) -> int:
    """Count captured tool calls from the agent scratchpad."""
    return sum(
        len(entries)
        for key, entries in state.scratch.items()
        if isinstance(entries, list) and key not in _NON_TOOL_SCRATCH_KEYS
    )


def tool_names_from_state(state: ItineraryState) -> list[str]:
    """Return tool names that appear in scratchpad insertion order."""
    names: list[str] = []
    for tool_name, entries in state.scratch.items():
        if isinstance(entries, list) and entries and tool_name not in _NON_TOOL_SCRATCH_KEYS:
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


def first_commit_call_step_from_state(state: ItineraryState) -> int | None:
    """Return the step index of the first commit_itinerary tool call, or None.

    INST-01 / D-12-03: primary commit-step metric. Reads
    ``state.scratch['commit_itinerary']`` entries (same key used by
    ``count_tool_calls`` / ``tool_names_from_state``). Each entry is a dict
    with a ``"step"`` key written by ``act()`` in graph.py at
    ``state.step_count`` (the pre-increment counter). Returns the minimum step
    index across all commit entries so it captures the FIRST commit call even
    when the model commits more than once. Returns None if no commit entry
    exists or if the scratch value is malformed.

    STEP-INDEX CONTRACT: commit_itinerary scratch entries are keyed by
    ``state.step_count`` (pre-increment), matching the step_telemetry and
    per-step viable-candidate index conventions from Plan 12-01.
    """
    entries = state.scratch.get("commit_itinerary")
    if not isinstance(entries, list) or not entries:
        return None
    steps = [e["step"] for e in entries if isinstance(e, dict) and "step" in e]
    return min(steps) if steps else None


def viable_candidates_per_step_from_state(
    state: ItineraryState,
    viability_threshold: float,
    requested_types: list[str],
) -> list[int]:
    """Return per-step viable-candidate counts (INST-02 / D-12-04).

    Returns a list of int, one element per plan step (indexed by step). Each
    element is the count of viable candidates found in scratch entries for that
    step. This list is PER-STEP (non-cumulative): element i counts only the
    hits from scratch entries whose ``"step"`` key equals i, NOT a running
    total. Cumulative accumulation is performed internally by
    ``rule8_met_per_step_from_state``.

    Viable = ``value_from_hit(hit, "similarity") >= viability_threshold`` AND
    ``value_from_hit(hit, "primary_type") in requested_types`` (or
    ``requested_types`` is empty — no type constraint, count on cosine alone).

    Reads ``semantic_search`` and ``nearby`` scratch entries grouped by their
    ``"step"`` key. Guards all reads with isinstance checks to handle malformed
    or empty state gracefully.

    STEP-INDEX CONTRACT: scratch entries use ``state.step_count`` (pre-increment)
    as the ``"step"`` key, matching step_telemetry from Plan 12-01.
    """
    # Build a map: step_index -> list[hits]
    step_hits: dict[int, list[Any]] = {}
    for tool_name in ("semantic_search", "nearby"):
        entries = state.scratch.get(tool_name)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            step = entry.get("step")
            if not isinstance(step, int):
                continue
            result = entry.get("result")
            if not isinstance(result, list):
                continue
            step_hits.setdefault(step, []).extend(result)

    if not step_hits:
        return []

    max_step = max(step_hits.keys())
    counts: list[int] = []
    for i in range(max_step + 1):
        hits = step_hits.get(i, [])
        viable = 0
        for hit in hits:
            sim = value_from_hit(hit, "similarity")
            if not isinstance(sim, (int, float)) or isinstance(sim, bool):
                continue
            if sim < viability_threshold:
                continue
            if requested_types:
                ptype = value_from_hit(hit, "primary_type")
                if ptype not in requested_types:
                    continue
            viable += 1
        counts.append(viable)
    return counts


def rule8_met_per_step_from_state(
    state: ItineraryState,
    viable_per_step: list[int],
    requested_types: list[str],
) -> list[bool]:
    """Return per-step boolean: did every requested stop have >=1 viable candidate cumulatively.

    INST-03 / D-12-05. A step where rule8_met=True but the model kept
    searching (no commit) is a decisiveness gap.

    CONTRACT: this helper performs CUMULATIVE accumulation internally. Element i
    is True iff, cumulatively across steps 0..i (inclusive), every requested
    type in ``requested_types`` has at least one viable candidate. Because
    per-type coverage cannot be reconstructed from the flat ``viable_per_step``
    int list, this helper re-reads the ``semantic_search``/``nearby`` scratch
    entries to accumulate the SET of covered ``primary_type``s.

    When ``requested_types`` is empty, falls back to the
    ``viable_per_step`` argument:
      - If ``state.constraints.num_stops`` is set: True iff
        ``sum(viable_per_step[0..i]) >= num_stops``.
      - Otherwise: True iff ``sum(viable_per_step[0..i]) >= 1``
        (at least one viable candidate exists).

    Guards every scratch read with isinstance checks (malformed/empty state
    returns an all-False list of the appropriate length).

    STEP-INDEX CONTRACT: scratch entries use ``state.step_count`` (pre-increment)
    as the ``"step"`` key, matching step_telemetry from Plan 12-01.
    """
    if not requested_types:
        # Fallback: use the cumulative viable_per_step count
        target = state.constraints.num_stops if state.constraints.num_stops is not None else 1
        result: list[bool] = []
        cumulative = 0
        for count in viable_per_step:
            cumulative += count
            result.append(cumulative >= target)
        return result

    # Build step -> hits map (same as viable_candidates_per_step_from_state)
    step_hits: dict[int, list[Any]] = {}
    viability_threshold = LOW_SIMILARITY_THRESHOLD
    for tool_name in ("semantic_search", "nearby"):
        entries = state.scratch.get(tool_name)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            step = entry.get("step")
            if not isinstance(step, int):
                continue
            result_list = entry.get("result")
            if not isinstance(result_list, list):
                continue
            step_hits.setdefault(step, []).extend(result_list)

    if not step_hits:
        # No scratch entries — return all-False of same length as viable_per_step
        return [False] * len(viable_per_step)

    max_step = max(step_hits.keys())
    covered_types: set[str] = set()
    bools: list[bool] = []
    for i in range(max_step + 1):
        hits = step_hits.get(i, [])
        for hit in hits:
            sim = value_from_hit(hit, "similarity")
            if not isinstance(sim, (int, float)) or isinstance(sim, bool):
                continue
            if sim < viability_threshold:
                continue
            ptype = value_from_hit(hit, "primary_type")
            if isinstance(ptype, str) and ptype in requested_types:
                covered_types.add(ptype)
        bools.append(covered_types >= set(requested_types))
    return bools


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
            raw = check(state)
            score = float(raw) if raw is not None else None
        except Exception as exc:  # noqa: BLE001
            results[name] = CheckResult(
                score=None,
                threshold=threshold,
                passed=False,
                error=str(exc),
            )
            continue
        if score is None:
            # D-11-03 / CR-01 abstain: the check carries no signal for this
            # run (e.g. zero-stop category_compliance). Not an error
            # (error=None keeps it out of check_error_count, so it can never
            # force exit 2) and not a violation (passed=True keeps it out of
            # violations_from_checks). The None score is excluded from
            # aggregate means downstream.
            results[name] = CheckResult(score=None, threshold=threshold, passed=True)
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
    # D-12-04: import threshold constant — never hardcode 0.55
    threshold = LOW_SIMILARITY_THRESHOLD
    requested_types = list(state.constraints.requested_primary_types)
    viable_per_step = viable_candidates_per_step_from_state(state, threshold, requested_types)
    rule8_per_step = rule8_met_per_step_from_state(state, viable_per_step, requested_types)
    rule8_met_steps = [i for i, met in enumerate(rule8_per_step) if met]
    # Steps where rule8 was met but the model did NOT commit (the decisiveness gap)
    commit_steps = {
        e["step"]
        for e in state.scratch.get("commit_itinerary", [])
        if isinstance(e, dict) and "step" in e
    }
    rule8_kept_searching = [s for s in rule8_met_steps if s not in commit_steps]
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
            # INST fields (D-12-03 to D-12-05)
            first_commit_call_step=first_commit_call_step_from_state(state),
            first_commit_mention_step=None,  # opaque by default; set when visible (D-12-03)
            viable_candidates_per_step=viable_per_step,
            rule8_met_per_step=rule8_per_step,
            rule8_met_but_kept_searching_steps=rule8_kept_searching,
            step_telemetry=list(state.step_telemetry),  # forwarded verbatim (INST-04 / D-12-02)
            viability_threshold=threshold,
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
    """Build deterministic eval constraints from checked-in YAML metadata.

    Phase 6 root-cause fix for D-06-09 turn-0 failure: mirror ``/chat``'s
    ``num_stops`` extraction (``app/main.py:781`` calls
    ``explicit_num_stops_from_conversation``). Without this, the eval
    prod-threading branch under-constrains the model relative to ``/chat``
    on identical queries — for "Plan a date night ... 3 stops" the model
    asks a clarifying question ("how many stops?") instead of committing,
    because the SystemMessage that previously suppressed that question is
    dropped by the N-1 fix in plan 06-06.

    Precedence: text extraction first (``"3 stops"`` in the query body),
    then YAML metadata (``expected_results.{min,max}_stops`` when they
    agree). Text extraction mirrors ``/chat`` byte-identically; the YAML
    fallback covers queries where the count is implied by the scenario
    rather than spoken in prose.
    """
    requested_primary_types = list(case.expected_constraints.requested_primary_types)
    num_stops: int | None = explicit_num_stops_from_text(case.query)
    if num_stops is None and case.expected_results is not None:
        min_s = case.expected_results.min_stops
        max_s = case.expected_results.max_stops
        if min_s is not None and max_s is not None and min_s == max_s:
            num_stops = min_s
    return UserConstraints(
        requested_primary_types=requested_primary_types,
        num_stops=num_stops,
    )


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
        except Exception as exc:  # noqa: BLE001 — D-11-06: mirror multi-turn error capture
            return make_error_record(case, "turn0", exc)
    finally:
        latency_seconds = time.monotonic() - start_time
    state = state_from_graph_output(raw)
    return query_result_from_state(case, state, latency_seconds=latency_seconds)


async def evaluate_multi_turn_case(graph: Any, case: EvalQuery) -> QueryEvalResult:
    """Drive a multi-turn eval case against a shared agent graph (EVAL-06).

    Branches on ``case.threading_mode`` (added in plan 06-04 / D-06-05):
      - ``legacy`` (default): the pre-Phase-6 turn loop — byte-identical to
        the existing behavior. Preserves Phase 3/4 baselines for every
        non-Phase-6 multi-turn case.
      - ``prod``: rebuilds messages per turn using ``messages_from_history``
        + the SHARED ``build_refinement_prompt_message`` helper from plan
        06-05, mirroring the production ``/chat`` injection EXACTLY so the
        merge gate measures parity with prod, not eval-only drift.

    Per D-06-06: only the ``refinement_cheaper`` scenario flips to
    ``threading_mode='prod'`` in plan 06-07. ``late_night_closure_cascade``
    stays ``legacy`` per ``project_eval_multi_turn_threading_bug``.

    Turn 1 = ``case.query``; each entry in ``case.turns`` is fed back to the
    same graph instance. The final reported QueryEvalResult is built from
    the LAST turn's state and ``latency_seconds`` is the SUM of per-turn
    latencies.

    Fail-open semantics mirror ``evaluate_cases`` in BOTH branches: if any
    turn raises, the helper records a synthetic ``multi_turn_runner`` entry
    in ``state.scratch`` (surfaced via :func:`tool_errors_from_state`) and
    returns the partial QueryEvalResult instead of crashing the whole run.
    """
    if case.threading_mode == "prod":
        result, _state = await _run_prod_threading(graph, case)
        return result
    eval_context = _eval_context_for(case)
    return await _run_legacy_threading(graph, case, eval_context)


async def _run_legacy_threading(
    graph: Any,
    case: EvalQuery,
    eval_context: str,
) -> QueryEvalResult:
    """Pre-Phase-6 legacy turn loop — byte-identical to the prior shape.

    Injects ``SystemMessage(eval_context)`` on every turn per the WR-06
    contract. The N-1 fix from plan 06-06 does NOT apply to this branch;
    only the prod branch drops the eval-context SystemMessage.
    """
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
            # D-10-02: exceptions never reach scorers. Return an ERROR-status
            # record instead of building a partial_state and scoring it.
            # Stage is "turn0" for index 0, "turnN" for any subsequent turn.
            total_latency += time.monotonic() - start_time
            stage = "turn0" if index == 0 else "turnN"
            return make_error_record(case, stage, exc)
        total_latency += time.monotonic() - start_time
        state = state_from_graph_output(raw)
    assert state is not None
    return query_result_from_state(case, state, latency_seconds=total_latency)


# fmt: off
async def _run_prod_threading(graph: Any, case: EvalQuery) -> tuple[QueryEvalResult, ItineraryState]:  # noqa: E501
    # fmt: on
    """Phase 6 plan 06-06 — prod-threading branch.

    Mirrors the production ``/chat`` invocation shape EXACTLY so the merge
    gate (``refinement_minimal_edit``) measures byte-identical behavior
    between production and eval. Key invariants:

    - **N-1 fix**: does NOT inject ``SystemMessage(eval_context)``. The
      ``graph.plan()`` node prepends ``SYSTEM_PROMPT`` naturally on first
      invocation per ``app/agent/graph.py:264`` when no ``SystemMessage``
      is present, matching ``/chat``.
    - **N-2 fix**: ALWAYS sets ``state.scratch['refinement_context'] = True``
      after turn 0, regardless of whether turn 0 commits any stops. This
      lets plan 06-03 Branch 2 fail-loud surface the false-pass path.
    - **HIGH-3 + Caveat #5**: turn N+1 ordering matches ``/chat``:
      ``[*messages_from_history(synthesized), build_refinement_prompt_message(prior), HumanMessage(turn_text)]``.
      The structured plan sits IMMEDIATELY before the user's turn-2 message.
    - **NEW HIGH-B**: ``REFINEMENT_STRUCTURED_PLAN_ENABLED`` env var is
      read INSIDE this function per OVR-05 / D-06-10 so a per-cell matrix
      override flips behavior on the next request. Injection is gated on
      the flag; scratch keys are written unconditionally so 06-03's regime
      detection still works when the flag is off.
    - **NEW HIGH-C**: helper call is gated on ``if prior_committed_stops:``
      so an empty turn-0 commit does NOT crash via the helper's strict
      raise-on-empty contract. Scratch keys still written.
    - **Caveat #6 (re-stamp)**: ``prior_scratch`` carrying all THREE keys
      (``refinement_context``, ``prior_committed_stops``,
      ``refinement_target_slot``) is re-applied after every fresh
      ``state_from_graph_output(raw)`` reset.
    - **MEDIUM (history shape)**: synthesized history includes BOTH the
      prior turn's user text AND the prior turn's assistant text (via
      ``state.final_reply``), paired as user/assistant entries — matches
      ``/chat``'s ``req.history`` shape, not just assistant-only.
    - **INFO-1**: uses ``types.SimpleNamespace`` to feed
      ``messages_from_history`` rather than importing ``app.main.ChatMessage``
      (forbidden cross-layer import).

    Returns BOTH the public ``QueryEvalResult`` AND the FINAL
    ``ItineraryState`` so unit tests can inspect ``state.scratch`` directly.
    The public ``evaluate_multi_turn_case`` discards the state.
    """
    if case.expected_refinement is None:
        raise ValueError(
            f"threading_mode='prod' on case {case.id} requires expected_refinement.target_slot"
        )

    all_turns: list[str] = [case.query, *(case.turns or [])]
    state: ItineraryState | None = None
    total_latency = 0.0
    # prior_scratch carries the THREE Phase-6 scratch keys across the
    # state_from_graph_output(raw) reset that happens after every turn.
    prior_scratch: dict[str, Any] = {}
    # Synthesized text history fed to messages_from_history for turn N >= 1.
    # Each entry is a SimpleNamespace with .role and .content — the duck
    # type required by `app.agent.io.messages_from_history`.
    synthesized_history: list[Any] = []

    for index, turn_text in enumerate(all_turns):
        if index == 0:
            # N-1 fix: NO SystemMessage(eval_context). graph.plan() will
            # prepend SYSTEM_PROMPT naturally on the first invocation.
            messages_in: list[Any] = [HumanMessage(content=turn_text)]
        else:
            # NEW HIGH-B: env var read INSIDE the function per OVR-05.
            # The default ("") matches the /chat injection guard exactly
            # (see app/main.py:753) so flag-off behavior is identical
            # between /chat and prod-mode eval.
            flag_raw = os.environ.get("REFINEMENT_STRUCTURED_PLAN_ENABLED", "")
            flag_enabled = flag_raw.strip().lower() in {"1", "true", "yes", "on"}

            prior_committed_stops = prior_scratch.get("prior_committed_stops", [])

            # NEW HIGH-C: gate helper call on `flag_enabled AND prior non-empty`.
            # Helper retains its strict raise-on-empty contract; the call site
            # protects against the empty-prior path here so the prod branch
            # runs to completion and 06-03 Branch 2 returns 0.0 on the final
            # state.
            refinement_messages: list[HumanMessage] = []
            if flag_enabled and prior_committed_stops:
                # Reconstruct Stop instances from prior_scratch's dict shape
                # so the helper sees the same Pydantic types /chat passes.
                prior_stops_models: list[Stop] = []
                # prior_scratch stores {slot, place_id} dicts; we need full
                # Stop instances. We carry the original Stop objects via a
                # separate `prior_stops_obj` key written at turn 0.
                for s in prior_scratch.get("prior_stops_obj", []):
                    prior_stops_models.append(s)
                if prior_stops_models:
                    refinement_messages.append(build_refinement_prompt_message(prior_stops_models))

            messages_in = [
                *messages_from_history(synthesized_history),
                *refinement_messages,
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
            # D-10-02: exceptions never reach scorers. Return an ERROR-status
            # record instead of building a partial_state and scoring it.
            # Stage is "turn0" for index 0, "turnN" for any subsequent turn.
            total_latency += time.monotonic() - start_time
            stage = "turn0" if index == 0 else "turnN"
            error_record = make_error_record(case, stage, exc)
            # Return the tuple shape _run_prod_threading always returns; the
            # state sentinel is a fresh ItineraryState (not scored).
            return (error_record, ItineraryState())
        total_latency += time.monotonic() - start_time
        state = state_from_graph_output(raw)

        # Caveat #6 re-stamp pattern: re-apply the prior scratch keys to the
        # FRESH state returned by the graph (graph reset scratch on its way
        # back). Without this, the final-turn state would lose refinement_context.
        if prior_scratch:
            state.scratch.update(prior_scratch)

        # After turn 0 we capture the THREE Phase-6 scratch keys.
        if index == 0:
            target_slot = case.expected_refinement.target_slot
            if state.stops:
                # Happy path — turn 0 committed.
                prior_scratch = {
                    "refinement_context": True,
                    "refinement_target_slot": target_slot,
                    "prior_committed_stops": [
                        {"slot": i + 1, "place_id": s.place_id, "primary_type": s.primary_type}
                        for i, s in enumerate(state.stops)
                    ],
                    # Carry the full Stop objects in a separate key so the
                    # helper (which needs full Stop instances) can be called
                    # without reconstructing them from the dict shape. This
                    # key is internal to _run_prod_threading and is never
                    # surfaced to the scorer (which only reads the three
                    # documented keys).
                    "prior_stops_obj": list(state.stops),
                }
            else:
                # N-2 fix empty-commit branch: still set refinement_context=True
                # so 06-03's Branch 2 returns 0.0 (fail-loud) rather than 1.0
                # (silent abstain).
                _log.warning(
                    "threading_mode=prod turn 0 produced no committed_stops; "
                    "refinement_minimal_edit will score 0.0 via Branch 2 "
                    "fail-loud (refinement_context=True + empty prior). "
                    "Inspect baseline for false-negative diagnosis."
                )
                prior_scratch = {
                    "refinement_context": True,
                    "refinement_target_slot": target_slot,
                    "prior_committed_stops": [],
                    "prior_stops_obj": [],
                }
            # Re-stamp on the current (turn-0 final) state too. Even though
            # turn-0 doesn't *need* it for the scorer (scorer reads the
            # final state), this keeps the invariant uniform across turns.
            state.scratch.update(prior_scratch)

        # MEDIUM (history shape): synthesize BOTH the user and assistant
        # turn for the next turn's messages_from_history call. This matches
        # /chat's req.history shape (user/assistant pairs) and ensures the
        # full-sequence parity test holds.
        synthesized_history.append(types.SimpleNamespace(role="user", content=turn_text))
        synthesized_history.append(
            types.SimpleNamespace(role="assistant", content=state.final_reply or "")
        )

    assert state is not None
    return (
        query_result_from_state(case, state, latency_seconds=total_latency),
        state,
    )


async def evaluate_cases(
    cases: Sequence[EvalQuery],
    llm: BaseChatModel,
    max_steps: int,
    provider: str,
) -> list[QueryEvalResult]:
    """Run the deterministic eval suite sequentially against one candidate LLM."""
    # D-08-16: thread provider for ProviderAdapter dispatch
    graph = build_agent_graph(llm, max_steps=max_steps, provider=provider)
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


def aggregate_results(results: Sequence[QueryEvalResult]) -> dict[str, float | int | list | None]:
    """Aggregate per-query deterministic eval results into flat metrics.

    D-10-03: scorer means are computed ONLY over results with status=="ok".
    Errored runs (status="error") are excluded from means and counted separately
    in n_errored. A cell with any errored run is INVALID_FOR_BASELINE.

    Distinct accounting:
      - n_scored: completed runs contributing to scorer means (status=="ok").
      - n_errored: whole-run infra/config failures (status="error").
      - check_error_count: individual scorer exceptions on COMPLETED runs.
        A completed run with one failing check is still status="ok"; the
        individual check's error is surfaced here, not via n_errored.
    """
    # D-10-03: split results into scored (status=="ok") and errored (status="error").
    scored_results = [r for r in results if r.status == "ok"]
    errored_results = [r for r in results if r.status == "error"]

    query_count = len(results)
    n_scored = len(scored_results)
    n_errored = len(errored_results)

    # Per-run error list for audit trail and eval_matrix summary.json threading.
    errors_list: list[dict[str, str]] = [
        r.error for r in errored_results if r.error is not None
    ]

    # All aggregate statistics below operate on scored_results only.
    queries_with_violations = sum(
        1 for result in scored_results if result.deterministic.violations
    )
    expected_results_mismatch_count = sum(
        1 for result in scored_results if result.deterministic.expected_results_met is False
    )
    queries_with_tool_errors = sum(
        1 for result in scored_results if result.deterministic.tool_errors
    )
    answer_coverage_scores = [
        score
        for result in scored_results
        if (score := answer_retrieved_place_coverage(result)) is not None
    ]
    latencies = [float(result.latency_seconds) for result in scored_results]
    aggregate: dict[str, float | int | list | None] = {
        # D-10-03: cell validity fields (read by eval_matrix summary threading).
        "n_scored": n_scored,
        "n_errored": n_errored,
        "cell_valid": n_errored == 0,
        "errors": errors_list,
        # Standard aggregate fields — computed over scored_results only.
        "query_count": query_count,
        "queries_with_violations": queries_with_violations,
        # WR-09 / D-11-04: guard all derived rates against n_scored == 0.
        # An all-errored cell publishes None, never the fail-open 1.0 that
        # rate(0, 0)==0.0 would produce via the (1.0 - 0.0) expression.
        "deterministic_pass_rate": (
            (1.0 - rate(queries_with_violations, n_scored)) if n_scored > 0 else None
        ),
        "deterministic_violation_rate": (
            rate(queries_with_violations, n_scored) if n_scored > 0 else None
        ),
        "expected_results_mismatch_count": expected_results_mismatch_count,
        "expected_results_mismatch_rate": (
            rate(expected_results_mismatch_count, n_scored) if n_scored > 0 else None
        ),
        "tool_error_count": sum(
            len(result.deterministic.tool_errors) for result in scored_results
        ),
        "queries_with_tool_errors": queries_with_tool_errors,
        "tool_error_rate": rate(queries_with_tool_errors, n_scored) if n_scored > 0 else None,
        "tool_success_rate": (
            (1.0 - rate(queries_with_tool_errors, n_scored)) if n_scored > 0 else None
        ),
        # check_error_count: individual scorer exceptions on COMPLETED runs.
        # Distinct from n_errored (whole-run failures) per D-10-03 / PATTERNS.md.
        "check_error_count": sum(
            1
            for result in scored_results
            for check in result.deterministic.checks.values()
            if check.error is not None
        ),
        "expected_results_match_rate": mean(
            [
                1.0 if result.deterministic.expected_results_met else 0.0
                for result in scored_results
                if result.deterministic.expected_results_met is not None
            ]
        ),
        "results_mean": mean([float(result.actual.result_count) for result in scored_results]),
        "committed_stops_mean": mean(
            [float(result.actual.committed_stop_count) for result in scored_results]
        ),
        # WR-01: committed_itinerary_rate is THE hard-gate metric — it gets the
        # same zero-n guard as the five derived rates above. An all-errored
        # cell must publish None, never the fabricated mean([]) == 0.0 that
        # would read as a hard decisiveness regression of the anchor.
        "committed_itinerary_rate": (
            mean(
                [
                    1.0 if result.actual.committed_stop_count > 0 else 0.0
                    for result in scored_results
                ]
            )
            if n_scored > 0
            else None
        ),
        "contexts_mean": mean([float(len(result.contexts)) for result in scored_results]),
        "context_presence_rate": mean(
            [1.0 if result.contexts else 0.0 for result in scored_results]
        ),
        "answer_retrieved_place_coverage_mean": mean(answer_coverage_scores),
        "answer_retrieved_place_coverage_count": len(answer_coverage_scores),
        "tool_calls_mean": mean(
            [float(result.deterministic.tool_calls) for result in scored_results]
        ),
        "revision_hints_mean": mean(
            [float(result.deterministic.revision_hints) for result in scored_results]
        ),
        "latency_total_seconds": sum(latencies),
        "latency_mean_seconds": mean(latencies),
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
        "latency_max_seconds": max(latencies) if latencies else 0.0,
    }
    # D-10-03: scorer means over scored_results only — errored runs excluded.
    # D-11-03 / CR-01: a check with zero non-None scores in the cell publishes
    # None, never mean([]) == 0.0 — "no signal" must not read as "zero score".
    # eval_matrix._scorer_means_from_cell skips non-numeric values, so a None
    # mean simply drops out of summary.json scorer stats (not-evaluable).
    for name in DETERMINISTIC_CHECKS:
        scores: list[float] = []
        for result in scored_results:
            score = result.deterministic.checks[name].score
            if score is not None:
                scores.append(score)
        aggregate[f"{name}_mean"] = mean(scores) if scores else None
    return aggregate


def report_to_dict(report: EvalRunReport) -> dict[str, Any]:
    """Convert a dataclass report to plain JSON-serializable containers."""
    return asdict(report)


def report_has_errors(report: EvalRunReport) -> bool:
    """Return True when any deterministic check raised an exception OR when
    any whole run failed with an infra/config error (status='error').

    D-10-03: n_errored > 0 means a cell is INVALID_FOR_BASELINE; the matrix
    exit code must be non-zero so operators know a re-run is needed before
    using results as a baseline. check_error_count covers individual scorer
    exceptions on completed runs; n_errored covers whole-run failures.
    """
    return (
        int(report.aggregate.get("check_error_count", 0)) > 0
        or int(report.aggregate.get("n_errored", 0)) > 0
    )


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
    results = await evaluate_cases(cases, llm, max_steps=args.max_steps, provider=provider)
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
    """Run deterministic evals from the command line.

    Exit codes:
        0 = clean — no infra failures, no model-behavior violations
        1 = model-behavior violations (expected in exploratory runs; rerun not needed)
        2 = infra failure (embedding quota, DB down, etc.; rerun needed)
    """
    args = parse_args(argv)
    try:
        report = asyncio.run(build_report(args))
    except Exception as exc:  # noqa: BLE001
        print(f"eval_agent failed: {exc}", file=sys.stderr)
        return 2  # D-11-16: infrastructure failure — was 1 before D-11-16

    rendered = json.dumps(report_to_dict(report), indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        write_report(args.output, report)

    if report_has_errors(report):
        return 2  # infra failure: errored runs need a rerun
    if report_has_violations(report):
        return 1  # model-behavior violations: expected, non-blocking
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
