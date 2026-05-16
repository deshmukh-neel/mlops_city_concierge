#!/usr/bin/env python3
"""Score eval-agent JSON reports with RAGAS metrics."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import mlflow
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings, resolve_llm_api_key

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


JudgeProvider = Literal["openai", "gemini", "anthropic"]
MetricName = Literal[
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

DEFAULT_EXPERIMENT_NAME = "city-concierge-ragas"
DEFAULT_METRICS: tuple[MetricName, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
GEMINI_DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
ANTHROPIC_DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_JUDGE_MAX_TOKENS = 4096
RAGAS_INSTALL_HINT = (
    "RAGAS is required for this scorer. Install it in the active environment with "
    "`poetry add 'ragas>=0.4'` or `pip install 'ragas>=0.4'`."
)
ProgressLogger = Callable[[str], None]


class RagasInputQuery(BaseModel):
    """One RAGAS-ready query row from scripts/eval_agent.py output."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = ""
    contexts: list[str] = Field(default_factory=list)
    reference: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    deterministic: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "question", "answer", "reference", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        """Normalize string fields before validation."""
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("contexts")
    @classmethod
    def clean_contexts(cls, value: list[str]) -> list[str]:
        """Drop empty contexts while preserving retrieval order."""
        return [
            context.strip() for context in value if isinstance(context, str) and context.strip()
        ]

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, value: list[str]) -> list[str]:
        """Drop empty tags while preserving declared order."""
        return [tag.strip() for tag in value if isinstance(tag, str) and tag.strip()]


class RagasInputReport(BaseModel):
    """The eval-agent report shape consumed by the RAGAS scorer."""

    model_config = ConfigDict(extra="allow")

    eval_queries_path: str = ""
    llm_provider: str = ""
    chat_model: str = ""
    query_count: int = Field(ge=0)
    aggregate: dict[str, Any] = Field(default_factory=dict)
    queries: list[RagasInputQuery] = Field(default_factory=list)

    @field_validator("queries")
    @classmethod
    def require_queries(cls, value: list[RagasInputQuery]) -> list[RagasInputQuery]:
        """Reject reports that contain no query rows to score."""
        if not value:
            raise ValueError("report must contain at least one query")
        return value


@dataclass
class RagasQueryScore:
    """RAGAS scores and metric errors for one eval query."""

    id: str
    question: str
    tags: list[str]
    metrics: dict[str, float | None]
    errors: dict[str, str]
    deterministic: dict[str, Any]


@dataclass
class RagasScoreReport:
    """Serializable RAGAS scoring report."""

    source_report_path: str
    eval_queries_path: str
    candidate_llm_provider: str
    candidate_chat_model: str
    judge_provider: str
    judge_model: str
    judge_embedding_model: str | None
    metrics: list[str]
    query_count: int
    aggregate: dict[str, float | int]
    deterministic_aggregate: dict[str, float | int]
    queries: list[RagasQueryScore]


@dataclass
class RagasRuntime:
    """Evaluator LLM and embedding handles used by RAGAS metrics."""

    llm: Any
    embeddings: Any


class JudgeSmokeResponse(BaseModel):
    """Minimal structured response used to validate judge API calls."""

    verdict: Literal["pass"]
    reason: str = Field(min_length=1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for RAGAS report scoring."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a JSON report produced by scripts/eval_agent.py, or '-' for stdin.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the scored RAGAS JSON report.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=list(DEFAULT_METRICS),
        default=None,
        help="RAGAS metric to run. Repeat to choose a subset; defaults to the core set.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["openai", "gemini", "anthropic"],
        default="openai",
        help="Provider used as the RAGAS judge, independent of the candidate report model.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model. Defaults to the configured chat model for the judge provider.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model for embedding-dependent metrics such as answer_relevancy.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--smoke-test-judge",
        action="store_true",
        help="Run one structured judge call and exit without scoring an eval report.",
    )
    args = parser.parse_args(argv)
    if not args.smoke_test_judge and not args.input:
        parser.error("--input is required unless --smoke-test-judge is set")
    return args


def resolve_path(path: str | Path) -> Path:
    """Resolve repo-relative paths without changing absolute paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def read_json_report(path: str | Path) -> dict[str, Any]:
    """Read a JSON report from disk or stdin."""
    raw = sys.stdin.read() if str(path) == "-" else resolve_path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("RAGAS input report must be a JSON object.")
    return data


def load_input_report(path: str | Path) -> RagasInputReport:
    """Load and validate an eval-agent report for RAGAS scoring."""
    return RagasInputReport.model_validate(read_json_report(path))


def selected_metrics(metrics: Sequence[str] | None) -> list[MetricName]:
    """Return the requested metric names in CLI order."""
    if not metrics:
        return list(DEFAULT_METRICS)
    return [metric for metric in metrics if metric in DEFAULT_METRICS]


def resolve_judge_model(provider: JudgeProvider, judge_model: str | None) -> str:
    """Resolve the RAGAS judge model from CLI input or environment settings."""
    if judge_model and judge_model.strip():
        return judge_model.strip()
    settings = get_settings()
    if provider == "openai":
        return settings.openai_chat_model
    if provider == "anthropic":
        return ANTHROPIC_DEFAULT_JUDGE_MODEL
    return settings.gemini_chat_model


def resolve_embedding_model(provider: JudgeProvider, embedding_model: str | None) -> str:
    """Resolve the embedding model used by embedding-dependent RAGAS metrics."""
    if embedding_model and embedding_model.strip():
        return embedding_model.strip()
    if provider in {"openai", "anthropic"}:
        return get_settings().openai_embedding_model
    return GEMINI_DEFAULT_EMBEDDING_MODEL


def build_openai_runtime(
    judge_model: str, embedding_model: str, temperature: float
) -> RagasRuntime:
    """Create RAGAS evaluator handles backed by OpenAI."""
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(RAGAS_INSTALL_HINT) from exc

    api_key = resolve_llm_api_key("openai")
    llm_client = AsyncOpenAI(api_key=api_key)
    llm = llm_factory(
        judge_model,
        provider="openai",
        client=llm_client,
        temperature=temperature,
    )
    try:
        embeddings = embedding_factory(
            "openai",
            model=embedding_model,
            client=llm_client,
            interface="modern",
        )
    except TypeError:
        embeddings = embedding_factory("openai", model=embedding_model, client=llm_client)
    return RagasRuntime(llm=llm, embeddings=embeddings)


def build_gemini_runtime(
    judge_model: str, embedding_model: str, temperature: float
) -> RagasRuntime:
    """Create RAGAS evaluator handles backed by Gemini."""
    try:
        from google import genai
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(RAGAS_INSTALL_HINT) from exc

    api_key = resolve_llm_api_key("gemini")
    client = genai.Client(api_key=api_key)
    llm = llm_factory(
        judge_model,
        provider="google",
        client=client,
        temperature=temperature,
    )
    try:
        from ragas.embeddings import GoogleEmbeddings

        embeddings = GoogleEmbeddings(client=client, model=embedding_model)
    except ImportError:
        try:
            from ragas.embeddings.base import embedding_factory
        except ImportError as exc:
            raise RuntimeError(RAGAS_INSTALL_HINT) from exc
        try:
            embeddings = embedding_factory(
                "google",
                model=embedding_model,
                client=client,
                interface="modern",
            )
        except TypeError:
            embeddings = embedding_factory("google", model=embedding_model, client=client)
    return RagasRuntime(llm=llm, embeddings=embeddings)


def configure_anthropic_model_args(llm: Any) -> None:
    """Tune RAGAS' instructor defaults for Anthropic structured judge calls."""
    model_args = getattr(llm, "model_args", None)
    if isinstance(model_args, dict):
        model_args.pop("top_p", None)
        model_args["max_tokens"] = ANTHROPIC_JUDGE_MAX_TOKENS


def build_anthropic_runtime(
    judge_model: str, embedding_model: str, temperature: float
) -> RagasRuntime:
    """Create a Claude judge runtime with OpenAI embeddings for relevancy metrics."""
    try:
        from anthropic import AsyncAnthropic
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic judging requires `anthropic`, `openai`, and `ragas` in the "
            "active scoring environment."
        ) from exc

    anthropic_client = AsyncAnthropic(api_key=resolve_llm_api_key("anthropic"))
    openai_client = AsyncOpenAI(api_key=resolve_llm_api_key("openai"))
    llm = llm_factory(
        judge_model,
        provider="anthropic",
        client=anthropic_client,
        temperature=temperature,
    )
    configure_anthropic_model_args(llm)
    try:
        embeddings = embedding_factory(
            "openai",
            model=embedding_model,
            client=openai_client,
            interface="modern",
        )
    except TypeError:
        embeddings = embedding_factory("openai", model=embedding_model, client=openai_client)
    return RagasRuntime(llm=llm, embeddings=embeddings)


def build_ragas_runtime(
    provider: JudgeProvider,
    judge_model: str,
    embedding_model: str,
    temperature: float,
) -> RagasRuntime:
    """Create the evaluator runtime for the selected judge provider."""
    if provider == "openai":
        return build_openai_runtime(judge_model, embedding_model, temperature)
    if provider == "anthropic":
        return build_anthropic_runtime(judge_model, embedding_model, temperature)
    return build_gemini_runtime(judge_model, embedding_model, temperature)


def build_metric_scorers(metrics: Sequence[MetricName], runtime: RagasRuntime) -> dict[str, Any]:
    """Instantiate the requested RAGAS metric scorers."""
    try:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(RAGAS_INSTALL_HINT) from exc

    scorers: dict[str, Any] = {}
    for metric in metrics:
        if metric == "faithfulness":
            scorers[metric] = Faithfulness(llm=runtime.llm)
        elif metric == "answer_relevancy":
            scorers[metric] = AnswerRelevancy(
                llm=runtime.llm,
                embeddings=runtime.embeddings,
                strictness=2,
            )
        elif metric == "context_precision":
            scorers[metric] = ContextPrecision(llm=runtime.llm)
        elif metric == "context_recall":
            scorers[metric] = ContextRecall(llm=runtime.llm)
    return scorers


def metric_result_value(result: Any) -> float:
    """Extract a finite float score from a RAGAS result object."""
    raw_value = getattr(result, "value", result)
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"RAGAS metric returned a non-finite score: {raw_value!r}")
    return value


def validate_metric_inputs(metric: MetricName, query: RagasInputQuery) -> None:
    """Raise early when a metric cannot be scored for this query row."""
    if metric in {"faithfulness", "answer_relevancy"} and not query.answer:
        raise ValueError("answer is required")
    if metric in {"faithfulness", "context_precision", "context_recall"} and not query.contexts:
        raise ValueError("contexts are required")
    if metric in {"context_precision", "context_recall"} and not query.reference:
        raise ValueError("reference is required")


def log_progress(message: str) -> None:
    """Emit RAGAS progress to stderr without corrupting JSON stdout."""
    print(message, file=sys.stderr, flush=True)


def progress_prefix(query_index: int | None, query_count: int | None, query_id: str) -> str:
    """Build a stable progress prefix for one query."""
    if query_index is None or query_count is None:
        return f"[?/?] {query_id}"
    return f"[{query_index}/{query_count}] {query_id}"


async def score_metric(metric: MetricName, scorer: Any, query: RagasInputQuery) -> float:
    """Run one RAGAS metric against one query row."""
    validate_metric_inputs(metric, query)
    if metric == "faithfulness":
        result = await scorer.ascore(
            user_input=query.question,
            response=query.answer,
            retrieved_contexts=query.contexts,
        )
    elif metric == "answer_relevancy":
        result = await scorer.ascore(
            user_input=query.question,
            response=query.answer,
        )
    elif metric in {"context_precision", "context_recall"}:
        result = await scorer.ascore(
            user_input=query.question,
            reference=query.reference,
            retrieved_contexts=query.contexts,
        )
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return metric_result_value(result)


async def score_query(
    query: RagasInputQuery,
    scorers: Mapping[str, Any],
    *,
    query_index: int | None = None,
    query_count: int | None = None,
    progress_logger: ProgressLogger | None = None,
) -> RagasQueryScore:
    """Score one eval query with every requested RAGAS metric."""
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for metric, scorer in scorers.items():
        prefix = progress_prefix(query_index, query_count, query.id)
        start_time = time.monotonic()
        if progress_logger:
            progress_logger(f"{prefix} {metric} ... running")
        try:
            scores[metric] = await score_metric(metric, scorer, query)
            elapsed = time.monotonic() - start_time
            if progress_logger:
                progress_logger(f"{prefix} {metric} ... ok {scores[metric]:.4f} ({elapsed:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            scores[metric] = None
            errors[metric] = str(exc)
            elapsed = time.monotonic() - start_time
            if progress_logger:
                progress_logger(f"{prefix} {metric} ... error ({elapsed:.1f}s): {exc}")
    return RagasQueryScore(
        id=query.id,
        question=query.question,
        tags=query.tags,
        metrics=scores,
        errors=errors,
        deterministic=query.deterministic,
    )


async def score_queries(
    queries: Sequence[RagasInputQuery],
    scorers: Mapping[str, Any],
    *,
    progress_logger: ProgressLogger | None = None,
) -> list[RagasQueryScore]:
    """Score query rows sequentially to keep evaluator rate limits predictable."""
    results: list[RagasQueryScore] = []
    query_count = len(queries)
    for index, query in enumerate(queries, start=1):
        results.append(
            await score_query(
                query,
                scorers,
                query_index=index,
                query_count=query_count,
                progress_logger=progress_logger,
            )
        )
    return results


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty input."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def numeric_aggregate(values: Mapping[str, Any]) -> dict[str, float | int]:
    """Keep finite numeric aggregate values from the source eval-agent report."""
    aggregate: dict[str, float | int] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) or (isinstance(value, float) and math.isfinite(value)):
            aggregate[key] = value
    return aggregate


def aggregate_scores(
    scores: Sequence[RagasQueryScore],
    metrics: Sequence[MetricName],
) -> dict[str, float | int]:
    """Aggregate per-query RAGAS scores into flat metrics for JSON and MLflow."""
    aggregate: dict[str, float | int] = {
        "query_count": len(scores),
        "metric_error_count": sum(len(score.errors) for score in scores),
        "queries_with_metric_errors": sum(1 for score in scores if score.errors),
    }
    for metric in metrics:
        values = [
            score.metrics[metric]
            for score in scores
            if metric in score.metrics and score.metrics[metric] is not None
        ]
        aggregate[f"{metric}_mean"] = mean([float(value) for value in values])
        aggregate[f"{metric}_scored_count"] = len(values)
    return aggregate


async def build_score_report(
    *,
    input_report: RagasInputReport,
    source_report_path: str,
    judge_provider: JudgeProvider,
    judge_model: str,
    embedding_model: str,
    metrics: Sequence[MetricName],
    temperature: float,
    progress_logger: ProgressLogger | None = None,
) -> RagasScoreReport:
    """Score an eval-agent report and return a serializable RAGAS report."""
    runtime = build_ragas_runtime(judge_provider, judge_model, embedding_model, temperature)
    scorers = build_metric_scorers(metrics, runtime)
    query_scores = await score_queries(
        input_report.queries,
        scorers,
        progress_logger=progress_logger,
    )
    return RagasScoreReport(
        source_report_path=source_report_path,
        eval_queries_path=input_report.eval_queries_path,
        candidate_llm_provider=input_report.llm_provider,
        candidate_chat_model=input_report.chat_model,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_embedding_model=embedding_model,
        metrics=list(metrics),
        query_count=len(query_scores),
        aggregate=aggregate_scores(query_scores, metrics),
        deterministic_aggregate=numeric_aggregate(input_report.aggregate),
        queries=query_scores,
    )


def score_report_to_dict(report: RagasScoreReport) -> dict[str, Any]:
    """Convert a RAGAS score report to plain JSON containers."""
    return asdict(report)


def write_score_report(path: str | Path, report: RagasScoreReport) -> None:
    """Write the scored RAGAS report to disk with stable formatting."""
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(score_report_to_dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def metric_log_name(name: str) -> str:
    """Return a namespaced MLflow metric name for a RAGAS aggregate key."""
    return f"ragas.{name}"


def default_run_name(report: RagasScoreReport) -> str:
    """Build a readable MLflow run name from candidate and judge metadata."""
    return (
        f"ragas-{report.candidate_llm_provider}-{report.candidate_chat_model}-"
        f"judge-{report.judge_provider}-{report.judge_model}"
    )


def log_score_report_to_mlflow(
    report: RagasScoreReport,
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None = None,
) -> str:
    """Log the scored RAGAS report, metrics, and model metadata to MLflow."""
    settings = get_settings()
    os.environ.setdefault("MLFLOW_ARTIFACTS_URI", settings.mlflow_artifacts_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params: dict[str, str | int] = {
        "candidate_llm_provider": report.candidate_llm_provider,
        "candidate_chat_model": report.candidate_chat_model,
        "judge_provider": report.judge_provider,
        "judge_model": report.judge_model,
        "metrics": ",".join(report.metrics),
        "query_count": report.query_count,
    }
    if report.judge_embedding_model:
        params["judge_embedding_model"] = report.judge_embedding_model

    with mlflow.start_run(run_name=run_name or default_run_name(report)) as run:
        mlflow.log_params(params)
        for key, value in report.aggregate.items():
            mlflow.log_metric(metric_log_name(key), float(value))
        for key, value in report.deterministic_aggregate.items():
            mlflow.log_metric(f"eval_agent.{key}", float(value))
        mlflow.log_dict(score_report_to_dict(report), "eval/ragas_report.json")
        return run.info.run_id


def report_has_metric_errors(report: RagasScoreReport) -> bool:
    """Return True when any RAGAS metric failed to score."""
    return int(report.aggregate.get("metric_error_count", 0)) > 0


async def build_report_from_args(args: argparse.Namespace) -> RagasScoreReport:
    """Load CLI inputs, run RAGAS scoring, and return the scored report."""
    metrics = selected_metrics(args.metric)
    judge_provider: JudgeProvider = args.judge_provider
    judge_model = resolve_judge_model(judge_provider, args.judge_model)
    embedding_model = resolve_embedding_model(judge_provider, args.embedding_model)
    input_report = load_input_report(args.input)
    return await build_score_report(
        input_report=input_report,
        source_report_path=str(args.input),
        judge_provider=judge_provider,
        judge_model=judge_model,
        embedding_model=embedding_model,
        metrics=metrics,
        temperature=args.temperature,
        progress_logger=log_progress,
    )


async def smoke_test_judge_call(
    *,
    judge_provider: JudgeProvider,
    judge_model: str,
    embedding_model: str,
    temperature: float,
) -> dict[str, str | float | None]:
    """Run one structured judge call through the same runtime RAGAS uses."""
    runtime = build_ragas_runtime(judge_provider, judge_model, embedding_model, temperature)
    response = await runtime.llm.agenerate(
        "Return verdict='pass' and a short reason confirming this judge smoke test works.",
        JudgeSmokeResponse,
    )
    smoke_response = JudgeSmokeResponse.model_validate(response)
    return {
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_embedding_model": embedding_model,
        "temperature": temperature,
        "verdict": smoke_response.verdict,
        "reason": smoke_response.reason,
    }


async def smoke_test_judge_from_args(args: argparse.Namespace) -> dict[str, str | float | None]:
    """Resolve CLI judge settings and run a single smoke-test call."""
    judge_provider: JudgeProvider = args.judge_provider
    judge_model = resolve_judge_model(judge_provider, args.judge_model)
    embedding_model = resolve_embedding_model(judge_provider, args.embedding_model)
    return await smoke_test_judge_call(
        judge_provider=judge_provider,
        judge_model=judge_model,
        embedding_model=embedding_model,
        temperature=args.temperature,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run RAGAS scoring from the command line."""
    args = parse_args(argv)
    try:
        if args.smoke_test_judge:
            smoke_report = asyncio.run(smoke_test_judge_from_args(args))
            print(json.dumps(smoke_report, indent=2, sort_keys=True))
            return 0
        report = asyncio.run(build_report_from_args(args))
        rendered = json.dumps(score_report_to_dict(report), indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            write_score_report(args.output, report)
        if args.log_mlflow:
            run_id = log_score_report_to_mlflow(
                report,
                tracking_uri=args.tracking_uri,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
            )
            print(f"Logged MLflow run: {run_id}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"eval_ragas failed: {exc}", file=sys.stderr)
        return 1
    return 1 if report_has_metric_errors(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
