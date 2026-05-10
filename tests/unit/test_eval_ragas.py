from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.eval_ragas import (
    DEFAULT_METRICS,
    RagasInputReport,
    RagasQueryScore,
    RagasRuntime,
    RagasScoreReport,
    aggregate_scores,
    build_ragas_runtime,
    build_score_report,
    load_input_report,
    log_score_report_to_mlflow,
    metric_result_value,
    numeric_aggregate,
    report_has_metric_errors,
    resolve_embedding_model,
    resolve_judge_model,
    score_metric,
    score_query,
    score_report_to_dict,
    selected_metrics,
    write_score_report,
)


class FakeRagasResult:
    """Minimal object shaped like a RAGAS metric result."""

    def __init__(self, value: float) -> None:
        """Store the score under the modern RAGAS result attribute."""
        self.value = value


class FakeScorer:
    """Collect ascore calls and return a configured score."""

    def __init__(self, value: float) -> None:
        """Initialize an in-memory fake scorer."""
        self.value = value
        self.calls: list[dict] = []

    async def ascore(self, **kwargs) -> FakeRagasResult:
        """Record the score inputs and return the configured fake result."""
        self.calls.append(kwargs)
        return FakeRagasResult(self.value)


def sample_query_payload(**overrides: object) -> dict:
    """Build a minimal eval-agent query row for RAGAS tests."""
    payload: dict = {
        "id": "case_one",
        "question": "coffee in soma",
        "answer": "Try Example Cafe.",
        "contexts": [" name: Example Cafe | snippet: quiet coffee ", ""],
        "reference": "Recommend a cafe in SOMA.",
        "tags": ["cafe", ""],
        "deterministic": {"violations": []},
    }
    payload.update(overrides)
    return payload


def sample_report_payload(**overrides: object) -> dict:
    """Build a minimal eval-agent report for RAGAS tests."""
    payload: dict = {
        "eval_queries_path": "configs/eval_queries.yaml",
        "llm_provider": "openai",
        "chat_model": "gpt-4o-mini",
        "query_count": 1,
        "aggregate": {"query_count": 1},
        "queries": [sample_query_payload()],
    }
    payload.update(overrides)
    return payload


def write_json(tmp_path: Path, payload: dict) -> Path:
    """Write JSON test data and return its path."""
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def score_report(**overrides: object) -> RagasScoreReport:
    """Build a scored RAGAS report for logging and serialization tests."""
    payload = {
        "source_report_path": "artifacts/eval_report.json",
        "eval_queries_path": "configs/eval_queries.yaml",
        "candidate_llm_provider": "openai",
        "candidate_chat_model": "gpt-4o-mini",
        "judge_provider": "openai",
        "judge_model": "gpt-4o-mini",
        "judge_embedding_model": "text-embedding-3-small",
        "metrics": ["faithfulness", "context_precision"],
        "query_count": 1,
        "aggregate": {
            "query_count": 1,
            "metric_error_count": 0,
            "queries_with_metric_errors": 0,
            "faithfulness_mean": 0.9,
            "faithfulness_scored_count": 1,
        },
        "deterministic_aggregate": {
            "tool_error_rate": 0.0,
            "expected_results_match_rate": 1.0,
        },
        "queries": [
            RagasQueryScore(
                id="case_one",
                question="coffee in soma",
                tags=["cafe"],
                metrics={"faithfulness": 0.9},
                errors={},
                deterministic={"violations": []},
            )
        ],
    }
    payload.update(overrides)
    return RagasScoreReport(**payload)


def test_load_input_report_validates_and_cleans_rows(tmp_path: Path) -> None:
    """Load an eval-agent JSON report into typed RAGAS input rows."""
    path = write_json(tmp_path, sample_report_payload())

    report = load_input_report(path)

    assert isinstance(report, RagasInputReport)
    assert report.queries[0].contexts == ["name: Example Cafe | snippet: quiet coffee"]
    assert report.queries[0].tags == ["cafe"]


def test_load_input_report_rejects_empty_queries(tmp_path: Path) -> None:
    """Reject eval reports that have no query rows to score."""
    path = write_json(tmp_path, sample_report_payload(queries=[]))

    with pytest.raises(ValueError, match="at least one query"):
        load_input_report(path)


def test_selected_metrics_defaults_to_core_set() -> None:
    """Use the core RAGAS metric set when the CLI does not narrow it."""
    assert selected_metrics(None) == list(DEFAULT_METRICS)


def test_selected_metrics_preserves_requested_order() -> None:
    """Preserve explicit metric ordering from the CLI."""
    assert selected_metrics(["context_recall", "faithfulness"]) == [
        "context_recall",
        "faithfulness",
    ]


def test_resolve_judge_model_defaults_anthropic_to_sonnet() -> None:
    """Use Sonnet as the default third-party judge model."""
    assert resolve_judge_model("anthropic", None) == "claude-sonnet-4-6"
    assert resolve_judge_model("anthropic", " custom-claude ") == "custom-claude"


def test_resolve_embedding_model_uses_openai_embeddings_for_anthropic() -> None:
    """Keep embedding-based metrics on the configured OpenAI embedding model."""
    assert resolve_embedding_model("anthropic", None) == "text-embedding-3-small"
    assert resolve_embedding_model("anthropic", " custom-embedding ") == "custom-embedding"


def test_build_ragas_runtime_dispatches_anthropic(mocker) -> None:
    """Route Anthropic judge requests to the Claude runtime builder."""
    build_anthropic_runtime = mocker.patch(
        "scripts.eval_ragas.build_anthropic_runtime",
        return_value=RagasRuntime(llm="llm", embeddings="embeddings"),
    )

    runtime = build_ragas_runtime(
        "anthropic",
        judge_model="claude-sonnet-4-6",
        embedding_model="text-embedding-3-small",
        temperature=0.0,
    )

    assert runtime == RagasRuntime(llm="llm", embeddings="embeddings")
    build_anthropic_runtime.assert_called_once_with(
        "claude-sonnet-4-6",
        "text-embedding-3-small",
        0.0,
    )


def test_metric_result_value_accepts_result_objects_and_floats() -> None:
    """Normalize RAGAS result objects and raw numeric scores."""
    assert metric_result_value(FakeRagasResult(0.75)) == 0.75
    assert metric_result_value(0.5) == 0.5


def test_metric_result_value_rejects_non_finite_scores() -> None:
    """Reject NaN or infinite scores before aggregation."""
    with pytest.raises(ValueError, match="non-finite"):
        metric_result_value(float("nan"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metric", "expected_call"),
    [
        (
            "faithfulness",
            {
                "user_input": "coffee in soma",
                "response": "Try Example Cafe.",
                "retrieved_contexts": ["name: Example Cafe | snippet: quiet coffee"],
            },
        ),
        (
            "answer_relevancy",
            {
                "user_input": "coffee in soma",
                "response": "Try Example Cafe.",
            },
        ),
        (
            "context_precision",
            {
                "user_input": "coffee in soma",
                "reference": "Recommend a cafe in SOMA.",
                "retrieved_contexts": ["name: Example Cafe | snippet: quiet coffee"],
            },
        ),
        (
            "context_recall",
            {
                "user_input": "coffee in soma",
                "reference": "Recommend a cafe in SOMA.",
                "retrieved_contexts": ["name: Example Cafe | snippet: quiet coffee"],
            },
        ),
    ],
)
async def test_score_metric_passes_required_fields(metric: str, expected_call: dict) -> None:
    """Call each RAGAS scorer with the fields that metric expects."""
    query = RagasInputReport.model_validate(sample_report_payload()).queries[0]
    scorer = FakeScorer(0.8)

    score = await score_metric(metric, scorer, query)

    assert score == 0.8
    assert scorer.calls == [expected_call]


@pytest.mark.asyncio
async def test_score_query_captures_metric_errors() -> None:
    """Keep one metric failure local to the query row."""
    query = RagasInputReport.model_validate(
        sample_report_payload(queries=[sample_query_payload(contexts=[])])
    ).queries[0]
    scorers = {
        "faithfulness": FakeScorer(1.0),
        "answer_relevancy": FakeScorer(0.5),
    }

    result = await score_query(query, scorers)

    assert result.metrics == {"faithfulness": None, "answer_relevancy": 0.5}
    assert result.errors == {"faithfulness": "contexts are required"}


def test_aggregate_scores_means_only_successful_scores() -> None:
    """Aggregate successful metric values without hiding metric errors."""
    scores = [
        RagasQueryScore(
            id="one",
            question="q1",
            tags=[],
            metrics={"faithfulness": 1.0, "context_recall": 0.5},
            errors={},
            deterministic={},
        ),
        RagasQueryScore(
            id="two",
            question="q2",
            tags=[],
            metrics={"faithfulness": None, "context_recall": 1.0},
            errors={"faithfulness": "contexts are required"},
            deterministic={},
        ),
    ]

    aggregate = aggregate_scores(scores, ["faithfulness", "context_recall"])

    assert aggregate["query_count"] == 2
    assert aggregate["metric_error_count"] == 1
    assert aggregate["queries_with_metric_errors"] == 1
    assert aggregate["faithfulness_mean"] == 1.0
    assert aggregate["faithfulness_scored_count"] == 1
    assert aggregate["context_recall_mean"] == 0.75
    assert aggregate["context_recall_scored_count"] == 2


def test_numeric_aggregate_keeps_finite_numbers_only() -> None:
    """Carry source eval-agent aggregate metrics without logging non-numeric values."""
    aggregate = numeric_aggregate(
        {
            "query_count": 2,
            "tool_error_rate": 0.0,
            "bad_bool": True,
            "bad_nan": float("nan"),
            "bad_text": "skip",
        }
    )

    assert aggregate == {"query_count": 2, "tool_error_rate": 0.0}


@pytest.mark.asyncio
async def test_build_score_report_uses_candidate_and_judge_metadata(mocker) -> None:
    """Build a complete scored report from one input report."""
    input_report = RagasInputReport.model_validate(
        sample_report_payload(aggregate={"query_count": 1, "tool_error_rate": 0.0})
    )
    mocker.patch(
        "scripts.eval_ragas.build_ragas_runtime",
        return_value=RagasRuntime(llm=object(), embeddings=object()),
    )
    mocker.patch(
        "scripts.eval_ragas.build_metric_scorers",
        return_value={"faithfulness": FakeScorer(0.91)},
    )

    report = await build_score_report(
        input_report=input_report,
        source_report_path="report.json",
        judge_provider="openai",
        judge_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        metrics=["faithfulness"],
        temperature=0.0,
    )

    assert report.candidate_llm_provider == "openai"
    assert report.candidate_chat_model == "gpt-4o-mini"
    assert report.judge_model == "gpt-4o-mini"
    assert report.aggregate["faithfulness_mean"] == 0.91
    assert report.deterministic_aggregate == {"query_count": 1, "tool_error_rate": 0.0}


def test_write_score_report_creates_parent_directories(tmp_path: Path) -> None:
    """Write scored reports with stable JSON formatting."""
    output_path = tmp_path / "nested" / "ragas_report.json"
    report = score_report()

    write_score_report(output_path, report)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["aggregate"]["faithfulness_mean"] == 0.9


def test_score_report_to_dict_serializes_nested_scores() -> None:
    """Convert scored reports to plain JSON containers."""
    payload = score_report_to_dict(score_report())

    assert payload["queries"][0]["metrics"] == {"faithfulness": 0.9}


def test_report_has_metric_errors_reads_aggregate() -> None:
    """Detect reports where at least one metric could not be scored."""
    report = score_report(aggregate={"metric_error_count": 1})

    assert report_has_metric_errors(report) is True


def test_log_score_report_to_mlflow_logs_params_metrics_and_report(mocker) -> None:
    """Log RAGAS scores and candidate metadata to MLflow."""
    report = score_report()
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
    start_run = mocker.patch("scripts.eval_ragas.mlflow.start_run")
    start_run.return_value.__enter__.return_value = run
    set_tracking_uri = mocker.patch("scripts.eval_ragas.mlflow.set_tracking_uri")
    set_experiment = mocker.patch("scripts.eval_ragas.mlflow.set_experiment")
    log_params = mocker.patch("scripts.eval_ragas.mlflow.log_params")
    log_metric = mocker.patch("scripts.eval_ragas.mlflow.log_metric")
    log_dict = mocker.patch("scripts.eval_ragas.mlflow.log_dict")

    run_id = log_score_report_to_mlflow(
        report,
        tracking_uri="http://mlflow.test",
        experiment_name="ragas-test",
        run_name="presentation-run",
    )

    assert run_id == "run-123"
    set_tracking_uri.assert_called_once_with("http://mlflow.test")
    set_experiment.assert_called_once_with("ragas-test")
    start_run.assert_called_once_with(run_name="presentation-run")
    assert log_params.call_args.kwargs == {}
    assert log_params.call_args.args[0]["candidate_chat_model"] == "gpt-4o-mini"
    log_metric.assert_any_call("ragas.faithfulness_mean", 0.9)
    log_metric.assert_any_call("eval_agent.tool_error_rate", 0.0)
    log_metric.assert_any_call("eval_agent.expected_results_match_rate", 1.0)
    log_dict.assert_called_once()
