#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.chain import build_rag_chain  # noqa: E402
from app.config import get_settings, resolve_llm_api_key  # noqa: E402

DEFAULT_EXPERIMENT_NAME = "city-concierge-rag-v2"
DEFAULT_CONFIG_PATH = Path("configs/experiments.yaml")


class RunConfig(BaseModel):
    llm_provider: Literal["openai", "gemini"]
    chat_model: str = Field(min_length=1)
    k: int = Field(gt=0)
    temperature: float

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _lowercase_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("chat_model", mode="before")
    @classmethod
    def _strip_chat_model(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("k", "temperature", mode="before")
    @classmethod
    def _reject_bools(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a number, not a bool")
        return value


class ExperimentConfig(BaseModel):
    experiment_name: str = Field(min_length=1)
    sample_queries: list[str] = Field(min_length=1)
    runs: list[RunConfig] = Field(min_length=1)

    @field_validator("experiment_name", mode="before")
    @classmethod
    def _strip_experiment_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("sample_queries")
    @classmethod
    def _sample_queries_non_empty(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for index, query in enumerate(value):
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"sample_queries[{index}] must be a non-empty string")
            cleaned.append(query.strip())
        return cleaned


class RagConfigPythonModel(mlflow.pyfunc.PythonModel):
    def __init__(self, config: dict[str, str | int | float]) -> None:
        self.config = config

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        del context, params
        row_count = len(model_input.index) if hasattr(model_input, "index") else 1
        return pd.DataFrame(
            [
                {
                    "status": "city-concierge-rag-config",
                    "llm_provider": self.config["llm_provider"],
                    "chat_model": self.config["chat_model"],
                    "k": self.config["k"],
                    "temperature": self.config["temperature"],
                }
            ]
            * row_count
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Log City Concierge RAG experiments to MLflow.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--k", type=int, default=settings.retriever_k)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--model-name", default=settings.mlflow_model_name)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--sample-query", action="append", dest="sample_queries", default=None)
    return parser.parse_args(argv)


def resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path)
    if config_path.is_absolute():
        return config_path
    return REPO_ROOT / config_path


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML mapping.")
    return ExperimentConfig.model_validate(raw)


def resolve_chat_model(llm_provider: str, chat_model: str | None) -> str:
    settings = get_settings()
    if chat_model:
        return chat_model
    if llm_provider.lower() == "openai":
        return settings.openai_chat_model
    return settings.gemini_chat_model


def format_sample_output(query: str, result: dict) -> str:
    response = result.get("result") or result.get("response") or ""
    source_documents = result.get("source_documents") or []

    lines = [f"Query: {query}", "", f"Response: {response}", "", "Sources:"]
    if not source_documents:
        lines.append("None")
        return "\n".join(lines)

    for document in source_documents:
        if isinstance(document, Document):
            metadata = document.metadata
        else:
            metadata = getattr(document, "metadata", {}) or {}

        lines.append(
            " | ".join(
                [
                    str(metadata.get("name") or "Unknown"),
                    str(metadata.get("address") or "Unknown address"),
                    f"similarity={metadata.get('similarity', 'n/a')}",
                ]
            )
        )

    return "\n".join(lines)


def log_rag_experiment(
    *,
    llm_provider: str,
    chat_model: str | None,
    k: int,
    temperature: float,
    sample_queries: list[str],
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    model_name: str | None = None,
    register_model: bool = False,
    nested: bool = False,
) -> str:
    settings = get_settings()
    resolved_chat_model = resolve_chat_model(llm_provider, chat_model)
    api_key = resolve_llm_api_key(llm_provider)
    tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    model_name = model_name or settings.mlflow_model_name

    os.environ.setdefault("MLFLOW_ARTIFACTS_URI", settings.mlflow_artifacts_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params: dict[str, str | int | float] = {
        "llm_provider": llm_provider,
        "chat_model": resolved_chat_model,
        "k": k,
        "temperature": temperature,
        "embedding_model": settings.openai_embedding_model,
        "vector_store": "pgvector",
    }

    run_name = f"{llm_provider}-{resolved_chat_model}-k{k}-t{temperature}"

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.log_params(params)
        mlflow.log_text(
            json.dumps(params, indent=2, sort_keys=True),
            "config/rag_config.json",
        )

        chain = build_rag_chain(
            connection_string=settings.database_url,
            api_key=api_key,
            llm_provider=llm_provider,
            chat_model=resolved_chat_model,
            k=k,
            temperature=temperature,
        )

        for index, query in enumerate(sample_queries, start=1):
            result = chain.invoke({"query": query})
            mlflow.log_text(
                format_sample_output(query, result),
                f"sample_outputs/query_{index}.txt",
            )

        if register_model:
            artifact_path = "rag_config_model"
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=RagConfigPythonModel(params),
            )
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/{artifact_path}",
                name=model_name,
            )

        return run.info.run_id


def log_rag_experiments_from_config(
    *,
    config: ExperimentConfig,
    tracking_uri: str | None = None,
    model_name: str | None = None,
    register_model: bool = False,
    experiment_name: str | None = None,
    sample_queries: list[str] | None = None,
    config_path: Path | None = None,
) -> list[str]:
    """Log every run in ``config`` as a nested MLflow run under one parent.

    Runs are logged sequentially to keep load on the shared MLflow server
    predictable. If batched experiments start feeling too slow during evals,
    revisit — options are ``chain.batch()`` for per-query parallelism (cheap)
    or a bounded ``ThreadPoolExecutor`` for per-run parallelism (requires an
    ``MlflowClient`` refactor so nested ``start_run`` isn't shared across
    threads).
    """
    settings = get_settings()
    resolved_experiment_name = experiment_name or config.experiment_name
    resolved_sample_queries = sample_queries or config.sample_queries
    resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri

    os.environ.setdefault("MLFLOW_ARTIFACTS_URI", settings.mlflow_artifacts_uri)
    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_experiment(resolved_experiment_name)

    run_ids: list[str] = []
    with mlflow.start_run(run_name=f"batch-{resolved_experiment_name}"):
        if config_path is not None:
            mlflow.log_artifact(str(config_path), artifact_path="config")
        for run in config.runs:
            run_ids.append(
                log_rag_experiment(
                    llm_provider=run.llm_provider,
                    chat_model=run.chat_model,
                    k=run.k,
                    temperature=run.temperature,
                    sample_queries=resolved_sample_queries,
                    experiment_name=resolved_experiment_name,
                    tracking_uri=resolved_tracking_uri,
                    model_name=model_name,
                    register_model=register_model,
                    nested=True,
                )
            )

    return run_ids


def run_config_mode(args: argparse.Namespace) -> None:
    config_path = resolve_config_path(args.config)
    config = load_experiment_config(config_path)
    run_ids = log_rag_experiments_from_config(
        config=config,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        register_model=args.register_model,
        experiment_name=args.experiment_name,
        sample_queries=args.sample_queries,
        config_path=config_path,
    )
    lines = [
        f"Logged MLflow runs: {len(run_ids)}",
        *(f"  - {run_id}" for run_id in run_ids),
        f"Config: {config_path}",
        f"Tracking URI: {args.tracking_uri}",
        f"Experiment: {args.experiment_name or config.experiment_name}",
        f"Model registry target: {args.model_name}",
    ]
    print("\n".join(lines))


def _load_default_sample_queries() -> list[str]:
    try:
        config = load_experiment_config(DEFAULT_CONFIG_PATH)
    except FileNotFoundError as exc:
        raise SystemExit(
            "No --sample-query provided and the default config "
            f"({DEFAULT_CONFIG_PATH}) is missing. Pass --sample-query or --config."
        ) from exc
    return config.sample_queries


def run_ad_hoc_mode(args: argparse.Namespace) -> None:
    sample_queries = args.sample_queries or _load_default_sample_queries()
    experiment_name = args.experiment_name or DEFAULT_EXPERIMENT_NAME
    run_id = log_rag_experiment(
        llm_provider=args.llm_provider,
        chat_model=args.chat_model,
        k=args.k,
        temperature=args.temperature,
        sample_queries=sample_queries,
        experiment_name=experiment_name,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        register_model=args.register_model,
    )
    print(
        "\n".join(
            [
                f"Logged MLflow run: {run_id}",
                f"Tracking URI: {args.tracking_uri}",
                f"Experiment: {experiment_name}",
                f"Model registry target: {args.model_name}",
            ]
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.config is not None:
        run_config_mode(args)
    else:
        run_ad_hoc_mode(args)


if __name__ == "__main__":
    main()
