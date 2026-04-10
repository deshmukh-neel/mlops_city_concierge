#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.chain import build_rag_chain  # noqa: E402
from app.config import get_settings, resolve_llm_api_key  # noqa: E402

DEFAULT_EXPERIMENT_NAME = "city-concierge-rag-v2"
DEFAULT_SAMPLE_QUERIES = [
    "Best tacos in the Mission",
    "Coffee shops near North Beach with good ratings",
    "Romantic dinner spots in Hayes Valley",
]


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


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Log City Concierge RAG experiments to MLflow.")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--k", type=int, default=settings.retriever_k)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--model-name", default=settings.mlflow_model_name)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--sample-query", action="append", dest="sample_queries", default=None)
    return parser.parse_args()


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
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
    model_name: str | None = None,
    register_model: bool = False,
    sample_queries: list[str] | None = None,
) -> str:
    settings = get_settings()
    resolved_chat_model = resolve_chat_model(llm_provider, chat_model)
    api_key = resolve_llm_api_key(llm_provider)
    tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    model_name = model_name or settings.mlflow_model_name

    import os

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

    queries = sample_queries or DEFAULT_SAMPLE_QUERIES
    run_name = f"{llm_provider}-{resolved_chat_model}-k{k}-t{temperature}"

    with mlflow.start_run(run_name=run_name) as run:
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

        for index, query in enumerate(queries, start=1):
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


def main() -> None:
    args = parse_args()
    run_id = log_rag_experiment(
        llm_provider=args.llm_provider,
        chat_model=args.chat_model,
        k=args.k,
        temperature=args.temperature,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        register_model=args.register_model,
        sample_queries=args.sample_queries,
    )
    print(
        "\n".join(
            [
                f"Logged MLflow run: {run_id}",
                f"Tracking URI: {args.tracking_uri}",
                f"Experiment: {args.experiment_name}",
                f"Model registry target: {args.model_name}",
            ]
        )
    )


if __name__ == "__main__":
    main()
