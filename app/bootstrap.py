from __future__ import annotations

from typing import Any

import mlflow

from .chain import build_rag_chain
from .config import get_settings, resolve_llm_api_key
from .providers import get_provider
from .schemas import ActiveModelConfig


def parse_active_model_config(
    params: dict[str, str], run_id: str, model_version: str
) -> ActiveModelConfig:
    settings = get_settings()
    llm_provider = (params.get("llm_provider") or "openai").lower()
    provider = get_provider(llm_provider)
    default_chat_model = provider.default_chat_model(settings)

    return ActiveModelConfig(
        llm_provider=llm_provider,
        chat_model=params.get("chat_model") or default_chat_model,
        k=int(params.get("k", settings.retriever_k)),
        temperature=float(params.get("temperature", "0.0")),
        run_id=run_id,
        model_version=model_version,
    )


def load_registered_rag_chain() -> tuple[Any, ActiveModelConfig]:
    settings = get_settings()
    tracking_uri = settings.mlflow_tracking_uri
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required (set it in .env).")

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    try:
        model_version = client.get_model_version_by_alias(
            settings.mlflow_model_name,
            "production",
        )
    except Exception as exc:  # pragma: no cover - exercised via startup tests
        raise RuntimeError(
            "Unable to load the MLflow production alias for "
            f"registered model '{settings.mlflow_model_name}'."
        ) from exc

    run = client.get_run(model_version.run_id)
    config = parse_active_model_config(
        run.data.params,
        run_id=model_version.run_id,
        model_version=str(model_version.version),
    )
    chain = build_rag_chain(
        api_key=resolve_llm_api_key(config.llm_provider),
        llm_provider=config.llm_provider,
        chat_model=config.chat_model,
        k=config.k,
        temperature=config.temperature,
    )
    return chain, config
