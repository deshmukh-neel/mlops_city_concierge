from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    api_title: str = "City Concierge API"
    api_version: str = "0.1.0"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    mlflow_tracking_uri: str = "http://35.223.147.177:5000"
    mlflow_artifacts_uri: str = "mlflow-artifacts://35.223.147.177:5000"
    mlflow_model_name: str = "city-concierge-rag"
    retriever_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def resolve_llm_api_key(llm_provider: str) -> str:
    """Return the API key for the given LLM provider, or raise if missing."""
    s = get_settings()
    provider = llm_provider.lower()

    if provider == "openai":
        api_key = s.openai_api_key
    elif provider == "gemini":
        api_key = s.gemini_api_key
    else:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    if not api_key:
        raise RuntimeError(f"Missing API key for llm_provider={llm_provider}.")
    return api_key
