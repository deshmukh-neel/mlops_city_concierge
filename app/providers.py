from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import Settings


@dataclass(frozen=True)
class ProviderSpec:
    default_chat_model: Callable[[Settings], str]
    api_key: Callable[[Settings], str]
    build_llm: Callable[[str, str, float], BaseChatModel]


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        default_chat_model=lambda s: s.openai_chat_model,
        api_key=lambda s: s.openai_api_key,
        build_llm=lambda model, key, temp: ChatOpenAI(
            model=model, api_key=SecretStr(key), temperature=temp
        ),
    ),
    "gemini": ProviderSpec(
        default_chat_model=lambda s: s.gemini_chat_model,
        api_key=lambda s: s.gemini_api_key,
        build_llm=lambda model, key, temp: ChatGoogleGenerativeAI(
            model=model, google_api_key=SecretStr(key), temperature=temp
        ),
    ),
}


def get_provider(name: str) -> ProviderSpec:
    spec = PROVIDERS.get(name.lower())
    if spec is None:
        raise ValueError(f"Unsupported llm_provider: {name}")
    return spec
