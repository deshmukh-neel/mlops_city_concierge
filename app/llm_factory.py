"""Single source of truth for provider -> chat model construction.

Reasoning models (DeepSeek, Kimi/Moonshot, Gemini) emit opaque reasoning
state with each tool call and require it replayed in history next turn.
`langchain_openai.ChatOpenAI` deliberately does NOT round-trip the
non-standard `reasoning_content` field (its docstring directs you to
provider-specific packages), so OpenAI-compatible reasoning models 400 on
the second tool turn. We therefore use the provider-specific LangChain
classes, which preserve reasoning state. Every provider->LLM construction
in the codebase routes through here so a provider is added in ONE place.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_moonshot import ChatMoonshot
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import resolve_llm_api_key

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "gemini", "deepseek", "kimi")


def build_chat_model(llm_provider: str, chat_model: str, temperature: float) -> BaseChatModel:
    """Construct a chat model for `llm_provider`. Raises ValueError for
    unsupported providers, RuntimeError if the provider's API key is missing
    (both via resolve_llm_api_key)."""
    provider = llm_provider.lower()
    api_key = resolve_llm_api_key(provider)
    if provider == "openai":
        return ChatOpenAI(model=chat_model, api_key=SecretStr(api_key), temperature=temperature)
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=chat_model,
            google_api_key=SecretStr(api_key),
            temperature=temperature,
        )
    if provider == "deepseek":
        return ChatDeepSeek(model=chat_model, api_key=SecretStr(api_key), temperature=temperature)
    if provider == "kimi":
        return ChatMoonshot(model=chat_model, api_key=SecretStr(api_key), temperature=temperature)
    raise ValueError(f"Unsupported llm_provider: {llm_provider}")
