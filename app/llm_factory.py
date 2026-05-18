"""Single source of truth for provider -> chat model construction.

DeepSeek (`deepseek-v4-pro`) and Kimi (`kimi-k2.6`) emit an opaque
`reasoning_content` field on assistant tool-call messages and require it
echoed back verbatim next turn. LangChain reconstructs assistant messages
via `_convert_message_to_dict`, which drops `reasoning_content`, so the
2nd tool turn 400s ("reasoning_content ... must be passed back"). Rather
than a fragile per-vendor round-trip shim, we disable thinking mode for
these providers in the agent (LangChain's documented Kimi tool-use path is
`ChatMoonshot(thinking=False)`; DeepSeek's verified equivalent is
`extra_body={"thinking": {"type": "disabled"}}`). With thinking off no
`reasoning_content` is emitted, so tool calls round-trip cleanly. This also
matches the W10 finding that reasoning-mode over-exploration is what broke
agent convergence; a decisive tool-caller is what the loop needs. Every
provider->LLM construction routes through here so a provider is added in
ONE place.
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
            thinking_budget=0,
        )
    if provider == "deepseek":
        return ChatDeepSeek(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
    if provider == "kimi":
        return ChatMoonshot(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=temperature,
            thinking=False,
        )
    raise ValueError(f"Unsupported llm_provider: {llm_provider}")
