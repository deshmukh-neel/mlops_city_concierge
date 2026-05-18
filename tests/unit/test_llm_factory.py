from __future__ import annotations

import pytest

from app.llm_factory import SUPPORTED_PROVIDERS, build_chat_model


@pytest.mark.parametrize(
    "provider,patch_path",
    [
        ("openai", "app.llm_factory.ChatOpenAI"),
        ("gemini", "app.llm_factory.ChatGoogleGenerativeAI"),
        ("deepseek", "app.llm_factory.ChatDeepSeek"),
        ("kimi", "app.llm_factory.ChatMoonshot"),
    ],
)
def test_build_chat_model_dispatches_per_provider(
    provider, patch_path, mocker, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch(patch_path, return_value=f"{provider}-llm")

    out = build_chat_model(provider, "some-model", temperature=0.3)

    assert out == f"{provider}-llm"
    cls.assert_called_once()
    _, kwargs = cls.call_args
    assert kwargs["model"] == "some-model"
    assert kwargs["temperature"] == 0.3


def test_build_chat_model_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        build_chat_model("anthropic", "claude", temperature=0.0)


def test_supported_providers_is_the_contract() -> None:
    assert SUPPORTED_PROVIDERS == ("openai", "gemini", "deepseek", "kimi")


def test_deepseek_disables_thinking_for_tool_calls(mocker, monkeypatch) -> None:
    """DeepSeek reasoning mode emits reasoning_content that langchain-openai's
    message reconstruction drops, 400-ing on the 2nd tool turn. Disabling
    thinking (verified raw-SDK form: extra_body thinking.type=disabled) makes
    the tool-calling agent loop work."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("app.llm_factory.ChatDeepSeek", return_value="ds")

    build_chat_model("deepseek", "deepseek-v4-pro", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_kimi_disables_thinking_for_tool_calls(mocker, monkeypatch) -> None:
    """Kimi has the same reasoning_content round-trip problem; LangChain's
    documented tool-use path is ChatMoonshot(thinking=False)."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("app.llm_factory.ChatMoonshot", return_value="kimi")

    build_chat_model("kimi", "kimi-k2.6", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["thinking"] is False


def test_gemini_disables_thinking(mocker, monkeypatch) -> None:
    """All providers run with reasoning OFF for an apples-to-apples agent
    comparison; reasoning-mode over-exploration breaks the tool loop.
    Gemini 2.5+ disables thinking via thinking_budget=0."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("app.llm_factory.ChatGoogleGenerativeAI", return_value="gem")

    build_chat_model("gemini", "gemini-3.1-pro-preview", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["thinking_budget"] == 0
