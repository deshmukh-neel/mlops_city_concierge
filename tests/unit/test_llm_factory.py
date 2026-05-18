from __future__ import annotations

import pytest

from app.llm_factory import SUPPORTED_PROVIDERS, build_chat_model


@pytest.mark.parametrize(
    "provider,patch_path",
    [
        ("openai", "app.llm_factory.ChatOpenAI"),
        ("gemini", "app.llm_factory.ChatGoogleGenerativeAI"),
        ("deepseek", "app.llm_factory.ChatDeepSeek"),
        ("kimi", "app.llm_factory._ToolLoopChatMoonshot"),
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


def test_build_chat_model_rejects_unknown_provider(mocker, monkeypatch) -> None:
    # anthropic is a provider resolve_llm_api_key KNOWS but the factory does
    # NOT support. Null every key so the contract is enforced by
    # SUPPORTED_PROVIDERS, not incidentally by a missing key (the bug: passed
    # locally with a .env anthropic key, failed in CI without one).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # If the factory ever calls resolve_llm_api_key for an unsupported
    # provider, that's the bug — it must reject via SUPPORTED_PROVIDERS first.
    mocker.patch(
        "app.llm_factory.resolve_llm_api_key",
        side_effect=AssertionError("must reject before key resolution"),
    )
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
    cls = mocker.patch("app.llm_factory._ToolLoopChatMoonshot", return_value="kimi")

    build_chat_model("kimi", "kimi-k2.6", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["thinking"] is False


def test_kimi_k2_6_temperature_clamped_to_0_6(mocker, monkeypatch) -> None:
    """HARD vendor constraint: the Moonshot API rejects any temperature other
    than 0.6 for kimi-k2.6 ('invalid temperature: only 0.6 is allowed for
    this model'). Clamp it regardless of the requested temperature."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("app.llm_factory._ToolLoopChatMoonshot", return_value="kimi")

    build_chat_model("kimi", "kimi-k2.6", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["temperature"] == 0.6


def test_gemini_flash_lite_disables_thinking(mocker, monkeypatch) -> None:
    """Gemini models that support it run reasoning OFF (thinking_budget=0)
    for the apples-to-apples agent comparison."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("app.llm_factory.ChatGoogleGenerativeAI", return_value="gem")

    build_chat_model("gemini", "gemini-3.1-flash-lite-preview", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["thinking_budget"] == 0


def test_gemini_pro_preview_minimizes_thinking_via_level(mocker, monkeypatch) -> None:
    """HARD vendor constraint: gemini-3.1-pro-preview rejects thinking_budget=0
    AND thinking_level='minimal' (both 400 — it has a hard reasoning floor).
    thinking_level='low' IS accepted (verified live) and minimizes reasoning
    depth, so pro-preview participates at minimized — not off — reasoning."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("app.llm_factory.ChatGoogleGenerativeAI", return_value="gem")

    build_chat_model("gemini", "gemini-3.1-pro-preview", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs.get("thinking_level") == "low"
    assert "thinking_budget" not in kwargs


def test_kimi_empty_assistant_content_gets_placeholder(monkeypatch) -> None:
    """Kimi emits pure tool-call turns with content='' but its own API then
    rejects empty assistant content on replay ('message at position N with
    role assistant must not be empty'). The factory's ChatMoonshot subclass
    must rewrite empty assistant content to a non-empty placeholder in the
    outbound payload (tool_calls preserved)."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    from langchain_core.messages import AIMessage, HumanMessage

    from app.config import get_settings

    get_settings.cache_clear()
    llm = build_chat_model("kimi", "kimi-k2.6", temperature=1.0)

    empty_tool_call_ai = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "c1", "type": "tool_call"}],
    )
    payload = llm._get_request_payload([HumanMessage(content="hi"), empty_tool_call_ai], stop=None)
    assistant = [m for m in payload["messages"] if m.get("role") == "assistant"]
    assert assistant, "expected an assistant message in the payload"
    assert assistant[0]["content"], "empty assistant content must be replaced"
    assert assistant[0]["tool_calls"], "tool_calls must be preserved"


def test_kimi_empty_content_only_assistant_gets_placeholder(monkeypatch) -> None:
    """The real agent failure: _prune_for_llm reconstructs Kimi's empty
    tool-call turns as content-only AIMessages (tool_calls dropped, content
    still ''). Kimi rejects THAT too. Empty assistant content must be
    backfilled even when there are no tool_calls."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    from langchain_core.messages import AIMessage, HumanMessage

    from app.config import get_settings

    get_settings.cache_clear()
    llm = build_chat_model("kimi", "kimi-k2.6", temperature=1.0)

    empty_content_only_ai = AIMessage(content="")
    payload = llm._get_request_payload(
        [HumanMessage(content="hi"), empty_content_only_ai], stop=None
    )
    assistant = [m for m in payload["messages"] if m.get("role") == "assistant"]
    assert assistant, "expected an assistant message in the payload"
    assert assistant[0]["content"], "empty content-only assistant must be replaced"
