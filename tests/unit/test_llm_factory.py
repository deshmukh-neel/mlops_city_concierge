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
    # PROV-03 (Phase 9 / Plan 09-03) added "anthropic" to SUPPORTED_PROVIDERS,
    # so the original unknown-provider sentinel (anthropic) is now supported.
    # Use a still-unsupported provider name to keep the test exercising the
    # "factory enforces its own contract before resolve_llm_api_key" path that
    # the original CI-vs-local regression (passed locally with a .env key,
    # failed in CI without one) was about.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # If the factory ever calls resolve_llm_api_key for an unsupported
    # provider, that's the bug — it must reject via SUPPORTED_PROVIDERS first.
    mocker.patch(
        "app.llm_factory.resolve_llm_api_key",
        side_effect=AssertionError("must reject before key resolution"),
    )
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        build_chat_model("definitely-not-a-provider", "x", temperature=0.0)


def test_supported_providers_is_the_contract() -> None:
    # Plan 03-05 extends the contract with 'scripted' (EVAL-09 / P4) — the
    # CI-safe deterministic branch that needs no API key and makes no network
    # calls. Plan 09-03 / PROV-03 appends 'anthropic' for the first-time
    # Claude wiring; both anthropic_api_key Setting + resolve_llm_api_key
    # branch were pre-existing per D-09-05, so this is a one-line tuple
    # append (no app/config.py edit needed).
    assert SUPPORTED_PROVIDERS == (
        "openai",
        "gemini",
        "deepseek",
        "kimi",
        "anthropic",
        "scripted",
    )


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


def test_deepseek_chat_keeps_thinking_disabled(mocker, monkeypatch) -> None:
    """PROV-02 / T-09-02-T4 regression guard: the model-level carve-out for
    deepseek-reasoner must NOT spill onto deepseek-chat. The default
    extra_body={'thinking': {'type': 'disabled'}} policy stays for every
    DeepSeek model not in `_DEEPSEEK_REASONER_THINKING_ENABLED`.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("app.llm_factory.ChatDeepSeek", return_value="ds")

    build_chat_model("deepseek", "deepseek-chat", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_reasoner_enables_thinking(mocker, monkeypatch) -> None:
    """PROV-02 / D-09-04: `deepseek-reasoner` is the model-level carve-out from
    the thinking-disabled default. The factory must construct ChatDeepSeek with
    `extra_body={'thinking': {'type': 'enabled'}}` so the API emits
    `reasoning_content` for `DeepSeekReasonerAdapter` to round-trip.

    Mirrors the `_KIMI_FORCED_TEMPERATURE` / `_GEMINI_THINKING_ONLY` per-model
    policy pattern at `app/llm_factory.py:65-78`.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("app.llm_factory.ChatDeepSeek", return_value="ds")

    build_chat_model("deepseek", "deepseek-reasoner", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    # temp=1.0 stays — feedback_temp1_reasoning_off_all_models general rule
    # is honored; the carve-out is thinking ENABLED for the reasoner family
    # ONLY, not a temperature change.
    assert kwargs["temperature"] == 1.0


def test_build_chat_model_anthropic_returns_chatanthropic_with_thinking_enabled(
    mocker, monkeypatch
) -> None:
    """PROV-03 / D-09-06: `build_chat_model("anthropic", "claude-sonnet-4-6", 1.0)`
    constructs `ChatAnthropic` with `temperature=1.0`, `model="claude-sonnet-4-6"`,
    and `thinking={"type": "enabled", "budget_tokens": 4096}` per the carve-out
    from `feedback_temp1_reasoning_off_all_models` documented in
    `_ANTHROPIC_THINKING_BUDGET`.

    Patches `langchain_anthropic.ChatAnthropic` at its import site (the lazy
    import inside the anthropic branch) and `resolve_llm_api_key` so the test
    needs no real `ANTHROPIC_API_KEY`.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    # Patch at the LAZY-import site — the anthropic branch does
    # `from langchain_anthropic import ChatAnthropic` INSIDE the branch, so
    # patching `app.llm_factory.ChatAnthropic` wouldn't intercept anything
    # (the name isn't bound at module level by design — that's what keeps
    # langchain-anthropic optional for environments that never construct an
    # Anthropic model).
    cls = mocker.patch("langchain_anthropic.ChatAnthropic", return_value="anthropic-llm")

    out = build_chat_model("anthropic", "claude-sonnet-4-6", temperature=1.0)

    assert out == "anthropic-llm"
    cls.assert_called_once()
    _, kwargs = cls.call_args
    assert kwargs["model_name"] == "claude-sonnet-4-6"
    assert kwargs["temperature"] == 1.0
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_anthropic_branch_sets_max_tokens_above_thinking_budget(mocker, monkeypatch) -> None:
    """PROV-03 live-probe correction (2026-06-05): the Anthropic Messages API
    requires `max_tokens > thinking.budget_tokens`. Without an explicit
    `max_tokens`, langchain-anthropic's default (1024) is ≤ the 4096 thinking
    budget and every live call 400s with `max_tokens must be greater than
    thinking.budget_tokens` (observed request_id req_011CbkjwQB58bHtNcShLSV59).

    Factory MUST pass `max_tokens` explicitly, MUST default to 8192 for
    claude-sonnet-4-6 (2× the 4096 budget, leaves 4096 for visible reply
    text), and MUST be strictly > the thinking budget.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("langchain_anthropic.ChatAnthropic", return_value="anthropic-llm")

    build_chat_model("anthropic", "claude-sonnet-4-6", temperature=1.0)

    _, kwargs = cls.call_args
    # max_tokens is passed (not relying on langchain-anthropic default).
    assert "max_tokens" in kwargs, (
        "anthropic branch must set max_tokens explicitly — without it, "
        "langchain-anthropic's default (1024) is ≤ the 4096 thinking budget "
        "and the API 400s with `max_tokens must be greater than "
        "thinking.budget_tokens` on every call."
    )
    # Default matches the _ANTHROPIC_MAX_TOKENS dict for claude-sonnet-4-6.
    assert kwargs["max_tokens"] == 8192
    # Strictly > thinking.budget_tokens — the Anthropic API constraint.
    assert kwargs["max_tokens"] > kwargs["thinking"]["budget_tokens"], (
        "max_tokens must be strictly greater than thinking.budget_tokens "
        "(Anthropic API constraint)."
    )


def test_anthropic_branch_clamps_temperature_to_1_0_when_thinking_enabled(
    mocker, monkeypatch
) -> None:
    """PROV-03 live-run matrix correction (2026-06-05): when thinking is
    enabled, Anthropic's API rejects any temperature != 1.0 with
    `temperature may only be set to 1 when thinking is enabled`
    (observed request_id req_011CbkpXMhQfXXSArRAzgVMP). The eval matrix
    runner does not pass --temperature for refinement cells, so it
    defaults to 0.0 — every anthropic cell in the local empirical gate
    400'd. Factory must clamp temperature to 1.0 unconditionally on the
    anthropic branch (D-09-06 already mandates temp=1.0; this enforces it
    mechanically so callers don't have to know the API constraint).

    Mirrors the `_KIMI_FORCED_TEMPERATURE` clamp pattern at
    `app/llm_factory.py:_KIMI_FORCED_TEMPERATURE`.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("langchain_anthropic.ChatAnthropic", return_value="anthropic-llm")

    # Caller passes temperature=0.0 (e.g. eval_matrix default); factory clamps to 1.0.
    build_chat_model("anthropic", "claude-sonnet-4-6", temperature=0.0)

    _, kwargs = cls.call_args
    assert kwargs["temperature"] == 1.0, (
        "Anthropic branch must clamp temperature to 1.0 when thinking is "
        "enabled — the API rejects any other value with a 400."
    )


def test_anthropic_branch_max_tokens_falls_back_for_unknown_model(mocker, monkeypatch) -> None:
    """The `_ANTHROPIC_MAX_TOKENS.get(chat_model, 8192)` fallback applies for any
    Claude model not explicitly listed (e.g. a future Opus / Haiku build) so
    new model strings don't silently regress to langchain-anthropic's too-low
    default. 8192 (2× the default 4096 budget) keeps the API invariant
    satisfied for any model that also falls through `_ANTHROPIC_THINKING_BUDGET`.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("langchain_anthropic.ChatAnthropic", return_value="anthropic-llm")

    build_chat_model("anthropic", "claude-opus-future", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["max_tokens"] == 8192
    assert kwargs["max_tokens"] > kwargs["thinking"]["budget_tokens"]


def test_build_chat_model_anthropic_uses_default_budget_when_model_not_in_dict(
    mocker, monkeypatch
) -> None:
    """PROV-03 / D-09-06: the `_ANTHROPIC_THINKING_BUDGET.get(chat_model, 4096)`
    fallback applies for any Claude model not explicitly listed (e.g. a future
    Opus / Haiku build). 4096 is the Claude's-Discretion default; Phase 10
    baseline regen may tune this per model.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    cls = mocker.patch("langchain_anthropic.ChatAnthropic", return_value="anthropic-llm")

    build_chat_model("anthropic", "claude-opus-future", temperature=1.0)

    _, kwargs = cls.call_args
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_supported_providers_contains_anthropic() -> None:
    """PROV-03 / D-09-05: `anthropic` is part of `SUPPORTED_PROVIDERS` after
    Plan 09-03. This is the contract the dict-comp at
    `app/agent/adapters/__init__.py:121` keys off — adding anthropic here
    auto-extends the `ADAPTERS` registry (then the Plan 09-03 swap line
    replaces the NoOp entry with `AnthropicAdapter()`).
    """
    assert "anthropic" in SUPPORTED_PROVIDERS


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
        tool_calls=[
            {
                "name": "search",
                "args": {"q": "x"},
                "id": "ChIJtest_c1_aaaaaaaa",
                "type": "tool_call",
            }
        ],
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


# ─── Plan 03-05 Task 1: scripted provider (EVAL-09 / P4) ─────────────────────


def test_scripted_provider_is_in_supported_providers() -> None:
    """SUPPORTED_PROVIDERS now includes 'scripted' — the CI-safe deterministic
    branch that needs no API key and makes no network calls."""
    from app.llm_factory import SUPPORTED_PROVIDERS

    assert "scripted" in SUPPORTED_PROVIDERS


def test_build_chat_model_scripted_needs_no_env_vars(monkeypatch) -> None:
    """EVAL-09 / P4: `build_chat_model('scripted', ...)` MUST succeed without
    any provider API key set. The CI matrix run sets no keys; if scripted
    leaked into resolve_llm_api_key, this test would crash."""
    for key in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "MOONSHOT_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    from app.llm_factory import build_chat_model

    llm = build_chat_model("scripted", "placeholder", temperature=0.0)
    assert llm is not None
    # Must satisfy the BaseChatModel contract so the agent graph can bind tools.
    from langchain_core.language_models import BaseChatModel

    assert isinstance(llm, BaseChatModel)


def test_scripted_chat_model_is_importable_directly() -> None:
    """`ScriptedChatModel` is importable from app.llm_factory so tests can
    construct it with custom scripted messages."""
    from app.llm_factory import ScriptedChatModel

    assert ScriptedChatModel is not None


def test_scripted_chat_model_returns_finalize_only_aimessage(monkeypatch) -> None:
    """Default-fallback script for unknown scenarios: emit one AIMessage with
    NO tool_calls, so the agent graph reaches a clean termination in one
    plan() step (plan -> critique -> finalize_as_is -> END)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    from langchain_core.messages import AIMessage, HumanMessage

    from app.llm_factory import build_chat_model

    llm = build_chat_model("scripted", "placeholder", temperature=0.0)
    response = llm.invoke([HumanMessage(content="anything")])
    assert isinstance(response, AIMessage)
    assert not response.tool_calls, "fallback script must NOT emit tool_calls"


def test_scripted_chat_model_bind_tools_returns_self() -> None:
    """The agent graph calls .bind_tools(...) on the LLM before invoking it.
    Scripted must support this (mirror the _ScriptedLLM pattern in
    test_chat_functional.py) so the graph wires up without crashing."""
    from app.llm_factory import build_chat_model

    llm = build_chat_model("scripted", "placeholder", temperature=0.0)
    bound = llm.bind_tools([])
    assert bound is llm


def test_scripted_scenarios_dict_exposed() -> None:
    """`SCRIPTED_SCENARIOS` is the per-scenario script registry — the matrix
    runner can route a specific scenario_id to a richer canned trajectory if
    needed. For Phase 3 we ship a minimal default; the dict's existence is
    the API contract."""
    from app.llm_factory import SCRIPTED_SCENARIOS

    assert isinstance(SCRIPTED_SCENARIOS, dict)


def test_supported_providers_includes_scripted() -> None:
    """SUPPORTED_PROVIDERS contract update — Plan 09-03 also adds 'anthropic'."""
    from app.llm_factory import SUPPORTED_PROVIDERS

    assert SUPPORTED_PROVIDERS == (
        "openai",
        "gemini",
        "deepseek",
        "kimi",
        "anthropic",
        "scripted",
    )


# ─── Plan 03-09 Task 1: CR-02 (fresh AIMessage) + IN-05 (self-documenting) ──


def test_scripted_chat_model_returns_fresh_aimessage_each_call() -> None:
    """CR-02 (BLOCKER) regression guard. Two `_generate` calls on an
    empty-scripted ScriptedChatModel must return AIMessages that are NOT the
    same Python object — otherwise LangGraph's `add_messages` reducer
    deduplicates them by identity and the agent's revision loop spins until
    `max_steps`. The previous module-level `_DEFAULT_SCRIPTED_FALLBACK`
    singleton failed this test; the fix constructs a fresh AIMessage per call.
    """
    from app.llm_factory import ScriptedChatModel

    m = ScriptedChatModel()
    first = m._generate(messages=[]).generations[0].message
    second = m._generate(messages=[]).generations[0].message
    assert first is not second, (
        "ScriptedChatModel._generate returned the same AIMessage instance twice "
        "— LangGraph add_messages will dedupe these by identity and the agent "
        "revision loop will spin to max_steps. Construct a fresh AIMessage per call."
    )


def test_scripted_chat_model_fallback_content_documents_ci_mode() -> None:
    """IN-05 regression guard. The fallback content must self-document as
    deterministic CI output so PR reviewers reading `summary.json` don't
    misread it as a real model failure. The string cites both the marker
    `[SCRIPTED CI MODE]` and the originating script `scripts/eval_matrix.py`.
    """
    from app.llm_factory import ScriptedChatModel

    m = ScriptedChatModel()
    msg = m._generate(messages=[]).generations[0].message
    assert "[SCRIPTED CI MODE]" in msg.content
    assert "scripts/eval_matrix.py" in msg.content


def test_scripted_chat_model_consumes_scripted_list_when_nonempty() -> None:
    """Existing pop-from-list-then-fallback semantics — unchanged behavior.
    When `scripted` is non-empty the first call pops it; subsequent calls fall
    back to the self-documenting `[SCRIPTED CI MODE]` placeholder.
    """
    from langchain_core.messages import AIMessage

    from app.llm_factory import ScriptedChatModel

    m = ScriptedChatModel(scripted=[AIMessage(content="hello")])
    first = m._generate(messages=[]).generations[0].message
    second = m._generate(messages=[]).generations[0].message
    assert first.content == "hello"
    assert "[SCRIPTED CI MODE]" in second.content


# ─── Plan 03-13 / IN-03: scripted default uses default_factory ───────────────


def test_scripted_chat_model_default_scripted_list_is_per_instance() -> None:
    """IN-03 consistency guard. `scripted` defaults to an empty list, but two
    fresh ScriptedChatModel instances must NOT share the same underlying list
    object — otherwise pop()s on one instance leak into the other and break
    the matrix runner's one-cell-per-subprocess isolation contract.

    Pydantic v2 deep-copies a `list[AIMessage] = []` class-level default per
    instance today, so this test currently passes either way; the test exists
    to lock the contract against a future refactor (Pydantic version bump or
    base-class swap) that flips that semantic.
    """
    from app.llm_factory import ScriptedChatModel

    a = ScriptedChatModel()
    b = ScriptedChatModel()
    assert a.scripted is not b.scripted, (
        "ScriptedChatModel.scripted must be per-instance; a shared default "
        "would leak pop()s across the matrix runner's parallel cells."
    )


# ─── Phase 9 / WR-01: OpenAIReasoningChatModel._lift_reasoning_blocks ────────
#
# The lift is the sole bridge between the OpenAI Responses-API content-block
# shape and the documented additional_kwargs["reasoning_content"] contract
# that OpenAIReasoningAdapter reads. PROV-01 SHIPPED-WITH-GAP at 0.4 vs ≥ 0.6
# because the conformance harness only planted additional_kwargs directly —
# the actual lift path was never asserted. These tests close that gap so a
# future langchain-openai version bump that changes content shape will
# fail loudly in unit tests instead of silently no-op'ing in production.


def test_openai_reasoning_chat_model_lifts_reasoning_blocks_into_kwargs() -> None:
    """WR-01 Test 1: `_lift_reasoning_blocks` copies `{"type": "reasoning"}`
    blocks from `AIMessage.content` to `additional_kwargs["reasoning_content"]`
    when content is a list. This is the path the gpt-5 family Responses-API
    response goes through to surface reasoning state on the documented
    adapter-contract key.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.llm_factory import OpenAIReasoningChatModel

    msg = AIMessage(
        content=[
            {"type": "reasoning", "summary": "thought process"},
            {"type": "text", "text": "the answer"},
        ]
    )
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    lifted = OpenAIReasoningChatModel._lift_reasoning_blocks(result)
    out_msg = lifted.generations[0].message
    # Only the reasoning block is lifted — the text block stays in content.
    assert out_msg.additional_kwargs["reasoning_content"] == [
        {"type": "reasoning", "summary": "thought process"},
    ]


def test_openai_reasoning_chat_model_lift_is_noop_on_str_content() -> None:
    """WR-01 Test 2: tool-call responses, refusals, and the Chat-Completions
    fallback shape all carry `str` content. The lift must NOT crash on this
    path — `isinstance(content, list)` short-circuits before the dict scan.
    No reasoning_content is added when none exists.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.llm_factory import OpenAIReasoningChatModel

    msg = AIMessage(content="plain string response")
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    lifted = OpenAIReasoningChatModel._lift_reasoning_blocks(result)
    assert "reasoning_content" not in lifted.generations[0].message.additional_kwargs


def test_openai_reasoning_chat_model_lift_isolates_inner_block_dicts() -> None:
    """WR-01 Test 3 + WR-05 contract: the docstring promises "downstream
    mutation of the additional_kwargs entry cannot reach back into `content`".
    Plain `list(...)` copy of the outer list is NOT enough — the inner block
    dicts must also be copied, otherwise mutating
    `additional_kwargs["reasoning_content"][0]["summary"]` mutates the same
    dict still aliased in `msg.content`. Mirrors the per-block-dict copy in
    `AnthropicAdapter.capture_reasoning_state`.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.llm_factory import OpenAIReasoningChatModel

    msg = AIMessage(content=[{"type": "reasoning", "summary": "original"}])
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    lifted = OpenAIReasoningChatModel._lift_reasoning_blocks(result)
    out_msg = lifted.generations[0].message
    # Mutate the lifted inner-dict — content's inner dict must NOT be aliased.
    out_msg.additional_kwargs["reasoning_content"][0]["summary"] = "TAMPERED"
    assert msg.content[0]["summary"] == "original", (
        "inner-dict alias leaked into msg.content — WR-05 mitigation broke. "
        f"Got: {msg.content[0]!r}"
    )


def test_openai_reasoning_chat_model_lift_noop_when_no_reasoning_blocks() -> None:
    """WR-01 Test 4: list content with only text/tool_use blocks (no
    `{"type": "reasoning"}` entry) yields no `reasoning_content` kwarg —
    matches the D-08-02 contract default for non-reasoning responses.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.llm_factory import OpenAIReasoningChatModel

    msg = AIMessage(content=[{"type": "text", "text": "answer"}])
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    lifted = OpenAIReasoningChatModel._lift_reasoning_blocks(result)
    assert "reasoning_content" not in lifted.generations[0].message.additional_kwargs


def test_openai_reasoning_chat_model_lift_handles_non_aimessage_generations() -> None:
    """WR-01 Test 5: defensive `isinstance(msg, AIMessage)` guard skips
    non-AIMessage generations without crashing. langchain-openai always
    yields AIMessage today; the guard documents the contract.
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.llm_factory import OpenAIReasoningChatModel

    # HumanMessage is not an AIMessage — lift should skip it without raising.
    msg = HumanMessage(content="not an AI message")
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    lifted = OpenAIReasoningChatModel._lift_reasoning_blocks(result)
    # Returned unchanged (identity preserved).
    assert lifted is result


def test_openai_reasoning_chat_model_generate_wires_through_lift(mocker) -> None:
    """WR-01 Test 6 (sync parity): `_generate` calls `_lift_reasoning_blocks`
    on the result returned by the parent `ChatOpenAI._generate`. Verifies
    the gpt-5 lift hook is wired into the sync codepath.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.llm_factory import OpenAIReasoningChatModel

    parent_result = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content=[
                        {"type": "reasoning", "summary": "lifted"},
                        {"type": "text", "text": "ok"},
                    ]
                )
            )
        ]
    )
    mocker.patch(
        "langchain_openai.ChatOpenAI._generate",
        return_value=parent_result,
    )
    model = OpenAIReasoningChatModel(model="gpt-5-mini", api_key="test")
    out = model._generate(messages=[])
    out_msg = out.generations[0].message
    assert out_msg.additional_kwargs["reasoning_content"] == [
        {"type": "reasoning", "summary": "lifted"},
    ]


async def test_openai_reasoning_chat_model_agenerate_wires_through_lift(mocker) -> None:
    """WR-01 Test 7 (async parity): `_agenerate` calls `_lift_reasoning_blocks`
    on the result returned by the parent `ChatOpenAI._agenerate`. Without
    this wire, async tool-loop traffic would silently drop reasoning state
    while sync traffic preserved it — exactly the asymmetric-coverage
    failure mode that bit Anthropic in 09-03.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.llm_factory import OpenAIReasoningChatModel

    parent_result = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content=[
                        {"type": "reasoning", "summary": "lifted-async"},
                        {"type": "text", "text": "ok"},
                    ]
                )
            )
        ]
    )

    async def _fake_agenerate(self, *args, **kwargs):  # noqa: ANN001
        return parent_result

    mocker.patch(
        "langchain_openai.ChatOpenAI._agenerate",
        _fake_agenerate,
    )
    model = OpenAIReasoningChatModel(model="gpt-5-mini", api_key="test")
    out = await model._agenerate(messages=[])
    out_msg = out.generations[0].message
    assert out_msg.additional_kwargs["reasoning_content"] == [
        {"type": "reasoning", "summary": "lifted-async"},
    ]
