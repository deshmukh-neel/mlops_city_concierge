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

from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_moonshot import ChatMoonshot
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

from app.config import resolve_llm_api_key

# Placeholder for assistant tool-call turns Kimi emits with empty content.
EMPTY_ASSISTANT_PLACEHOLDER = "(tool call)"


class ToolLoopChatMoonshot(ChatMoonshot):
    """ChatMoonshot that survives the agent tool loop.

    Kimi emits pure tool-call assistant turns with `content=""`. Its own API
    then rejects ANY empty assistant content on replay ("the message at
    position N with role 'assistant' must not be empty") — both the original
    tool-call turn AND the content-only reconstruction the agent's history
    pruner produces from it (which drops tool_calls, leaving empty content).
    Backfill a non-empty placeholder on every outbound assistant message that
    has empty content; tool_calls, when present, are preserved untouched.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for message in payload.get("messages", []):
            if message.get("role") == "assistant" and not (message.get("content") or "").strip():
                message["content"] = EMPTY_ASSISTANT_PLACEHOLDER
        return payload


class OpenAIReasoningChatModel(ChatOpenAI):
    """`ChatOpenAI` subclass for the gpt-5 family that surfaces reasoning state
    on `AIMessage.additional_kwargs["reasoning_content"]`.

    The gpt-5 family uses the Responses API so reasoning blocks survive the
    agent tool loop. gpt-4o-mini stays on plain `ChatOpenAI` to preserve the
    existing Chat Completions message shape.
    """

    @staticmethod
    def lift_reasoning_blocks(result: ChatResult) -> ChatResult:
        """Copy any `{"type": "reasoning", ...}` content blocks into
        `AIMessage.additional_kwargs["reasoning_content"]`.

        We COPY rather than move so LangChain's Responses-API outbound
        serializer (which round-trips reasoning blocks in `content`) keeps
        working on the wire while the adapter contract still has access via
        the documented `additional_kwargs` path.

        Use per-block dict copies instead of `list(...)`. Plain list-copy
        preserves inner-dict aliasing: a downstream consumer that mutates
        `msg.additional_kwargs["reasoning_content"][0]["summary"]` would
        mutate the same dict still living in `msg.content`. Per-block
        shallow copies keep the two views isolated.
        """
        for gen in result.generations:
            msg = gen.message
            if not isinstance(msg, AIMessage):
                continue
            content = msg.content
            if not isinstance(content, list):
                continue
            reasoning_blocks = [
                dict(block)
                for block in content
                if isinstance(block, dict) and block.get("type") == "reasoning"
            ]
            if reasoning_blocks:
                msg.additional_kwargs["reasoning_content"] = reasoning_blocks
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self.lift_reasoning_blocks(result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self.lift_reasoning_blocks(result)


SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "openai",
    "gemini",
    "deepseek",
    "kimi",
    "anthropic",
    "scripted",
)

# OpenAI model families wired through `OpenAIReasoningChatModel`. Keep this
# scoped to reasoning models; routing regular chat models through the Responses
# API changes their message shape.


def is_openai_reasoning_model(chat_model: str) -> bool:
    """Return True when `chat_model` needs the OpenAIReasoningChatModel subclass."""
    return chat_model.startswith("gpt-5")


# Hard vendor constraints discovered against the live APIs (2026-05-17).
# Default policy is temp=1.0 + reasoning OFF for an apples-to-apples agent
# comparison, but some models physically reject those settings — clamp per
# model and keep the exact API error here as the rationale.

# Moonshot rejects any temperature != 0.6 for these models:
#   400 "invalid temperature: only 0.6 is allowed for this model"
KIMI_FORCED_TEMPERATURE: dict[str, float] = {"kimi-k2.6": 0.6}

# Gemini models with a hard reasoning floor — both thinking_budget=0 and
# thinking_level="minimal" yield 400 ("Budget 0 is invalid. This model only
# works in thinking mode"). thinking_level="low" IS accepted and minimizes
# reasoning depth, so these participate at minimized (not off) reasoning.
GEMINI_THINKING_ONLY: frozenset[str] = frozenset({"gemini-3.1-pro-preview"})

# DeepSeek models that should run with thinking enabled so the adapter has
# reasoning_content to round-trip. Keep the carve-out exact, not prefix-based,
# so regular DeepSeek chat models stay on the thinking-disabled policy.
DEEPSEEK_REASONER_THINKING_ENABLED: frozenset[str] = frozenset({"deepseek-reasoner"})

# Claude models that run with thinking enabled. Sonnet without thinking emits
# no thinking_blocks, so there is nothing for the Anthropic adapter to preserve.
ANTHROPIC_THINKING_BUDGET: dict[str, int] = {"claude-sonnet-4-6": 4096}

# Anthropic requires max_tokens > thinking.budget_tokens. Pin max_tokens high
# enough to leave room for the visible reply after the thinking budget is used.
ANTHROPIC_MAX_TOKENS: dict[str, int] = {"claude-sonnet-4-6": 8192}


# Scripted provider
#
# CI runs the eval matrix with `--llm-provider scripted` so it does not depend
# on any external API key. The class below is a minimal BaseChatModel that
# pops AIMessages from a per-instance script, with a safe fallback that emits
# one finalize-only AIMessage so the agent graph terminates cleanly.
#
# The fallback content begins with the `[SCRIPTED CI MODE]` marker and cites
# `scripts/eval_matrix.py` so reviewers can recognize deterministic placeholder
# output in CI reports.
#
# SCRIPTED_SCENARIOS is the per-scenario script registry. It is empty by
# default; callers can pass a custom script list or populate a scenario entry.


class ScriptedChatModel(BaseChatModel):
    """Deterministic, no-network BaseChatModel for CI / matrix scripted runs.

    Pops AIMessages from `scripted` on each `_generate` call. When the list is
    exhausted (or empty), returns a fallback finalize-only AIMessage so the
    agent graph always reaches a clean termination — the matrix runner can
    never deadlock on this LLM.

    When the scripted list is empty, a freshly-constructed `AIMessage` is
    returned on each call. Reusing a module-level singleton lets LangGraph's
    `add_messages` reducer deduplicate consecutive fallbacks by identity.

    See SCRIPTED_SCENARIOS for the per-scenario script registry. The matrix
    runner subprocess-fans-out one cell per (provider, model, scenario_id, n)
    and each subprocess builds its own ScriptedChatModel instance — there is
    no shared state across cells.
    """

    scripted: list[AIMessage] = Field(default_factory=list)
    scenario_id: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.scripted:
            msg = self.scripted.pop(0)
        else:
            # Construct a fresh AIMessage on every call so LangGraph's
            # `add_messages` reducer does not dedupe consecutive fallbacks.
            msg = AIMessage(
                content=(
                    "[SCRIPTED CI MODE] Deterministic no-network finalize; "
                    "see scripts/eval_matrix.py."
                ),
                tool_calls=[],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        # The agent graph calls .bind_tools(...) on the LLM. We return self
        # so the binding is a no-op.
        return self


# Per-scenario canned trajectory registry. The fallback path is enough for
# no-key CI matrix runs; callers can add realistic tool-call trajectories here
# when a scenario needs them.
SCRIPTED_SCENARIOS: dict[str, list[AIMessage]] = {}


def build_scripted_chat_model(
    chat_model: str, temperature: float, scenario_id: str | None = None
) -> ScriptedChatModel:
    """Construct a ScriptedChatModel for a given scenario_id (or fallback).

    `chat_model` and `temperature` are accepted for signature parity with the
    other build_chat_model branches; scripted mode does not consult either
    (the chat_model becomes an informational label in the eval report).
    """
    script = list(SCRIPTED_SCENARIOS.get(scenario_id or "", []))
    return ScriptedChatModel(scripted=script, scenario_id=scenario_id)


def build_chat_model(llm_provider: str, chat_model: str, temperature: float) -> BaseChatModel:
    """Construct a chat model for `llm_provider`. Raises ValueError for
    unsupported providers, RuntimeError if the provider's API key is missing
    (both via resolve_llm_api_key)."""
    provider = llm_provider.lower()
    # Enforce the factory's own contract BEFORE key resolution. resolve_llm_api_key
    # knows other providers (e.g. anthropic) and would reject them via a
    # missing-key RuntimeError instead of the unsupported-provider ValueError
    # callers expect — and only when the key happens to be absent (passes
    # locally with a .env key, fails in CI without one).
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")
    # Scripted short-circuits before key resolution; it intentionally runs
    # without provider credentials.
    if provider == "scripted":
        return build_scripted_chat_model(chat_model, temperature)
    api_key = resolve_llm_api_key(provider)
    if provider == "openai":
        if is_openai_reasoning_model(chat_model):
            # gpt-5 routes through the Responses API so reasoning blocks
            # survive the agent's tool loop.
            return OpenAIReasoningChatModel(
                model=chat_model,
                api_key=SecretStr(api_key),
                temperature=temperature,
                use_responses_api=True,
            )
        return ChatOpenAI(model=chat_model, api_key=SecretStr(api_key), temperature=temperature)
    if provider == "gemini":
        gemini_kwargs: dict[str, object] = {}
        if chat_model in GEMINI_THINKING_ONLY:
            gemini_kwargs["thinking_level"] = "low"  # hard floor; minimize it
        else:
            gemini_kwargs["thinking_budget"] = 0  # reasoning OFF where supported
        return ChatGoogleGenerativeAI(
            model=chat_model,
            google_api_key=SecretStr(api_key),
            temperature=temperature,
            **gemini_kwargs,
        )
    if provider == "anthropic":
        # Lazy import keeps the dependency optional for environments that
        # never construct an Anthropic model.
        from langchain_anthropic import ChatAnthropic

        # max_tokens MUST be > thinking.budget_tokens or Anthropic returns 400
        # ("max_tokens must be greater than thinking.budget_tokens"). The
        # langchain-anthropic default is too low (1024); pin explicitly here.
        budget_tokens = ANTHROPIC_THINKING_BUDGET.get(chat_model, 4096)
        max_tokens = ANTHROPIC_MAX_TOKENS.get(chat_model, 8192)
        # Anthropic requires temperature=1.0 when thinking is enabled.
        anthropic_temperature = 1.0
        anthropic_kwargs: dict[str, Any] = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            },
            "max_tokens": max_tokens,
        }
        return ChatAnthropic(
            model_name=chat_model,
            api_key=SecretStr(api_key),
            temperature=anthropic_temperature,
            **anthropic_kwargs,
        )
    if provider == "deepseek":
        # Default to thinking-disabled for tool-loop stability. Enable it only
        # for models whose adapter round-trips reasoning_content.
        extra_body: dict[str, Any] = {"thinking": {"type": "disabled"}}
        if chat_model in DEEPSEEK_REASONER_THINKING_ENABLED:
            extra_body = {"thinking": {"type": "enabled"}}
        return ChatDeepSeek(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=temperature,
            extra_body=extra_body,
        )
    if provider == "kimi":
        kimi_temp = KIMI_FORCED_TEMPERATURE.get(chat_model, temperature)
        return ToolLoopChatMoonshot(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=kimi_temp,
            thinking=False,
        )
    raise ValueError(f"Unsupported llm_provider: {llm_provider}")
