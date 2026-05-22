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

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_moonshot import ChatMoonshot
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import resolve_llm_api_key

# Placeholder for assistant tool-call turns Kimi emits with empty content.
_EMPTY_ASSISTANT_PLACEHOLDER = "(tool call)"


class _ToolLoopChatMoonshot(ChatMoonshot):
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
                message["content"] = _EMPTY_ASSISTANT_PLACEHOLDER
        return payload


SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "gemini", "deepseek", "kimi", "scripted")

# Hard vendor constraints discovered against the live APIs (2026-05-17).
# Default policy is temp=1.0 + reasoning OFF for an apples-to-apples agent
# comparison, but some models physically reject those settings — clamp per
# model and keep the exact API error here as the rationale.

# Moonshot rejects any temperature != 0.6 for these models:
#   400 "invalid temperature: only 0.6 is allowed for this model"
_KIMI_FORCED_TEMPERATURE: dict[str, float] = {"kimi-k2.6": 0.6}

# Gemini models with a hard reasoning floor — both thinking_budget=0 and
# thinking_level="minimal" yield 400 ("Budget 0 is invalid. This model only
# works in thinking mode"). thinking_level="low" IS accepted and minimizes
# reasoning depth, so these participate at minimized (not off) reasoning.
_GEMINI_THINKING_ONLY: frozenset[str] = frozenset({"gemini-3.1-pro-preview"})


# ─── Scripted provider (EVAL-09 / P4) ────────────────────────────────────────
#
# CI runs the eval matrix with `--llm-provider scripted` so it does not depend
# on any external API key (cf. P4 in PITFALLS.md). The class below is a
# minimal BaseChatModel that pops AIMessages from a per-instance script, with
# a safe fallback that emits one finalize-only AIMessage so the agent graph
# terminates cleanly in a single plan() step (plan -> critique ->
# finalize_as_is -> END).
#
# SCRIPTED_SCENARIOS is the per-scenario script registry. For Phase 3 we keep
# it empty (the matrix runner subprocess-fans-out with --scenario-ids and the
# scripted LLM emits the generic finalize on each cell); a future plan may
# populate scenario-specific tool-call/commit_itinerary trajectories per
# scenario_id. The dict is exposed so callers can override per-instance via
# `scripted_llm.scenario_id` or by passing `scripted` directly with a custom
# scripted list (mirroring the test_chat_functional._ScriptedLLM pattern).
#
# Project-memory invariants honored here:
#   - project_aimessage_tool_call_args_json_safe: all canned tool_call args
#     are dict-shaped, NEVER pydantic models — they survive json.dumps for the
#     EVAL-08 wire-contract guard.
#   - project_full_suite_db_pool_contamination: a finalize-only trajectory
#     keeps stops=[] so no DB-touching scorer fires during the cell run.

_DEFAULT_SCRIPTED_FALLBACK = AIMessage(
    content=("Sorry, I don't have specific information for that scenario in scripted mode."),
    tool_calls=[],
)


class ScriptedChatModel(BaseChatModel):
    """Deterministic, no-network BaseChatModel for CI / matrix scripted runs.

    Pops AIMessages from `scripted` on each `_generate` call. When the list is
    exhausted (or empty), returns a fallback finalize-only AIMessage so the
    agent graph always reaches a clean termination — the matrix runner can
    never deadlock on this LLM.

    See SCRIPTED_SCENARIOS for the per-scenario script registry. The matrix
    runner subprocess-fans-out one cell per (provider, model, scenario_id, n)
    and each subprocess builds its own ScriptedChatModel instance — there is
    no shared state across cells.
    """

    scripted: list[AIMessage] = []
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
        msg = self.scripted.pop(0) if self.scripted else _DEFAULT_SCRIPTED_FALLBACK
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        # The agent graph calls .bind_tools(...) on the LLM. We return self
        # so the binding is a no-op (mirror the _ScriptedLLM in
        # tests/unit/test_chat_functional.py).
        return self


# Per-scenario canned trajectory registry. Phase 3 ships an empty dict and the
# fallback path is sufficient for CI matrix-scripted runs — D-08 isolation
# guarantees one cell per subprocess, and the matrix runner's job is to run
# the harness end-to-end without keys, not to produce baseline-grade outputs.
# Future plans (e.g. populating realistic tool-call trajectories per scenario)
# can append entries here; the existing tests assert the dict's existence and
# type only, not its contents.
SCRIPTED_SCENARIOS: dict[str, list[AIMessage]] = {}


def _build_scripted_chat_model(
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
    # Scripted short-circuits BEFORE resolve_llm_api_key — CI / matrix
    # scripted runs set NO provider keys (EVAL-09 / P4); calling
    # resolve_llm_api_key('scripted') would raise even though we don't need
    # a key.
    if provider == "scripted":
        return _build_scripted_chat_model(chat_model, temperature)
    api_key = resolve_llm_api_key(provider)
    if provider == "openai":
        return ChatOpenAI(model=chat_model, api_key=SecretStr(api_key), temperature=temperature)
    if provider == "gemini":
        gemini_kwargs: dict[str, object] = {}
        if chat_model in _GEMINI_THINKING_ONLY:
            gemini_kwargs["thinking_level"] = "low"  # hard floor; minimize it
        else:
            gemini_kwargs["thinking_budget"] = 0  # reasoning OFF where supported
        return ChatGoogleGenerativeAI(
            model=chat_model,
            google_api_key=SecretStr(api_key),
            temperature=temperature,
            **gemini_kwargs,
        )
    if provider == "deepseek":
        return ChatDeepSeek(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
    if provider == "kimi":
        kimi_temp = _KIMI_FORCED_TEMPERATURE.get(chat_model, temperature)
        return _ToolLoopChatMoonshot(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=kimi_temp,
            thinking=False,
        )
    raise ValueError(f"Unsupported llm_provider: {llm_provider}")
