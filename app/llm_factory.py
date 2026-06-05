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


class OpenAIReasoningChatModel(ChatOpenAI):
    """`ChatOpenAI` subclass for the gpt-5 family that surfaces reasoning state
    on `AIMessage.additional_kwargs["reasoning_content"]` so the Phase 9
    `OpenAIReasoningAdapter` (and the Phase 8 ProviderAdapter contract more
    broadly) can round-trip it across agent turns.

    **Why a subclass exists (Phase 9 / PROV-01 / D-09-03 Path B):**

    The probe at `.planning/phases/09-.../09-PROV-01-PROBE.md` confirmed that
    against `langchain-openai==1.2.2`'s **Chat Completions** wrapper,
    `gpt-5-mini` returns ONLY a `reasoning_tokens` counter in `usage` — the
    `additional_kwargs` field is bare (only `refusal`), and the reasoning
    text never reaches `AIMessage`. This matches the historical diagnosis in
    memory `project_agent_loses_reasoning_state_all_providers` (the
    pre-Phase-9 architectural bug Phase 8 contract + Phase 9 impls fix).

    The fix is to switch the gpt-5 family onto OpenAI's **Responses API**
    (`use_responses_api=True`), which DOES expose reasoning items as content
    blocks of `type: "reasoning"`. LangChain's Responses-API serializer
    (`_construct_responses_api_input`) round-trips reasoning blocks natively
    on the wire when they remain in `AIMessage.content`. Concurrently we
    surface a copy of those blocks on `additional_kwargs["reasoning_content"]`
    so the Phase 8 adapter contract (which reads `additional_kwargs`) sees
    them through the documented interface — that copy is what
    `OpenAIReasoningAdapter.capture_reasoning_state` reads.

    **gpt-4o-mini stays out (CLAUDE.md / v2.0 anchor):** dispatch in
    `build_chat_model` only routes `chat_model.startswith("gpt-5")` here.
    `gpt-4o-mini` continues to use plain `ChatOpenAI` — its
    Chat-Completions/str-content shape is what the v2.0 production anchor
    runs and must not regress.
    """

    @staticmethod
    def _lift_reasoning_blocks(result: ChatResult) -> ChatResult:
        """Copy any `{"type": "reasoning", ...}` content blocks into
        `AIMessage.additional_kwargs["reasoning_content"]`.

        We COPY rather than move so LangChain's Responses-API outbound
        serializer (which round-trips reasoning blocks in `content`) keeps
        working on the wire while the adapter contract still has access via
        the documented `additional_kwargs` path. The list is shallow-copied
        so downstream mutation of the additional_kwargs entry cannot reach
        back into `content`.
        """
        for gen in result.generations:
            msg = gen.message
            if not isinstance(msg, AIMessage):
                continue
            content = msg.content
            if not isinstance(content, list):
                continue
            reasoning_blocks = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "reasoning"
            ]
            if reasoning_blocks:
                msg.additional_kwargs["reasoning_content"] = list(reasoning_blocks)
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._lift_reasoning_blocks(result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._lift_reasoning_blocks(result)


SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "openai",
    "gemini",
    "deepseek",
    "kimi",
    "anthropic",
    "scripted",
)

# Phase 9 / PROV-01 (D-09-03): OpenAI model families that should be wired
# through `OpenAIReasoningChatModel` so reasoning content survives the agent's
# tool loop. The v2.0 anchor `gpt-4o-mini` is intentionally EXCLUDED — it is
# not a reasoning model and routing it through the Responses-API subclass
# would change content shape on the prod anchor path. Currently scoped to
# names starting with "gpt-5"; extend explicitly here if future gpt-5
# sibling models (mini, nano, etc.) need the same wiring.


def _is_openai_reasoning_model(chat_model: str) -> bool:
    """Return True when `chat_model` needs the OpenAIReasoningChatModel subclass.

    Phase 9 / PROV-01 scope: gpt-5 family only. CLAUDE.md locked v2.0 anchor
    rule — gpt-4o-mini MUST keep plain ChatOpenAI (no Responses-API switch).
    """
    return chat_model.startswith("gpt-5")


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

# PROV-02 (D-09-04): DeepSeek models that should run with thinking ENABLED
# so they emit reasoning_content for DeepSeekReasonerAdapter to round-trip.
# Default policy (above) is reasoning OFF; reasoner family is the carve-out
# because the whole point of the model is its reasoning trace. The
# carve-out is scoped to this frozenset lookup (NOT a startswith match) so
# the existing deepseek-chat / deepseek-v4-pro paths stay on the documented
# thinking-disabled policy that the v2.0 production agent relies on.
_DEEPSEEK_REASONER_THINKING_ENABLED: frozenset[str] = frozenset({"deepseek-reasoner"})

# PROV-03 (D-09-06): Claude models that run with thinking ENABLED + temp=1.0
# despite feedback_temp1_reasoning_off_all_models. Rationale: Sonnet 4.6 with
# thinking disabled is just regular Sonnet — no thinking_blocks to round-trip
# means no AnthropicAdapter signal at all. budget_tokens is Claude's Discretion
# (4096 default; Phase 10 baseline regen may tune).
_ANTHROPIC_THINKING_BUDGET: dict[str, int] = {"claude-sonnet-4-6": 4096}


# ─── Scripted provider (EVAL-09 / P4) ────────────────────────────────────────
#
# CI runs the eval matrix with `--llm-provider scripted` so it does not depend
# on any external API key (cf. P4 in PITFALLS.md). The class below is a
# minimal BaseChatModel that pops AIMessages from a per-instance script, with
# a safe fallback that emits one finalize-only AIMessage so the agent graph
# terminates cleanly in a single plan() step (plan -> critique ->
# finalize_as_is -> END).
#
# The fallback content begins with the `[SCRIPTED CI MODE]` marker and cites
# `scripts/eval_matrix.py` (IN-05) so PR reviewers reading CI `summary.json`
# files immediately recognize it as deterministic placeholder output, not a
# real LLM failure.
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


class ScriptedChatModel(BaseChatModel):
    """Deterministic, no-network BaseChatModel for CI / matrix scripted runs.

    Pops AIMessages from `scripted` on each `_generate` call. When the list is
    exhausted (or empty), returns a fallback finalize-only AIMessage so the
    agent graph always reaches a clean termination — the matrix runner can
    never deadlock on this LLM.

    When the scripted list is empty, a freshly-constructed `AIMessage` is
    returned on each call (CR-02 fix — the previous module-level singleton
    was identity-deduplicated by LangGraph's `add_messages` reducer, which
    caused multi-turn revision loops to spin until `max_steps`).

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
            # CR-02: construct a fresh AIMessage on every call so LangGraph's
            # `add_messages` reducer does not dedupe consecutive fallbacks by
            # identity. IN-05: self-documenting marker makes the output
            # unambiguous in CI summary.json diffs.
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
        if _is_openai_reasoning_model(chat_model):
            # Phase 9 / PROV-01 (D-09-03 Path B): gpt-5 family routes through
            # the Responses API so reasoning blocks survive the agent's tool
            # loop. gpt-4o-mini intentionally skips this branch — keeping it
            # on plain ChatOpenAI / Chat Completions preserves the v2.0
            # production anchor path byte-for-byte (CLAUDE.md).
            return OpenAIReasoningChatModel(
                model=chat_model,
                api_key=SecretStr(api_key),
                temperature=temperature,
                use_responses_api=True,
            )
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
    if provider == "anthropic":
        # PROV-03 (D-09-05 + D-09-06): first-time Anthropic wiring. ``anthropic``
        # joined ``SUPPORTED_PROVIDERS`` in Phase 9 / PROV-03; ``app/config.py``
        # already shipped the ``anthropic_api_key`` Setting + ``resolve_llm_api_key``
        # branch pre-Phase-9 (verified by re-grep, no edits needed here).
        #
        # Carve-out: thinking ENABLED + temp=1.0 even though
        # ``feedback_temp1_reasoning_off_all_models`` says otherwise — Claude
        # Sonnet 4.6 with thinking disabled is just regular Sonnet and emits
        # NO ``thinking_blocks``, so ``AnthropicAdapter.capture_reasoning_state``
        # has nothing to round-trip. The signed thinking blocks Anthropic emits
        # MUST round-trip byte-identical on the next request or the API 400s,
        # which is exactly the wire contract Phase 9 PROV-03 exercises.
        #
        # Lazy import: keeps the dependency optional for environments that
        # never construct an Anthropic model (CI scripted runs, tests).
        from langchain_anthropic import ChatAnthropic

        anthropic_kwargs: dict[str, object] = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": _ANTHROPIC_THINKING_BUDGET.get(chat_model, 4096),
            },
        }
        return ChatAnthropic(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=temperature,
            **anthropic_kwargs,
        )
    if provider == "deepseek":
        # PROV-02 / D-09-04: model-level conditional. Default policy is
        # thinking-disabled (the v2.0 deepseek-chat / deepseek-v4-pro path
        # the agent loop has shipped against since W7). The reasoner family
        # is a deliberate carve-out: DeepSeekReasonerAdapter rounds-trips
        # reasoning_content cross-turn, so the model must actually emit
        # reasoning state to capture — flip thinking ON only for entries in
        # _DEEPSEEK_REASONER_THINKING_ENABLED.
        extra_body: dict[str, Any] = {"thinking": {"type": "disabled"}}
        if chat_model in _DEEPSEEK_REASONER_THINKING_ENABLED:
            extra_body = {"thinking": {"type": "enabled"}}
        return ChatDeepSeek(
            model=chat_model,
            api_key=SecretStr(api_key),
            temperature=temperature,
            extra_body=extra_body,
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
