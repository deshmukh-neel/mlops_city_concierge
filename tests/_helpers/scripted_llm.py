"""Shared scripted-LLM test helpers (Plan 03-11 / WR-03 hoist).

Consolidates the two near-identical scripted-LLM-with-recording test classes
that previously lived in:
  - tests/unit/test_chat_functional.py (`_ScriptedLLM`)
  - tests/unit/test_eval_agent.py (`_RecordingScriptedLLM`)

Production `app.llm_factory.ScriptedChatModel` is intentionally NOT
consolidated into this helper. It has scenario-registry semantics
(`SCRIPTED_SCENARIOS`, `scenario_id`) and a fresh-AIMessage fallback on
exhaustion (CR-02) that don't belong in a test helper — tests should be
exhaustive about their scripts and surface mis-counts loudly.

Contract differences from the production class:
  - Empty `scripted` list raises `IndexError` (loud-fail). The production
    fallback synthesizes a "[SCRIPTED CI MODE]" AIMessage so the CI matrix
    runner never deadlocks; tests do not need that affordance.
  - `RecordingScriptedLLM.seen` uses `Field(default_factory=list)` so callers
    do NOT need to pass `seen=outer_var` (which was misleading anyway since
    Pydantic deep-copies during validation — see WR-04 in 03-VERIFICATION.md).
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedLLM(BaseChatModel):
    """Deterministic test LLM that pops AIMessages from a scripted list.

    Use this in tests that exercise the agent graph or `/chat` endpoint
    without hitting a real provider. The test author is responsible for
    providing enough scripted messages for the full trajectory; exhaustion
    raises `IndexError` so the mis-count localizes immediately.
    """

    scripted: list[AIMessage]

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
        if not self.scripted:
            raise IndexError(
                "ScriptedLLM exhausted; provide more AIMessages in the scripted list "
                "(test wrote fewer scripted messages than the agent graph consumed)"
            )
        msg = self.scripted.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedLLM:
        # The agent graph calls .bind_tools(...) on the LLM; tests don't care
        # about real tool binding, so this is a no-op returning self.
        return self


class RecordingScriptedLLM(ScriptedLLM):
    """`ScriptedLLM` that also captures every messages-list it saw on
    `_generate`, so tests can directly assert on what the LLM was prompted
    with on a given turn.

    The `seen` field uses `Field(default_factory=list)` so each instance
    gets its own fresh list — callers should NOT pass `seen=outer_var`
    (Pydantic deep-copies during validation, which was the root cause of
    the dead outer-scope `seen` vars closed under WR-04).
    """

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Snapshot BEFORE delegating; if `super()._generate` raises on
        # exhaustion we still want to see what the LLM was asked.
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
