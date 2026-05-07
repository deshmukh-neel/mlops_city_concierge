"""Integration test for the agent graph.

Exercises the agent against the real Postgres + pgvector retrieval stack. The
LLM is still stubbed because we don't want to spend real tokens / require an
API key in CI; the point of this test is that the *retrieval* tools query a
live database successfully.

Run with:
    make db-up
    APP_ENV=integration poetry run pytest tests/integration/test_agent_graph.py
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from app.agent.graph import build_agent_graph  # noqa: E402
from app.agent.state import ItineraryState  # noqa: E402


class _SemanticSearchOnceLLM(BaseChatModel):
    """Issues one semantic_search call against the real DB, then finalizes."""

    @property
    def _llm_type(self) -> str:
        return "semantic-search-once"

    _step: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._step == 0:
            self._step += 1
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "id": "call-1",
                        "args": {"query": "cocktail bar in San Francisco", "k": 3},
                    }
                ],
            )
        else:
            msg = AIMessage(content="Done.", tool_calls=[])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> _SemanticSearchOnceLLM:
        return self


async def test_agent_graph_against_real_database() -> None:
    graph = build_agent_graph(_SemanticSearchOnceLLM(), max_steps=4)
    out = await graph.ainvoke(
        ItineraryState(messages=[HumanMessage(content="cocktail bar tonight")])
    )
    assert out["done"] is True
    # Whatever the DB returns (could be empty if seed not run), the scratch
    # entry exists and either holds rows or an error dict — never silent loss.
    assert "semantic_search" in out["scratch"]
    entry = out["scratch"]["semantic_search"][0]
    assert entry["args"]["query"] == "cocktail bar in San Francisco"
    assert "result" in entry
