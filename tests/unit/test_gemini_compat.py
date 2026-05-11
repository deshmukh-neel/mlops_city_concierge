from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.gemini_compat import (
    GEMINI3_THOUGHT_SIGNATURE_BYPASS,
    patch_langchain_google_genai_for_gemini3,
)


def test_patch_adds_bypass_signature_to_model_function_call_history() -> None:
    """Backfill Gemini 3 thought signatures for older LangChain adapters."""
    import langchain_google_genai.chat_models as chat_models

    patch_langchain_google_genai_for_gemini3()
    _, messages = chat_models._parse_chat_history(
        [
            HumanMessage(content="find tacos"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "args": {"query": "tacos"},
                        "id": "call-1",
                    }
                ],
            ),
            ToolMessage(content="[]", tool_call_id="call-1"),
        ]
    )

    function_call_part = messages[1].parts[0]
    assert function_call_part.function_call.name == "semantic_search"
    assert function_call_part.thought_signature == GEMINI3_THOUGHT_SIGNATURE_BYPASS


def test_patch_is_idempotent() -> None:
    """Allow every Gemini constructor path to call the patch safely."""
    import langchain_google_genai.chat_models as chat_models

    patch_langchain_google_genai_for_gemini3()
    first = chat_models._parse_chat_history
    patch_langchain_google_genai_for_gemini3()

    assert chat_models._parse_chat_history is first
