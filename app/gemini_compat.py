"""Compatibility helpers for Gemini 3 tool-calling through LangChain.

Gemini 3 requires thought signatures on function-call history. Older
langchain-google-genai releases do not preserve them, so we add the documented
validator-bypass signature until the project can move to the newer LangChain
1.x-compatible adapter.
"""

from __future__ import annotations

from typing import Any

GEMINI3_THOUGHT_SIGNATURE_BYPASS = b"skip_thought_signature_validator"


def patch_langchain_google_genai_for_gemini3() -> None:
    """Patch older langchain-google-genai history conversion for Gemini 3 tools."""
    try:
        import langchain_google_genai.chat_models as chat_models
    except ImportError:
        return

    chat_models_dynamic: Any = chat_models
    if getattr(chat_models_dynamic, "_city_concierge_gemini3_patch", False):
        return

    original_parse_chat_history = chat_models_dynamic._parse_chat_history

    def parse_chat_history_with_signatures(*args: Any, **kwargs: Any) -> Any:
        system_instruction, messages = original_parse_chat_history(*args, **kwargs)
        for content in messages:
            if getattr(content, "role", None) != "model":
                continue
            signature_added = False
            for part in content.parts:
                if not getattr(part, "function_call", None):
                    continue
                if signature_added:
                    continue
                if not getattr(part, "thought_signature", None):
                    part.thought_signature = GEMINI3_THOUGHT_SIGNATURE_BYPASS
                signature_added = True
        return system_instruction, messages

    chat_models_dynamic._parse_chat_history = parse_chat_history_with_signatures
    chat_models_dynamic._city_concierge_gemini3_patch = True
