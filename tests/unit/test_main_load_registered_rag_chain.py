"""Unit tests for the RAG_MODEL_OVERRIDE plumbing in app.main.

Parser cases live alongside the helper module (app.main._parse_model_override);
integration cases for load_registered_rag_chain are added in Task 2.
"""

from __future__ import annotations

import pytest

from app.main import _parse_model_override


def test_parse_model_override_version_returns_version_kind() -> None:
    assert _parse_model_override("version:7") == ("version", "7")


def test_parse_model_override_alias_returns_alias_kind() -> None:
    assert _parse_model_override("alias:my-test-alias") == ("alias", "my-test-alias")


def test_parse_model_override_strips_whitespace_from_value() -> None:
    # Shell env exports often have stray spaces; treat them as harmless.
    assert _parse_model_override("version: 7 ") == ("version", "7")


@pytest.mark.parametrize(
    "raw",
    ["", "garbage", "version:", "alias:", "run_id:abc", "versionNoColon"],
)
def test_parse_model_override_rejects_malformed_value(raw: str) -> None:
    with pytest.raises(ValueError, match=r"version:N.*alias:NAME"):
        _parse_model_override(raw)


def test_parse_model_override_error_message_contains_offending_value() -> None:
    with pytest.raises(ValueError) as exc_info:
        _parse_model_override("garbage")
    # repr("garbage") -> "'garbage'" — the developer sees exactly what they typed.
    assert repr("garbage") in str(exc_info.value)
