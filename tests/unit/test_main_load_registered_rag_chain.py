"""Unit tests for the RAG_MODEL_OVERRIDE plumbing in app.main.

Parser cases live alongside the helper module (app.main._parse_model_override);
integration cases exercise load_registered_rag_chain's override branch end-to-end
with mocked MLflow and downstream chain construction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.main import _parse_model_override, load_registered_rag_chain


def _stub_mlflow_for_load(mocker, *, version: str = "7", run_id: str = "run-xyz"):
    """Patch mlflow + downstream dependencies so load_registered_rag_chain runs
    purely against fakes. Returns the MlflowClient instance so callers can assert
    on get_model_version / get_model_version_by_alias calls.
    """
    fake_version = SimpleNamespace(version=version, run_id=run_id)
    fake_run = SimpleNamespace(
        data=SimpleNamespace(
            params={
                "llm_provider": "openai",
                "chat_model": "gpt-4o-mini",
                "k": "5",
                "temperature": "0.0",
            }
        )
    )
    fake_client = mocker.Mock()
    fake_client.get_model_version.return_value = fake_version
    fake_client.get_model_version_by_alias.return_value = fake_version
    fake_client.get_run.return_value = fake_run

    mocker.patch("app.main.mlflow.set_tracking_uri")
    mocker.patch("app.main.mlflow.MlflowClient", return_value=fake_client)
    mocker.patch("app.main.resolve_llm_api_key", return_value="fake-key")
    mocker.patch(
        "app.main.build_rag_chain",
        return_value=SimpleNamespace(chain=object(), llm=object()),
    )
    return fake_client


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


# ---------------------------------------------------------------------------
# Integration tests for load_registered_rag_chain — exercise the override
# branch end-to-end with mocked MLflow + downstream chain construction.
# ---------------------------------------------------------------------------


def test_load_registered_rag_chain_uses_production_alias_when_override_unset(
    mocker, monkeypatch
) -> None:
    """OVR-03 regression sentinel: unset override -> existing production-alias path.

    Must call get_model_version_by_alias("city-concierge-rag", "production") exactly
    once and never call get_model_version.
    """
    monkeypatch.delenv("RAG_MODEL_OVERRIDE", raising=False)
    get_settings.cache_clear()
    client = _stub_mlflow_for_load(mocker)

    loaded = load_registered_rag_chain()

    client.get_model_version_by_alias.assert_called_once_with("city-concierge-rag", "production")
    client.get_model_version.assert_not_called()
    assert loaded.params.llm_provider == "openai"
    assert loaded.params.chat_model == "gpt-4o-mini"
    assert loaded.params.model_version == "7"


def test_load_registered_rag_chain_uses_version_when_override_is_version(
    mocker, monkeypatch
) -> None:
    """OVR-02: 'version:N' routes to client.get_model_version(name, N)."""
    monkeypatch.setenv("RAG_MODEL_OVERRIDE", "version:7")
    get_settings.cache_clear()
    client = _stub_mlflow_for_load(mocker)

    load_registered_rag_chain()

    client.get_model_version.assert_called_once_with("city-concierge-rag", "7")
    client.get_model_version_by_alias.assert_not_called()


def test_load_registered_rag_chain_uses_alias_when_override_is_named_alias(
    mocker, monkeypatch
) -> None:
    """OVR-02: 'alias:NAME' routes to client.get_model_version_by_alias(name, NAME)."""
    monkeypatch.setenv("RAG_MODEL_OVERRIDE", "alias:my-test-alias")
    get_settings.cache_clear()
    client = _stub_mlflow_for_load(mocker)

    load_registered_rag_chain()

    client.get_model_version_by_alias.assert_called_once_with("city-concierge-rag", "my-test-alias")
    client.get_model_version.assert_not_called()


def test_load_registered_rag_chain_treats_empty_override_as_unset(mocker, monkeypatch) -> None:
    """OVR-04: empty string is treated as unset (must NOT raise).

    Falls back to the 'production' alias path so an accidentally-blank env var
    in CI/Cloud Run doesn't crash the lifespan.
    """
    monkeypatch.setenv("RAG_MODEL_OVERRIDE", "")
    get_settings.cache_clear()
    client = _stub_mlflow_for_load(mocker)

    load_registered_rag_chain()

    client.get_model_version_by_alias.assert_called_once_with("city-concierge-rag", "production")
    client.get_model_version.assert_not_called()


def test_load_registered_rag_chain_raises_on_malformed_override(mocker, monkeypatch) -> None:
    """OVR-04: malformed override surfaces the parser ValueError unwrapped so the
    actionable 'version:N or alias:NAME' message reaches lifespan logs.
    """
    monkeypatch.setenv("RAG_MODEL_OVERRIDE", "garbage")
    get_settings.cache_clear()
    _stub_mlflow_for_load(mocker)

    with pytest.raises(ValueError, match=r"version:N.*alias:NAME"):
        load_registered_rag_chain()
