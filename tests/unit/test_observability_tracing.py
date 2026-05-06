"""Tests for app.observability tracing helpers.

Layers covered:
  - unit/mock      → pure logic, Langfuse client mocked.
  - smoke          → module imports, helpers callable without raising.
  - functional     → context manager yields trace id from a stubbed client,
                     flushes on exit, treats exceptions transparently.

Integration test (real Langfuse Cloud) lives in
  tests/integration/test_observability_tracing_integration.py
and is gated on APP_ENV=integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_module_imports_cleanly() -> None:
    import app.observability as obs

    assert hasattr(obs, "get_client")
    assert hasattr(obs, "langgraph_callbacks")
    assert hasattr(obs, "trace_request")


def test_helpers_callable_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    from app.observability import get_client, langgraph_callbacks, trace_request

    assert get_client() is None
    assert langgraph_callbacks() == []
    with trace_request("smoke") as trace_id:
        assert trace_id is None


# ---------------------------------------------------------------------------
# Unit / mock — no env
# ---------------------------------------------------------------------------


def test_get_client_returns_none_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from app.observability import get_client

    assert get_client() is None


def test_langgraph_callbacks_empty_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from app.observability import langgraph_callbacks

    assert langgraph_callbacks() == []


def test_trace_request_no_op_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from app.observability import trace_request

    with trace_request("test", metadata_key="value") as trace_id:
        assert trace_id is None


# ---------------------------------------------------------------------------
# Unit / mock — env present, Langfuse client stubbed
# ---------------------------------------------------------------------------


def _set_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:9999")


def test_get_client_constructs_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_langfuse_env(monkeypatch)
    fake_client = MagicMock(name="LangfuseClient")
    fake_ctor = MagicMock(return_value=fake_client)

    with patch("app.observability.Langfuse", fake_ctor):
        from app.observability import get_client

        client = get_client()

    assert client is fake_client
    fake_ctor.assert_called_once_with(
        secret_key="sk-test",
        public_key="pk-test",
        host="http://localhost:9999",
    )


def test_get_client_default_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    fake_ctor = MagicMock(return_value=MagicMock())
    with patch("app.observability.Langfuse", fake_ctor):
        from app.observability import get_client

        get_client()

    _, kwargs = fake_ctor.call_args
    assert kwargs["host"] == "https://cloud.langfuse.com"


def test_langgraph_callbacks_returns_handler_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_langfuse_env(monkeypatch)
    sentinel = object()

    with patch("app.observability.CallbackHandler", return_value=sentinel):
        from app.observability import langgraph_callbacks

        callbacks = langgraph_callbacks()

    assert callbacks == [sentinel]


# ---------------------------------------------------------------------------
# Functional — context-manager behavior end-to-end through public API
# ---------------------------------------------------------------------------


def test_trace_request_yields_trace_id_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_langfuse_env(monkeypatch)
    fake_trace = MagicMock(id="trace-abc-123")
    fake_client = MagicMock()
    fake_client.trace.return_value = fake_trace

    with patch("app.observability.Langfuse", return_value=fake_client):
        from app.observability import trace_request

        with trace_request("chat", message="hello") as trace_id:
            assert trace_id == "trace-abc-123"

    fake_client.trace.assert_called_once_with(name="chat", metadata={"message": "hello"})
    fake_client.flush.assert_called_once()


def test_trace_request_flushes_even_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_langfuse_env(monkeypatch)
    fake_client = MagicMock()
    fake_client.trace.return_value = MagicMock(id="trace-exc")

    with patch("app.observability.Langfuse", return_value=fake_client):
        from app.observability import trace_request

        with pytest.raises(ValueError, match="boom"), trace_request("chat"):
            raise ValueError("boom")

    fake_client.flush.assert_called_once()


def test_trace_request_metadata_threaded_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_langfuse_env(monkeypatch)
    fake_client = MagicMock()
    fake_client.trace.return_value = MagicMock(id="trace-meta")

    with patch("app.observability.Langfuse", return_value=fake_client):
        from app.observability import trace_request

        with trace_request("chat", user_id="u1", session_id="s1"):
            pass

    fake_client.trace.assert_called_once_with(
        name="chat",
        metadata={"user_id": "u1", "session_id": "s1"},
    )
