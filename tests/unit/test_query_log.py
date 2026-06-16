"""Unit tests for app.query_log.log_user_query.

Uses a FakeCursor/FakeConnection seam (adapted from test_tools_retrieval.py)
to test the INSERT SQL, bound params, commit, and fail-open behaviour without
touching a real database.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from app.query_log import log_user_query

# ---------------------------------------------------------------------------
# Fake cursor / connection plumbing
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self) -> None:
        self.executed_sql: str = ""
        self.executed_params: list = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: list) -> None:
        self.executed_sql = sql
        self.executed_params = list(params)


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.commit_called: bool = False

    def cursor(self, **_kwargs: Any) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commit_called = True


@pytest.fixture
def fake_conn_and_cursor(mocker):
    """Patch app.query_log.get_conn with a FakeConnection context manager.

    Returns (conn, cursor) so tests can assert both commit and SQL state.
    """
    cursor = FakeCursor()
    conn = FakeConnection(cursor)

    @contextmanager
    def fake_get_conn():
        yield conn

    mocker.patch("app.query_log.get_conn", fake_get_conn)
    return conn, cursor


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_happy_path_populated(fake_conn_and_cursor) -> None:
    """Populated types + explicit stop count insert correct SQL/params/commit."""
    conn, cur = fake_conn_and_cursor

    log_user_query(
        message="date night in North Beach",
        requested_primary_types=["restaurant", "bar"],
        num_stops=3,
        rag_label="openai:gpt-4o-mini",
    )

    assert "INSERT INTO user_query_log" in cur.executed_sql
    assert cur.executed_params == [
        "date night in North Beach",
        ["restaurant", "bar"],
        3,
        "openai:gpt-4o-mini",
        None,  # session_id default
    ]
    assert conn.commit_called is True


def test_happy_path_empty_array(fake_conn_and_cursor) -> None:
    """Empty requested_primary_types binds as [] (not None), num_stops as None — dominant free-text shape."""
    conn, cur = fake_conn_and_cursor

    log_user_query(
        message="find me good tacos",
        requested_primary_types=[],
        num_stops=None,
        rag_label="openai:gpt-4o-mini",
    )

    # empty list, NOT None
    assert cur.executed_params[1] == []
    # num_stops is None
    assert cur.executed_params[2] is None
    # commit still called
    assert conn.commit_called is True
    # full params check
    assert cur.executed_params == [
        "find me good tacos",
        [],
        None,
        "openai:gpt-4o-mini",
        None,
    ]


# ---------------------------------------------------------------------------
# SQL-safety (parameterisation) test
# ---------------------------------------------------------------------------


def test_param_safety_message_not_interpolated(fake_conn_and_cursor) -> None:
    """The raw message must NOT appear in the SQL string itself (proves %s parameterisation, not interpolation)."""
    conn, cur = fake_conn_and_cursor

    message = "INJECTION_MARKER_DO_NOT_APPEAR_IN_SQL"
    log_user_query(
        message=message,
        requested_primary_types=[],
        num_stops=None,
        rag_label="openai:gpt-4o-mini",
    )

    # Message must be in params, NOT in the SQL template string
    assert message not in cur.executed_sql
    # SQL must use 5 positional %s placeholders
    assert cur.executed_sql.count("%s") == 5


# ---------------------------------------------------------------------------
# Fail-open tests
# ---------------------------------------------------------------------------


def test_fail_open_get_conn_raises(mocker) -> None:
    """When get_conn() raises, log_user_query returns None without propagating."""
    mocker.patch("app.query_log.get_conn", side_effect=RuntimeError("db down"))
    mock_logger = mocker.patch("app.query_log.logger")

    result = log_user_query(
        message="any message",
        requested_primary_types=[],
        num_stops=None,
        rag_label="openai:gpt-4o-mini",
    )

    assert result is None
    mock_logger.warning.assert_called_once()
    # exc_info=True must be passed so the traceback is captured
    _, kwargs = mock_logger.warning.call_args
    assert kwargs.get("exc_info") is True


def test_fail_open_cursor_execute_raises(mocker) -> None:
    """When cur.execute() raises, log_user_query returns None without propagating."""
    cursor = FakeCursor()
    cursor.execute = mocker.Mock(side_effect=RuntimeError("execute failed"))  # type: ignore[method-assign]
    conn = FakeConnection(cursor)

    @contextmanager
    def fake_get_conn():
        yield conn

    mocker.patch("app.query_log.get_conn", fake_get_conn)
    mock_logger = mocker.patch("app.query_log.logger")

    result = log_user_query(
        message="trigger execute error",
        requested_primary_types=["cafe"],
        num_stops=2,
        rag_label="openai:gpt-4o-mini",
    )

    assert result is None
    mock_logger.warning.assert_called_once()
    _, kwargs = mock_logger.warning.call_args
    assert kwargs.get("exc_info") is True
