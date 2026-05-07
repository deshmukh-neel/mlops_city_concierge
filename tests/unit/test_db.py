from __future__ import annotations

import pytest

from app.db import get_conn, get_db


def test_get_db_returns_connection_to_pool(mocker) -> None:
    conn = mocker.Mock()
    conn.closed = 0
    get_connection = mocker.patch("app.db.get_connection", return_value=conn)
    return_connection = mocker.patch("app.db.return_connection")

    db = get_db()
    assert next(db) is conn
    with pytest.raises(StopIteration):
        next(db)

    get_connection.assert_called_once_with()
    conn.rollback.assert_called_once_with()
    return_connection.assert_called_once_with(conn, close=False)


def test_get_db_returns_connection_after_caller_error(mocker) -> None:
    conn = mocker.Mock()
    conn.closed = 0
    mocker.patch("app.db.get_connection", return_value=conn)
    return_connection = mocker.patch("app.db.return_connection")

    db = get_db()
    next(db)
    with pytest.raises(ValueError, match="boom"):
        db.throw(ValueError("boom"))

    conn.rollback.assert_called_once_with()
    return_connection.assert_called_once_with(conn, close=False)


def test_get_db_closes_connection_when_reset_fails(mocker) -> None:
    conn = mocker.Mock()
    conn.closed = 0
    conn.rollback.side_effect = RuntimeError("connection reset failed")
    mocker.patch("app.db.get_connection", return_value=conn)
    return_connection = mocker.patch("app.db.return_connection")
    logger = mocker.patch("app.db.logger")

    db = get_db()
    next(db)
    with pytest.raises(StopIteration):
        next(db)

    logger.warning.assert_called_once()
    return_connection.assert_called_once_with(conn, close=True)


def test_get_db_discards_already_closed_connection(mocker) -> None:
    conn = mocker.Mock()
    conn.closed = 1
    mocker.patch("app.db.get_connection", return_value=conn)
    return_connection = mocker.patch("app.db.return_connection")

    db = get_db()
    next(db)
    with pytest.raises(StopIteration):
        next(db)

    conn.rollback.assert_not_called()
    return_connection.assert_called_once_with(conn, close=True)


def test_get_conn_context_manager_uses_pool(mocker) -> None:
    conn = mocker.Mock()
    conn.closed = 0
    get_connection = mocker.patch("app.db.get_connection", return_value=conn)
    return_connection = mocker.patch("app.db.return_connection")

    with get_conn() as borrowed:
        assert borrowed is conn

    get_connection.assert_called_once_with()
    conn.rollback.assert_called_once_with()
    return_connection.assert_called_once_with(conn, close=False)
