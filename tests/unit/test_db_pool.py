from __future__ import annotations

import pytest

from app.config import get_settings
from app.db_pool import close_db_pool, get_connection, init_db_pool, return_connection

TEST_DATABASE_URL = "postgresql://postgres:test@localhost:5432/city_concierge_test"


@pytest.fixture(autouse=True)
def _reset_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    get_settings.cache_clear()
    close_db_pool()
    yield
    close_db_pool()
    get_settings.cache_clear()


def test_init_db_pool_creates_threaded_pool(mocker) -> None:
    pool = mocker.Mock()
    pool_cls = mocker.patch("app.db_pool.ThreadedConnectionPool", return_value=pool)

    result = init_db_pool(
        TEST_DATABASE_URL,
        0,
        10,
    )

    assert result is pool
    pool_cls.assert_called_once_with(
        0,
        10,
        dsn=TEST_DATABASE_URL,
    )


def test_init_db_pool_reuses_existing_pool_when_params_match(mocker) -> None:
    pool = mocker.Mock()
    pool_cls = mocker.patch("app.db_pool.ThreadedConnectionPool", return_value=pool)

    first = init_db_pool("postgresql://example", 0, 10)
    second = init_db_pool("postgresql://example", 0, 10)

    assert first is second
    pool_cls.assert_called_once()


def test_init_db_pool_raises_when_params_differ(mocker) -> None:
    """A second init with different params would silently swap the database
    underneath in-flight callers — refuse instead."""
    mocker.patch("app.db_pool.ThreadedConnectionPool", return_value=mocker.Mock())

    init_db_pool("postgresql://example", 0, 10)
    with pytest.raises(RuntimeError, match="different parameters"):
        init_db_pool("postgresql://different", 0, 5)


@pytest.mark.parametrize(
    ("min_connections", "max_connections", "match"),
    [
        (-1, 10, "greater than or equal to 0"),
        (0, 0, "greater than or equal to 1"),
        (11, 10, "cannot exceed"),
    ],
)
def test_init_db_pool_validates_pool_size(
    min_connections: int,
    max_connections: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        init_db_pool("postgresql://example", min_connections, max_connections)


def test_get_connection_lazily_initializes_pool_from_settings(mocker) -> None:
    conn = mocker.Mock()
    pool = mocker.Mock()
    pool.getconn.return_value = conn
    pool_cls = mocker.patch("app.db_pool.ThreadedConnectionPool", return_value=pool)

    result = get_connection()

    assert result is conn
    pool_cls.assert_called_once_with(
        0,
        10,
        dsn=TEST_DATABASE_URL,
    )
    pool.getconn.assert_called_once_with()


def test_return_connection_returns_to_pool(mocker) -> None:
    conn = mocker.Mock()
    pool = mocker.Mock()
    mocker.patch("app.db_pool.ThreadedConnectionPool", return_value=pool)
    init_db_pool("postgresql://example", 0, 10)

    return_connection(conn)

    pool.putconn.assert_called_once_with(conn, close=False)


def test_return_connection_closes_when_pool_is_missing(mocker) -> None:
    conn = mocker.Mock()

    return_connection(conn)

    conn.close.assert_called_once_with()


def test_close_db_pool_closes_all_connections(mocker) -> None:
    pool = mocker.Mock()
    mocker.patch("app.db_pool.ThreadedConnectionPool", return_value=pool)
    init_db_pool("postgresql://example", 0, 10)

    close_db_pool()

    pool.closeall.assert_called_once_with()
