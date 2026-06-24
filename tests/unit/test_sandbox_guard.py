"""Unit tests for scripts/sandbox_guard.py — sandbox-write guard.

The guard checks the live current_database() on the write connection,
NOT env-var equality. A mis-set SANDBOX_DATABASE_URL that names prod
must still trigger a raise (REVIEW ROUND-2 H3 refinement).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Capturing-stub connection (reusing _InsertConn/_InsertCursor pattern)
# ---------------------------------------------------------------------------


class StubCursor:
    """Cursor that returns a controllable current_database() value."""

    def __init__(self, dbname: str) -> None:
        self.dbname_value = dbname
        self.executes: list[tuple] = []

    def __enter__(self) -> StubCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> None:
        self.executes.append((sql, params))

    def fetchone(self) -> tuple:
        return (self.dbname_value,)


class StubConn:
    """Capturing connection that reports a fixed current_database()."""

    def __init__(self, dbname: str) -> None:
        self.dbname_value = dbname
        self.cursor_obj = StubCursor(dbname)
        self.committed = False

    def __enter__(self) -> StubConn:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> StubCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sandbox_url(dbname: str = "city_concierge_sandbox") -> str:
    return f"postgresql://user:pw@localhost:5433/{dbname}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssertSandboxWriteTarget:
    """Covers the five behaviors from the plan <behavior> block."""

    def test_sandbox_name_passes(self, monkeypatch) -> None:
        """Test 1: current_database() == 'city_concierge_sandbox' → no raise."""
        import os

        monkeypatch.setitem(os.environ, "SANDBOX_DATABASE_URL", make_sandbox_url())
        conn = StubConn("city_concierge_sandbox")
        from scripts.sandbox_guard import assert_sandbox_write_target

        assert_sandbox_write_target(conn=conn)  # must not raise

    def test_prod_name_raises(self, monkeypatch) -> None:
        """Test 2: current_database() == 'city_concierge' → raises with offending name."""
        import os

        monkeypatch.setitem(os.environ, "SANDBOX_DATABASE_URL", make_sandbox_url("city_concierge"))
        conn = StubConn("city_concierge")
        from scripts.sandbox_guard import assert_sandbox_write_target

        with pytest.raises((SystemExit, RuntimeError)) as exc_info:
            assert_sandbox_write_target(conn=conn)
        # The error message must name the offending dbname
        assert "city_concierge" in str(exc_info.value)

    def test_h3_misset_env_var_cannot_whitelist_prod(self, monkeypatch) -> None:
        """Test 3 (H3 refinement): even when SANDBOX_DATABASE_URL names 'city_concierge',
        if current_database() returns 'city_concierge' the guard STILL RAISES.
        The pass decision depends on the live current_database(), NOT env-var equality.
        """
        import os

        # mis-set: env var points at prod
        monkeypatch.setitem(
            os.environ,
            "SANDBOX_DATABASE_URL",
            "postgresql://user:pw@localhost:5432/city_concierge",
        )
        conn = StubConn("city_concierge")  # live DB is prod
        from scripts.sandbox_guard import assert_sandbox_write_target

        with pytest.raises((SystemExit, RuntimeError)):
            assert_sandbox_write_target(conn=conn)

    def test_sandbox_pattern_name_passes(self, monkeypatch) -> None:
        """Test 4: a name CONTAINING 'sandbox' that is NOT a known-prod name passes."""
        import os

        monkeypatch.setitem(
            os.environ,
            "SANDBOX_DATABASE_URL",
            "postgresql://user:pw@localhost:5433/my_sandbox_db",
        )
        conn = StubConn("my_sandbox_db")
        from scripts.sandbox_guard import assert_sandbox_write_target

        assert_sandbox_write_target(conn=conn)  # must not raise

    def test_non_sandbox_name_without_sandbox_pattern_raises(self, monkeypatch) -> None:
        """Test 4 (corollary): a name that does NOT contain 'sandbox' raises."""
        import os

        monkeypatch.setitem(
            os.environ,
            "SANDBOX_DATABASE_URL",
            "postgresql://user:pw@localhost:5433/some_other_db",
        )
        conn = StubConn("some_other_db")
        from scripts.sandbox_guard import assert_sandbox_write_target

        with pytest.raises((SystemExit, RuntimeError)):
            assert_sandbox_write_target(conn=conn)

    def test_connection_reuse_uses_passed_conn(self, monkeypatch) -> None:
        """Test 5: when a connection is passed, the guard runs SELECT current_database()
        on THAT connection (no new pool open).
        """
        import os

        import scripts.sandbox_guard as guard_mod

        monkeypatch.setitem(os.environ, "SANDBOX_DATABASE_URL", make_sandbox_url())

        pool_opened = []

        def fake_get_conn():
            pool_opened.append(True)
            from contextlib import contextmanager

            @contextmanager
            def ctx():
                yield StubConn("city_concierge_sandbox")

            return ctx()

        monkeypatch.setattr(guard_mod, "get_conn", fake_get_conn)

        conn = StubConn("city_concierge_sandbox")
        guard_mod.assert_sandbox_write_target(conn=conn)

        assert not pool_opened, "No pool should be opened when conn is passed"
        # The passed conn's cursor must have received the SELECT
        assert any("current_database" in sql for sql, _ in conn.cursor_obj.executes)
