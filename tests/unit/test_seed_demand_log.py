"""Unit tests for scripts/seed_demand_log.py.

Covers:
  Test 1: seed_demand_rows() returns catalog-valid rows
  Test 2: at least one row targets ("Outer Sunset", "vietnamese")
  Test 3: INSERT path uses parameterised %s placeholders (no string interpolation)
  Test 4: assert_sandbox_write_target is called before any INSERT (HIGH-3)
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Capturing-stub connection (reusing _InsertConn/_InsertCursor pattern)
# ---------------------------------------------------------------------------


class InsertCursor:
    """Captures INSERT execute calls for assertion."""

    def __init__(self) -> None:
        self.executes: list[tuple] = []
        self.rowcount = 1

    def __enter__(self) -> InsertCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> None:
        self.executes.append((sql, params))

    def fetchone(self) -> tuple:
        return ("city_concierge_sandbox",)


class InsertConn:
    """Capturing connection that records commits."""

    def __init__(self) -> None:
        self.cursor_obj = InsertCursor()
        self.commits: int = 0

    def __enter__(self) -> InsertConn:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> InsertCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSeedDemandRows:
    """Test 1 + Test 2: catalog-valid fixture rows."""

    def test_returns_non_empty_list(self) -> None:
        from scripts.seed_demand_log import seed_demand_rows

        rows = seed_demand_rows()
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_all_rows_have_required_keys(self) -> None:
        from scripts.seed_demand_log import seed_demand_rows

        rows = seed_demand_rows()
        required = {"message", "requested_primary_types", "num_stops", "rag_label", "session_id"}
        for row in rows:
            assert required <= set(row.keys()), f"Row missing keys: {row}"

    def test_cuisine_types_are_catalog_valid(self) -> None:
        """Test 1: every requested_primary_types entry maps to a CUISINES member."""
        from scripts.ingest_places_sf import CUISINES
        from scripts.seed_demand_log import seed_demand_rows

        rows = seed_demand_rows()
        for row in rows:
            for pt in row["requested_primary_types"]:
                # The mapping: "Vietnamese Restaurant" → "vietnamese"
                # Each entry must have a corresponding CUISINES member
                cuisine_key = pt.replace(" Restaurant", "").replace(" restaurant", "").lower()
                assert cuisine_key in CUISINES, (
                    f"{pt!r} → {cuisine_key!r} not in CUISINES. "
                    "Fix seed row to use catalog-valid types."
                )

    def test_message_contains_neighborhood(self) -> None:
        """Test 1: every message contains a NEIGHBORHOODS member."""
        from scripts.ingest_places_sf import NEIGHBORHOODS
        from scripts.seed_demand_log import seed_demand_rows

        rows = seed_demand_rows()
        for row in rows:
            msg = row["message"]
            found = any(n in msg for n in NEIGHBORHOODS)
            assert found, (
                f"message {msg!r} contains no NEIGHBORHOODS member. "
                "Fix seed row to reference a catalog neighborhood."
            )

    def test_outer_sunset_vietnamese_row_present(self) -> None:
        """Test 2: at least one row targets the thin ('Outer Sunset', 'vietnamese') bucket."""
        from scripts.seed_demand_log import seed_demand_rows

        rows = seed_demand_rows()
        found = False
        for row in rows:
            has_outer_sunset = "Outer Sunset" in row["message"]
            cuisine_keys = [
                pt.replace(" Restaurant", "").replace(" restaurant", "").lower()
                for pt in row["requested_primary_types"]
            ]
            has_vietnamese = "vietnamese" in cuisine_keys
            if has_outer_sunset and has_vietnamese:
                found = True
                break
        assert found, "No seed row targets ('Outer Sunset', 'vietnamese') — required by plan spec."


class TestInsertDemandRows:
    """Test 3 + Test 4: INSERT shape and write-guard enforcement."""

    def test_insert_uses_parameterised_placeholders(self, monkeypatch) -> None:
        """Test 3: each INSERT uses %s params (no f-string interpolation of message)."""
        import scripts.seed_demand_log as seed_mod
        from scripts.seed_demand_log import insert_demand_rows, seed_demand_rows

        conn = InsertConn()
        # Patch guard to pass (sandbox context)
        monkeypatch.setattr(seed_mod, "assert_sandbox_write_target", lambda conn=None: None)

        rows = seed_demand_rows()
        insert_demand_rows(rows, conn=conn)

        insert_executes = [
            (sql, params) for sql, params in conn.cursor_obj.executes if "INSERT" in sql.upper()
        ]
        assert len(insert_executes) == len(rows), "Expected one INSERT per row"
        for sql, params in insert_executes:
            # Must use %s placeholders (parameterised)
            assert "%s" in sql, f"INSERT SQL missing %s placeholder: {sql!r}"
            # Params must be a list/tuple (not None)
            assert params is not None, "INSERT called without params (string interpolation risk)"
            assert len(params) == 5, f"Expected 5 params, got {len(params)}"

    def test_exactly_one_commit_after_loop(self, monkeypatch) -> None:
        """Test 3: exactly one commit after inserting all rows."""
        import scripts.seed_demand_log as seed_mod
        from scripts.seed_demand_log import insert_demand_rows, seed_demand_rows

        conn = InsertConn()
        monkeypatch.setattr(seed_mod, "assert_sandbox_write_target", lambda conn=None: None)

        rows = seed_demand_rows()
        insert_demand_rows(rows, conn=conn)

        assert conn.commits == 1, f"Expected exactly 1 commit, got {conn.commits}"

    def test_insert_returns_count(self, monkeypatch) -> None:
        """Test 3: insert_demand_rows returns the number of inserted rows."""
        import scripts.seed_demand_log as seed_mod
        from scripts.seed_demand_log import insert_demand_rows, seed_demand_rows

        conn = InsertConn()
        monkeypatch.setattr(seed_mod, "assert_sandbox_write_target", lambda conn=None: None)

        rows = seed_demand_rows()
        count = insert_demand_rows(rows, conn=conn)

        assert count == len(rows)

    def test_guard_called_before_insert(self, monkeypatch) -> None:
        """Test 4 (HIGH-3): assert_sandbox_write_target called before any INSERT."""
        import scripts.seed_demand_log as seed_mod
        from scripts.seed_demand_log import insert_demand_rows, seed_demand_rows

        call_order: list[str] = []

        def tracking_guard(conn=None) -> None:
            call_order.append("guard")

        conn = InsertConn()
        orig_cursor_execute = conn.cursor_obj.execute

        def tracking_execute(sql: str, params: object = None) -> None:
            if "INSERT" in sql.upper():
                call_order.append("insert")
            orig_cursor_execute(sql, params)

        conn.cursor_obj.execute = tracking_execute
        monkeypatch.setattr(seed_mod, "assert_sandbox_write_target", tracking_guard)

        rows = seed_demand_rows()
        insert_demand_rows(rows, conn=conn)

        assert "guard" in call_order, "assert_sandbox_write_target was never called"
        assert call_order[0] == "guard", "guard must be called BEFORE any INSERT"

    def test_raising_guard_prevents_all_inserts(self, monkeypatch) -> None:
        """Test 4 (HIGH-3): when guard raises, zero INSERTs are executed."""
        import scripts.seed_demand_log as seed_mod
        from scripts.seed_demand_log import insert_demand_rows, seed_demand_rows

        def raising_guard(conn=None) -> None:
            raise RuntimeError("not a sandbox — refusing write")

        conn = InsertConn()
        monkeypatch.setattr(seed_mod, "assert_sandbox_write_target", raising_guard)

        rows = seed_demand_rows()
        with pytest.raises(RuntimeError, match="not a sandbox"):
            insert_demand_rows(rows, conn=conn)

        insert_executes = [sql for sql, _ in conn.cursor_obj.executes if "INSERT" in sql.upper()]
        assert insert_executes == [], "No INSERTs should execute when guard raises"

    def test_guard_imported_from_sandbox_guard(self) -> None:
        """Verify assert_sandbox_write_target is imported from scripts.sandbox_guard (MEDIUM-2)."""
        import scripts.seed_demand_log as seed_mod
        from scripts.sandbox_guard import assert_sandbox_write_target as canonical_guard

        # The module-level attribute must be the canonical guard function
        assert seed_mod.assert_sandbox_write_target is canonical_guard, (
            "seed_demand_log.assert_sandbox_write_target must be the SAME object "
            "as scripts.sandbox_guard.assert_sandbox_write_target (no re-definition)"
        )
