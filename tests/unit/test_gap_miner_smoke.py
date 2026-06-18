"""Smoke + functional tests for the demand-driven gap miner (Phase 18).

Smoke: module imports cleanly, gap_mine_main can be invoked.
Functional: with stubbed DB + no-op LLM, gap_mine_main walks the full
  demand→supply→gap→proposal path and emits the expected MLflow artifact
  (demand_gaps.json) proving cuisine-recall from message (ROUND-3 HIGH).
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Smoke 1: module contract
# ---------------------------------------------------------------------------


def test_smoke_module_imports() -> None:
    """coverage_agent exposes the demand-miner entrypoints + guard importable."""
    mod = importlib.import_module("scripts.coverage_agent")
    assert hasattr(mod, "gap_mine_main")
    assert hasattr(mod, "gather_demand")
    assert hasattr(mod, "gather_pair_supply")
    assert hasattr(mod, "find_demand_gaps")
    assert hasattr(mod, "ingested_query_texts")
    assert hasattr(mod, "gap_to_seed_query")
    assert hasattr(mod, "get_demand_conn")
    # Sandbox guard importable from its own module
    from scripts.sandbox_guard import assert_sandbox_write_target  # noqa: F401

    assert callable(assert_sandbox_write_target)


# ---------------------------------------------------------------------------
# Stub infrastructure — branched on SQL substrings
# ---------------------------------------------------------------------------


class _StubCursor:
    """Returns rows based on which table the SQL references.

    Branches:
      FROM user_query_log         → one demand row (empty types, message with cuisine+hood)
      FROM place_query_hits       → pair-supply row with count below min_places
      FROM places_ingest_query_checkpoints → COMPLETED prefixed row that does NOT match seed
      FROM places_ingest_query_proposals  → explicit empty list (exercises both branches)
      otherwise                   → []

    The branching is required so the 4+ selects inside gap_mine_main don't
    cross-contaminate (demand rows returned for a supply query would silently
    hide real regressions).

    For the cold-start tests, pass empty_demand=True; this changes the
    user_query_log branch to return [] so gather_demand yields no counts.
    """

    def __init__(self, *, empty_demand: bool = False) -> None:
        self._empty_demand = empty_demand
        self.executed: list[tuple[str, Any]] = []
        self._last_sql = ""

    def __enter__(self) -> _StubCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        self._last_sql = sql

    def fetchall(self) -> list[tuple]:
        sql = self._last_sql
        if "FROM user_query_log" in sql:
            if self._empty_demand:
                return []
            # Functional demand row:
            # message names catalog cuisine "vietnamese" + catalog neighborhood "Outer Sunset"
            # requested_primary_types is EMPTY [] — the free-text case (app/main.py produces this)
            # This exercises Plan 02's _lexical_cuisines fallback (ROUND-3 HIGH).
            return [("vietnamese restaurants in Outer Sunset", [])]
        if "FROM place_query_hits" in sql:
            # Return a pair-supply row with count=0 (below any min_places threshold)
            seed = "vietnamese restaurants in Outer Sunset San Francisco"
            return [(seed, 0)]
        if "FROM places_ingest_query_checkpoints" in sql:
            # COMPLETED prefixed `all::...` checkpoint that does NOT match our seed —
            # so the proposal for (Outer Sunset, vietnamese) survives dedup.
            # The prefix strip in ingested_query_texts splits on "::" → "some other query"
            # which is NOT "vietnamese restaurants in Outer Sunset San Francisco".
            return [("all::korean restaurants in Mission District San Francisco",)]
        if "FROM places_ingest_query_proposals" in sql:
            # Explicit empty — exercises the second branch of ingested_query_texts
            return []
        return []

    @property
    def rowcount(self) -> int:
        return 0


class _StubConn:
    def __init__(self, *, empty_demand: bool = False) -> None:
        self._empty_demand = empty_demand
        self.commits = 0
        self.cursors: list[_StubCursor] = []
        self._executed_sqls: list[str] = []

    def cursor(self) -> _StubCursor:
        cur = _StubCursor(empty_demand=self._empty_demand)
        self.cursors.append(cur)
        return cur

    def commit(self) -> None:
        self.commits += 1


def _make_get_conn(*, empty_demand: bool = False):
    """Return a context-manager factory yielding a _StubConn."""

    @contextmanager
    def _ctx():
        yield _StubConn(empty_demand=empty_demand)

    return _ctx


def _stub_mlflow(monkeypatch: pytest.MonkeyPatch, coverage_agent):
    """Monkeypatch all mlflow.* calls on the module; return the log_dict mock."""
    log_dict = MagicMock()
    start_run = MagicMock()
    start_run.return_value.__enter__ = MagicMock(return_value=None)
    start_run.return_value.__exit__ = MagicMock(return_value=None)
    monkeypatch.setattr(coverage_agent.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "start_run", start_run)
    monkeypatch.setattr(coverage_agent.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_dict", log_dict)
    return log_dict


# ---------------------------------------------------------------------------
# Smoke 2: cold-start no-op (D-04)
# ---------------------------------------------------------------------------


def test_smoke_cold_start_empty_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """gap_mine_main on an empty user_query_log returns 0 and logs gaps_found=0."""
    from scripts import coverage_agent

    # Empty demand (no user_query_log rows → gather_demand returns no counts)
    monkeypatch.setattr(coverage_agent, "get_conn", _make_get_conn(empty_demand=True))
    monkeypatch.setattr(coverage_agent, "get_demand_conn", _make_get_conn(empty_demand=True))
    monkeypatch.setattr(coverage_agent.vibe, "make_judge", lambda: None)
    monkeypatch.setattr(coverage_agent, "assert_sandbox_write_target", lambda conn=None: None)

    log_metric = MagicMock()
    _stub_mlflow(monkeypatch, coverage_agent)
    monkeypatch.setattr(coverage_agent.mlflow, "log_metric", log_metric)

    rc = coverage_agent.gap_mine_main(["--dry-run", "--days", "1"])

    assert rc == 0
    # gaps_found metric must be 0
    metric_calls = {call.args[0]: call.args[1] for call in log_metric.call_args_list}
    assert metric_calls.get("gaps_found") == 0


# ---------------------------------------------------------------------------
# Functional 1: cuisine-recall path + completed/prefix checkpoint dedup
# ---------------------------------------------------------------------------


def test_functional_cuisine_recall_emits_demand_gaps_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Populated path with EMPTY requested_primary_types + message naming a catalog
    cuisine — proves cuisine is recovered from the message (ROUND-3 HIGH).

    Also exercises the COMPLETED PREFIXED all::... checkpoint dedup:
    the stub returns a completed all::korean... checkpoint that does NOT match
    the mined (Outer Sunset, vietnamese) seed → the proposal survives dedup.

    Asserts: gap_mine_main returns 0 AND mlflow.log_dict is called with a
    demand_gaps.json artifact containing the (Outer Sunset, vietnamese) gap.
    """
    from scripts import coverage_agent

    monkeypatch.setattr(coverage_agent, "get_conn", _make_get_conn(empty_demand=False))
    monkeypatch.setattr(coverage_agent, "get_demand_conn", _make_get_conn(empty_demand=False))
    # No real LLM needed — the lexical pre-passes fully cover "vietnamese" + "Outer Sunset"
    monkeypatch.setattr(coverage_agent.vibe, "make_judge", lambda: None)
    # Monkeypatch sandbox guard to no-op (unit context has no real sandbox connection)
    monkeypatch.setattr(coverage_agent, "assert_sandbox_write_target", lambda conn=None: None)

    log_dict = _stub_mlflow(monkeypatch, coverage_agent)

    rc = coverage_agent.gap_mine_main(["--dry-run", "--days", "1"])

    assert rc == 0

    # demand_gaps.json artifact must have been logged
    artifact_names = [call.args[1] for call in log_dict.call_args_list]
    assert "demand_gaps.json" in artifact_names, (
        f"expected demand_gaps.json in logged artifacts; got {artifact_names}"
    )

    # The artifact must contain the (Outer Sunset, vietnamese) gap
    demand_gaps_payload = next(
        call.args[0] for call in log_dict.call_args_list if call.args[1] == "demand_gaps.json"
    )
    gaps = demand_gaps_payload["demand_gaps"]
    assert len(gaps) > 0, "demand_gaps.json must contain at least one gap"
    gap = gaps[0]
    assert gap["neighborhood"] == "Outer Sunset", (
        f"expected neighborhood='Outer Sunset', got {gap['neighborhood']!r}"
    )
    assert gap["cuisine"] == "vietnamese", f"expected cuisine='vietnamese', got {gap['cuisine']!r}"


# ---------------------------------------------------------------------------
# Functional 2: dry-run never calls real INSERT
# ---------------------------------------------------------------------------


def test_functional_dry_run_no_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run functional path must NOT execute any INSERT statement on the stub conn."""
    from scripts import coverage_agent

    stub_conn = _StubConn(empty_demand=False)

    @contextmanager
    def _fixed_conn():
        yield stub_conn

    monkeypatch.setattr(coverage_agent, "get_conn", _fixed_conn)
    monkeypatch.setattr(coverage_agent, "get_demand_conn", _fixed_conn)
    monkeypatch.setattr(coverage_agent.vibe, "make_judge", lambda: None)
    monkeypatch.setattr(coverage_agent, "assert_sandbox_write_target", lambda conn=None: None)
    _stub_mlflow(monkeypatch, coverage_agent)

    rc = coverage_agent.gap_mine_main(["--dry-run", "--days", "1"])
    assert rc == 0

    # Collect every SQL statement executed on every cursor
    all_sqls = []
    for cur in stub_conn.cursors:
        for sql, _ in cur.executed:
            all_sqls.append(sql.strip().upper())

    insert_sqls = [s for s in all_sqls if s.startswith("INSERT")]
    assert insert_sqls == [], f"--dry-run must not execute any INSERT, but found: {insert_sqls}"
