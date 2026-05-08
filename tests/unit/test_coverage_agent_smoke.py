"""Smoke + functional tests for the W5 coverage agent.

Smoke: module imports cleanly, main() can be invoked.
Functional: with stubbed DB + LLM, main() walks the full path and emits
the expected MLflow artifacts.
"""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest


def test_smoke_module_imports() -> None:
    """Coverage agent module is importable without side effects."""
    mod = importlib.import_module("scripts.coverage_agent")
    assert hasattr(mod, "main")
    assert hasattr(mod, "gather_stats")
    assert hasattr(mod, "find_gaps")
    assert hasattr(mod, "propose_queries")
    assert hasattr(mod, "insert_pending")


class _StubCursor:
    """Returns gather_stats rows for the WITH-CTE SELECT, [] for everything else.

    Without this branching, every fetchall() returns the same gather_stats rows,
    which means the existing-query SELECT silently picks up bucket strings as
    if they were query texts. The smoke test would still pass by coincidence,
    but a real regression in filter_already_covered would be hidden.
    """

    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
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
        if "WITH neighborhoods" in self._last_sql:
            return self._rows
        return []

    @property
    def rowcount(self) -> int:
        return 1


class _StubConn:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.commits = 0
        self.cursors: list[_StubCursor] = []

    def cursor(self) -> _StubCursor:
        cur = _StubCursor(self._rows)
        self.cursors.append(cur)
        return cur

    def commit(self) -> None:
        self.commits += 1


def _make_get_conn(rows: list[tuple]):
    @contextmanager
    def _ctx():
        yield _StubConn(rows)

    return _ctx


def test_smoke_main_dry_run_empty_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """main(['--dry-run']) on a DB with no places returns 0 cleanly."""
    from scripts import coverage_agent

    monkeypatch.setattr(coverage_agent, "get_conn", _make_get_conn([]))
    monkeypatch.setattr(coverage_agent.vibe, "make_judge", lambda: None)

    set_exp = MagicMock()
    start_run = MagicMock()
    start_run.return_value.__enter__ = MagicMock(return_value=None)
    start_run.return_value.__exit__ = MagicMock(return_value=None)
    monkeypatch.setattr(coverage_agent.mlflow, "set_experiment", set_exp)
    monkeypatch.setattr(coverage_agent.mlflow, "start_run", start_run)
    monkeypatch.setattr(coverage_agent.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_dict", MagicMock())

    rc = coverage_agent.main(["--dry-run", "--days", "1"])
    assert rc == 0
    set_exp.assert_called_once_with("coverage_agent")


def test_functional_dry_run_emits_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end with a sparse bucket and a stub LLM:
    - main() returns 0
    - mlflow.log_dict is called with proposals.json containing the LLM's query
    - insert_pending is dry-run, so no real INSERT runs
    """
    from scripts import coverage_agent

    rows = [
        ("neighborhood:Mission", 200, 0, None),
        ("cuisine:burmese", 1, 0, None),
        ("recent_query", 50, 30, None),
    ]
    monkeypatch.setattr(coverage_agent, "get_conn", _make_get_conn(rows))

    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = json.dumps(
        [
            {
                "query_text": "burmese restaurants in San Francisco",
                "field_mode": "enriched",
                "rationale": "burmese only has 1 place",
            }
        ]
    )
    monkeypatch.setattr(coverage_agent.vibe, "make_judge", lambda: fake_llm)

    log_dict = MagicMock()
    start_run = MagicMock()
    start_run.return_value.__enter__ = MagicMock(return_value=None)
    start_run.return_value.__exit__ = MagicMock(return_value=None)
    monkeypatch.setattr(coverage_agent.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "start_run", start_run)
    monkeypatch.setattr(coverage_agent.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_dict", log_dict)

    rc = coverage_agent.main(["--dry-run", "--days", "1"])
    assert rc == 0

    artifact_names = [call.args[1] for call in log_dict.call_args_list]
    assert "proposals.json" in artifact_names
    proposals_payload = next(
        call.args[0] for call in log_dict.call_args_list if call.args[1] == "proposals.json"
    )
    assert proposals_payload["proposals"][0]["query_text"] == (
        "burmese restaurants in San Francisco"
    )

    stats_payload = next(
        call.args[0] for call in log_dict.call_args_list if call.args[1] == "stats.json"
    )
    assert any(s["bucket"] == "cuisine:burmese" for s in stats_payload["stats"])
