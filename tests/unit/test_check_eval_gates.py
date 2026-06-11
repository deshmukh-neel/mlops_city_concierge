"""Unit tests for scripts/check_eval_gates.py (EVAL-03 / D-10-05).

Gate-check script that reads a matrix summary.json and exits non-zero on
any hard-gate violation.

Exit-code conventions (matching check_baselines_fresh.py exactly):
    0 = all hard gates passed (aspirational misses printed, non-blocking)
    1 = one or more hard-gate violations
    2 = infrastructure failure (missing YAML, malformed summary.json)

Tests use tmp_path fixtures with synthetic YAML + JSON so they never
need a live eval run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_eval_gates.py"


def _load_script() -> ModuleType:
    """Load scripts/check_eval_gates.py as a module without sys.path mutation."""
    spec = importlib.util.spec_from_file_location("check_eval_gates", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_eval_gates"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    """The check_eval_gates module under test."""
    return _load_script()


# ---------------------------------------------------------------------------
# Synthetic YAML / JSON helpers
# ---------------------------------------------------------------------------

_MINIMAL_GATES_YAML = """\
gates:
  - family: openai/gpt-4o-mini
    status: active
    rationale: "D-10-07: anchor"
    hard:
      metric: committed_itinerary_rate
      op: ">="
      value: 0.8
    advisory: []

  - family: openai/gpt-5-mini
    status: aspirational
    rationale: "D-10-07: v2.2 target"
    hard:
      metric: committed_itinerary_rate
      op: ">="
      value: 0.6
    advisory: []

  - family: deepseek/deepseek-chat
    status: logged
    rationale: "D-10-07: logged-not-gated"
    hard: null
    advisory: []

  - family: late_night_closure_cascade
    status: quarantined-legacy-threading
    rationale: "D-10-09: legacy"
    hard: null
    advisory: []
"""


_SYNTHETIC_SCENARIO_ID = "refinement_cheaper"

_INTEGRATION_GATES_YAML = """\
gates:
  - family: openai/gpt-4o-mini
    status: active
    rationale: "integration test gate on a whitelisted scorer"
    hard:
      metric: category_compliance
      op: ">="
      value: 0.8
    advisory: []
"""


def _make_summary(
    provider_cells: dict[str, dict],
    extra_top_level: dict | None = None,
) -> dict:
    """Build a minimal summary.json-shaped dict using the real nested shape.

    provider_cells: mapping of provider_key → per-cell dict (at minimum
    {"scorers": {}, "n_scored": N, "n_errored": 0}).

    The real ``aggregate_cell_jsons`` shape nests the provider map under a
    ``scenarios`` block — this helper mirrors that so every exit-code test
    exercises the actual code path:

        {"scenarios": {"<id>": {"providers": {...}}}, "errors": []}
    """
    out: dict = {
        "scenarios": {
            _SYNTHETIC_SCENARIO_ID: {
                "providers": provider_cells,
            }
        },
        "errors": [],
    }
    if extra_top_level:
        out.update(extra_top_level)
    return out


def _cell_with_rate(rate: float, n: int = 5) -> dict:
    """A cell dict with committed_itinerary_rate populated."""
    return {
        "scorers": {
            "committed_itinerary_rate": {"median": rate, "mean": rate},
        },
        "n_scored": n,
        "n_errored": 0,
        "cell_valid": True,
    }


def _cell_no_rate(n: int = 5) -> dict:
    """A cell dict WITHOUT committed_itinerary_rate — metric not yet wired."""
    return {
        "scorers": {
            "refinement_minimal_edit": {"median": 0.5, "mean": 0.5},
        },
        "n_scored": n,
        "n_errored": 0,
        "cell_valid": True,
    }


# ---------------------------------------------------------------------------
# Exit-code test: exit 0 when all active gates pass
# ---------------------------------------------------------------------------


def test_all_gates_pass_returns_exit_0(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """All active/provisional hard gates satisfied → exit 0."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    summary = _make_summary(
        {
            "openai/gpt-4o-mini": _cell_with_rate(1.0),
            "openai/gpt-5-mini": _cell_with_rate(0.7),  # above aspirational floor too
            "deepseek/deepseek-chat": _cell_with_rate(0.2),  # logged; ignored
            "late_night_closure_cascade": _cell_with_rate(0.0),  # quarantined; ignored
        }
    )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 0, "all active gates passing must return exit 0"
    captured = capsys.readouterr()
    assert "OK" in captured.out or "ok" in captured.out.lower()


# ---------------------------------------------------------------------------
# Exit-code test: exit 1 when active gate fails
# ---------------------------------------------------------------------------


def test_active_hard_gate_violation_returns_exit_1_with_family_in_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """Active gpt-4o-mini cell below gate → exit 1 with family name in stderr."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    summary = _make_summary(
        {
            "openai/gpt-4o-mini": _cell_with_rate(0.4),  # BELOW 0.8 gate
            "openai/gpt-5-mini": _cell_with_rate(0.7),
        }
    )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 1, "active gate violation must return exit 1"
    captured = capsys.readouterr()
    assert "openai/gpt-4o-mini" in captured.err, (
        f"family name must appear in stderr: {captured.err!r}"
    )
    assert "HARD GATE VIOLATION" in captured.err or "VIOLATION" in captured.err, (
        f"violation signal must appear in stderr: {captured.err!r}"
    )


# ---------------------------------------------------------------------------
# Exit-code test: aspirational miss returns exit 0 with ASPIRATIONAL line
# ---------------------------------------------------------------------------


def test_aspirational_gate_miss_returns_exit_0_with_aspirational_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """Aspirational gpt-5-mini below its floor → exit 0, ASPIRATIONAL line in stdout."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    summary = _make_summary(
        {
            "openai/gpt-4o-mini": _cell_with_rate(1.0),  # passes active gate
            "openai/gpt-5-mini": _cell_with_rate(0.4),  # BELOW aspirational 0.6
        }
    )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 0, "aspirational gate miss must NOT hard-fail (exit 0)"
    captured = capsys.readouterr()
    assert "ASPIRATIONAL" in captured.out, (
        f"ASPIRATIONAL miss must be reported to stdout: {captured.out!r}"
    )
    # Must NOT appear in stderr (would imply it's being treated as a hard fail)
    assert "openai/gpt-5-mini" not in captured.err, (
        f"aspirational family must not appear in stderr: {captured.err!r}"
    )


# ---------------------------------------------------------------------------
# Exit-code test: infrastructure failure returns exit 2
# ---------------------------------------------------------------------------


def test_missing_gates_yaml_returns_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """Missing gates YAML → exit 2 (infrastructure failure)."""
    summary = _make_summary({"openai/gpt-4o-mini": _cell_with_rate(1.0)})
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    nonexistent = tmp_path / "nonexistent_gates.yaml"

    rc = script.main([str(summary_file), "--gates-config", str(nonexistent)])
    assert rc == 2, "missing gates YAML must return exit 2"


def test_missing_summary_json_returns_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """Missing summary.json → exit 2 (infrastructure failure)."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    nonexistent = tmp_path / "nonexistent_summary.json"

    rc = script.main([str(nonexistent), "--gates-config", str(gates_file)])
    assert rc == 2, "missing summary.json must return exit 2"


def test_malformed_summary_json_returns_exit_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """Malformed (non-JSON) summary.json → exit 2 (fail-closed, not silent pass)."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    summary_file = tmp_path / "summary.json"
    summary_file.write_text("this is not valid JSON {{{")

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 2, "malformed summary.json must fail closed with exit 2"


# ---------------------------------------------------------------------------
# logged and quarantined families never block
# ---------------------------------------------------------------------------


def test_logged_family_with_low_rate_never_blocks(
    tmp_path: Path,
    script: ModuleType,
) -> None:
    """logged families are skipped entirely — even a 0.0 rate must not block."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    summary = _make_summary(
        {
            "openai/gpt-4o-mini": _cell_with_rate(1.0),
            "deepseek/deepseek-chat": _cell_with_rate(0.0),  # logged; must not block
        }
    )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 0, "logged family with low rate must not block (exit 0)"


def test_quarantined_family_never_blocks(
    tmp_path: Path,
    script: ModuleType,
) -> None:
    """quarantined-legacy-threading families are skipped entirely."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    summary = _make_summary(
        {
            "openai/gpt-4o-mini": _cell_with_rate(1.0),
            "late_night_closure_cascade": _cell_with_rate(0.0),  # quarantined; must not block
        }
    )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 0, "quarantined family must not block (exit 0)"


# ---------------------------------------------------------------------------
# Not-evaluable: metric absent from summary → reported, not silent pass
# ---------------------------------------------------------------------------


def test_metric_not_in_summary_is_not_evaluable_not_silent_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """When committed_itinerary_rate is absent from the cell, the gate is
    reported as not-evaluable — not treated as a silent pass (T-10-04-01)."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    # Cell only has refinement_minimal_edit; committed_itinerary_rate absent
    summary = _make_summary(
        {
            "openai/gpt-4o-mini": _cell_no_rate(),
        }
    )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    script.main([str(summary_file), "--gates-config", str(gates_file)])
    # Must NOT silently pass (exit 0 without reporting) — must report not-evaluable
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert (
        "not-evaluable" in combined.lower()
        or "not evaluable" in combined.lower()
        or "evaluable" in combined.lower()
    ), f"not-evaluable condition must be reported: out={captured.out!r} err={captured.err!r}"
    # Not-evaluable is a non-zero exit (not a silent pass) — either 0-with-warning
    # or 1/2; the key contract is it's REPORTED, not silently pass.
    # Per plan: "reported (not a silent pass)" — checking the report above is sufficient.


# ---------------------------------------------------------------------------
# Family absent from summary (cell not run) → treated as not-evaluable
# ---------------------------------------------------------------------------


def test_family_absent_from_summary_is_not_evaluable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """If a family with an active hard gate has no cell in the summary,
    the gate is not-evaluable (not a silent pass)."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    # gpt-4o-mini is missing from summary entirely
    summary = _make_summary(
        {
            "openai/gpt-5-mini": _cell_with_rate(0.7),
        }
    )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    script.main([str(summary_file), "--gates-config", str(gates_file)])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert (
        "not-evaluable" in combined.lower()
        or "evaluable" in combined.lower()
        or "missing" in combined.lower()
    ), f"absent family must not be a silent pass: out={captured.out!r} err={captured.err!r}"


# ---------------------------------------------------------------------------
# main() signature: returns int, callable with argv list
# ---------------------------------------------------------------------------


def test_main_returns_int(tmp_path: Path, script: ModuleType) -> None:
    """main(argv) must return an int (not raise SystemExit directly)."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)
    summary = _make_summary({"openai/gpt-4o-mini": _cell_with_rate(1.0)})
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    result = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert isinstance(result, int), f"main() must return int, got {type(result)}"


# ---------------------------------------------------------------------------
# Integration test: real aggregate_cell_jsons output fires hard gate (CR-01)
# ---------------------------------------------------------------------------


def test_integration_real_aggregate_output_fires_hard_gate(
    tmp_path: Path,
    script: ModuleType,
) -> None:
    """End-to-end: aggregate_cell_jsons() output fed to check_eval_gates → exit 1.

    This is the test that would have caught CR-01.  The old flat
    summary.get('providers', {}) lookup never found any cell in the real
    aggregate_cell_jsons shape, so make eval-gates-check always exited 0.

    Uses category_compliance as the gated scorer because it is registered in
    CRITIQUE_THRESHOLDS and therefore survives the _scorer_means_from_cell
    whitelist (committed_itinerary_rate is wired in Phase 11 BASE-01; until
    then, testing the end-to-end path with a whitelisted scorer is sufficient
    to prove the nested-shape fix closes CR-01).
    """
    from scripts.eval_matrix import aggregate_cell_jsons

    # Write the per-cell JSON that _write_cell_with_aggregate would produce.
    # The aggregator reads aggregate.{scorer}_mean and strips the _mean suffix.
    cell_dir = tmp_path / "cells"
    cell_dir.mkdir()
    cell_path = cell_dir / "openai--gpt-4o-mini--refinement_cheaper--run-0.json"
    cell_payload = {
        "llm_provider": "openai",
        "chat_model": "gpt-4o-mini",
        "query_count": 1,
        "aggregate": {
            # 0.4 < 0.8 gate — must trigger violation
            "category_compliance_mean": 0.4,
        },
        "queries": [{"id": "refinement_cheaper"}],
    }
    cell_path.write_text(json.dumps(cell_payload), encoding="utf-8")

    # Build the real aggregate_cell_jsons summary.
    summary = aggregate_cell_jsons(cell_dir)

    # Confirm the aggregator produced the nested shape with the scorer present.
    assert "scenarios" in summary, "aggregate_cell_jsons must produce a 'scenarios' top-level key"
    scenario_block = summary["scenarios"]["refinement_cheaper"]
    provider_block = scenario_block["providers"]["openai/gpt-4o-mini"]
    assert "category_compliance" in provider_block["scorers"], (
        "category_compliance scorer must survive the _scorer_means_from_cell whitelist"
    )
    assert provider_block["scorers"]["category_compliance"]["median"] == pytest.approx(0.4)

    # Write the summary to disk.
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary), encoding="utf-8")

    # Write the integration gate config.
    gates_file = tmp_path / "integration_gates.yaml"
    gates_file.write_text(_INTEGRATION_GATES_YAML)

    # The gate must fire (exit 1) against the real aggregator output.
    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 1, (
        f"Hard gate must fire (exit 1) against real aggregate_cell_jsons output; got exit {rc}. "
        "This is the CR-01 regression: if the checker reads a flat top-level 'providers' key "
        "it will find nothing and exit 0 instead of 1."
    )


# ---------------------------------------------------------------------------
# TDD RED: _check_gate must walk nested scenarios->providers shape (CR-01)
# ---------------------------------------------------------------------------


def test_check_gate_fires_on_nested_scenarios_providers_shape(script: ModuleType) -> None:
    """_check_gate must locate a cell under summary['scenarios'][*]['providers'][family].

    CR-01: the old flat summary.get('providers', {}) lookup never finds the cell
    because aggregate_cell_jsons writes the nested shape.  This test is the RED
    gate — it fails against the broken implementation and passes after the fix.
    """
    gate = {
        "family": "openai/gpt-4o-mini",
        "status": "active",
        "hard": {
            "metric": "committed_itinerary_rate",
            "op": ">=",
            "value": 0.8,
        },
    }
    # Real aggregate_cell_jsons shape — nested under scenarios->providers.
    nested_summary = {
        "generated_at": "2026-01-01T00:00:00Z",
        "scenarios": {
            "refinement_cheaper": {
                "providers": {
                    "openai/gpt-4o-mini": _cell_with_rate(0.4),
                }
            }
        },
    }
    result = script._check_gate(gate, nested_summary)
    assert result == "violation", (
        f"_check_gate must return 'violation' for below-gate cell in nested shape; got {result!r}"
    )


def test_check_gate_passes_on_nested_shape_above_gate(script: ModuleType) -> None:
    """_check_gate returns 'pass' when cell is at or above gate in nested shape."""
    gate = {
        "family": "openai/gpt-4o-mini",
        "status": "active",
        "hard": {
            "metric": "committed_itinerary_rate",
            "op": ">=",
            "value": 0.8,
        },
    }
    nested_summary = {
        "scenarios": {
            "refinement_cheaper": {
                "providers": {
                    "openai/gpt-4o-mini": _cell_with_rate(1.0),
                }
            }
        }
    }
    result = script._check_gate(gate, nested_summary)
    assert result == "pass", (
        f"_check_gate must return 'pass' for above-gate cell in nested shape; got {result!r}"
    )


def test_check_gate_not_evaluable_when_family_absent_from_all_scenarios(
    script: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_check_gate returns 'not_evaluable' when no scenario's providers map contains the family."""
    gate = {
        "family": "openai/gpt-4o-mini",
        "status": "active",
        "hard": {
            "metric": "committed_itinerary_rate",
            "op": ">=",
            "value": 0.8,
        },
    }
    nested_summary = {
        "scenarios": {
            "refinement_cheaper": {
                "providers": {
                    "deepseek/deepseek-chat": _cell_with_rate(0.9),
                }
            }
        }
    }
    result = script._check_gate(gate, nested_summary)
    assert result == "not_evaluable", f"absent family must return 'not_evaluable', got {result!r}"
    captured = capsys.readouterr()
    assert "NOT-EVALUABLE" in captured.out, (
        f"NOT-EVALUABLE must be printed to stdout; got: {captured.out!r}"
    )
    assert "openai/gpt-4o-mini" in captured.out, (
        f"family name must appear in NOT-EVALUABLE message; got: {captured.out!r}"
    )


def test_check_gate_skips_quarantined_scenario_for_cell_lookup(script: ModuleType) -> None:
    """_check_gate must skip scenarios where baseline_eligible is explicitly False.

    A quarantined scenario's cell must not satisfy a gate — if the only cell
    for the family is in a quarantined scenario, the result is not_evaluable.
    """
    gate = {
        "family": "openai/gpt-4o-mini",
        "status": "active",
        "hard": {
            "metric": "committed_itinerary_rate",
            "op": ">=",
            "value": 0.8,
        },
    }
    nested_summary = {
        "scenarios": {
            "late_night_closure_cascade": {
                "baseline_eligible": False,
                "providers": {
                    "openai/gpt-4o-mini": _cell_with_rate(1.0),  # above gate but quarantined
                },
            }
        }
    }
    result = script._check_gate(gate, nested_summary)
    assert result == "not_evaluable", (
        f"quarantined scenario's cell must not satisfy gate; got {result!r}"
    )


# ---------------------------------------------------------------------------
# WR-02: cell lookup must evaluate every eligible scenario, not first-match-wins
# ---------------------------------------------------------------------------


def test_hard_gate_fires_when_any_eligible_scenario_violates_regardless_of_order(
    tmp_path: Path,
    script: ModuleType,
) -> None:
    """WR-02: when a family has cells in more than one eligible scenario and
    ANY of them fails the hard gate, the checker must exit 1 — in both
    scenario-id orders. First-match-wins lookup makes the verdict depend on
    alphabetical scenario naming, which a merge gate must never do.
    """
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    passing = _cell_with_rate(1.0)
    failing = _cell_with_rate(0.4)  # below the 0.8 active gate

    for label, (first_cell, second_cell) in {
        "pass_then_fail": (passing, failing),
        "fail_then_pass": (failing, passing),
    }.items():
        summary = {
            "scenarios": {
                "a_first": {"providers": {"openai/gpt-4o-mini": first_cell}},
                "b_second": {"providers": {"openai/gpt-4o-mini": second_cell}},
            },
            "errors": [],
        }
        summary_file = tmp_path / f"summary_{label}.json"
        summary_file.write_text(json.dumps(summary, sort_keys=True))

        rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
        assert rc == 1, (
            f"{label}: a failing cell in ANY eligible scenario must exit 1 "
            f"(got rc={rc}) — gate verdict must not depend on scenario-id order"
        )


def test_hard_gate_passes_when_all_eligible_scenarios_pass(
    tmp_path: Path,
    script: ModuleType,
) -> None:
    """WR-02 complement: multiple eligible scenarios all above the gate → exit 0."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    summary = {
        "scenarios": {
            "a_first": {"providers": {"openai/gpt-4o-mini": _cell_with_rate(0.9)}},
            "b_second": {"providers": {"openai/gpt-4o-mini": _cell_with_rate(1.0)}},
        },
        "errors": [],
    }
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary, sort_keys=True))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 0, f"all eligible scenarios above gate must exit 0 (got rc={rc})"


# ---------------------------------------------------------------------------
# WR-03: unknown gate status must be rejected at load time, never fail-open
# ---------------------------------------------------------------------------


def test_unknown_gate_status_exits_2_not_silent_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    script: ModuleType,
) -> None:
    """WR-03: a typo'd status ('activ') must NOT silently disable a hard gate.
    Unknown status vocabulary is an infrastructure failure → exit 2 with a
    diagnostic, never exit 0.
    """
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(
        """\
gates:
  - family: openai/gpt-4o-mini
    status: activ
    rationale: "typo'd status"
    hard:
      metric: committed_itinerary_rate
      op: ">="
      value: 0.8
    advisory: []
"""
    )

    summary = _make_summary({"openai/gpt-4o-mini": _cell_with_rate(0.0)})
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 2, (
        f"unknown status 'activ' must exit 2 (infra failure), got rc={rc} — "
        "a one-character typo must never disable a hard gate"
    )
    captured = capsys.readouterr()
    assert "activ" in captured.err, "diagnostic must name the unknown status"


# ---------------------------------------------------------------------------
# WR-04: malformed gates YAML / summary values → exit 2, never exit 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "gates_yaml"),
    [
        ("null_gates_list", "gates:\n"),
        (
            "entry_missing_family",
            """\
gates:
  - status: active
    rationale: "no family key"
    hard:
      metric: committed_itinerary_rate
      op: ">="
      value: 0.8
""",
        ),
        (
            "hard_missing_metric",
            """\
gates:
  - family: openai/gpt-4o-mini
    status: active
    rationale: "hard block missing metric"
    hard:
      op: ">="
      value: 0.8
""",
        ),
        (
            "unknown_op",
            """\
gates:
  - family: openai/gpt-4o-mini
    status: active
    rationale: "bad op"
    hard:
      metric: committed_itinerary_rate
      op: "~="
      value: 0.8
""",
        ),
    ],
)
def test_malformed_gates_yaml_exits_2_not_1(
    tmp_path: Path,
    script: ModuleType,
    label: str,
    gates_yaml: str,
) -> None:
    """WR-04: structural defects in the gates config are infrastructure
    failures (exit 2), not hard-gate violations (exit 1) and not raw
    tracebacks. CI callers distinguish the two.
    """
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(gates_yaml)

    summary = _make_summary({"openai/gpt-4o-mini": _cell_with_rate(1.0)})
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 2, f"{label}: malformed gates config must exit 2, got rc={rc}"


def test_null_median_in_summary_exits_2_not_traceback(
    tmp_path: Path,
    script: ModuleType,
) -> None:
    """WR-04: a null median in the summary (float(None) → TypeError) must be
    reported as infra failure exit 2, not escape as a traceback."""
    gates_file = tmp_path / "eval_gates.yaml"
    gates_file.write_text(_MINIMAL_GATES_YAML)

    cell = {
        "scorers": {"committed_itinerary_rate": {"median": None}},
        "n_scored": 5,
        "n_errored": 0,
        "cell_valid": True,
    }
    summary = _make_summary({"openai/gpt-4o-mini": cell})
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(json.dumps(summary))

    rc = script.main([str(summary_file), "--gates-config", str(gates_file)])
    assert rc == 2, f"null median must exit 2 (infra failure), got rc={rc}"
