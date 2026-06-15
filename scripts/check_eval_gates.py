#!/usr/bin/env python3
"""Gate-check script for eval_gates.yaml (EVAL-03 / D-10-05).

Reads a matrix summary.json and exits non-zero on any hard-gate violation.

Usage:
    poetry run python scripts/check_eval_gates.py eval_reports/{ts}/summary.json
    make eval-gates-check SUMMARY=eval_reports/{ts}/summary.json
    make eval-gates-check-baselines   # D-11-15: reads committed baselines (no live keys)

Exit codes (matching check_baselines_fresh.py exactly):
    0 = all hard gates passed (aspirational misses printed, non-blocking)
    1 = one or more hard-gate violations
    2 = infrastructure failure (missing YAML, bad summary.json shape, unknown
        gate status, malformed gate entry, or null/non-numeric metric values)

Gate semantics:
    active / provisional-n1 — hard gate; violation → exit 1
    aspirational — reported to stdout but non-blocking; exit 0 on miss
    logged / quarantined-legacy-threading — skipped entirely

Not-evaluable condition:
    When committed_itinerary_rate is absent from a cell's scorers block
    (Phase 10 — the metric is wired by Phase 11 BASE-01), the gate is
    reported as not-evaluable rather than silently passing (T-10-04-01).
    Similarly, if a family with a hard gate has no cell in the summary at
    all, it is treated as not-evaluable and reported.

    The cells for a family are located by walking each scenario block under
    ``summary['scenarios']`` and looking up the family under
    ``scenario_block['providers']``.  Scenarios whose ``baseline_eligible``
    flag is explicitly ``False`` (D-10-09 quarantined scenarios) are skipped
    during this walk — a quarantined scenario's cell neither satisfies nor
    violates a gate.  The gate is evaluated against EVERY eligible scenario
    carrying the family (fail-closed): if any eligible scenario's cell fails,
    the gate fails, regardless of scenario-id ordering.

Baselines mode (D-11-15):
    When --baselines-mode is set, the script reads committed
    configs/eval_baselines/*.json and synthesises a summary-shaped dict via
    _build_summary_from_baselines.  The _check_gate loop is unchanged — only
    the input source is swapped.  _snapshots/ subdirectory files are skipped.
    Per-scenario ``baseline_eligible`` flags are resolved from
    --eval-queries (WR-03) the same way summary-mode's aggregate_cell_jsons
    does, so D-10-09 quarantined scenarios neither satisfy nor violate gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# No top-level LLM SDK imports — this script is the checker, not the caller.

_HARD_STATUSES = {"active", "provisional-n1"}
_SKIP_STATUSES = {"logged", "quarantined-legacy-threading"}
# WR-03: the full status vocabulary. Anything else is rejected at load time —
# a typo'd status must surface as exit 2, never silently disable a hard gate.
_VALID_STATUSES = _HARD_STATUSES | _SKIP_STATUSES | {"aspirational"}


def _load_gates(gates_path: str) -> dict:
    """Load and return the gates YAML as a Python dict.

    Raises OSError on missing file, ValueError on parse failure.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise OSError("PyYAML not available — install with 'poetry install'") from exc

    p = Path(gates_path)
    if not p.exists():
        raise OSError(f"gates config not found: {gates_path}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse gates YAML at {gates_path}: {exc}") from exc
    if not isinstance(data, dict) or "gates" not in data:
        raise ValueError(f"gates YAML at {gates_path} missing top-level 'gates' key")
    if not isinstance(data["gates"], list):
        raise ValueError(f"gates YAML at {gates_path}: 'gates' must be a list of gate entries")
    for idx, gate in enumerate(data["gates"]):
        if not isinstance(gate, dict):
            raise ValueError(f"gates YAML at {gates_path}: gate entry #{idx} is not a mapping")
        status = gate.get("status", "logged")
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"gates YAML at {gates_path}: unknown status {status!r} on gate entry "
                f"{gate.get('family', f'#{idx}')!r}; valid statuses: {sorted(_VALID_STATUSES)}"
            )
    return data


def _load_baseline_eligibility(eval_queries_path: str) -> dict[str, bool]:
    """Resolve scenario_id -> baseline_eligible from eval_queries.yaml (D-10-09).

    WR-03: baselines-mode must honor the quarantine the same way summary-mode
    does (aggregate_cell_jsons resolves the flag from the eval-queries config).
    Scenario IDs absent from the lookup default to True downstream — failing
    toward enforcement, not silent exclusion (T-10-03-03).

    Raises OSError on missing file, ValueError on parse failure.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise OSError("PyYAML not available — install with 'poetry install'") from exc

    p = Path(eval_queries_path)
    if not p.exists():
        raise OSError(f"eval queries config not found: {eval_queries_path}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"could not parse eval queries YAML at {eval_queries_path}: {exc}"
        ) from exc
    lookup: dict[str, bool] = {}
    if isinstance(data, dict):
        for case in data.get("hand_written") or []:
            if isinstance(case, dict) and case.get("id"):
                lookup[str(case["id"])] = bool(case.get("baseline_eligible", True))
    return lookup


def _build_summary_from_baselines(
    baselines_dir: Path,
    baseline_eligibility: dict[str, bool] | None = None,
) -> dict:
    """Synthesise a summary-shaped dict from committed baseline JSONs (D-11-15).

    Shape matches aggregate_cell_jsons output exactly so _check_gate can
    consume it without modification.  Files inside a ``_snapshots``
    subdirectory are skipped.

    Fail-closed contract (CR-02): this feeds the HARD CI gate. A missing
    baselines directory raises OSError and an empty result raises ValueError —
    both route through main's (OSError, ValueError) handler to exit 2. The
    T-10-04-01 "not-evaluable is non-blocking" design covers a single missing
    metric in an otherwise-present summary; an entirely missing input must
    never read as green (deleted baselines, renamed dir, typo'd
    --baselines-dir would all silently pass otherwise).

    Per-scenario ``baseline_eligible`` flags come from ``baseline_eligibility``
    (WR-03: resolved from eval_queries.yaml via _load_baseline_eligibility, so
    D-10-09 quarantined scenarios are skipped by _check_gate exactly as in
    summary-mode). Unknown scenario IDs default to True (T-10-03-03).

    Raises OSError on missing/unreadable directory, ValueError on malformed
    JSON or an empty baselines directory.
    """
    if not baselines_dir.is_dir():
        raise OSError(f"baselines directory not found: {baselines_dir}")
    eligibility = baseline_eligibility or {}
    scenarios_out: dict[str, Any] = {}
    for baseline_path in sorted(baselines_dir.glob("*.json")):
        # Skip any file whose immediate parent is _snapshots (pre-phase snapshots).
        if baseline_path.parent.name == "_snapshots":
            continue
        try:
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse baseline JSON at {baseline_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"baseline JSON at {baseline_path} is not an object")
        # Prefer the explicit scenario_id key; fall back to the filename stem
        # for pre-write_baselines legacy JSONs (e.g. refinement_cheaper.json
        # predates the scenario_id field added by write_baselines.py).
        scenario_id = payload.get("scenario_id") or baseline_path.stem
        if not scenario_id:
            raise ValueError(
                f"baseline JSON at {baseline_path} missing 'scenario_id' key "
                "and has no usable filename stem"
            )
        providers_out: dict[str, Any] = {}
        for provider_key, cell in payload.get("providers", {}).items():
            scorers = cell.get("scorers", {})
            # Derive n_scored from scorers.<any_metric>.n for completeness
            # (fallback to cell-level n_scored if the key is present).
            any_n = next(
                (v["n"] for v in scorers.values() if isinstance(v, dict) and "n" in v),
                0,
            )
            providers_out[provider_key] = {
                "scorers": scorers,  # verbatim — shape already correct
                "n_scored": cell.get("n_scored", any_n),
                "n_errored": 0,
                "cell_valid": True,
            }
        scenarios_out[scenario_id] = {
            "baseline_eligible": eligibility.get(scenario_id, True),
            "providers": providers_out,
        }
    if not scenarios_out:
        raise ValueError(
            f"no baseline JSONs found in {baselines_dir} — refusing to pass "
            "hard gates against an empty record (D-11-15 fail-closed)"
        )
    return {"scenarios": scenarios_out}


def _load_summary(summary_path: str) -> dict:
    """Load and return the summary.json as a Python dict.

    Raises OSError on missing file, ValueError on parse/shape failure.
    """
    p = Path(summary_path)
    if not p.exists():
        raise OSError(f"summary.json not found: {summary_path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse summary.json at {summary_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"summary.json at {summary_path} is not a JSON object")
    return data


def _get_metric_value(cell: dict, metric: str) -> float | None:
    """Extract metric value from a cell dict.

    Returns None if the metric is not present in the cell's scorers block.
    The cell shape produced by aggregate_cell_jsons nests the metric under
    ``scorers → <metric> → median``.
    """
    scorers = cell.get("scorers", {})
    if metric not in scorers:
        return None
    metric_block = scorers[metric]

    def _num(v: object) -> float | None:
        # Codex PR#110 finding B: bool is a subclass of int — float(True)==1.0
        # would let a malformed boolean median satisfy a >= gate. Reject it so a
        # malformed metric surfaces as not-present (caller treats as not-evaluable)
        # rather than a vacuous pass.
        if isinstance(v, bool):
            return None
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    if isinstance(metric_block, dict):
        # Prefer median; fall back to mean for robustness.
        if "median" in metric_block:
            return _num(metric_block["median"])
        if "mean" in metric_block:
            return _num(metric_block["mean"])
        return None
    # Scalar value (less common but handle gracefully)
    return _num(metric_block)


def _evaluate_op(value: float, op: str, threshold: float) -> bool:
    """Evaluate ``value op threshold``."""
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op == "==":
        return value == threshold
    raise ValueError(f"unknown gate op: {op!r}")


def _check_gate(
    gate: dict,
    summary: dict,
) -> str:
    """Check a single gate entry against the summary.

    Returns one of:
        "pass"                  — gate satisfied or gate is null
        "violation"             — active/provisional-n1 hard gate failed
        "aspirational_miss"     — aspirational gate failed (non-blocking)
        "not_evaluable"         — metric absent or cell absent (non-blocking)
        "skip"                  — logged or quarantined family

    The caller is responsible for routing each result to the correct output.
    """
    status = gate.get("status", "logged")

    # logged and quarantined-legacy-threading families are skipped entirely.
    if status in _SKIP_STATUSES:
        return "skip"

    hard = gate.get("hard")
    family = gate["family"]

    # Locate the cells for this family by walking the nested scenarios->providers shape
    # produced by aggregate_cell_jsons:
    #   summary['scenarios'][<scenario_id>]['providers'][<family>]
    # Quarantined scenarios (baseline_eligible=False, D-10-09) are skipped so their
    # cells neither satisfy nor violate a gate.
    #
    # The gate is evaluated against EVERY eligible scenario carrying the family —
    # never first-match-wins. A merge gate's verdict must not depend on scenario-id
    # ordering: if the family fails in any eligible scenario, the gate fails.
    #
    # WR-05: cell location is hoisted ABOVE the `hard is None` early return —
    # advisory entries need cells regardless of whether a hard block exists.
    cells: list[tuple[str, dict]] = []
    for scenario_id, scenario_block in summary.get("scenarios", {}).items():
        if scenario_block.get("baseline_eligible", True) is False:
            continue
        candidate = scenario_block.get("providers", {}).get(family)
        if candidate is not None:
            cells.append((scenario_id, candidate))

    # WR-05 / D-11-17: evaluate advisory entries — report-only WARN, never
    # blocking. Advisory results MUST NOT be added to violations or change the
    # exit code. Evaluated BEFORE the hard-gate early returns so an
    # advisory-only entry (`hard: null` + non-empty advisory — e.g. the
    # documented anthropic promotion path's intermediate state) still reports.
    advisory_entries = gate.get("advisory") or []
    for adv in advisory_entries:
        adv_metric = adv.get("metric", "")
        # D-11-17: resolve the advisory metric name alias.
        # 'refinement_minimal_edit_median' is the gate-YAML name for the
        # refinement_minimal_edit scorer's median value.
        if adv_metric == "refinement_minimal_edit_median":
            adv_metric = "refinement_minimal_edit"
        adv_op = adv.get("op", ">=")
        adv_threshold = adv.get("value", 0.0)
        # Evaluate advisory against every eligible cell for this family.
        for _sid, cell in cells:
            adv_value = _get_metric_value(cell, adv_metric)
            if adv_value is not None and not _evaluate_op(adv_value, adv_op, adv_threshold):
                print(
                    f"check_eval_gates: ADVISORY miss — {family} "
                    f"{adv_metric} {adv_op} {adv_threshold} "
                    f"(actual={adv_value:.3f}) [non-blocking]"
                )
                break  # one ADVISORY line per advisory entry is sufficient

    # Gates with null hard block are trivially passing for the purpose of
    # exit-code calculation (advisory-only entries report above, never block).
    if hard is None:
        return "pass"

    metric = hard["metric"]
    op = hard["op"]
    threshold = hard["value"]
    # Optional: restrict gate evaluation to a specific subset of scenarios.
    # When hard.scenarios is present, only those scenario_ids are evaluated;
    # other eligible scenarios carrying the family are skipped for this gate.
    # This supports omakase-anchored gates that should not fail-close on
    # observational scenarios (e.g. refinement_cheaper) whose results are
    # tracked but not gated (D-15-06/07).
    hard_scenarios: list[str] | None = hard.get("scenarios") if isinstance(hard, dict) else None

    scoped_cells = (
        [(sid, cell) for sid, cell in cells if sid in hard_scenarios]
        if hard_scenarios is not None
        else cells
    )

    if not scoped_cells:
        # Family has no cell in any gated scenario — not-evaluable, not a silent pass.
        note = f"check_eval_gates: NOT-EVALUABLE — family '{family}' has no cell in summary"
        print(note)
        # Codex PR#110 finding 1: a HARD-status gate that cannot be evaluated is an
        # infrastructure failure (exit 2), never a silent pass. Only logged/aspirational
        # gates may be non-blocking when not-evaluable.
        return "not_evaluable_hard" if status in _HARD_STATUSES else "not_evaluable"

    # Extract the metric from every located cell. Cells lacking the metric do not
    # count toward the verdict; if NO cell carries it, the gate is not-evaluable.
    evaluable: list[tuple[str, float]] = []
    for scenario_id, cell in scoped_cells:
        value = _get_metric_value(cell, metric)
        if value is not None:
            evaluable.append((scenario_id, value))

    if not evaluable:
        # Metric absent from every eligible cell — not-evaluable, not a silent pass.
        # Codex PR#110 finding A: the Phase 11 "pre-wiring allowance" (metric not
        # yet added → non-blocking) is now STALE — committed_itinerary_rate is
        # wired (eval_gates.yaml header / D-11-02), and it is the only hard-gated
        # metric. So a HARD-status gate whose metric is absent from a present cell
        # is a fail-closed infrastructure failure (exit 2), never a vacuous pass.
        # Non-hard statuses (logged/aspirational) remain non-blocking.
        note = f"check_eval_gates: NOT-EVALUABLE — metric '{metric}' absent from cell '{family}'"
        print(note)
        return "not_evaluable_hard" if status in _HARD_STATUSES else "not_evaluable"

    # Evaluate the gate against every evaluable scenario — fail-closed on any miss.
    failing_scenarios = [
        scenario_id for scenario_id, value in evaluable if not _evaluate_op(value, op, threshold)
    ]

    if not failing_scenarios:
        gate_result = "pass"
    else:
        print(
            f"check_eval_gates: gate '{family}' {metric} {op} {threshold} failed in "
            f"scenario(s): {sorted(failing_scenarios)}"
        )

        # Gate failed — classify by status.
        if status in _HARD_STATUSES:
            gate_result = "violation"
        elif status == "aspirational":
            gate_result = "aspirational_miss"
        else:
            # Statuses are validated at load time (_load_gates, WR-03); reaching here
            # means a caller bypassed validation — fail closed, never silently pass.
            raise ValueError(f"unknown gate status {status!r} for family {family!r}")

    return gate_result


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse argv. Both positional summary and --gates-config flag are accepted."""
    parser = argparse.ArgumentParser(
        prog="check_eval_gates",
        description=(
            "Gate-check script for eval_gates.yaml (EVAL-03 / D-10-05). "
            "Reads a matrix summary.json and exits non-zero on any hard-gate violation. "
            "Use --baselines-mode to read committed configs/eval_baselines/*.json instead "
            "(D-11-15: live-key-free CI enforcement)."
        ),
    )
    parser.add_argument(
        "summary",
        nargs="?",
        default=None,
        help=(
            "Path to summary.json from an eval_matrix run. Required unless --baselines-mode is set."
        ),
    )
    parser.add_argument(
        "--gates-config",
        default="configs/eval_gates.yaml",
        help="Path to eval_gates.yaml (default: configs/eval_gates.yaml).",
    )
    parser.add_argument(
        "--baselines-mode",
        action="store_true",
        default=False,
        help=(
            "Read committed configs/eval_baselines/*.json instead of a live summary.json "
            "(D-11-15: CI live-key-free gate enforcement)."
        ),
    )
    parser.add_argument(
        "--baselines-dir",
        default="configs/eval_baselines",
        help="Baseline JSON directory (used with --baselines-mode; default: configs/eval_baselines).",
    )
    parser.add_argument(
        "--eval-queries",
        default="configs/eval_queries.yaml",
        help=(
            "Path to eval_queries.yaml for D-10-09 baseline_eligible resolution "
            "(used with --baselines-mode; default: configs/eval_queries.yaml)."
        ),
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the script's exit code.

    Exit codes:
        0 = all hard gates passed (aspirational misses reported, non-blocking)
        1 = one or more hard-gate violations (active or provisional-n1)
        2 = infrastructure failure (missing YAML, unreadable or malformed summary.json)
    """
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    try:
        gates_cfg = _load_gates(args.gates_config)
        if args.baselines_mode:
            # D-11-15: read committed baseline JSONs instead of a live summary.json.
            # WR-03: resolve D-10-09 quarantine flags from eval_queries.yaml so a
            # quarantined baseline file neither satisfies nor violates a gate.
            eligibility = _load_baseline_eligibility(args.eval_queries)
            summary = _build_summary_from_baselines(Path(args.baselines_dir), eligibility)
        else:
            if args.summary is None:
                sys.stderr.write(
                    "check_eval_gates: error: the 'summary' argument is required "
                    "unless --baselines-mode is set\n"
                )
                return 2
            summary = _load_summary(args.summary)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"check_eval_gates: {exc}\n")
        return 2

    violations: list[str] = []
    aspirational_misses: list[str] = []
    not_evaluable_hard: list[str] = []

    # WR-04: structural defects in a gate entry or in summary cell values
    # (missing keys, null medians, non-dict scenario blocks, unknown ops) are
    # infrastructure failures — they must exit 2, never masquerade as a
    # hard-gate violation (exit 1) or escape as a raw traceback.
    try:
        for gate in gates_cfg["gates"]:
            result = _check_gate(gate, summary)
            if result == "violation":
                violations.append(gate["family"])
            elif result == "aspirational_miss":
                aspirational_misses.append(gate["family"])
            elif result == "not_evaluable_hard":
                not_evaluable_hard.append(gate["family"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        sys.stderr.write(f"check_eval_gates: malformed gates config or summary shape: {exc!r}\n")
        return 2

    if aspirational_misses:
        print(f"check_eval_gates: ASPIRATIONAL miss (not blocking): {sorted(aspirational_misses)}")

    # Codex PR#110 finding 1: a hard-status gate that could not be evaluated (its
    # scoped scenario cell is absent) is an infrastructure failure, never a silent
    # pass. Surfaces before the violation check so a misconfigured scoped gate is
    # caught even when no other gate violated.
    if not_evaluable_hard:
        sys.stderr.write(
            f"check_eval_gates: HARD GATE NOT EVALUABLE (scoped cell missing): "
            f"{sorted(not_evaluable_hard)}\n"
        )
        return 2

    if violations:
        sys.stderr.write(f"check_eval_gates: HARD GATE VIOLATION: {sorted(violations)}\n")
        return 1

    print(f"check_eval_gates: OK — {len(gates_cfg['gates'])} gates checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
