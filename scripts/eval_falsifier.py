#!/usr/bin/env python3
"""Falsifier report for Phase 12 / v2.2 milestone (INST-05 / D-12-06..08).

Reads a completed eval_reports run directory (latest by default) and answers:
  - Did openai/gpt-5-mini hit >= 0.6 committed_itinerary_rate (median-weighted
    across scenarios; see _pooled_commit_rate) pooled over
    all scored scenarios? (D-12-08: pooled across configs/eval_matrix.yaml cells)
  - Did openai/gpt-4o-mini hold >= its honest baseline floor?

Does NOT fan out live API calls (D-12-06). Run AFTER `make eval-matrix`.

Usage:
    poetry run python scripts/eval_falsifier.py
    poetry run python scripts/eval_falsifier.py --run-dir eval_reports/{ts}
    poetry run python scripts/eval_falsifier.py --baselines-mode --baselines-dir configs/eval_baselines
    make eval-falsifier
    make eval-falsifier RUN_DIR=eval_reports/{ts}

Exit codes:
    0 = PASS — all falsifier checks met
    1 = FAIL — one or more checks failed (expected; not an infra error)
    2 = infrastructure failure (missing run dir, malformed JSON)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# No top-level LLM SDK imports — artifact-reading only (D-12-06)

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_BASE = REPO_ROOT / "eval_reports"

# INST-05 / D-12-08: pooled committed_itinerary_rate bar for gpt-5-mini
_FALSIFIER_BAR = 0.6

# Provider keys used in falsifier checks
_GPT5_KEY = "openai/gpt-5-mini"
_ANCHOR_KEY = "openai/gpt-4o-mini"
_ANCHOR_METRIC = "committed_itinerary_rate"


def _latest_run_dir(base: Path) -> Path:
    """Return the most recently created subdirectory in base (latest eval run).

    ISO8601 filenames sort lexicographically, so the last entry is the newest.
    Raises OSError if no subdirectories are found.
    """
    if not base.is_dir():
        raise OSError(f"output base directory not found: {base}")
    candidates = sorted(
        (d for d in base.iterdir() if d.is_dir()),
        key=lambda d: d.name,  # ISO8601 filenames sort lexicographically
    )
    if not candidates:
        raise OSError(f"no run directories found under {base}")
    return candidates[-1]


def _pooled_commit_rate(
    summary: dict[str, Any],
    provider_key: str,
    scenario_ids: set[str] | None = None,
) -> tuple[float | None, dict[str, float | None]]:
    """Return (pooled_rate, per_scenario_rates) for one provider across all scored scenarios.

    D-12-08: pool across ALL scenarios in summary['scenarios'] whose
    baseline_eligible is not explicitly False. Weights each scenario's median
    by its n (number of runs) so scenarios with more runs have proportionally
    more influence on the pooled rate.

    WR-03 HONESTY NOTE: this is a MEDIAN-WEIGHTED average
    (``sum(median_s * n_s) / sum(n_s)``), NOT a true pooled per-run rate.
    summary.json scorer blocks carry only {median,min,max,stdev,n} — per-run
    values are unavailable here — so every run in a scenario is treated as if
    it scored the scenario median. With a single scenario and binary per-run
    rates the verdict is equivalent to a true pool; with multiple scenarios or
    fractional rates it can diverge (e.g. run-rates [0,0,0.7,0.8,1.0] pool to
    0.7 via median vs 0.5 true mean). Plan 12-03 mandates median-weighting;
    the printed label says "median-weighted" so operators are not misled.

    When ``scenario_ids`` is provided, pooling is restricted to that set —
    used by the anchor non-regression check to compare the run and the
    committed baselines over the SAME scenario universe (CR-01: an
    eval_matrix.yaml run covers fewer scenarios than the committed baselines,
    e.g. refinement_cheaper exists only as a baseline, so pooling each side
    over its own full universe gives apples-to-oranges floors).

    Returns (None, per_scenario) when the provider has no evaluable cells at all.
    """
    total_commits = 0.0
    total_runs = 0
    per_scenario: dict[str, float | None] = {}

    for scenario_id, scenario_block in summary.get("scenarios", {}).items():
        if scenario_ids is not None and scenario_id not in scenario_ids:
            continue
        # D-10-09: skip quarantined scenarios
        if scenario_block.get("baseline_eligible", True) is False:
            continue
        cell = scenario_block.get("providers", {}).get(provider_key)
        if cell is None:
            per_scenario[scenario_id] = None
            continue

        # committed_itinerary_rate is wired into scorers by eval_matrix.py D-11-02
        scorers = cell.get("scorers", {})
        cir_block = scorers.get("committed_itinerary_rate")
        if cir_block is None:
            per_scenario[scenario_id] = None
            continue

        median = cir_block.get("median")
        n = cir_block.get("n", 0)

        # Guard against None/bool/non-numeric values
        if median is None or isinstance(median, bool) or not isinstance(median, (int, float)):
            per_scenario[scenario_id] = None
            continue

        per_scenario[scenario_id] = float(median)
        total_commits += float(median) * n
        total_runs += n

    pooled = total_commits / total_runs if total_runs > 0 else None
    return pooled, per_scenario


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse argv for the falsifier report."""
    parser = argparse.ArgumentParser(
        prog="eval_falsifier",
        description=__doc__,
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to a specific eval_reports run dir. Defaults to latest.",
    )
    parser.add_argument(
        "--baselines-mode",
        action="store_true",
        default=False,
        help=(
            "Read committed configs/eval_baselines/*.json instead of a live summary.json "
            "(live-key-free; enables CI use)."
        ),
    )
    parser.add_argument(
        "--baselines-dir",
        default="configs/eval_baselines",
        help="Baseline JSON directory for anchor non-regression (default: configs/eval_baselines).",
    )
    parser.add_argument(
        "--gates-config",
        default="configs/eval_gates.yaml",
        help="Path to eval_gates.yaml (default: configs/eval_gates.yaml).",
    )
    parser.add_argument(
        "--eval-queries",
        default="configs/eval_queries.yaml",
        help=(
            "Path to eval_queries.yaml for D-10-09 baseline_eligible resolution "
            "(default: configs/eval_queries.yaml)."
        ),
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Exit codes:
        0 = PASS — gpt-5-mini >= 0.6 pooled and gpt-4o-mini held baseline
        1 = FAIL — one or more checks failed (expected; not an infra error)
        2 = infrastructure failure (missing run dir, malformed JSON)
    """
    # Lazy import — only used here, not at module level (D-12-06 artifact-reading only)
    from scripts.check_eval_gates import (  # noqa: PLC0415
        _build_summary_from_baselines,
        _load_baseline_eligibility,
    )

    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # ── Resolve the summary dict ────────────────────────────────────────────
    try:
        if args.baselines_mode:
            # D-11-15 pattern: read committed baseline JSONs live-key-free
            eligibility = _load_baseline_eligibility(args.eval_queries)
            summary = _build_summary_from_baselines(Path(args.baselines_dir), eligibility)
        else:
            run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(_DEFAULT_OUTPUT_BASE)

            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                raise OSError(f"summary.json not found in {run_dir}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError(f"summary.json at {summary_path} is not a JSON object")
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"eval_falsifier: {exc}\n")
        return 2

    failed = False

    # ── Check 1: gpt-5-mini pooled committed_itinerary_rate >= 0.6 ─────────
    gpt5_pooled, gpt5_per_scenario = _pooled_commit_rate(summary, _GPT5_KEY)

    print(f"\n{'=' * 60}")
    print("eval_falsifier: INST-05 Milestone Falsifier Report")
    print(f"{'=' * 60}")

    # Per-scenario breakdown (D-12-08: always print)
    print(f"\n[{_GPT5_KEY}] committed_itinerary_rate per scenario:")
    for scenario_id, rate in gpt5_per_scenario.items():
        rate_str = f"{rate:.3f}" if rate is not None else "N/A"
        print(f"  {scenario_id}: {rate_str}")

    if gpt5_pooled is None:
        print(
            f"\n{_GPT5_KEY}: median-weighted committed_itinerary_rate = N/A (no evaluable cells)  FAIL"
        )
        failed = True
    elif gpt5_pooled >= _FALSIFIER_BAR:
        print(
            f"\n{_GPT5_KEY}: median-weighted committed_itinerary_rate = {gpt5_pooled:.3f}"
            f" >= {_FALSIFIER_BAR}  PASS"
        )
    else:
        print(
            f"\n{_GPT5_KEY}: median-weighted committed_itinerary_rate = {gpt5_pooled:.3f}"
            f" < {_FALSIFIER_BAR}  FAIL"
        )
        failed = True

    # ── Check 2: gpt-4o-mini anchor non-regression ──────────────────────────
    # Use the committed baselines as the reference floor — always, regardless of mode.
    # This means the anchor check is always grounded in the honest committed baseline,
    # not compared to itself when in --baselines-mode.
    try:
        baseline_eligibility_for_anchor = _load_baseline_eligibility(args.eval_queries)
        baselines_summary = _build_summary_from_baselines(
            Path(args.baselines_dir), baseline_eligibility_for_anchor
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"eval_falsifier: could not load baselines for anchor check: {exc}\n")
        return 2

    _, anchor_per_scenario = _pooled_commit_rate(summary, _ANCHOR_KEY)
    baseline_pooled_full, baseline_per_scenario = _pooled_commit_rate(
        baselines_summary, _ANCHOR_KEY
    )

    print(f"\n[{_ANCHOR_KEY}] committed_itinerary_rate per scenario (run vs baseline):")
    all_scenario_ids = set(anchor_per_scenario) | set(baseline_per_scenario)
    for scenario_id in sorted(all_scenario_ids):
        run_rate = anchor_per_scenario.get(scenario_id)
        base_rate = baseline_per_scenario.get(scenario_id)
        run_str = f"{run_rate:.3f}" if run_rate is not None else "N/A"
        base_str = f"{base_rate:.3f}" if base_rate is not None else "N/A"
        print(f"  {scenario_id}: run={run_str}  baseline={base_str}")

    # In baselines mode, both summaries are the same source — skip anchor regression check
    # (comparing baselines to themselves would always pass and is vacuous).
    if args.baselines_mode:
        # Use the baseline value itself as the anchor floor for display
        if baseline_pooled_full is not None:
            print(
                f"\n{_ANCHOR_KEY}: baselines-mode — median-weighted baseline = {baseline_pooled_full:.3f}"
                f" (anchor regression check skipped in baselines-mode)"
            )
        else:
            print(f"\n{_ANCHOR_KEY}: baselines-mode — median-weighted baseline = N/A")
    else:
        # Run-dir mode: compare run results against committed baseline floor.
        # CR-01: the run and the committed baselines cover DIFFERENT scenario
        # universes by construction (eval_matrix.yaml runs only a subset of the
        # baseline-eligible scenarios; refinement_cheaper exists only as a
        # baseline). Pool both sides over the INTERSECTION of scenarios that
        # have a rate in BOTH summaries, so the floor is apples-to-apples.
        common = {
            sid
            for sid in set(anchor_per_scenario) & set(baseline_per_scenario)
            if anchor_per_scenario[sid] is not None and baseline_per_scenario[sid] is not None
        }
        excluded = sorted((set(anchor_per_scenario) | set(baseline_per_scenario)) - common)
        if excluded:
            print(
                f"\n  note: scenarios excluded from anchor comparison"
                f" (missing on one side): {', '.join(excluded)}"
            )
        anchor_pooled, _ = _pooled_commit_rate(summary, _ANCHOR_KEY, scenario_ids=common)
        baseline_pooled, _ = _pooled_commit_rate(
            baselines_summary, _ANCHOR_KEY, scenario_ids=common
        )

        anchor_has_cells = any(v is not None for v in anchor_per_scenario.values())
        if not anchor_has_cells:
            print(
                f"\n{_ANCHOR_KEY}: median-weighted committed_itinerary_rate = N/A (no evaluable cells)  FAIL"
            )
            failed = True
        elif not common or anchor_pooled is None or baseline_pooled is None:
            print(
                f"\n{_ANCHOR_KEY}: WARNING — no scenario overlap between run and committed"
                f" baselines; anchor floor is not comparable (treating as PASS — no floor set)"
            )
        elif anchor_pooled >= baseline_pooled:
            print(
                f"\n{_ANCHOR_KEY}: median-weighted = {anchor_pooled:.3f}"
                f" >= baseline {baseline_pooled:.3f}  PASS"
            )
        else:
            print(
                f"\n{_ANCHOR_KEY}: median-weighted = {anchor_pooled:.3f}"
                f" < baseline {baseline_pooled:.3f}  FAIL (anchor regression)"
            )
            failed = True

    # ── Final verdict ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if failed:
        print("eval_falsifier: VERDICT = FAIL")
        print(f"{'=' * 60}\n")
        return 1
    else:
        print("eval_falsifier: VERDICT = PASS")
        print(f"{'=' * 60}\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
