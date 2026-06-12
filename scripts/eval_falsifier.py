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
    2 = infrastructure failure (missing run dir, malformed JSON,
        or a run-dir summary that shares zero scenarios with
        configs/eval_matrix.yaml (wrong-matrix run))
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


# WR-06: the scope of this falsifier is configs/eval_matrix.yaml (D-12-08).
_DEFAULT_MATRIX_CONFIG = REPO_ROOT / "configs" / "eval_matrix.yaml"


def _expected_matrix_scenarios(matrix_path: Path = _DEFAULT_MATRIX_CONFIG) -> set[str]:
    """Return the scenario ids declared in configs/eval_matrix.yaml (WR-06).

    Used only to WARN when a graded summary contains none of the expected
    scenarios (e.g. default-latest mode grabbed an eval_matrix_refinement run,
    which writes to the same eval_reports/ base). Returns an empty set when
    the config cannot be read — the warning is best-effort, never an error.
    """
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — warn-only helper, never fatal
        return set()
    if not isinstance(data, dict):
        return set()
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list):
        return set()
    return {str(s) for s in scenarios if isinstance(s, str)}


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

    WR-04: every nested read is isinstance-guarded so a malformed summary.json
    (non-dict scenario/providers/cell/scorers blocks, null/string ``n``)
    degrades to an unevaluable (None) scenario instead of raising — an
    uncaught traceback exits the interpreter with code 1, which the documented
    contract reserves for a legitimate FAIL verdict.
    """
    total_commits = 0.0
    total_runs = 0
    per_scenario: dict[str, float | None] = {}

    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, dict):
        return None, per_scenario

    for scenario_id, scenario_block in scenarios.items():
        if scenario_ids is not None and scenario_id not in scenario_ids:
            continue
        if not isinstance(scenario_block, dict):
            per_scenario[scenario_id] = None
            continue
        # D-10-09: skip quarantined scenarios
        if scenario_block.get("baseline_eligible", True) is False:
            continue
        providers = scenario_block.get("providers")
        cell = providers.get(provider_key) if isinstance(providers, dict) else None
        if not isinstance(cell, dict):
            per_scenario[scenario_id] = None
            continue

        # committed_itinerary_rate is wired into scorers by eval_matrix.py D-11-02
        scorers = cell.get("scorers")
        cir_block = scorers.get("committed_itinerary_rate") if isinstance(scorers, dict) else None
        if not isinstance(cir_block, dict):
            per_scenario[scenario_id] = None
            continue

        median = cir_block.get("median")
        n = cir_block.get("n", 0)

        # Guard against None/bool/non-numeric values (median AND n — WR-04)
        if median is None or isinstance(median, bool) or not isinstance(median, (int, float)):
            per_scenario[scenario_id] = None
            continue
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            per_scenario[scenario_id] = None
            continue

        per_scenario[scenario_id] = float(median)
        total_commits += float(median) * n
        total_runs += n

    pooled = total_commits / total_runs if total_runs > 0 else None
    return pooled, per_scenario


def _commit_split_from_run_dir(
    run_dir: Path,
    provider_key: str,
    scenario_ids: set[str] | None = None,
) -> tuple[int, int]:
    """Return (model_initiated_commits, forced_commits) for one provider.

    Reads individual *.json run files (not summary.json) because commit_forced
    is a per-run field, not aggregated into scorers blocks.
    D-13-04: forced commits count toward committed_itinerary_rate (product-honest)
    but the split must be printed explicitly in the A2 verdict line.

    T-13-05-01: malformed JSON or OSError → file silently skipped; never raises.
    """
    model_initiated = 0
    forced = 0
    # Convert "openai/gpt-5-mini" to "openai--gpt-5-mini" for glob matching
    provider_slug = provider_key.replace("/", "--")
    for path in run_dir.glob(f"{provider_slug}--*.json"):
        # Skip the summary file (not an individual run file)
        if path.name == "summary.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        det = data.get("deterministic") or {}
        if scenario_ids:
            # Filter by scenario parsed from the filename
            # Filename pattern: provider--model--scenario--run-N.json
            # e.g. openai--gpt-5-mini--omakase_mission_open_ended--run-0
            # Since provider_slug may contain --, strip it from the front
            suffix = path.stem[len(provider_slug) + 2 :]  # skip "provider_slug--"
            # suffix is now "scenario--run-N"
            scenario_in_name = suffix.rsplit("--", 1)[0] if "--" in suffix else suffix
            if scenario_in_name not in scenario_ids:
                continue
        if det.get("commit_forced"):
            forced += 1
        elif det.get("first_commit_call_step") is not None:
            model_initiated += 1
    return model_initiated, forced


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
    # WR-05: no --gates-config flag. The 0.6 bar is _FALSIFIER_BAR (INST-05 /
    # D-12-08) and the anchor floor comes from --baselines-dir; accepting a
    # gates-config path here would imply eval_gates.yaml drives the falsifier
    # when it does not.
    parser.add_argument(
        "--eval-queries",
        default="configs/eval_queries.yaml",
        help=(
            "Path to eval_queries.yaml for D-10-09 baseline_eligible resolution "
            "(default: configs/eval_queries.yaml)."
        ),
    )
    # Phase 13 / D-13-02: arm matrix config path for the zero-overlap exit-2 guard.
    # When grading an arm run dir (which covers both omakase + refinement_cheaper),
    # pass --matrix-config configs/eval_matrix_arm.yaml so the guard reads the arm
    # scenario universe instead of the default eval_matrix.yaml (omakase-only).
    parser.add_argument(
        "--matrix-config",
        default=None,
        help=(
            "Path to the matrix YAML whose scenario universe the zero-overlap exit-2 guard "
            "reads (default: configs/eval_matrix.yaml). "
            "Use configs/eval_matrix_arm.yaml for Phase 13 arm run dirs."
        ),
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Exit codes:
        0 = PASS — gpt-5-mini >= 0.6 pooled and gpt-4o-mini held baseline
        1 = FAIL — one or more checks failed (expected; not an infra error)
        2 = infrastructure failure (missing run dir, malformed JSON,
            or a run-dir summary that shares zero scenarios with
            configs/eval_matrix.yaml (wrong-matrix run))
    """
    # Lazy import — only used here, not at module level (D-12-06 artifact-reading only)
    from scripts.check_eval_gates import (  # noqa: PLC0415
        _build_summary_from_baselines,
        _load_baseline_eligibility,
    )

    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # ── Resolve the summary dict ────────────────────────────────────────────
    run_dir: Path | None = None
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

    # ── Resolve the matrix config for the zero-overlap guard ───────────────
    # Phase 13 / D-13-02: --matrix-config lets arm run dirs be graded against
    # the arm scenario universe (omakase + refinement_cheaper) instead of the
    # default eval_matrix.yaml (omakase-only). The guard reads whichever config
    # is given; the default is the production matrix.
    matrix_config_path: Path = (
        Path(args.matrix_config) if args.matrix_config else _DEFAULT_MATRIX_CONFIG
    )

    # ── Check 1: gpt-5-mini pooled committed_itinerary_rate >= 0.6 ─────────
    gpt5_pooled, gpt5_per_scenario = _pooled_commit_rate(summary, _GPT5_KEY)

    print(f"\n{'=' * 60}")
    print("eval_falsifier: INST-05 Milestone Falsifier Report")
    print(f"{'=' * 60}")
    # WR-06: always show which artifact is being graded. Both eval-matrix and
    # eval-matrix-refinement write to eval_reports/, so default-latest mode can
    # silently grab the wrong matrix's run dir — name it so the operator can tell.
    if args.baselines_mode:
        print(f"source: committed baselines at {args.baselines_dir}")
    else:
        print(f"source: run dir {run_dir}")
        expected_scenarios = _expected_matrix_scenarios(matrix_config_path)
        scenarios_block = summary.get("scenarios")
        found_scenarios = set(scenarios_block) if isinstance(scenarios_block, dict) else set()
        if expected_scenarios and not (found_scenarios & expected_scenarios):
            matrix_label = matrix_config_path.name
            print(
                f"  WARNING: none of {matrix_label}'s scenarios "
                f"({', '.join(sorted(expected_scenarios))}) appear in this summary "
                f"({', '.join(sorted(found_scenarios)) or 'none'}). The latest run dir may "
                "belong to a different matrix (e.g. eval_matrix_refinement.yaml) — "
                "pass --run-dir explicitly if so."
            )
            sys.stderr.write(
                f"eval_falsifier: refusing to grade — resolved run dir shares zero scenarios "
                f"with {matrix_label} (wrong-matrix run); exit 2, no verdict.\n"
            )
            return 2

    # Per-scenario breakdown (D-12-08: always print)
    print(f"\n[{_GPT5_KEY}] committed_itinerary_rate per scenario:")
    for scenario_id, rate in gpt5_per_scenario.items():
        rate_str = f"{rate:.3f}" if rate is not None else "N/A"
        print(f"  {scenario_id}: {rate_str}")

    # D-13-04: forced-commit split annotation (run-dir mode only — baselines mode has no per-run files)
    gpt5_split_str = ""
    if run_dir is not None:
        gpt5_mi, gpt5_fc = _commit_split_from_run_dir(run_dir, _GPT5_KEY)
        gpt5_total = gpt5_mi + gpt5_fc
        gpt5_split_str = f" (model-initiated {gpt5_mi}/{gpt5_total}, forced {gpt5_fc}/{gpt5_total})"

    if gpt5_pooled is None:
        print(
            f"\n{_GPT5_KEY}: median-weighted committed_itinerary_rate = N/A (no evaluable cells)  FAIL"
        )
        failed = True
    elif gpt5_pooled >= _FALSIFIER_BAR:
        print(
            f"\n{_GPT5_KEY}: median-weighted committed_itinerary_rate = {gpt5_pooled:.3f}"
            f" >= {_FALSIFIER_BAR}{gpt5_split_str}  PASS"
        )
    else:
        print(
            f"\n{_GPT5_KEY}: median-weighted committed_itinerary_rate = {gpt5_pooled:.3f}"
            f" < {_FALSIFIER_BAR}{gpt5_split_str}  FAIL"
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

        # D-13-04: forced-commit split annotation for anchor (run-dir mode only)
        anchor_split_str = ""
        if run_dir is not None:
            anchor_mi, anchor_fc = _commit_split_from_run_dir(run_dir, _ANCHOR_KEY)
            anchor_total = anchor_mi + anchor_fc
            anchor_split_str = (
                f" (model-initiated {anchor_mi}/{anchor_total}, forced {anchor_fc}/{anchor_total})"
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
                f" >= baseline {baseline_pooled:.3f}{anchor_split_str}  PASS"
            )
        else:
            print(
                f"\n{_ANCHOR_KEY}: median-weighted = {anchor_pooled:.3f}"
                f" < baseline {baseline_pooled:.3f}{anchor_split_str}  FAIL (anchor regression)"
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
