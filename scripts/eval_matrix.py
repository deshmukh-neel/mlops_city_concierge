#!/usr/bin/env python3
"""Cross-(provider, model, scenario, run) eval matrix runner.

EVAL-05 / D-08 / D-09 / D-10. Subprocess fan-out: one fresh
`python scripts/eval_agent.py ...` per cell so DB pools, LLM SDK client
state, and `@lru_cache` settings are isolated between providers
(project memory: full_suite_db_pool_contamination + Gemini/DeepSeek
reasoning-state pruning). Execution is sequential in Phase 3 (D-09);
parallel ProcessPoolExecutor is deferred to v2.1.

Output layout (D-10):
    eval_reports/{ISO8601-Z}/
        openai--gpt-4o-mini--omakase_mission_open_ended--run-0.json
        ...
        summary.json    (cross-provider median table per scorer per scenario)

CI safety (EVAL-09 / P4):
- Real-provider matrix runs require APP_ENV=eval. The gate is enforced
  BEFORE any subprocess.run, so a misconfigured key cannot rate-limit CI.
- `--llm-provider-override scripted` is the single source of truth for the
  CI gate: it maps every entry to the deterministic ScriptedChatModel
  (no API keys, no network).

This module intentionally has NO top-level LLM SDK imports — it is the
orchestrator, not the LLM caller. Each cell shells out to a fresh
scripts/eval_agent.py subprocess which imports the SDKs in isolation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.critique.checks import CRITIQUE_THRESHOLDS
from app.eval.config import (
    DEFAULT_EVAL_MATRIX_PATH,
    EvalMatrixConfig,
    MatrixEntry,
    load_eval_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default eval-queries YAML to forward to each subprocess invocation.
_DEFAULT_EVAL_QUERIES_REL = "configs/eval_queries.yaml"

# Default output base under the repository root.
_DEFAULT_OUTPUT_BASE = REPO_ROOT / "eval_reports"

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatrixCell:
    """One (provider, model, scenario_id, run_n) cell in the matrix.

    `env` (Phase 6 / plan 06-06 / D-06-10): optional per-cell env override
    threaded from `MatrixEntry.env`. When set, `run_matrix` composes it
    into the subprocess `env=` kwarg AND applies it to `os.environ` for
    the cell's lifetime (NEW HIGH-B) so in-process consumers like
    `_run_prod_threading` see the override.
    """

    provider: str
    model: str
    scenario_id: str
    run_n: int
    env: dict[str, str] | None = None

    def cell_filename(self) -> str:
        """Per-cell JSON filename (D-10 layout)."""
        return f"{self.provider}--{self.model}--{self.scenario_id}--run-{self.run_n}.json"


def iter_cells(matrix: EvalMatrixConfig, runs: int) -> Iterator[MatrixCell]:
    """Yield cells in deterministic order.

    Order: entry-outer (matches D-06 anchors), scenario-middle (YAML),
    run-inner (0..runs-1). This ordering makes the dry-run printout and
    summary aggregation human-readable; the subprocess fan-out doesn't
    care about order since each cell is independent.

    Phase 6 plan 06-06: thread `entry.env` through to every cell yielded
    for that entry so the per-cell env override (D-06-10) reaches
    `run_matrix`. `MatrixEntry.env` is `dict[StrictStr, StrictStr] | None`
    per plan 06-04; we coerce the Pydantic field to a plain dict here
    (the `frozen=True` dataclass needs hashability-friendly inputs, but
    a regular `dict` field on the dataclass is fine since the dataclass
    is constructed fresh per cell).
    """
    for entry in matrix.entries:
        # entry.env is `dict[StrictStr, StrictStr] | None` (Pydantic).
        # Coerce to a plain dict so the dataclass field is the plain shape.
        entry_env = dict(entry.env) if entry.env is not None else None
        for scenario_id in matrix.scenarios:
            for run_n in range(runs):
                yield MatrixCell(
                    provider=entry.provider,
                    model=entry.model,
                    scenario_id=scenario_id,
                    run_n=run_n,
                    env=entry_env,
                )


def _iso_timestamp_filename_safe() -> str:
    """ISO8601 UTC timestamp with colons replaced (Windows + URL safe)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def resolve_run_dir(base: Path | None = None) -> Path:
    """Build the per-run output directory under `base` (default eval_reports/).

    The directory is created eagerly so subprocess invocations can write
    their cell JSONs without each having to mkdir-p.
    """
    base_path = base if base is not None else _DEFAULT_OUTPUT_BASE
    run_dir = base_path / _iso_timestamp_filename_safe()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _provider_label(provider: str, model: str) -> str:
    """The cross-provider summary keys by 'provider/model' for diffing."""
    return f"{provider}/{model}"


def _parse_cell_filename(name: str) -> tuple[str, str, str, int] | None:
    """Parse `{provider}--{model}--{scenario_id}--run-{n}.json` back to its
    parts. Returns None if the filename does not match the expected shape
    (so a stray file in the output dir does not crash the aggregator)."""
    if not name.endswith(".json"):
        return None
    stem = name[:-5]  # drop ".json"
    parts = stem.split("--")
    if len(parts) != 4:
        return None
    provider, model, scenario_id, run_part = parts
    if not run_part.startswith("run-"):
        return None
    try:
        run_n = int(run_part[len("run-") :])
    except ValueError:
        return None
    return provider, model, scenario_id, run_n


def _scorer_means_from_cell(payload: dict[str, Any]) -> dict[str, float]:
    """Extract `{scorer_name}_mean` keys from one cell's aggregate dict.

    Plan 03-07 baselines diff on per-scorer median across runs; we only
    aggregate scorer-mean values here. The cell-level mean already collapses
    per-query variance to a single number per scorer.

    Whitelist contract (plan 03-08 / CR-01): only emits scorer names whose
    key is registered in `app.agent.critique.checks.CRITIQUE_THRESHOLDS`. The
    existing `results_mean`, `tool_calls_mean`, `contexts_mean`,
    `revision_hints_mean`, `committed_stops_mean`, and
    `answer_retrieved_place_coverage_mean` aggregate keys are intentionally
    excluded — they are cell-level diagnostics, not scorers, and including
    them would pollute summary.json and the Phase 4-6 baseline-diff target.

    bool exclusion (plan 03-08 / IN-04): `bool` is a subclass of `int`, so
    `isinstance(value, int | float)` matches `True`/`False`. Exclude bools
    explicitly so a stray bool in the aggregate dict does not become a
    scorer score of 1.0 or 0.0.
    """
    aggregate = payload.get("aggregate") or {}
    out: dict[str, float] = {}
    for key, value in aggregate.items():
        if not isinstance(key, str) or not key.endswith("_mean"):
            continue
        # bool is a subclass of int — exclude it before the numeric check (IN-04).
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        scorer_name = key[: -len("_mean")]
        # Whitelist against CRITIQUE_THRESHOLDS — admit only registered scorers (CR-01).
        if scorer_name in CRITIQUE_THRESHOLDS:
            out[scorer_name] = float(value)
    return out


def _stats_for_values(values: Sequence[float]) -> dict[str, float | int]:
    """Compute the {median, min, max, stdev, n} table for one cell-stack."""
    n = len(values)
    if n == 0:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0, "n": 0}
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        # statistics.stdev requires n >= 2; for n=1 stdev is 0 by convention.
        "stdev": float(statistics.stdev(values)) if n >= 2 else 0.0,
        "n": n,
    }


def aggregate_cell_jsons(
    output_dir: Path,
    llm_provider_override: str | None = None,
) -> dict[str, Any]:
    """Build the cross-provider summary.json content from a directory of
    per-cell JSON reports (D-10 shape).

    The aggregator walks every `*.json` in `output_dir` EXCLUDING
    `summary.json` itself (so re-running over an already-aggregated dir
    does not double-count). Filenames carry the (provider, model,
    scenario_id, run_n) tuple — the cell JSON payload is only consulted
    for `aggregate.{scorer}_mean` values.

    Output shape::

        {
          "generated_at": "2026-05-21T18-00-00Z",
          "scenarios": {
            "<scenario_id>": {
              "providers": {
                "<provider>/<model>": {
                  "scorers": {
                    "<scorer>": {"median": ..., "min": ..., "max": ...,
                                  "stdev": ..., "n": ...}
                  }
                }
              }
            }
          },
          "overridden_to": "<provider>"   # optional, IN-02
        }

    When `llm_provider_override` is non-None, a top-level `overridden_to`
    field is recorded so downstream diffs (PR review tooling, plan 03-07
    baseline comparisons) can detect that per-provider keys reflect the
    override target instead of the originally configured provider. The
    per-provider scorer keys are NOT re-mapped by this field — they keep
    whatever name `_apply_override` wrote into the cell filenames.
    """
    # Group raw scorer scores per (scenario_id, provider_label, scorer).
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        parsed = _parse_cell_filename(path.name)
        if parsed is None:
            # WR-01: surface unparseable filenames so a stray '--' in a
            # provider/model name (or a foreign file dropped into the run dir)
            # is observable instead of silently zeroing a cell.
            _log.warning(
                "eval_matrix: skipping unparseable cell file %s in %s",
                path.name,
                output_dir,
            )
            continue
        provider, model, scenario_id, _run_n = parsed
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        provider_key = _provider_label(provider, model)
        scenario_block = grouped.setdefault(scenario_id, {})
        provider_block = scenario_block.setdefault(provider_key, {})
        for scorer_name, value in _scorer_means_from_cell(payload).items():
            provider_block.setdefault(scorer_name, []).append(value)

    # Compute per-(scenario, provider, scorer) stats.
    scenarios_out: dict[str, Any] = {}
    for scenario_id, scenario_providers in grouped.items():
        providers_out: dict[str, Any] = {}
        for provider_key, scorers in scenario_providers.items():
            providers_out[provider_key] = {
                "scorers": {scorer: _stats_for_values(values) for scorer, values in scorers.items()}
            }
        scenarios_out[scenario_id] = {"providers": providers_out}

    result: dict[str, Any] = {
        "generated_at": _iso_timestamp_filename_safe(),
        "scenarios": scenarios_out,
    }
    if llm_provider_override is not None:
        result["overridden_to"] = llm_provider_override
    return result


def _build_subprocess_cmd(
    cell: MatrixCell,
    cell_path: Path,
    eval_queries_path: str,
    llm_provider_override: str | None,
) -> list[str]:
    """Construct the `python scripts/eval_agent.py ...` command for one cell.

    When llm_provider_override is set, it replaces cell.provider in the cmd
    (single source of truth for the CI gate; the override is typically
    'scripted'). cell.model is preserved as a label; eval_agent.py threads
    it through the report.
    """
    provider = llm_provider_override or cell.provider
    eval_agent_path = str(REPO_ROOT / "scripts" / "eval_agent.py")
    return [
        sys.executable,
        eval_agent_path,
        "--eval-queries",
        eval_queries_path,
        "--llm-provider",
        provider,
        "--chat-model",
        cell.model,
        "--scenario-ids",
        cell.scenario_id,
        "--max-queries",
        "1",
        "--output",
        str(cell_path),
    ]


def _gate_blocks(matrix: EvalMatrixConfig, llm_provider_override: str | None) -> bool:
    """Return True when APP_ENV=eval is required but not set.

    The gate fires when ANY entry uses a non-scripted provider AND the
    --llm-provider-override is also not 'scripted'. CI scripted runs bypass
    the gate entirely (P4 / EVAL-09).
    """
    if llm_provider_override == "scripted":
        return False
    if os.environ.get("APP_ENV") == "eval":
        return False
    real_providers = [
        e for e in matrix.entries if (llm_provider_override or e.provider) != "scripted"
    ]
    return bool(real_providers)


def _apply_override(
    entries: Sequence[MatrixEntry], llm_provider_override: str | None
) -> Sequence[MatrixEntry]:
    """Return entries with `provider` rewritten to llm_provider_override when set.

    MEDIUM-1 fix (plan 06-06): preserves `entry.env` so the per-cell
    Phase-6 env override (e.g. `REFINEMENT_STRUCTURED_PLAN_ENABLED`) still
    propagates after `--llm-provider-override scripted` rebinds entries.
    Without this preservation, CI scripted-mode runs of the refinement
    matrix would silently drop the flag and the prod-threading branch
    would never inject the structured plan.
    """
    if not llm_provider_override:
        return entries
    return [MatrixEntry(provider=llm_provider_override, model=e.model, env=e.env) for e in entries]


def run_matrix(
    matrix: EvalMatrixConfig,
    runs: int,
    output_dir: Path,
    llm_provider_override: str | None,
    eval_queries_path: str,
) -> tuple[int, list[dict[str, Any]]]:
    """Fan-out one subprocess per cell. Returns (returncode, failures).

    `failures` is a list of dicts shaped:
        {"cell": "<cell_filename>", "stderr": "...", "returncode": N}

    Subprocess failures do NOT short-circuit the matrix (D-08 / plan task
    3 behavior). The caller checks `failures` to know whether the matrix
    completed cleanly; the returncode is non-zero iff any cell failed.
    """
    if llm_provider_override:
        # IN-02: surface the rebind in CI logs so operators reading the run
        # output can correlate per-provider summary keys to the override target.
        _log.info(
            "eval_matrix: --llm-provider-override=%s active; entries rebound",
            llm_provider_override,
        )
    effective_matrix = EvalMatrixConfig(
        entries=list(_apply_override(matrix.entries, llm_provider_override)),
        scenarios=list(matrix.scenarios),
    )
    failures: list[dict[str, Any]] = []
    # Phase 6 plan 06-06 / D-06-10: each cell composes its own env from
    # `parent_env` + any per-cell overrides (`MatrixEntry.env` from
    # `eval_matrix.yaml`). The per-cell composition replaces the prior
    # shared snapshot to support flag-on-refinement-cell-only patterns.
    # NEW HIGH-B: the per-cell env is ALSO applied to `os.environ` during
    # the cell's run with try/finally cleanup, so in-process readers
    # (e.g., unit-test invocations of `_run_prod_threading`'s
    # `os.environ.get('REFINEMENT_STRUCTURED_PLAN_ENABLED')` read) see
    # the override. Without the apply step, the env propagated to the
    # subprocess child but was invisible to in-process consumers.
    parent_env = os.environ.copy()
    for cell in iter_cells(effective_matrix, runs=runs):
        cell_path = output_dir / cell.cell_filename()
        cell_env = {**parent_env, **(cell.env or {})}
        # NEW HIGH-B: apply cell.env to os.environ for in-process callers.
        # Capture prior values so cleanup restores them (or pops keys that
        # had no prior value). Wrapped in try/finally so cells don't leak
        # env into each other (T-06-06-06 mitigation).
        saved_env: dict[str, str | None] = {k: os.environ.get(k) for k in (cell.env or {})}
        try:
            if cell.env:
                os.environ.update(cell.env)
            cmd = _build_subprocess_cmd(
                cell=cell,
                cell_path=cell_path,
                eval_queries_path=eval_queries_path,
                llm_provider_override=None,  # already applied via _apply_override
            )
            # cmd is built from a closed allowlist (sys.executable + repo-relative
            # script path + structured matrix/scenario fields) — no shell, no
            # untrusted input. The S603 lint here is a noqa-by-design.
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                env=cell_env,
                check=False,
            )
            if result.returncode != 0:
                failures.append(
                    {
                        "cell": cell.cell_filename(),
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                    }
                )
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    rc = 0 if not failures else 1
    return rc, failures


def _validate_override(value: str | None) -> str | None:
    """argparse type for --llm-provider-override (WR-04).

    `--` is reserved as the cell-filename separator in `scripts/eval_matrix.py`.
    The `MatrixEntry.reject_double_dash` validator already enforces this at
    YAML-load time, but without an upfront argparse-level check, a malformed
    --llm-provider-override value bubbles into `_apply_override` as a
    `pydantic.ValidationError` mid-`run_matrix()` — no actionable CLI error.
    Rejecting `--` at parse time keeps the operator-facing error path clean
    (argparse rc=2 + usage) instead of a pydantic traceback.
    """
    if value is None:
        return None
    if "--" in value:
        raise argparse.ArgumentTypeError(
            f"--llm-provider-override='{value}' contains '--'; '--' is "
            "reserved as the cell-filename separator. Use a single-dash "
            "or alphanumeric provider name."
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for scripts/eval_matrix.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-config",
        default=str(DEFAULT_EVAL_MATRIX_PATH),
        help="YAML matrix config to load (default: configs/eval_matrix.yaml).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per (provider, model, scenario) cell (D-05; default 3).",
    )
    parser.add_argument(
        "--llm-provider-override",
        default=None,
        type=_validate_override,
        help=(
            "Force ALL entries to this provider (typical CI: 'scripted'). "
            "Bypasses the APP_ENV=eval gate when set to 'scripted'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Per-run output directory base. Defaults to "
            "eval_reports/{ISO8601-Z}/ under the repo root."
        ),
    )
    parser.add_argument(
        "--eval-queries",
        default=_DEFAULT_EVAL_QUERIES_REL,
        help="Eval-queries YAML to forward to each subprocess.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the (provider, model, scenario, run) cell list without "
            "invoking any subprocess. Useful for plan 03-07 baseline preview."
        ),
    )
    return parser.parse_args(argv)


def _print_dry_run(matrix: EvalMatrixConfig, runs: int) -> None:
    """Print one line per cell in deterministic order. Used by --dry-run."""
    for cell in iter_cells(matrix, runs=runs):
        print(f"{cell.provider}/{cell.model} :: {cell.scenario_id} :: --run-{cell.run_n}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    if args.runs < 1:
        print(
            f"eval_matrix: --runs must be >= 1 (got {args.runs})",
            file=sys.stderr,
        )
        return 2
    matrix = load_eval_matrix(args.matrix_config)

    # Apply override for the dry-run printout so users see the effective
    # provider list before any subprocess fires.
    effective_for_print = EvalMatrixConfig(
        entries=list(_apply_override(matrix.entries, args.llm_provider_override)),
        scenarios=list(matrix.scenarios),
    )

    if _gate_blocks(matrix, args.llm_provider_override):
        print(
            "eval_matrix: APP_ENV=eval required for real-provider matrix runs; "
            "pass --llm-provider-override scripted for CI or set APP_ENV=eval",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        _print_dry_run(effective_for_print, runs=args.runs)
        return 0

    output_dir = Path(args.output_dir) if args.output_dir is not None else resolve_run_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    rc, failures = run_matrix(
        matrix=matrix,
        runs=args.runs,
        output_dir=output_dir,
        llm_provider_override=args.llm_provider_override,
        eval_queries_path=args.eval_queries,
    )

    # Write summary.json regardless of failures (partial results are still
    # informative for debugging). Thread the override through so summary.json
    # records `overridden_to` for IN-02 traceability.
    summary = aggregate_cell_jsons(output_dir, llm_provider_override=args.llm_provider_override)
    summary["failures"] = failures
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"eval_matrix: wrote {summary_path}", file=sys.stderr)
    if failures:
        print(
            f"eval_matrix: {len(failures)} cell(s) failed; see {summary_path}#/failures",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
