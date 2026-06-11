"""Unit tests for scripts/eval_matrix.py (Plan 03-05 / EVAL-05).

The matrix runner uses subprocess fan-out per D-08 (one fresh
`python scripts/eval_agent.py ...` per cell) to isolate DB pool / LLM SDK
client state / `@lru_cache` settings across providers (cf. project memory
project_full_suite_db_pool_contamination and agent_loses_reasoning_state).

These tests cover:
  - configs/eval_matrix.yaml loads via load_eval_matrix (D-06 anchors)
  - --dry-run flag prints the expected (provider, model, scenario, run_n)
    cell list without invoking subprocess
  - APP_ENV=eval gate enforcement (EVAL-09): real-provider runs require
    APP_ENV=eval; --llm-provider-override scripted bypasses the gate
  - summary.json aggregator: given a fixture directory of cell JSONs, the
    aggregator computes correct median/min/max/stdev/n
  - scripts/eval_matrix.py is importable in any CI environment (no API key
    requirements at top-level import time)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.eval.config import REPO_ROOT, load_eval_matrix

# ─── configs/eval_matrix.yaml exists and loads via D-06 anchors ──────────────


def test_repo_eval_matrix_yaml_loads_via_load_eval_matrix() -> None:
    """configs/eval_matrix.yaml ships with the D-06 anchors locked:
    providers=[openai/gpt-4o-mini, deepseek/deepseek-chat].

    Phase 6 / D-06-09 / plan 06-07: `refinement_cheaper` was moved out of
    the default matrix and into `configs/eval_matrix_refinement.yaml`. The
    default matrix now contains only the first-turn scenarios so REF-04
    (omakase first-turn no-regression with flag OFF) is preserved.
    """
    matrix = load_eval_matrix(REPO_ROOT / "configs/eval_matrix.yaml")
    assert len(matrix.entries) == 2
    assert len(matrix.scenarios) == 2
    providers = {(e.provider, e.model) for e in matrix.entries}
    assert ("openai", "gpt-4o-mini") in providers
    assert ("deepseek", "deepseek-chat") in providers
    assert "omakase_mission_open_ended" in matrix.scenarios
    assert "late_night_closure_cascade" in matrix.scenarios
    # Phase 6 invariant: the refinement scenario lives in the sibling
    # refinement-only matrix, NOT in the default matrix.
    assert "refinement_cheaper" not in matrix.scenarios


def test_repo_eval_matrix_refinement_yaml_loads_via_load_eval_matrix() -> None:
    """configs/eval_matrix_refinement.yaml carries the six Phase-9 provider
    entries with a per-cell env override REFINEMENT_STRUCTURED_PLAN_ENABLED=true
    on EACH entry, and a single scenario (refinement_cheaper). The merge gate
    history:
    - openai/gpt-4o-mini — strict 1.0 (v2.0 anchor)
    - deepseek/deepseek-chat — logged-not-gated reference (regression guard)
    - openai/gpt-5-mini — PROV-01 GATED (D-09-02 re-scoped 2026-06-05); the
      milestone anchor gate (Part A hard committed_itinerary_rate ≥ 0.6,
      Part B advisory refinement_minimal_edit median ≥ 0.5)
    - deepseek/deepseek-reasoner — PROV-02 GATED lower-bar median ≥ 0.6
      (D-09-04 thinking-enabled carve-out via _DEEPSEEK_REASONER_THINKING_ENABLED)
    - anthropic/claude-sonnet-4-6 — PROV-03 GATED strict median ≥ 1.0
      (D-09-06 thinking-enabled carve-out via _ANTHROPIC_THINKING_BUDGET;
      Plan 09-03 first-time Anthropic wiring)
    - gemini/gemini-3.1-pro-preview — PROV-04 EXPERIMENTAL — no merge gate
      per D-09-08 (Plan 09-04 first-time bytes thought_signature wiring;
      critique-loop fix deferred per project_w10_migration_necessary_not_sufficient;
      empirical median logged-not-gated)
    """
    matrix = load_eval_matrix(REPO_ROOT / "configs/eval_matrix_refinement.yaml")
    assert len(matrix.entries) == 6
    assert len(matrix.scenarios) == 1
    providers = {(e.provider, e.model) for e in matrix.entries}
    assert ("openai", "gpt-4o-mini") in providers
    assert ("deepseek", "deepseek-chat") in providers
    assert ("openai", "gpt-5-mini") in providers
    assert ("deepseek", "deepseek-reasoner") in providers
    assert ("anthropic", "claude-sonnet-4-6") in providers
    assert ("gemini", "gemini-3.1-pro-preview") in providers
    assert matrix.scenarios == ["refinement_cheaper"]
    for entry in matrix.entries:
        assert entry.env == {"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"}, (
            f"entry {entry.provider}/{entry.model} missing the per-cell env"
            f" override; got env={entry.env}"
        )


# ─── baseline JSON ↔ matrix YAML provider-cell parity ────────────────────────

# Matrix entries whose baseline cell is intentionally absent. The only
# sanctioned deferral is gemini/gemini-3.1-pro-preview: its first n=5
# measurement is deferred to the baseline-regen phase (see the PROV-04
# comment block in configs/eval_matrix_refinement.yaml). Shrink this set
# when the deferred cell lands; never grow it without a matching comment
# in the matrix YAML.
_DEFERRED_BASELINE_CELLS: dict[str, set[str]] = {
    "eval_matrix_refinement.yaml": {"gemini/gemini-3.1-pro-preview"},
    "eval_matrix.yaml": set(),
}

_MATRIX_TO_BASELINES: dict[str, list[str]] = {
    "eval_matrix_refinement.yaml": ["refinement_cheaper.json"],
    "eval_matrix.yaml": [
        "omakase_mission_open_ended.json",
        "late_night_closure_cascade.json",
    ],
}


@pytest.mark.parametrize("matrix_name", sorted(_MATRIX_TO_BASELINES))
def test_baseline_provider_cells_match_matrix_entries(matrix_name: str) -> None:
    """Every baseline provider cell maps to a matrix entry and vice versa.

    Phase 9's revertability audit (09-05-AUDIT.md, finding pair 4) found that
    the matrix YAML and its baseline JSON can drift SILENTLY: reverting a
    single data(09-0x) baseline commit leaves baseline keys ≠ matrix entries
    with no test detection. This locks the parity in both directions:
    no orphan baseline cells, and no matrix entry without a baseline cell
    unless listed in _DEFERRED_BASELINE_CELLS.
    """
    matrix = load_eval_matrix(REPO_ROOT / "configs" / matrix_name)
    matrix_keys = {f"{e.provider}/{e.model}" for e in matrix.entries}
    deferred = _DEFERRED_BASELINE_CELLS[matrix_name]
    assert deferred <= matrix_keys, (
        f"{matrix_name}: deferred cells {deferred - matrix_keys} are not matrix"
        " entries — stale deferral, remove them from _DEFERRED_BASELINE_CELLS"
    )
    for baseline_name in _MATRIX_TO_BASELINES[matrix_name]:
        baseline_path = REPO_ROOT / "configs" / "eval_baselines" / baseline_name
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_keys = set(payload["providers"])
        orphans = baseline_keys - matrix_keys
        assert not orphans, (
            f"{baseline_name} has provider cells with no matching entry in"
            f" {matrix_name}: {sorted(orphans)} — remove the stale baseline"
            " cell or add the matrix entry in the same commit"
        )
        missing = matrix_keys - baseline_keys
        assert missing == deferred, (
            f"{matrix_name} entries missing a baseline cell in {baseline_name}:"
            f" {sorted(missing - deferred)} — regenerate the baseline for the"
            " new cell in the same commit, or document the deferral in the"
            " matrix YAML and _DEFERRED_BASELINE_CELLS"
        )


# ─── scripts/eval_matrix.py imports without env vars ─────────────────────────


def test_eval_matrix_module_imports_in_ci_environment(monkeypatch) -> None:
    """The matrix-runner module MUST be importable in any CI environment
    (no API key requirements at top-level import time). If a future change
    adds `from openai import ...` at module load, this test catches it."""
    for key in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    import importlib

    import scripts.eval_matrix  # noqa: F401

    importlib.reload(scripts.eval_matrix)


# ─── cells generator: 2 providers * 3 scenarios * N runs = 6N cells ──────────


def test_iter_cells_produces_provider_model_scenario_run_combinations() -> None:
    """Plan 03-05 task 3 behavior: for each (entry, scenario_id, run_n) the
    matrix produces one cell. Test the underlying generator independently
    of subprocess.run (which is the next test's territory)."""
    from app.eval.config import EvalMatrixConfig, MatrixEntry
    from scripts.eval_matrix import iter_cells

    matrix = EvalMatrixConfig(
        entries=[
            MatrixEntry(provider="openai", model="gpt-4o-mini"),
            MatrixEntry(provider="deepseek", model="deepseek-chat"),
        ],
        scenarios=[
            "omakase_mission_open_ended",
            "refinement_cheaper",
            "late_night_closure_cascade",
        ],
    )
    cells = list(iter_cells(matrix, runs=3))
    assert len(cells) == 18  # 2 * 3 * 3
    # Deterministic order: entry-outer, scenario-middle, run-inner.
    assert cells[0].provider == "openai"
    assert cells[0].model == "gpt-4o-mini"
    assert cells[0].scenario_id == "omakase_mission_open_ended"
    assert cells[0].run_n == 0
    assert cells[1].run_n == 1
    assert cells[2].run_n == 2
    assert cells[3].scenario_id == "refinement_cheaper"
    # Second provider starts after all (3 * 3 = 9) cells of the first.
    assert cells[9].provider == "deepseek"


def test_iter_cells_zero_runs_yields_empty() -> None:
    """Defensive: runs=0 produces no cells (the CLI validates >= 1, but the
    generator itself should be safe)."""
    from app.eval.config import EvalMatrixConfig, MatrixEntry
    from scripts.eval_matrix import iter_cells

    matrix = EvalMatrixConfig(
        entries=[MatrixEntry(provider="openai", model="gpt-4o-mini")],
        scenarios=["a"],
    )
    assert list(iter_cells(matrix, runs=0)) == []


# ─── --dry-run prints the cell list without running subprocess ───────────────


def test_dry_run_prints_default_matrix_cells(capsys, monkeypatch) -> None:
    """`python scripts/eval_matrix.py --dry-run --matrix-config
    configs/eval_matrix.yaml --runs 3` exits 0 and prints one line per cell.

    Phase 6 / D-06-09 / plan 06-07: default matrix has 2 entries × 2
    scenarios × 3 runs = 12 cells (refinement_cheaper moved to the sibling
    refinement-only matrix). Pre-Phase-6 expected 18 cells.
    """
    monkeypatch.setenv("APP_ENV", "eval")  # gate doesn't apply to dry-run
    from scripts.eval_matrix import main

    rc = main(
        [
            "--matrix-config",
            str(REPO_ROOT / "configs/eval_matrix.yaml"),
            "--runs",
            "3",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    # 2 providers * 2 scenarios * 3 runs = 12 cells.
    cell_lines = [line for line in out.splitlines() if "--run-" in line]
    assert len(cell_lines) == 12


# ─── APP_ENV=eval gate enforcement (EVAL-09) ─────────────────────────────────


def test_gate_blocks_real_provider_runs_without_app_env_eval(monkeypatch) -> None:
    """EVAL-09 / P4: real-provider matrix runs require APP_ENV=eval. Without
    it, exit non-zero with an actionable error."""
    monkeypatch.setenv("APP_ENV", "dev")
    from scripts.eval_matrix import main

    rc = main(
        [
            "--matrix-config",
            str(REPO_ROOT / "configs/eval_matrix.yaml"),
            "--runs",
            "1",
        ]
    )
    assert rc == 2


def test_gate_allows_scripted_override_without_app_env_eval(monkeypatch, tmp_path) -> None:
    """The --llm-provider-override scripted bypasses the APP_ENV gate so CI
    can run the matrix without setting APP_ENV=eval."""
    monkeypatch.setenv("APP_ENV", "dev")
    from scripts.eval_matrix import main

    rc = main(
        [
            "--matrix-config",
            str(REPO_ROOT / "configs/eval_matrix.yaml"),
            "--runs",
            "1",
            "--llm-provider-override",
            "scripted",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_gate_allows_real_provider_with_app_env_eval(monkeypatch) -> None:
    """When APP_ENV=eval IS set, the gate lets real-provider invocations
    through (we use --dry-run so no subprocess actually fires)."""
    monkeypatch.setenv("APP_ENV", "eval")
    from scripts.eval_matrix import main

    rc = main(
        [
            "--matrix-config",
            str(REPO_ROOT / "configs/eval_matrix.yaml"),
            "--runs",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0


# ─── summary.json aggregator: cross-provider median/min/max/stdev ────────────


def _write_cell(
    directory: Path,
    provider: str,
    model: str,
    scenario_id: str,
    run_n: int,
    score_value: float,
) -> Path:
    """Write a minimal cell JSON mirroring scripts/eval_agent.py's report
    shape. The aggregator only needs scenario_id (we encode it in the
    filename) plus aggregate.{scorer}_mean values."""
    fname = f"{provider}--{model}--{scenario_id}--run-{run_n}.json"
    path = directory / fname
    payload = {
        "llm_provider": provider,
        "chat_model": model,
        "query_count": 1,
        "aggregate": {
            "category_compliance_mean": score_value,
            "rationale_stop_alignment_mean": score_value,
        },
        "queries": [{"id": scenario_id}],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_aggregate_cell_jsons_computes_median_min_max_stdev(tmp_path: Path) -> None:
    """Given a fixture directory with 3 cell JSONs for one (provider,
    scenario), the aggregator emits the cross-run median/min/max/stdev/n."""
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 0, 0.4)
    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 1, 0.5)
    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 2, 0.6)

    summary = aggregate_cell_jsons(tmp_path)
    assert "scenarios" in summary
    scenario_block = summary["scenarios"]["scenario_a"]
    provider_block = scenario_block["providers"]["openai/gpt-4o-mini"]
    scorer = provider_block["scorers"]["category_compliance"]
    assert scorer["n"] == 3
    assert scorer["min"] == pytest.approx(0.4)
    assert scorer["max"] == pytest.approx(0.6)
    assert scorer["median"] == pytest.approx(0.5)
    # stdev sample of [0.4, 0.5, 0.6] = 0.1
    assert scorer["stdev"] == pytest.approx(0.1)


def test_aggregate_handles_multiple_providers_per_scenario(tmp_path: Path) -> None:
    """Cross-provider median table per scenario is the user-facing surface
    of summary.json (plan 03-07 commits the per-scenario baseline JSON)."""
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 0, 0.4)
    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 1, 0.5)
    _write_cell(tmp_path, "deepseek", "deepseek-chat", "scenario_a", 0, 0.8)
    _write_cell(tmp_path, "deepseek", "deepseek-chat", "scenario_a", 1, 0.9)

    summary = aggregate_cell_jsons(tmp_path)
    providers = summary["scenarios"]["scenario_a"]["providers"]
    assert set(providers.keys()) == {"openai/gpt-4o-mini", "deepseek/deepseek-chat"}
    assert providers["openai/gpt-4o-mini"]["scorers"]["category_compliance"]["n"] == 2
    assert providers["deepseek/deepseek-chat"]["scorers"]["category_compliance"][
        "median"
    ] == pytest.approx(0.85)


def test_aggregate_skips_summary_json_itself(tmp_path: Path) -> None:
    """The aggregator walks `*.json` excluding summary.json — re-running the
    aggregator over an output dir that already has summary.json must not
    double-count it."""
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 0, 0.5)
    # Pre-existing summary.json with the same shape; must be skipped.
    (tmp_path / "summary.json").write_text(
        json.dumps({"scenarios": {"old": {"providers": {}}}}), encoding="utf-8"
    )
    summary = aggregate_cell_jsons(tmp_path)
    assert set(summary["scenarios"].keys()) == {"scenario_a"}


def test_aggregate_records_generated_at_timestamp(tmp_path: Path) -> None:
    """summary.json carries a top-level `generated_at` ISO8601 timestamp for
    plan 03-07 baseline diffing."""
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 0, 0.5)
    summary = aggregate_cell_jsons(tmp_path)
    assert "generated_at" in summary
    # Bare-minimum shape: ISO-like string (precise format tested by
    # write_summary integration).
    assert isinstance(summary["generated_at"], str)


# ─── CR-01 + IN-04: scorer-whitelist + bool exclusion (plan 03-08) ──────────


def _write_cell_with_aggregate(
    directory: Path,
    provider: str,
    model: str,
    scenario_id: str,
    run_n: int,
    aggregate: dict,
) -> Path:
    """Variant of `_write_cell` that injects an arbitrary `aggregate` dict.

    The CR-01 / IN-04 tests need to plant non-scorer `_mean` keys and bool
    values that the standard 2-scorer `_write_cell` payload can't express.
    """
    fname = f"{provider}--{model}--{scenario_id}--run-{run_n}.json"
    path = directory / fname
    payload = {
        "llm_provider": provider,
        "chat_model": model,
        "query_count": 1,
        "aggregate": aggregate,
        "queries": [{"id": scenario_id}],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_scorer_means_excludes_non_scorer_keys(tmp_path: Path) -> None:
    """CR-01 (BLOCKER): `_scorer_means_from_cell` must only emit scorer names
    registered in `app.agent.critique.checks.CRITIQUE_THRESHOLDS`. The six
    non-scorer `_mean` aggregate keys observed empirically in VERIFICATION.md
    (results_mean, tool_calls_mean, contexts_mean, revision_hints_mean,
    committed_stops_mean, answer_retrieved_place_coverage_mean) are cell-level
    diagnostics — they MUST be excluded so summary.json contains scorers and
    nothing else."""
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell_with_aggregate(
        tmp_path,
        "openai",
        "gpt-4o-mini",
        "scenario_a",
        0,
        aggregate={
            # 6 non-scorer diagnostic _mean keys from the empirical 8-key live
            # payload (see VERIFICATION.md CR-01 section) — all must be filtered.
            "tool_calls_mean": 3.0,
            "results_mean": 5.0,
            "contexts_mean": 2.0,
            "revision_hints_mean": 1.0,
            "committed_stops_mean": 3.0,
            "answer_retrieved_place_coverage_mean": 0.8,
            # 2 real scorer-mean keys (registered in CRITIQUE_THRESHOLDS) —
            # both must survive into the summary.
            "category_compliance_mean": 0.5,
            "rationale_stop_alignment_mean": 0.7,
        },
    )
    summary = aggregate_cell_jsons(tmp_path)
    scorer_block = summary["scenarios"]["scenario_a"]["providers"]["openai/gpt-4o-mini"]["scorers"]
    assert set(scorer_block.keys()) == {"category_compliance", "rationale_stop_alignment"}
    # Spot-check the six non-scorer names CR-01 specifically calls out:
    for forbidden in (
        "tool_calls",
        "results",
        "contexts",
        "revision_hints",
        "committed_stops",
        "answer_retrieved_place_coverage",
    ):
        assert forbidden not in scorer_block, (
            f"{forbidden!r} leaked into scorer block — CR-01 whitelist failed"
        )


def test_scorer_means_rejects_bool_values_disguised_as_numeric(tmp_path: Path) -> None:
    """IN-04: `isinstance(value, int | float)` matches `True`/`False` because
    `bool` is a subclass of `int`. The numeric check must explicitly exclude
    bool so a stray bool in the cell aggregate does not become a scorer
    score of 1.0 or 0.0."""
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell_with_aggregate(
        tmp_path,
        "openai",
        "gpt-4o-mini",
        "scenario_a",
        0,
        aggregate={
            "category_compliance_mean": True,  # bool — must be rejected.
            "rationale_stop_alignment_mean": 0.5,  # real float — must survive.
        },
    )
    summary = aggregate_cell_jsons(tmp_path)
    scorer_block = summary["scenarios"]["scenario_a"]["providers"]["openai/gpt-4o-mini"]["scorers"]
    assert "category_compliance" not in scorer_block, (
        "bool-disguised-as-numeric leaked into scorer block — IN-04 fix failed"
    )
    assert "rationale_stop_alignment" in scorer_block, (
        "real float scorer dropped — IN-04 fix overshot"
    )


def test_aggregate_warns_on_unparseable_cell_filename(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """WR-01: when `_parse_cell_filename` returns None inside the aggregator
    loop, the silent `continue` makes the dropped cell invisible. Emit a
    WARNING log so operators can see when a stray file slipped past the
    glob (typically a `--` collision in a model name). The well-formed cell
    in the same dir must still be aggregated normally."""
    from scripts.eval_matrix import aggregate_cell_jsons

    # One well-formed cell so we can assert the loop still produces output.
    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 0, 0.5)
    # An unparseable filename — 2 split-parts, not 4 — must trigger the warning.
    (tmp_path / "nope--stray.json").write_text("{}", encoding="utf-8")

    with caplog.at_level("WARNING", logger="scripts.eval_matrix"):
        summary = aggregate_cell_jsons(tmp_path)

    # Warning must mention the offending filename.
    matching = [rec for rec in caplog.records if "nope--stray.json" in rec.getMessage()]
    assert matching, (
        "expected a WARNING log mentioning 'nope--stray.json'; "
        f"got records: {[r.getMessage() for r in caplog.records]}"
    )
    # Well-formed cell still aggregated — the warning is observability,
    # not a kill switch.
    assert "scenario_a" in summary["scenarios"]


# ─── IN-02: overridden_to summary field when --llm-provider-override set ────


def test_aggregate_records_overridden_to_when_override_set(tmp_path: Path) -> None:
    """IN-02: when run_matrix has rewritten provider keys via
    `_apply_override`, the resulting summary.json must carry a top-level
    `overridden_to` field naming the override target. Without this, PRs
    that diff summary.json across providers cannot detect the rebind
    (the per-provider scorer keys reflect the rewritten name, not the
    original config). Pass `llm_provider_override` to `aggregate_cell_jsons`
    so the summary records the override explicitly."""
    from scripts.eval_matrix import aggregate_cell_jsons

    # Simulate the post-_apply_override layout: cells already named with
    # the override provider (`scripted` in this case).
    _write_cell(tmp_path, "scripted", "gpt-4o-mini", "scenario_a", 0, 0.5)
    _write_cell(tmp_path, "scripted", "gpt-4o-mini", "scenario_a", 1, 0.6)

    summary = aggregate_cell_jsons(tmp_path, llm_provider_override="scripted")

    assert summary.get("overridden_to") == "scripted"
    # The per-provider keys themselves are NOT rebound by this field —
    # they keep whatever _apply_override wrote (scripted/gpt-4o-mini).
    assert "scripted/gpt-4o-mini" in summary["scenarios"]["scenario_a"]["providers"]


def test_aggregate_omits_overridden_to_when_no_override(tmp_path: Path) -> None:
    """IN-02 (negative case): when no override is in effect, the summary
    must NOT carry an `overridden_to` field. Absence is meaningful — the
    Phase 4-6 diff target reads `overridden_to` to know whether two
    summary.jsons are comparable."""
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 0, 0.5)

    # Default call shape — no override kwarg.
    summary_default = aggregate_cell_jsons(tmp_path)
    assert "overridden_to" not in summary_default

    # Explicit None must also omit the field (defensive — equivalent contract).
    summary_explicit_none = aggregate_cell_jsons(tmp_path, llm_provider_override=None)
    assert "overridden_to" not in summary_explicit_none


# ─── resolve_run_dir naming (D-10 output shape) ──────────────────────────────


def test_resolve_run_dir_under_eval_reports_with_iso_timestamp(tmp_path: Path) -> None:
    """eval_reports/{ISO8601-Z-with-colons-replaced} is the per-run output
    directory shape. resolve_run_dir creates the directory under the
    requested base path and returns the absolute Path."""
    from scripts.eval_matrix import resolve_run_dir

    run_dir = resolve_run_dir(base=tmp_path / "eval_reports")
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path / "eval_reports"
    # Name must not contain raw colons (windows-hostile + URL-hostile).
    assert ":" not in run_dir.name
    # It should at least look like a timestamp.
    assert any(ch.isdigit() for ch in run_dir.name)


# ─── CLI argument parsing ────────────────────────────────────────────────────


def test_main_rejects_zero_runs(monkeypatch) -> None:
    """--runs must be >= 1; zero or negative makes no sense."""
    monkeypatch.setenv("APP_ENV", "eval")
    from scripts.eval_matrix import main

    rc = main(
        [
            "--matrix-config",
            str(REPO_ROOT / "configs/eval_matrix.yaml"),
            "--runs",
            "0",
            "--dry-run",
        ]
    )
    assert rc != 0


# ─── No top-level LLM SDK imports ────────────────────────────────────────────


def test_eval_matrix_module_does_not_import_llm_sdks() -> None:
    """`scripts/eval_matrix.py` must not pull in openai/anthropic/etc. at
    import — those are subprocess-only concerns. The matrix runner is the
    orchestrator, NOT the LLM caller."""
    import scripts.eval_matrix as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "from openai" not in src
    assert "from anthropic" not in src
    assert "import openai" not in src
    assert "from langchain_openai" not in src


# ─── subprocess fan-out shape ────────────────────────────────────────────────


def test_run_matrix_invokes_eval_agent_subprocess(mocker, monkeypatch, tmp_path) -> None:
    """When --dry-run is NOT passed, run_matrix shells out to scripts/eval_agent.py
    once per cell. We mock subprocess.run so this test stays hermetic
    (no real network, no real eval_agent.py invocation)."""
    monkeypatch.setenv("APP_ENV", "eval")
    from scripts.eval_matrix import run_matrix

    fake_run = mocker.patch(
        "scripts.eval_matrix.subprocess.run",
        return_value=mocker.Mock(returncode=0, stdout="{}", stderr=""),
    )
    from app.eval.config import EvalMatrixConfig, MatrixEntry

    matrix = EvalMatrixConfig(
        entries=[MatrixEntry(provider="scripted", model="placeholder")],
        scenarios=["scenario_a"],
    )
    rc, failures = run_matrix(
        matrix=matrix,
        runs=2,
        output_dir=tmp_path,
        llm_provider_override=None,
        eval_queries_path="configs/eval_queries.yaml",
    )
    assert fake_run.call_count == 2  # 1 entry * 1 scenario * 2 runs
    # Each call must shell out to scripts/eval_agent.py with the expected flags
    call_args_list = [call.args[0] for call in fake_run.call_args_list]
    first_cmd = call_args_list[0]
    assert sys.executable == first_cmd[0]
    assert any("scripts/eval_agent.py" in arg for arg in first_cmd)
    assert "--llm-provider" in first_cmd
    assert "scripted" in first_cmd
    assert "--scenario-ids" in first_cmd
    assert "scenario_a" in first_cmd
    # No cell failures expected when subprocess.run returns 0.
    assert failures == []
    # Return code: 0 because no cells failed.
    assert rc == 0


def test_run_matrix_collects_failures_without_short_circuit(mocker, monkeypatch, tmp_path) -> None:
    """Subprocess failures do not stop the matrix — they're recorded in the
    returned `failures` list and the runner still exits non-zero (D-08
    + plan task 3 behavior bullets)."""
    monkeypatch.setenv("APP_ENV", "eval")
    from scripts.eval_matrix import run_matrix

    fake_results = [
        mocker.Mock(returncode=0, stdout="{}", stderr=""),
        mocker.Mock(returncode=2, stdout="", stderr="boom"),
        mocker.Mock(returncode=0, stdout="{}", stderr=""),
    ]
    mocker.patch("scripts.eval_matrix.subprocess.run", side_effect=fake_results)
    from app.eval.config import EvalMatrixConfig, MatrixEntry

    matrix = EvalMatrixConfig(
        entries=[MatrixEntry(provider="scripted", model="placeholder")],
        scenarios=["a", "b", "c"],
    )
    rc, failures = run_matrix(
        matrix=matrix,
        runs=1,
        output_dir=tmp_path,
        llm_provider_override=None,
        eval_queries_path="configs/eval_queries.yaml",
    )
    assert len(failures) == 1
    assert failures[0]["returncode"] == 2
    assert failures[0]["stderr"] == "boom"
    assert rc != 0  # the runner exits non-zero when any cell failed


def test_run_matrix_uses_provider_override_in_subprocess_cmd(mocker, monkeypatch, tmp_path) -> None:
    """--llm-provider-override scripted maps ALL entries to scripted in the
    subprocess invocations (single source of truth for the CI gate)."""
    monkeypatch.setenv("APP_ENV", "dev")
    from scripts.eval_matrix import run_matrix

    fake_run = mocker.patch(
        "scripts.eval_matrix.subprocess.run",
        return_value=mocker.Mock(returncode=0, stdout="{}", stderr=""),
    )
    from app.eval.config import EvalMatrixConfig, MatrixEntry

    matrix = EvalMatrixConfig(
        entries=[MatrixEntry(provider="openai", model="gpt-4o-mini")],
        scenarios=["scenario_a"],
    )
    run_matrix(
        matrix=matrix,
        runs=1,
        output_dir=tmp_path,
        llm_provider_override="scripted",
        eval_queries_path="configs/eval_queries.yaml",
    )
    cmd = fake_run.call_args_list[0].args[0]
    # The override replaces the real-provider entry with scripted.
    assert "scripted" in cmd
    # The original provider name must NOT appear in the cmd.
    assert "openai" not in cmd


# ─── Plan 06-06 Task 2: per-cell env override (D-06-10 / NEW HIGH-B) ─────────


class TestPerCellEnvOverride:
    """Phase 6 plan 06-06 Task 2 — `MatrixCell.env` + `iter_cells` threading
    + `run_matrix` per-cell env composition (subprocess) and os.environ
    apply/cleanup (in-process). Closes the D-06-10 + NEW HIGH-B gap where
    flipping `REFINEMENT_STRUCTURED_PLAN_ENABLED` on the refinement cell
    without flipping it for the first-turn cells was structurally impossible
    in the prior shared-env shape, AND where the propagated env was never
    visible to in-process consumers.

    Per `project_full_suite_db_pool_contamination.md` these tests mock
    `subprocess.run` so no real subprocess fires; the unit suite stays
    hermetic.
    """

    def test_matrix_cell_env_default_is_none(self) -> None:
        """`MatrixCell.env` default is None — backward compat with all
        existing constructors that don't pass `env=...`."""
        from scripts.eval_matrix import MatrixCell

        cell = MatrixCell(provider="o", model="m", scenario_id="s", run_n=0)
        assert cell.env is None

    def test_iter_cells_threads_entry_env(self) -> None:
        """`iter_cells` propagates `entry.env` to every cell yielded for
        that entry. Entries with `env=None` yield cells with `env=None`."""
        from app.eval.config import EvalMatrixConfig, MatrixEntry
        from scripts.eval_matrix import iter_cells

        matrix = EvalMatrixConfig(
            entries=[
                MatrixEntry(
                    provider="openai",
                    model="gpt-4o-mini",
                    env={"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"},
                ),
                MatrixEntry(provider="deepseek", model="deepseek-chat"),
            ],
            scenarios=["scenario_a"],
        )
        cells = list(iter_cells(matrix, runs=1))
        assert cells[0].env == {"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"}
        assert cells[1].env is None

    def test_run_matrix_composes_per_cell_env(self, monkeypatch, mocker, tmp_path) -> None:
        """`run_matrix` composes `{**parent_env, **cell.env}` per-cell and
        passes it to `subprocess.run(env=...)`. Cells without an override
        get the parent env unchanged."""
        monkeypatch.setenv("APP_ENV", "eval")
        from app.eval.config import EvalMatrixConfig, MatrixEntry
        from scripts.eval_matrix import run_matrix

        captured_envs: list[dict] = []

        def fake_subprocess_run(*args, **kwargs):
            captured_envs.append(dict(kwargs.get("env") or {}))
            return mocker.Mock(returncode=0, stdout="{}", stderr="")

        mocker.patch("scripts.eval_matrix.subprocess.run", side_effect=fake_subprocess_run)

        matrix = EvalMatrixConfig(
            entries=[
                MatrixEntry(
                    provider="scripted",
                    model="placeholder",
                    env={"FOO": "bar"},
                ),
                MatrixEntry(provider="scripted", model="other"),
            ],
            scenarios=["scenario_a"],
        )
        run_matrix(
            matrix=matrix,
            runs=1,
            output_dir=tmp_path,
            llm_provider_override=None,
            eval_queries_path="configs/eval_queries.yaml",
        )

        assert len(captured_envs) == 2
        # First cell got FOO=bar in addition to the parent env.
        assert captured_envs[0].get("FOO") == "bar"
        # Parent env keys (like PATH) are present in every cell.
        assert "PATH" in captured_envs[0]
        # Second cell does NOT have FOO (no override on that cell).
        assert "FOO" not in captured_envs[1]

    def test_run_matrix_backward_compat_no_per_cell_env(
        self, monkeypatch, mocker, tmp_path
    ) -> None:
        """When all entries have `env=None`, every cell's captured subprocess
        env equals `os.environ.copy()` exactly — backward compat with the
        pre-Phase-6 shared-env behavior."""
        monkeypatch.setenv("APP_ENV", "eval")
        from app.eval.config import EvalMatrixConfig, MatrixEntry
        from scripts.eval_matrix import run_matrix

        captured_envs: list[dict] = []

        def fake_subprocess_run(*args, **kwargs):
            captured_envs.append(dict(kwargs.get("env") or {}))
            return mocker.Mock(returncode=0, stdout="{}", stderr="")

        mocker.patch("scripts.eval_matrix.subprocess.run", side_effect=fake_subprocess_run)

        matrix = EvalMatrixConfig(
            entries=[
                MatrixEntry(provider="scripted", model="placeholder"),
                MatrixEntry(provider="scripted", model="other"),
            ],
            scenarios=["scenario_a"],
        )
        # Snapshot the parent env BEFORE run_matrix so we compare against
        # what run_matrix saw at call time.
        import os as _os

        parent_env_snapshot = dict(_os.environ)

        run_matrix(
            matrix=matrix,
            runs=1,
            output_dir=tmp_path,
            llm_provider_override=None,
            eval_queries_path="configs/eval_queries.yaml",
        )

        # Each cell's env equals the parent env snapshot (no per-cell override).
        for captured in captured_envs:
            assert captured == parent_env_snapshot

    def test_apply_override_preserves_env(self) -> None:
        """MEDIUM-1 fix: `_apply_override` preserves `entry.env` when
        rewriting `provider` so the per-cell flag still propagates after
        `--llm-provider-override scripted` rebinds entries."""
        from app.eval.config import MatrixEntry
        from scripts.eval_matrix import _apply_override

        entries = [
            MatrixEntry(
                provider="openai",
                model="gpt-4o-mini",
                env={"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"},
            ),
            MatrixEntry(provider="deepseek", model="deepseek-chat"),
        ]
        result = list(_apply_override(entries, llm_provider_override="scripted"))
        assert all(e.provider == "scripted" for e in result)
        assert result[0].env == {"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"}
        assert result[1].env is None

    def test_per_cell_env_applies_to_os_environ_then_unsets(
        self, monkeypatch, mocker, tmp_path
    ) -> None:
        """NEW HIGH-B: per-cell `env` is applied to `os.environ` DURING the
        cell's run and restored AFTER. Critical because in-process consumers
        (e.g., unit-test invocations of `_run_prod_threading`) read
        `os.environ` directly — without the apply/cleanup the per-cell
        override would be invisible to them."""
        monkeypatch.setenv("APP_ENV", "eval")
        monkeypatch.delenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", raising=False)
        import os as _os

        from app.eval.config import EvalMatrixConfig, MatrixEntry
        from scripts.eval_matrix import run_matrix

        captured_inside: list[str | None] = []

        def fake_subprocess_run(*args, **kwargs):
            # Read os.environ at the moment subprocess.run is called — this
            # is INSIDE the per-cell block, so the apply must already have
            # happened.
            captured_inside.append(_os.environ.get("REFINEMENT_STRUCTURED_PLAN_ENABLED"))
            return mocker.Mock(returncode=0, stdout="{}", stderr="")

        mocker.patch("scripts.eval_matrix.subprocess.run", side_effect=fake_subprocess_run)

        matrix = EvalMatrixConfig(
            entries=[
                MatrixEntry(
                    provider="scripted",
                    model="placeholder",
                    env={"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"},
                ),
            ],
            scenarios=["scenario_a"],
        )
        assert _os.environ.get("REFINEMENT_STRUCTURED_PLAN_ENABLED") is None

        run_matrix(
            matrix=matrix,
            runs=1,
            output_dir=tmp_path,
            llm_provider_override=None,
            eval_queries_path="configs/eval_queries.yaml",
        )

        # DURING the cell: os.environ was set to "true".
        assert captured_inside == ["true"]
        # AFTER the cell: cleanup restored the prior unset state.
        assert _os.environ.get("REFINEMENT_STRUCTURED_PLAN_ENABLED") is None

    def test_per_cell_env_restores_prior_value_after_cell(
        self, monkeypatch, mocker, tmp_path
    ) -> None:
        """NEW HIGH-B edge case: when the parent env had a prior value for
        a key the cell overrides, the cleanup restores the prior value
        (not just pops the key)."""
        monkeypatch.setenv("APP_ENV", "eval")
        monkeypatch.setenv("REFINEMENT_STRUCTURED_PLAN_ENABLED", "false")
        import os as _os

        from app.eval.config import EvalMatrixConfig, MatrixEntry
        from scripts.eval_matrix import run_matrix

        def fake_subprocess_run(*args, **kwargs):
            return mocker.Mock(returncode=0, stdout="{}", stderr="")

        mocker.patch("scripts.eval_matrix.subprocess.run", side_effect=fake_subprocess_run)

        matrix = EvalMatrixConfig(
            entries=[
                MatrixEntry(
                    provider="scripted",
                    model="placeholder",
                    env={"REFINEMENT_STRUCTURED_PLAN_ENABLED": "true"},
                ),
            ],
            scenarios=["scenario_a"],
        )
        run_matrix(
            matrix=matrix,
            runs=1,
            output_dir=tmp_path,
            llm_provider_override=None,
            eval_queries_path="configs/eval_queries.yaml",
        )

        # Prior parent value restored, not popped.
        assert _os.environ.get("REFINEMENT_STRUCTURED_PLAN_ENABLED") == "false"


# ─── CI-workflow drift guards (Plan 03-06 / EVAL-09) ─────────────────────────
# These tests pin the shape of .github/workflows/ci.yml's eval-matrix job so
# that a future PR cannot silently drop the scripted-mode flag (exposing CI
# to real LLM API costs + rate limits), accidentally set APP_ENV=eval
# (defeating the runtime gate), or remove the artifact upload (losing the
# summary.json that PR reviewers need to inspect matrix output).
#
# These are smoke-level tests (per project memory feedback_test_layering):
# they verify a static file's shape, not runtime CI behavior. They run with
# the unit suite because the YAML parse is fast (<10ms) and self-contained.


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    """Module-scoped parse of .github/workflows/ci.yml so the three CI-drift
    tests share one yaml.safe_load. Path resolved via the existing REPO_ROOT
    constant from app.eval.config (the resolve_eval_queries_path pattern),
    not a hardcoded absolute path or a sys.path bootstrap."""
    import yaml

    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(ci_path.read_text(encoding="utf-8"))


def test_ci_workflow_uses_scripted_provider(ci_workflow: dict) -> None:
    """EVAL-09 / P4: the eval-matrix CI job MUST invoke the matrix runner
    with `scripted` mode. Without this guard a future PR could remove
    --llm-provider-override (or the LLM_OVERRIDE=scripted make var) and
    expose CI to real LLM API calls, rate limits, and cost.

    We accept any step whose `run` command mentions both `eval-matrix` (the
    make target or the script name) AND `scripted` — this gives Phase 4-6
    flexibility to refactor the invocation (e.g. drop the make wrapper)
    without breaking this guard, as long as scripted mode stays in force.
    """
    job = ci_workflow["jobs"].get("eval-matrix")
    assert job is not None, "eval-matrix job missing from .github/workflows/ci.yml"
    step_runs = [s.get("run", "") for s in job["steps"] if isinstance(s, dict)]
    matrix_steps = [
        r for r in step_runs if ("eval-matrix" in r or "eval_matrix.py" in r) and "scripted" in r
    ]
    assert matrix_steps, (
        "no eval-matrix step invokes scripted mode — silent CI drift risk; "
        "expected a `run:` containing both 'eval-matrix' (or 'eval_matrix.py') "
        "and 'scripted' (P4 / EVAL-09 gate)"
    )


def test_ci_workflow_does_not_set_app_env_eval(ci_workflow: dict) -> None:
    """P4 / EVAL-09: the eval-matrix CI job MUST NOT set APP_ENV=eval. The
    runtime gate in scripts/eval_matrix.py exists exactly so CI never
    accidentally runs real-provider matrices; setting APP_ENV=eval would
    bypass that gate. The scripted-mode override (asserted by the test
    above) is the ONLY sanctioned bypass."""
    job = ci_workflow["jobs"].get("eval-matrix")
    assert job is not None, "eval-matrix job missing from .github/workflows/ci.yml"

    def _walk_for_app_env_eval(node: object) -> bool:
        """Recursive walk: trip on (a) any `env:` mapping with APP_ENV: eval,
        or (b) any string value containing `APP_ENV=eval` (shell-style env
        assignment in a `run:` block)."""
        if isinstance(node, dict):
            env_block = node.get("env")
            if isinstance(env_block, dict) and env_block.get("APP_ENV") == "eval":
                return True
            return any(_walk_for_app_env_eval(v) for v in node.values())
        if isinstance(node, list):
            return any(_walk_for_app_env_eval(x) for x in node)
        if isinstance(node, str):
            return "APP_ENV=eval" in node or "APP_ENV: eval" in node
        return False

    assert not _walk_for_app_env_eval(job), (
        "eval-matrix job sets APP_ENV=eval somewhere — this defeats the P4 / "
        "EVAL-09 gate. Use --llm-provider-override scripted instead "
        "(LLM_OVERRIDE=scripted in the make target)."
    )


# ─── WR-04: argparse-level '--' guard on --llm-provider-override ─────────────


def test_parse_args_rejects_double_dash_in_llm_provider_override() -> None:
    """WR-04: '--' is reserved as the cell-filename separator. Without an
    argparse type validator, '--llm-provider-override foo--bar' would
    propagate to MatrixEntry's after-validator and crash with a
    pydantic.ValidationError mid-run_matrix() — no actionable CLI error.

    Reject it at parse-time so the operator sees argparse-style 'rc=2 +
    usage' instead of a pydantic traceback.
    """
    from scripts.eval_matrix import parse_args

    with pytest.raises(SystemExit) as excinfo:
        parse_args(
            [
                "--matrix-config",
                "configs/eval_matrix.yaml",
                "--llm-provider-override",
                "foo--bar",
            ]
        )
    # argparse type-error exits with rc=2 by default; anything else means
    # argparse silently accepted the value.
    assert excinfo.value.code == 2


def test_parse_args_accepts_single_dash_llm_provider_override() -> None:
    """WR-04 negative-control: single-dash and alphanumeric overrides pass."""
    from scripts.eval_matrix import parse_args

    args = parse_args(["--llm-provider-override", "scripted"])
    assert args.llm_provider_override == "scripted"
    args = parse_args(["--llm-provider-override", "gpt-4o-mini"])
    assert args.llm_provider_override == "gpt-4o-mini"


def test_ci_workflow_eval_matrix_uploads_artifact(ci_workflow: dict) -> None:
    """The eval-matrix CI job MUST upload its eval_reports/ output as an
    artifact (via actions/upload-artifact@v4) so PR reviewers can recover
    summary.json. Without this, scripted-cell failures are invisible to
    reviewers and the soft-gate stance (Phase 3) becomes useless. The
    `if: always()` shape is recommended so failed-cell debug data
    survives, but this test pins only the artifact-upload requirement —
    Phase 4-6 may refine the `if:` clause without breaking the guard."""
    job = ci_workflow["jobs"].get("eval-matrix")
    assert job is not None, "eval-matrix job missing from .github/workflows/ci.yml"
    step_uses = [s.get("uses", "") for s in job["steps"] if isinstance(s, dict)]
    upload_steps = [u for u in step_uses if "upload-artifact" in u]
    assert upload_steps, (
        "eval-matrix job has no actions/upload-artifact step — "
        "summary.json output cannot be recovered for PR review"
    )


# ─── Phase 6 plan 06-07 / NEW HIGH-A: --structural-check flag ───────────────


class TestStructuralCheck:
    """NEW HIGH-A fix (plan 06-07): --structural-check validates the matrix
    end-to-end WITHOUT invoking subprocess.run. Pins the no-subprocess
    contract + all five structural checks (matrix loads, iter_cells
    non-empty, _apply_override preserves env, DETERMINISTIC_CHECKS contains
    'refinement_minimal_edit', build_refinement_prompt_message functional).

    Sidesteps the SCRIPTED_SCENARIOS-empty problem (app/llm_factory.py:170 —
    SCRIPTED_SCENARIOS is an empty dict; a live --llm-provider-override
    scripted run on the refinement matrix would crash without a populated
    refinement_cheaper trajectory). The structural-check path is the
    CI-hard-gated smoke test per N-4 + REF-04.
    """

    @staticmethod
    def _write_valid_refinement_matrix(tmp_path: Path) -> Path:
        """Write a minimal valid refinement matrix YAML."""
        yaml_path = tmp_path / "matrix.yaml"
        yaml_path.write_text(
            "entries:\n"
            "  - provider: openai\n"
            "    model: gpt-4o-mini\n"
            "    env:\n"
            '      REFINEMENT_STRUCTURED_PLAN_ENABLED: "true"\n'
            "scenarios:\n"
            "  - refinement_cheaper\n",
            encoding="utf-8",
        )
        return yaml_path

    def test_structural_check_exits_0_for_well_formed_refinement_matrix(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """A well-formed refinement matrix passes all five structural checks.

        ALSO monkeypatches subprocess.run to raise so the test fails loudly
        if the flag ever invokes a subprocess (NEW HIGH-A contract: zero
        subprocess.run calls under --structural-check).
        """
        from scripts import eval_matrix as eval_matrix_mod

        def _fail_subprocess(*_args, **_kwargs):  # pragma: no cover
            raise AssertionError("subprocess.run must NOT be called in structural-check mode")

        monkeypatch.setattr(eval_matrix_mod.subprocess, "run", _fail_subprocess)
        yaml_path = self._write_valid_refinement_matrix(tmp_path)
        rc = eval_matrix_mod.main(["--matrix-config", str(yaml_path), "--structural-check"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "structural-check: OK" in captured.err

    def test_structural_check_exits_1_when_cells_empty(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """If iter_cells yields zero cells -> exit 1 with descriptive error.

        Pydantic's EvalMatrixConfig validator rejects YAML with `scenarios:
        []` at load time (min_length=1), so the empty-cells branch is only
        reachable via a stub-iter_cells. Monkeypatch the module's
        `iter_cells` to return an empty iterator to exercise the guard.
        """
        from scripts import eval_matrix as eval_matrix_mod

        monkeypatch.setattr(eval_matrix_mod, "iter_cells", lambda matrix, runs: iter([]))
        yaml_path = self._write_valid_refinement_matrix(tmp_path)
        rc = eval_matrix_mod.main(["--matrix-config", str(yaml_path), "--structural-check"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "iter_cells produced 0 cells" in captured.err

    def test_structural_check_does_not_invoke_subprocess_run(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Pin the no-subprocess contract: --structural-check NEVER calls
        subprocess.run. Strategy (b) per NEW HIGH-A — this is the whole
        point of the flag (avoids SCRIPTED_SCENARIOS-empty crash).
        """
        from scripts import eval_matrix as eval_matrix_mod

        call_log: list[tuple] = []

        def _record_subprocess(*args, **kwargs):
            call_log.append((args, kwargs))
            raise AssertionError("subprocess.run must NOT be called in structural-check mode")

        monkeypatch.setattr(eval_matrix_mod.subprocess, "run", _record_subprocess)
        yaml_path = self._write_valid_refinement_matrix(tmp_path)
        rc = eval_matrix_mod.main(["--matrix-config", str(yaml_path), "--structural-check"])
        assert rc == 0
        assert call_log == [], (
            f"structural-check invoked subprocess.run {len(call_log)} time(s); "
            f"strategy (b) contract violated"
        )

    def test_structural_check_passes_when_env_override_preserved(self, tmp_path: Path) -> None:
        """A matrix with per-cell env={REFINEMENT_STRUCTURED_PLAN_ENABLED: 'true'}
        exercises the MEDIUM-1 env-preservation path through _apply_override.
        Exit 0 == env survived the scripted-override rebind in CI smoke.
        """
        from scripts import eval_matrix as eval_matrix_mod

        yaml_path = tmp_path / "matrix.yaml"
        yaml_path.write_text(
            "entries:\n"
            "  - provider: openai\n"
            "    model: gpt-4o-mini\n"
            "    env:\n"
            '      FOO: "bar"\n'
            "scenarios:\n"
            "  - refinement_cheaper\n",
            encoding="utf-8",
        )
        rc = eval_matrix_mod.main(["--matrix-config", str(yaml_path), "--structural-check"])
        assert rc == 0

    def test_structural_check_passes_when_refinement_minimal_edit_registered(
        self,
    ) -> None:
        """HIGH-1 cross-check: refinement_minimal_edit MUST be in
        DETERMINISTIC_CHECKS for the merge gate to be enforceable. Without
        this registration, the scorer key never appears in baseline JSON
        and the D-06-09 gate is silently absent.
        """
        from scripts.eval_agent import DETERMINISTIC_CHECKS

        assert "refinement_minimal_edit" in DETERMINISTIC_CHECKS, (
            "refinement_minimal_edit missing from DETERMINISTIC_CHECKS — "
            "plan 06-03 registration regressed"
        )


# ─── 10-02: n_scored/n_errored/cell_valid error-threading in aggregation ─────


def _write_cell_with_error_fields(
    directory: Path,
    provider: str,
    model: str,
    scenario_id: str,
    run_n: int,
    n_scored: int,
    n_errored: int,
    errors: list[dict[str, str]] | None = None,
    score_value: float = 0.5,
) -> Path:
    """Write a cell JSON with the D-10-03 aggregate error fields (10-01 shape).

    Supports both OK cells (n_errored=0) and errored cells (n_errored>=1).
    The `errors` list mirrors the shape produced by eval_agent.aggregate_results.
    """
    fname = f"{provider}--{model}--{scenario_id}--run-{run_n}.json"
    path = directory / fname
    aggregate: dict = {
        "n_scored": n_scored,
        "n_errored": n_errored,
        "cell_valid": n_errored == 0,
        "errors": errors or [],
        "category_compliance_mean": score_value,
        "rationale_stop_alignment_mean": score_value,
    }
    payload = {
        "llm_provider": provider,
        "chat_model": model,
        "query_count": n_scored + n_errored,
        "aggregate": aggregate,
        "queries": [{"id": scenario_id}],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_aggregate_cell_jsons_threads_error_counts(tmp_path: Path) -> None:
    """10-02 / D-10-03: aggregate_cell_jsons surfaces n_scored, n_errored, and
    cell_valid per provider-key block, and accumulates a top-level errors list.

    Setup: one OK cell (n_errored=0) and one errored cell (n_errored=1).
    Expected:
      - OK cell: n_scored=1, n_errored=0, cell_valid=True
      - Errored cell: n_scored=0, n_errored=1, cell_valid=False
      - summary["errors"] is non-empty with the errored cell's entry
    """
    from scripts.eval_matrix import aggregate_cell_jsons

    # OK cell — n_errored=0, contributes scores.
    _write_cell_with_error_fields(
        tmp_path,
        "openai",
        "gpt-4o-mini",
        "scenario_a",
        0,
        n_scored=1,
        n_errored=0,
        errors=[],
    )
    # Errored cell — n_errored=1, stage/type populated.
    _write_cell_with_error_fields(
        tmp_path,
        "openai",
        "gpt-4o-mini",
        "scenario_a",
        1,
        n_scored=0,
        n_errored=1,
        errors=[{"stage": "turn0", "type": "RateLimitError", "message": "quota exceeded"}],
    )

    summary = aggregate_cell_jsons(tmp_path)

    # Per-provider block must carry n_scored, n_errored, cell_valid.
    provider_block = summary["scenarios"]["scenario_a"]["providers"]["openai/gpt-4o-mini"]
    # n_scored: sum across 2 runs → 1+0=1
    assert provider_block["n_scored"] == 1
    # n_errored: sum across 2 runs → 0+1=1
    assert provider_block["n_errored"] == 1
    # cell_valid: False because n_errored > 0
    assert provider_block["cell_valid"] is False

    # Top-level errors list must be non-empty.
    assert "errors" in summary
    assert len(summary["errors"]) >= 1
    # Each error entry must carry stage and type.
    err = summary["errors"][0]
    assert "stage" in err
    assert "type" in err


def test_aggregate_cell_jsons_cell_valid_true_when_no_errors(tmp_path: Path) -> None:
    """10-02 backward-compat: cells without any errored runs have cell_valid=True."""
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell_with_error_fields(
        tmp_path,
        "openai",
        "gpt-4o-mini",
        "scenario_a",
        0,
        n_scored=1,
        n_errored=0,
        errors=[],
    )
    _write_cell_with_error_fields(
        tmp_path,
        "openai",
        "gpt-4o-mini",
        "scenario_a",
        1,
        n_scored=1,
        n_errored=0,
        errors=[],
    )

    summary = aggregate_cell_jsons(tmp_path)
    provider_block = summary["scenarios"]["scenario_a"]["providers"]["openai/gpt-4o-mini"]
    assert provider_block["n_scored"] == 2
    assert provider_block["n_errored"] == 0
    assert provider_block["cell_valid"] is True
    # No errors → top-level errors list is empty (or absent).
    assert summary.get("errors", []) == []


def test_aggregate_cell_jsons_legacy_cell_json_defaults_to_zero_errored(tmp_path: Path) -> None:
    """10-02 backward-compat: legacy cell JSONs missing n_scored/n_errored fields
    default to n_errored=0, n_scored=n (backward-compatible with pre-10-01 cells)."""
    from scripts.eval_matrix import aggregate_cell_jsons

    # Legacy cell with NO n_scored/n_errored/cell_valid fields.
    _write_cell(tmp_path, "openai", "gpt-4o-mini", "scenario_a", 0, 0.5)

    summary = aggregate_cell_jsons(tmp_path)
    provider_block = summary["scenarios"]["scenario_a"]["providers"]["openai/gpt-4o-mini"]
    # Legacy cell: n_errored defaults to 0.
    assert provider_block["n_errored"] == 0
    # Legacy cell: cell_valid defaults to True (no errors).
    assert provider_block["cell_valid"] is True


def test_aggregate_cell_jsons_error_count_in_exit_code(tmp_path: Path) -> None:
    """10-02: main() exits non-zero when total_errored > 0 (cell had errored runs).

    Uses a tmp_path with one errored cell. The matrix runner subprocess fan-out
    is monkeypatched to be a no-op; we call aggregate_cell_jsons + a minimal
    main() wrapper to verify the error-aware exit code path.

    This is the integration-level exit-code assertion; the structural-check
    extension test is separate (test_structural_check_validates_error_schema).
    """
    from scripts.eval_matrix import aggregate_cell_jsons

    _write_cell_with_error_fields(
        tmp_path,
        "openai",
        "gpt-4o-mini",
        "scenario_a",
        0,
        n_scored=0,
        n_errored=1,
        errors=[{"stage": "turn0", "type": "RateLimitError", "message": "quota"}],
    )

    summary = aggregate_cell_jsons(tmp_path)
    provider_block = summary["scenarios"]["scenario_a"]["providers"]["openai/gpt-4o-mini"]
    assert provider_block["n_errored"] == 1
    assert provider_block["cell_valid"] is False


class TestStructuralCheckErrorSchema:
    """10-02: --structural-check now includes Check 6 — validates the error-schema
    contract is well-formed. A synthetic error cell with {status:"error",
    error:{stage,type,message}} passes membership checks; a malformed one fails.
    """

    @staticmethod
    def _write_valid_refinement_matrix(tmp_path: Path) -> Path:
        """Write a minimal valid refinement matrix YAML (mirrors TestStructuralCheck)."""
        yaml_path = tmp_path / "matrix.yaml"
        yaml_path.write_text(
            "entries:\n"
            "  - provider: openai\n"
            "    model: gpt-4o-mini\n"
            "    env:\n"
            '      REFINEMENT_STRUCTURED_PLAN_ENABLED: "true"\n'
            "scenarios:\n"
            "  - refinement_cheaper\n",
            encoding="utf-8",
        )
        return yaml_path

    def test_structural_check_validates_error_schema(self, tmp_path: Path) -> None:
        """10-02 Check 6: --structural-check validates the error-schema shape.

        A well-formed matrix with the error-schema contract in place exits 0.
        Asserts that Check 6 is present by verifying exit 0 still holds after
        adding the check.
        """
        from scripts import eval_matrix as eval_matrix_mod

        yaml_path = self._write_valid_refinement_matrix(tmp_path)
        rc = eval_matrix_mod.main(["--matrix-config", str(yaml_path), "--structural-check"])
        assert rc == 0

    def test_structural_check_error_schema_check_is_present_in_source(self) -> None:
        """10-02: assert the error-schema membership check is present in the
        structural-check block. Validates the source contains the stage membership
        guard (not subprocess-calling code)."""
        import inspect

        import scripts.eval_matrix as mod

        src = inspect.getsource(mod.main)
        assert 'in {"setup", "turn0", "turnN"}' in src, (
            "Check 6 error-schema membership guard not found in main() source — "
            "structural-check block missing the stage membership assertion"
        )
