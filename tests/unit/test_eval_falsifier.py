"""Unit + smoke tests for scripts/eval_falsifier.py (INST-05 / D-12-06..08).

Falsifier report that reads eval artifacts and answers:
  - Did gpt-5-mini hit >= 0.6 committed_itinerary_rate pooled across scenarios?
  - Did gpt-4o-mini hold >= its honest baseline floor?

Exit-code conventions:
    0 = PASS — all falsifier checks met
    1 = FAIL — one or more checks failed (expected; not an infra error)
    2 = infrastructure failure (missing run dir, malformed JSON)

Tests use synthetic summary dicts and tmp_path fixtures so they never need
live eval runs or real API keys. A committed smoke test exercises the script
against the real configs/eval_baselines directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_falsifier.py"


def _load_script() -> ModuleType:
    """Load scripts/eval_falsifier.py as a module without sys.path mutation."""
    spec = importlib.util.spec_from_file_location("eval_falsifier", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_falsifier"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    """The eval_falsifier module under test."""
    return _load_script()


# ---------------------------------------------------------------------------
# Synthetic summary helpers
# ---------------------------------------------------------------------------

_GPT5_KEY = "openai/gpt-5-mini"
_ANCHOR_KEY = "openai/gpt-4o-mini"


def _make_summary(
    scenarios: dict[str, dict],
) -> dict:
    """Build a summary.json-shaped dict from a {scenario_id: provider_map} input.

    Each provider_map is a {provider_key: cell_dict} mapping.
    """
    out: dict = {"scenarios": {}, "errors": []}
    for scenario_id, providers in scenarios.items():
        out["scenarios"][scenario_id] = {
            "providers": providers,
            "baseline_eligible": True,
        }
    return out


def _cell_with_cir(median: float, n: int = 5) -> dict:
    """A cell dict with committed_itinerary_rate populated."""
    return {
        "scorers": {
            "committed_itinerary_rate": {"median": median, "mean": median, "n": n},
        },
        "n_scored": n,
        "n_errored": 0,
        "cell_valid": True,
    }


def _cell_no_cir(n: int = 5) -> dict:
    """A cell dict WITHOUT committed_itinerary_rate (metric absent)."""
    return {
        "scorers": {
            "category_compliance": {"median": 1.0, "mean": 1.0, "n": n},
        },
        "n_scored": n,
        "n_errored": 0,
        "cell_valid": True,
    }


# ---------------------------------------------------------------------------
# Unit tests: _pooled_commit_rate
# ---------------------------------------------------------------------------


class TestPooledCommitRate:
    """Tests for the _pooled_commit_rate helper function."""

    def test_two_scenarios_pool_to_average_when_equal_n(self, script: ModuleType) -> None:
        """Two scenarios with medians [1.0, 0.0] at equal n pool to 0.5 (D-12-08)."""
        summary = _make_summary(
            {
                "scenario_a": {_GPT5_KEY: _cell_with_cir(1.0, n=5)},
                "scenario_b": {_GPT5_KEY: _cell_with_cir(0.0, n=5)},
            }
        )
        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert pooled == pytest.approx(0.5, abs=1e-9)
        assert per_scenario["scenario_a"] == pytest.approx(1.0)
        assert per_scenario["scenario_b"] == pytest.approx(0.0)

    def test_per_scenario_breakdown_contains_both_entries(self, script: ModuleType) -> None:
        """Per-scenario breakdown includes an entry for every scenario."""
        summary = _make_summary(
            {
                "omakase": {_GPT5_KEY: _cell_with_cir(1.0, n=5)},
                "refinement": {_GPT5_KEY: _cell_with_cir(0.0, n=5)},
            }
        )
        _, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert set(per_scenario.keys()) == {"omakase", "refinement"}

    def test_baseline_eligible_false_scenario_excluded_from_pool(self, script: ModuleType) -> None:
        """Scenarios with baseline_eligible: False are excluded from the pooled rate."""
        summary = _make_summary(
            {
                "eligible_scenario": {_GPT5_KEY: _cell_with_cir(0.4, n=5)},
                "quarantined_scenario": {_GPT5_KEY: _cell_with_cir(1.0, n=5)},
            }
        )
        # Mark the second scenario as quarantined
        summary["scenarios"]["quarantined_scenario"]["baseline_eligible"] = False

        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        # Only eligible scenario counts → pooled = 0.4
        assert pooled == pytest.approx(0.4, abs=1e-9)
        # Quarantined scenario not in per_scenario (skipped entirely)
        assert "quarantined_scenario" not in per_scenario
        assert per_scenario["eligible_scenario"] == pytest.approx(0.4)

    def test_absent_provider_cell_yields_none_entry(self, script: ModuleType) -> None:
        """When a provider has no cell in a scenario, per_scenario entry is None."""
        summary = _make_summary(
            {
                "scenario_a": {_ANCHOR_KEY: _cell_with_cir(1.0, n=5)},  # no gpt-5-mini
                "scenario_b": {_GPT5_KEY: _cell_with_cir(0.8, n=5)},
            }
        )
        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert per_scenario["scenario_a"] is None
        assert per_scenario["scenario_b"] == pytest.approx(0.8)
        # Only scenario_b contributes to the pooled rate
        assert pooled == pytest.approx(0.8, abs=1e-9)

    def test_no_eligible_scenarios_returns_none_pooled(self, script: ModuleType) -> None:
        """When all scenarios are quarantined, returns (None, empty_dict)."""
        summary = _make_summary({"scenario_a": {_GPT5_KEY: _cell_with_cir(1.0, n=5)}})
        summary["scenarios"]["scenario_a"]["baseline_eligible"] = False
        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert pooled is None
        assert per_scenario == {}

    def test_empty_scenarios_returns_none_pooled(self, script: ModuleType) -> None:
        """An empty scenarios dict returns (None, {})."""
        summary: dict = {"scenarios": {}, "errors": []}
        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert pooled is None
        assert per_scenario == {}

    def test_weighted_pool_with_unequal_n(self, script: ModuleType) -> None:
        """Pool is weighted by n, not a simple average of medians."""
        # scenario_a: median=1.0, n=1; scenario_b: median=0.0, n=9
        # weighted: (1.0*1 + 0.0*9) / 10 = 0.1
        summary = _make_summary(
            {
                "scenario_a": {_GPT5_KEY: _cell_with_cir(1.0, n=1)},
                "scenario_b": {_GPT5_KEY: _cell_with_cir(0.0, n=9)},
            }
        )
        pooled, _ = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert pooled == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests: bar logic (0.6 threshold)
# ---------------------------------------------------------------------------


class TestBarLogic:
    """Tests for the 0.6 threshold bar logic via main()."""

    def _write_summary(self, tmp_path: Path, summary: dict) -> Path:
        """Write a summary.json to a tmp run dir, return the run dir path."""
        run_dir = tmp_path / "2026-01-01T00-00-00Z"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps(summary))
        return run_dir

    def test_pooled_05_fails_06_bar(
        self, tmp_path: Path, script: ModuleType, capsys: pytest.CaptureFixture
    ) -> None:
        """Pooled 0.5 is below the 0.6 bar → exit 1."""
        summary = _make_summary(
            {
                "scenario_a": {
                    _GPT5_KEY: _cell_with_cir(1.0, n=5),
                    _ANCHOR_KEY: _cell_with_cir(1.0, n=5),
                },
                "scenario_b": {
                    _GPT5_KEY: _cell_with_cir(0.0, n=5),
                    _ANCHOR_KEY: _cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = self._write_summary(tmp_path, summary)
        rc2 = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
            ]
        )
        assert rc2 == 1, f"pooled 0.5 < 0.6 should return exit 1, got {rc2}"
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_pooled_08_passes_06_bar(self, tmp_path: Path, script: ModuleType) -> None:
        """Pooled 0.8 is above the 0.6 bar → exit 0 (assuming anchor holds)."""
        summary = _make_summary(
            {
                "scenario_a": {
                    _GPT5_KEY: _cell_with_cir(0.8, n=5),
                    _ANCHOR_KEY: _cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = self._write_summary(tmp_path, summary)
        rc = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
            ]
        )
        assert rc == 0, f"pooled 0.8 >= 0.6 with anchor holding should return 0, got {rc}"

    def test_missing_summary_json_returns_exit_2(self, tmp_path: Path, script: ModuleType) -> None:
        """Missing summary.json in run dir → exit 2 (infra failure)."""
        run_dir = tmp_path / "2026-01-01T00-00-00Z"
        run_dir.mkdir()
        # No summary.json written
        rc = script.main(["--run-dir", str(run_dir)])
        assert rc == 2, f"missing summary.json must return exit 2, got {rc}"

    def test_nonexistent_run_dir_returns_exit_2(self, script: ModuleType) -> None:
        """Nonexistent --run-dir → exit 2."""
        rc = script.main(["--run-dir", "/nonexistent/path/that/does/not/exist"])
        assert rc == 2, f"nonexistent run dir must return exit 2, got {rc}"


# ---------------------------------------------------------------------------
# Unit tests: --baselines-mode with synthetic baselines dir
# ---------------------------------------------------------------------------


def _write_minimal_baseline(
    baselines_dir: Path,
    scenario_id: str,
    provider_rates: dict[str, float],
    n: int = 5,
) -> None:
    """Write a minimal baseline JSON the falsifier can read via --baselines-mode."""
    providers: dict = {}
    for provider_key, rate in provider_rates.items():
        providers[provider_key] = {
            "scorers": {
                "committed_itinerary_rate": {
                    "median": rate,
                    "mean": rate,
                    "n": n,
                    "min": rate,
                    "max": rate,
                    "stdev": 0.0,
                }
            },
            "n_scored": n,
        }
    payload = {
        "scenario_id": scenario_id,
        "providers": providers,
    }
    (baselines_dir / f"{scenario_id}.json").write_text(json.dumps(payload))


class TestBaselinesModeUnit:
    """Tests for the --baselines-mode code path via synthetic baselines dir."""

    def test_baselines_mode_with_synthetic_dir_returns_real_verdict(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """--baselines-mode against synthetic baselines dir returns 0 or 1 (real verdict, not 2)."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        # Write minimal baseline with gpt-5-mini above bar and anchor present
        _write_minimal_baseline(
            baselines_dir,
            "scenario_a",
            {
                _GPT5_KEY: 0.8,  # above 0.6 bar
                _ANCHOR_KEY: 1.0,
            },
        )

        rc = script.main(
            [
                "--baselines-mode",
                "--baselines-dir",
                str(baselines_dir),
            ]
        )
        assert rc in {0, 1}, f"--baselines-mode must return 0 or 1, got {rc}"

    def test_baselines_mode_gpt5_above_bar_returns_exit_0(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """--baselines-mode with gpt-5-mini >= 0.6 returns exit 0."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        _write_minimal_baseline(
            baselines_dir,
            "scenario_a",
            {_GPT5_KEY: 0.8, _ANCHOR_KEY: 1.0},
        )
        rc = script.main(["--baselines-mode", "--baselines-dir", str(baselines_dir)])
        assert rc == 0

    def test_baselines_mode_gpt5_below_bar_returns_exit_1(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """--baselines-mode with gpt-5-mini < 0.6 returns exit 1."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        _write_minimal_baseline(
            baselines_dir,
            "scenario_a",
            {_GPT5_KEY: 0.3, _ANCHOR_KEY: 1.0},
        )
        rc = script.main(["--baselines-mode", "--baselines-dir", str(baselines_dir)])
        assert rc == 1

    def test_baselines_mode_empty_dir_returns_exit_2(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """--baselines-mode with empty baselines dir returns exit 2 (infra failure)."""
        baselines_dir = tmp_path / "empty_baselines"
        baselines_dir.mkdir()
        # No JSON files written
        rc = script.main(["--baselines-mode", "--baselines-dir", str(baselines_dir)])
        assert rc == 2, f"empty baselines dir must return exit 2, got {rc}"

    def test_baselines_mode_missing_dir_returns_exit_2(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """--baselines-mode with nonexistent baselines dir returns exit 2."""
        rc = script.main(
            [
                "--baselines-mode",
                "--baselines-dir",
                str(tmp_path / "nonexistent_dir"),
            ]
        )
        assert rc == 2


# ---------------------------------------------------------------------------
# Unit tests: malformed summary shapes must not crash (WR-04)
# ---------------------------------------------------------------------------


class TestMalformedSummaryShapes:
    """WR-04: malformed summary.json degrades to N/A, never an uncaught traceback.

    An uncaught traceback exits the interpreter with code 1, which the
    documented exit-code contract reserves for a legitimate FAIL verdict —
    tooling keyed on exit codes would misread a corrupt artifact as a true
    falsifier failure.
    """

    def test_null_n_treated_as_unevaluable(self, script: ModuleType) -> None:
        summary = _make_summary(
            {
                "s": {
                    _GPT5_KEY: {"scorers": {"committed_itinerary_rate": {"median": 1.0, "n": None}}}
                }
            }
        )
        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert pooled is None
        assert per_scenario["s"] is None

    def test_string_n_treated_as_unevaluable(self, script: ModuleType) -> None:
        summary = _make_summary(
            {"s": {_GPT5_KEY: {"scorers": {"committed_itinerary_rate": {"median": 1.0, "n": "5"}}}}}
        )
        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert pooled is None
        assert per_scenario["s"] is None

    def test_non_dict_scenario_block_treated_as_unevaluable(self, script: ModuleType) -> None:
        summary = {"scenarios": {"s": ["not", "a", "dict"]}}
        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert pooled is None
        assert per_scenario["s"] is None

    def test_non_dict_scorers_treated_as_unevaluable(self, script: ModuleType) -> None:
        summary = _make_summary({"s": {_GPT5_KEY: {"scorers": "corrupt"}}})
        pooled, per_scenario = script._pooled_commit_rate(summary, _GPT5_KEY)
        assert pooled is None
        assert per_scenario["s"] is None

    def test_non_dict_scenarios_returns_none(self, script: ModuleType) -> None:
        pooled, per_scenario = script._pooled_commit_rate({"scenarios": []}, _GPT5_KEY)
        assert pooled is None
        assert per_scenario == {}

    def test_main_with_malformed_n_returns_verdict_not_traceback(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """main() over a summary with null n must return a deliberate exit code,
        not raise TypeError (pre-fix behavior: uncaught crash → interpreter exit 1)."""
        summary = _make_summary(
            {
                "s": {
                    _GPT5_KEY: {"scorers": {"committed_itinerary_rate": {"median": 1.0, "n": None}}}
                }
            }
        )
        run_dir = tmp_path / "2026-01-01T00-00-00Z"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps(summary))
        rc = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
            ]
        )
        assert rc in {0, 1, 2}, f"must return a deliberate exit code, got {rc}"
        assert rc == 1, "all cells unevaluable → N/A → FAIL verdict (exit 1)"


# ---------------------------------------------------------------------------
# Unit tests: anchor non-regression over the COMMON scenario set (CR-01)
# ---------------------------------------------------------------------------


def _write_run_summary(tmp_path: Path, summary: dict) -> Path:
    """Write a summary.json to a tmp run dir, return the run dir path."""
    run_dir = tmp_path / "run" / "2026-01-01T00-00-00Z"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


class TestAnchorCommonScenarioPooling:
    """CR-01: anchor floor must be pooled over the run∩baseline scenario set."""

    def _baselines_with_two_scenarios(self, tmp_path: Path) -> Path:
        """Baselines: omakase anchor=0.8, refinement anchor=1.0 (floor 0.9 if mis-pooled)."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        _write_minimal_baseline(
            baselines_dir, "omakase_mission_open_ended", {_ANCHOR_KEY: 0.8, _GPT5_KEY: 0.8}
        )
        _write_minimal_baseline(
            baselines_dir, "refinement_cheaper", {_ANCHOR_KEY: 1.0, _GPT5_KEY: 0.8}
        )
        return baselines_dir

    def test_run_matching_its_own_scenario_baseline_passes(self, tmp_path: Path, script) -> None:
        """A run scoring exactly its omakase baseline (0.8) must NOT fail against a
        floor inflated by the refinement-only baseline (0.9). False-FAIL repro."""
        baselines_dir = self._baselines_with_two_scenarios(tmp_path)
        summary = _make_summary(
            {
                "omakase_mission_open_ended": {
                    _GPT5_KEY: _cell_with_cir(0.8, n=5),
                    _ANCHOR_KEY: _cell_with_cir(0.8, n=5),
                },
            }
        )
        run_dir = _write_run_summary(tmp_path, summary)
        rc = script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        assert rc == 0, (
            "anchor at its own per-scenario baseline must PASS; pooling against the "
            f"refinement-only baseline is a false regression (got exit {rc})"
        )

    def test_anchor_regression_on_common_scenario_fails(self, tmp_path: Path, script) -> None:
        """Run anchor 0.6 < omakase baseline 0.8 → exit 1 via the anchor branch."""
        baselines_dir = self._baselines_with_two_scenarios(tmp_path)
        summary = _make_summary(
            {
                "omakase_mission_open_ended": {
                    _GPT5_KEY: _cell_with_cir(0.8, n=5),  # check 1 passes
                    _ANCHOR_KEY: _cell_with_cir(0.6, n=5),  # below 0.8 floor
                },
            }
        )
        run_dir = _write_run_summary(tmp_path, summary)
        rc = script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        assert rc == 1, f"anchor 0.6 < common-scenario floor 0.8 must FAIL, got exit {rc}"

    def test_anchor_regression_message_printed(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        baselines_dir = self._baselines_with_two_scenarios(tmp_path)
        summary = _make_summary(
            {
                "omakase_mission_open_ended": {
                    _GPT5_KEY: _cell_with_cir(0.8, n=5),
                    _ANCHOR_KEY: _cell_with_cir(0.6, n=5),
                },
            }
        )
        run_dir = _write_run_summary(tmp_path, summary)
        script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        captured = capsys.readouterr()
        assert "anchor regression" in captured.out

    def test_asymmetric_scenarios_are_reported_as_excluded(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        """Scenarios present on only one side must be named in the output."""
        baselines_dir = self._baselines_with_two_scenarios(tmp_path)
        summary = _make_summary(
            {
                "omakase_mission_open_ended": {
                    _GPT5_KEY: _cell_with_cir(0.8, n=5),
                    _ANCHOR_KEY: _cell_with_cir(0.8, n=5),
                },
            }
        )
        run_dir = _write_run_summary(tmp_path, summary)
        script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        captured = capsys.readouterr()
        assert "excluded from anchor comparison" in captured.out
        assert "refinement_cheaper" in captured.out

    def test_no_common_scenarios_warns_and_passes(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        """Empty intersection → loud warning, treated as no-floor PASS (not a crash/FAIL)."""
        baselines_dir = self._baselines_with_two_scenarios(tmp_path)
        summary = _make_summary(
            {
                "scenario_not_in_baselines": {
                    _GPT5_KEY: _cell_with_cir(0.8, n=5),
                    _ANCHOR_KEY: _cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = _write_run_summary(tmp_path, summary)
        rc = script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "no scenario overlap" in captured.out

    def test_anchor_with_no_evaluable_cells_still_fails(self, tmp_path: Path, script) -> None:
        """A run where the anchor provider has no cells at all remains a FAIL."""
        baselines_dir = self._baselines_with_two_scenarios(tmp_path)
        summary = _make_summary(
            {
                "omakase_mission_open_ended": {_GPT5_KEY: _cell_with_cir(0.8, n=5)},
            }
        )
        run_dir = _write_run_summary(tmp_path, summary)
        rc = script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        assert rc == 1


# ---------------------------------------------------------------------------
# Unit tests: resolved-source visibility + matrix identity warning (WR-06)
# ---------------------------------------------------------------------------


class TestResolvedSourceVisibility:
    """WR-06: the report must name the artifact it graded and warn on a
    scenario set that doesn't look like a configs/eval_matrix.yaml run."""

    def _run_with_scenario(self, tmp_path: Path, script, scenario_id: str) -> tuple[int, str, Path]:
        summary = _make_summary(
            {
                scenario_id: {
                    _GPT5_KEY: _cell_with_cir(0.8, n=5),
                    _ANCHOR_KEY: _cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = _write_run_summary(tmp_path, summary)
        rc = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
            ]
        )
        return rc, scenario_id, run_dir

    def test_run_dir_is_printed_in_report(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        _, _, run_dir = self._run_with_scenario(tmp_path, script, "omakase_mission_open_ended")
        captured = capsys.readouterr()
        assert f"source: run dir {run_dir}" in captured.out

    def test_warns_when_no_expected_matrix_scenario_present(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        """A summary covering only a non-eval_matrix scenario (e.g. refinement)
        must produce a visible wrong-matrix warning."""
        self._run_with_scenario(tmp_path, script, "refinement_cheaper")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "eval_matrix_refinement" in captured.out

    def test_no_warning_when_expected_scenario_present(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        self._run_with_scenario(tmp_path, script, "omakase_mission_open_ended")
        captured = capsys.readouterr()
        assert "may belong to a different matrix" not in captured.out

    def test_baselines_mode_prints_baselines_source(
        self, script, capsys: pytest.CaptureFixture
    ) -> None:
        real_baselines_dir = REPO_ROOT / "configs" / "eval_baselines"
        script.main(["--baselines-mode", "--baselines-dir", str(real_baselines_dir)])
        captured = capsys.readouterr()
        assert "source: committed baselines" in captured.out

    def test_expected_matrix_scenarios_reads_real_config(self, script) -> None:
        expected = script._expected_matrix_scenarios()
        assert "omakase_mission_open_ended" in expected

    def test_expected_matrix_scenarios_missing_file_returns_empty(
        self, tmp_path: Path, script
    ) -> None:
        assert script._expected_matrix_scenarios(tmp_path / "nope.yaml") == set()


# ---------------------------------------------------------------------------
# Smoke test: real configs/eval_baselines (COMMITTED TEST ARTIFACT)
# Satisfies feedback_test_layering: smoke + unit + functional coverage.
# ---------------------------------------------------------------------------


def test_smoke_runs_against_real_baselines(script: ModuleType) -> None:
    """SMOKE: falsifier runs against real configs/eval_baselines without infra error.

    This is the committed smoke test artifact (feedback_test_layering). It
    exercises the --baselines-mode code path end-to-end against the real
    checked-in baseline JSONs and asserts:
      - the script produces a real verdict (exit code 0 or 1)
      - NOT an infra failure (exit code 2)

    Does NOT call any live API. Does NOT read eval_reports run dirs.
    Only requires the checked-in configs/eval_baselines directory.
    """
    real_baselines_dir = REPO_ROOT / "configs" / "eval_baselines"
    assert real_baselines_dir.is_dir(), (
        f"configs/eval_baselines not found at {real_baselines_dir}; is the repo root correct?"
    )

    rc = script.main(
        [
            "--baselines-mode",
            "--baselines-dir",
            str(real_baselines_dir),
        ]
    )
    assert rc in {0, 1}, (
        f"smoke test against real baselines returned exit {rc} (expected 0 or 1). "
        "Exit 2 means an infrastructure failure — check configs/eval_baselines/ content."
    )
