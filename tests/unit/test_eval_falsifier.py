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


def load_script() -> ModuleType:
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
    return load_script()


# ---------------------------------------------------------------------------
# Synthetic summary helpers
# ---------------------------------------------------------------------------

GPT5_KEY = "openai/gpt-5-mini"
ANCHOR_KEY = "openai/gpt-4o-mini"


def make_summary(
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


def cell_with_cir(median: float, n: int = 5) -> dict:
    """A cell dict with committed_itinerary_rate populated."""
    return {
        "scorers": {
            "committed_itinerary_rate": {"median": median, "mean": median, "n": n},
        },
        "n_scored": n,
        "n_errored": 0,
        "cell_valid": True,
    }


def cell_no_cir(n: int = 5) -> dict:
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
# Unit tests: pooled_commit_rate
# ---------------------------------------------------------------------------


class TestPooledCommitRate:
    """Tests for the pooled_commit_rate helper function."""

    def test_two_scenarios_pool_to_average_when_equal_n(self, script: ModuleType) -> None:
        """Two scenarios with medians [1.0, 0.0] at equal n pool to 0.5 (D-12-08)."""
        summary = make_summary(
            {
                "scenario_a": {GPT5_KEY: cell_with_cir(1.0, n=5)},
                "scenario_b": {GPT5_KEY: cell_with_cir(0.0, n=5)},
            }
        )
        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert pooled == pytest.approx(0.5, abs=1e-9)
        assert per_scenario["scenario_a"] == pytest.approx(1.0)
        assert per_scenario["scenario_b"] == pytest.approx(0.0)

    def test_per_scenario_breakdown_contains_both_entries(self, script: ModuleType) -> None:
        """Per-scenario breakdown includes an entry for every scenario."""
        summary = make_summary(
            {
                "omakase": {GPT5_KEY: cell_with_cir(1.0, n=5)},
                "refinement": {GPT5_KEY: cell_with_cir(0.0, n=5)},
            }
        )
        _, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert set(per_scenario.keys()) == {"omakase", "refinement"}

    def test_baseline_eligible_false_scenario_excluded_from_pool(self, script: ModuleType) -> None:
        """Scenarios with baseline_eligible: False are excluded from the pooled rate."""
        summary = make_summary(
            {
                "eligible_scenario": {GPT5_KEY: cell_with_cir(0.4, n=5)},
                "quarantined_scenario": {GPT5_KEY: cell_with_cir(1.0, n=5)},
            }
        )
        # Mark the second scenario as quarantined
        summary["scenarios"]["quarantined_scenario"]["baseline_eligible"] = False

        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        # Only eligible scenario counts → pooled = 0.4
        assert pooled == pytest.approx(0.4, abs=1e-9)
        # Quarantined scenario not in per_scenario (skipped entirely)
        assert "quarantined_scenario" not in per_scenario
        assert per_scenario["eligible_scenario"] == pytest.approx(0.4)

    def test_absent_provider_cell_yields_none_entry(self, script: ModuleType) -> None:
        """When a provider has no cell in a scenario, per_scenario entry is None."""
        summary = make_summary(
            {
                "scenario_a": {ANCHOR_KEY: cell_with_cir(1.0, n=5)},  # no gpt-5-mini
                "scenario_b": {GPT5_KEY: cell_with_cir(0.8, n=5)},
            }
        )
        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert per_scenario["scenario_a"] is None
        assert per_scenario["scenario_b"] == pytest.approx(0.8)
        # Only scenario_b contributes to the pooled rate
        assert pooled == pytest.approx(0.8, abs=1e-9)

    def test_no_eligible_scenarios_returns_none_pooled(self, script: ModuleType) -> None:
        """When all scenarios are quarantined, returns (None, empty_dict)."""
        summary = make_summary({"scenario_a": {GPT5_KEY: cell_with_cir(1.0, n=5)}})
        summary["scenarios"]["scenario_a"]["baseline_eligible"] = False
        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert pooled is None
        assert per_scenario == {}

    def test_empty_scenarios_returns_none_pooled(self, script: ModuleType) -> None:
        """An empty scenarios dict returns (None, {})."""
        summary: dict = {"scenarios": {}, "errors": []}
        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert pooled is None
        assert per_scenario == {}

    def test_weighted_pool_with_unequal_n(self, script: ModuleType) -> None:
        """Pool is weighted by n, not a simple average of medians."""
        # scenario_a: median=1.0, n=1; scenario_b: median=0.0, n=9
        # weighted: (1.0*1 + 0.0*9) / 10 = 0.1
        summary = make_summary(
            {
                "scenario_a": {GPT5_KEY: cell_with_cir(1.0, n=1)},
                "scenario_b": {GPT5_KEY: cell_with_cir(0.0, n=9)},
            }
        )
        pooled, _ = script.pooled_commit_rate(summary, GPT5_KEY)
        assert pooled == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests: bar logic (0.6 threshold)
# ---------------------------------------------------------------------------


class TestBarLogic:
    """Tests for the 0.6 threshold bar logic via main()."""

    def write_summary(self, tmp_path: Path, summary: dict) -> Path:
        """Write a summary.json to a tmp run dir, return the run dir path."""
        run_dir = tmp_path / "2026-01-01T00-00-00Z"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps(summary))
        return run_dir

    def test_pooled_05_fails_06_bar(
        self, tmp_path: Path, script: ModuleType, capsys: pytest.CaptureFixture
    ) -> None:
        """Pooled 0.5 is below the 0.6 bar → exit 1.

        Uses omakase_mission_open_ended (the in-matrix scenario) for both slots
        so the zero-overlap guard (12-05) does not fire before the bar check.
        """
        summary = make_summary(
            {
                "omakase_mission_open_ended": {
                    GPT5_KEY: cell_with_cir(0.5, n=5),  # below 0.6 bar
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = self.write_summary(tmp_path, summary)
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
        """Pooled 0.8 is above the 0.6 bar → exit 0 (assuming anchor holds).

        Uses omakase_mission_open_ended (the in-matrix scenario) so the
        zero-overlap guard (12-05) does not fire before the bar check.
        """
        summary = make_summary(
            {
                "omakase_mission_open_ended": {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = self.write_summary(tmp_path, summary)
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


def write_minimal_baseline(
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
        write_minimal_baseline(
            baselines_dir,
            "scenario_a",
            {
                GPT5_KEY: 0.8,  # above 0.6 bar
                ANCHOR_KEY: 1.0,
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
        write_minimal_baseline(
            baselines_dir,
            "scenario_a",
            {GPT5_KEY: 0.8, ANCHOR_KEY: 1.0},
        )
        rc = script.main(["--baselines-mode", "--baselines-dir", str(baselines_dir)])
        assert rc == 0

    def test_baselines_mode_gpt5_below_bar_returns_exit_1(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """--baselines-mode with gpt-5-mini < 0.6 returns exit 1."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        write_minimal_baseline(
            baselines_dir,
            "scenario_a",
            {GPT5_KEY: 0.3, ANCHOR_KEY: 1.0},
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
        summary = make_summary(
            {
                "s": {
                    GPT5_KEY: {"scorers": {"committed_itinerary_rate": {"median": 1.0, "n": None}}}
                }
            }
        )
        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert pooled is None
        assert per_scenario["s"] is None

    def test_string_n_treated_as_unevaluable(self, script: ModuleType) -> None:
        summary = make_summary(
            {"s": {GPT5_KEY: {"scorers": {"committed_itinerary_rate": {"median": 1.0, "n": "5"}}}}}
        )
        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert pooled is None
        assert per_scenario["s"] is None

    def test_non_dict_scenario_block_treated_as_unevaluable(self, script: ModuleType) -> None:
        summary = {"scenarios": {"s": ["not", "a", "dict"]}}
        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert pooled is None
        assert per_scenario["s"] is None

    def test_non_dict_scorers_treated_as_unevaluable(self, script: ModuleType) -> None:
        summary = make_summary({"s": {GPT5_KEY: {"scorers": "corrupt"}}})
        pooled, per_scenario = script.pooled_commit_rate(summary, GPT5_KEY)
        assert pooled is None
        assert per_scenario["s"] is None

    def test_non_dict_scenarios_returns_none(self, script: ModuleType) -> None:
        pooled, per_scenario = script.pooled_commit_rate({"scenarios": []}, GPT5_KEY)
        assert pooled is None
        assert per_scenario == {}

    def test_main_with_malformed_n_returns_verdict_not_traceback(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """main() over a summary with null n must return a deliberate exit code,
        not raise TypeError (pre-fix behavior: uncaught crash → interpreter exit 1).

        Uses omakase_mission_open_ended (the in-matrix scenario) so the
        zero-overlap guard (12-05) does not fire; the cell's null n is the
        tested defect path.
        """
        summary = make_summary(
            {
                "omakase_mission_open_ended": {
                    GPT5_KEY: {"scorers": {"committed_itinerary_rate": {"median": 1.0, "n": None}}}
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


def write_run_summary(tmp_path: Path, summary: dict) -> Path:
    """Write a summary.json to a tmp run dir, return the run dir path."""
    run_dir = tmp_path / "run" / "2026-01-01T00-00-00Z"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


class TestAnchorCommonScenarioPooling:
    """CR-01: anchor floor must be pooled over the run∩baseline scenario set."""

    def baselines_with_two_scenarios(self, tmp_path: Path) -> Path:
        """Baselines: omakase anchor=0.8, refinement anchor=1.0 (floor 0.9 if mis-pooled)."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        write_minimal_baseline(
            baselines_dir, "omakase_mission_open_ended", {ANCHOR_KEY: 0.8, GPT5_KEY: 0.8}
        )
        write_minimal_baseline(
            baselines_dir, "refinement_cheaper", {ANCHOR_KEY: 1.0, GPT5_KEY: 0.8}
        )
        return baselines_dir

    def test_run_matching_its_own_scenario_baseline_passes(self, tmp_path: Path, script) -> None:
        """A run scoring exactly its omakase baseline (0.8) must NOT fail against a
        floor inflated by the refinement-only baseline (0.9). False-FAIL repro."""
        baselines_dir = self.baselines_with_two_scenarios(tmp_path)
        summary = make_summary(
            {
                "omakase_mission_open_ended": {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(0.8, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        rc = script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        assert rc == 0, (
            "anchor at its own per-scenario baseline must PASS; pooling against the "
            f"refinement-only baseline is a false regression (got exit {rc})"
        )

    def test_anchor_regression_on_common_scenario_fails(self, tmp_path: Path, script) -> None:
        """Run anchor 0.6 < omakase baseline 0.8 → exit 1 via the anchor branch."""
        baselines_dir = self.baselines_with_two_scenarios(tmp_path)
        summary = make_summary(
            {
                "omakase_mission_open_ended": {
                    GPT5_KEY: cell_with_cir(0.8, n=5),  # check 1 passes
                    ANCHOR_KEY: cell_with_cir(0.6, n=5),  # below 0.8 floor
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        rc = script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        assert rc == 1, f"anchor 0.6 < common-scenario floor 0.8 must FAIL, got exit {rc}"

    def test_anchor_regression_message_printed(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        baselines_dir = self.baselines_with_two_scenarios(tmp_path)
        summary = make_summary(
            {
                "omakase_mission_open_ended": {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(0.6, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        captured = capsys.readouterr()
        assert "anchor regression" in captured.out

    def test_asymmetric_scenarios_are_reported_as_excluded(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        """Scenarios present on only one side must be named in the output."""
        baselines_dir = self.baselines_with_two_scenarios(tmp_path)
        summary = make_summary(
            {
                "omakase_mission_open_ended": {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(0.8, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        captured = capsys.readouterr()
        assert "excluded from anchor comparison" in captured.out
        assert "refinement_cheaper" in captured.out

    def test_no_common_scenarios_warns_and_passes(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        """Empty run∩baseline intersection → loud warning, treated as no-floor PASS.

        Uses omakase_mission_open_ended in the run (passes the 12-05 zero-overlap
        matrix guard) but synthetic baselines that contain only refinement_cheaper
        (no overlap with the run on the baselines side) — so the anchor comparison
        hits the "no scenario overlap" branch.
        """
        # Synthetic baselines with NO omakase scenario — so run∩baselines = empty
        baselines_dir = tmp_path / "baselines_no_omakase"
        baselines_dir.mkdir()
        write_minimal_baseline(
            baselines_dir, "refinement_cheaper", {ANCHOR_KEY: 1.0, GPT5_KEY: 0.8}
        )
        summary = make_summary(
            {
                "omakase_mission_open_ended": {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        rc = script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "no scenario overlap" in captured.out

    def test_anchor_with_no_evaluable_cells_still_fails(self, tmp_path: Path, script) -> None:
        """A run where the anchor provider has no cells at all remains a FAIL."""
        baselines_dir = self.baselines_with_two_scenarios(tmp_path)
        summary = make_summary(
            {
                "omakase_mission_open_ended": {GPT5_KEY: cell_with_cir(0.8, n=5)},
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        rc = script.main(["--run-dir", str(run_dir), "--baselines-dir", str(baselines_dir)])
        assert rc == 1


# ---------------------------------------------------------------------------
# Unit tests: resolved-source visibility + matrix identity warning (WR-06)
# ---------------------------------------------------------------------------


class TestResolvedSourceVisibility:
    """WR-06: the report must name the artifact it graded and warn on a
    scenario set that doesn't look like a configs/eval_matrix.yaml run."""

    def run_with_scenario(self, tmp_path: Path, script, scenario_id: str) -> tuple[int, str, Path]:
        summary = make_summary(
            {
                scenario_id: {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
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
        _, _, run_dir = self.run_with_scenario(tmp_path, script, "omakase_mission_open_ended")
        captured = capsys.readouterr()
        assert f"source: run dir {run_dir}" in captured.out

    def test_warns_when_no_expected_matrix_scenario_present(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        """A summary covering only a non-eval_matrix scenario (e.g. refinement)
        must produce a visible wrong-matrix warning."""
        self.run_with_scenario(tmp_path, script, "refinement_cheaper")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "eval_matrix_refinement" in captured.out

    def test_no_warning_when_expected_scenario_present(
        self, tmp_path: Path, script, capsys: pytest.CaptureFixture
    ) -> None:
        self.run_with_scenario(tmp_path, script, "omakase_mission_open_ended")
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
        expected = script.expected_matrix_scenarios()
        assert "omakase_mission_open_ended" in expected

    def test_expected_matrix_scenarios_missing_file_returns_empty(
        self, tmp_path: Path, script
    ) -> None:
        assert script.expected_matrix_scenarios(tmp_path / "nope.yaml") == set()


# ---------------------------------------------------------------------------
# Unit tests: zero-overlap run-dir summary refuses with exit 2 (12-05 / UAT Test 1)
# ---------------------------------------------------------------------------


class TestZeroOverlapRefusesWithExit2:
    """12-05: run-dir mode must exit 2 with no VERDICT when the summary shares
    zero scenarios with configs/eval_matrix.yaml (wrong-matrix refusal).

    Coverage plan:
      (A) exit code == 2 on zero-overlap summary with otherwise-passing cells
      (B) no VERDICT line in stdout; existing diagnosis text is preserved
      (C) in-matrix scenario still grades to a real verdict (negative control)
      (D) monkeypatched empty expected set does NOT trigger the guard
    """

    def test_zero_overlap_run_dir_returns_exit_2(self, tmp_path: Path, script: ModuleType) -> None:
        """(A) A zero-overlap summary with otherwise-passing gpt-5-mini + anchor
        cells must make main() return exit 2 — proving the guard prevents a
        spurious milestone PASS."""
        chosen_id = "refinement_cheaper"
        # Safety assertion: chosen scenario must NOT be in the real matrix config
        assert chosen_id not in script.expected_matrix_scenarios(), (
            f"'{chosen_id}' is now in eval_matrix.yaml — pick a different test scenario id"
        )
        # Build a summary with otherwise-passing cells: gpt-5-mini 0.8, anchor 1.0
        summary = make_summary(
            {
                chosen_id: {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        rc = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
            ]
        )
        assert rc == 2, (
            f"zero-overlap run-dir summary (scenario='{chosen_id}') must return exit 2; "
            f"got exit {rc}. Without the guard a spurious milestone PASS could occur."
        )

    def test_zero_overlap_emits_no_verdict_line(
        self, tmp_path: Path, script: ModuleType, capsys: pytest.CaptureFixture
    ) -> None:
        """(B) On the zero-overlap path, stdout must contain NO 'VERDICT' line
        and the existing wrong-matrix diagnosis text must still be printed."""
        chosen_id = "scenario_not_in_matrix"
        assert chosen_id not in script.expected_matrix_scenarios(), (
            f"'{chosen_id}' is now in eval_matrix.yaml — pick a different test scenario id"
        )
        summary = make_summary(
            {
                chosen_id: {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
            ]
        )
        captured = capsys.readouterr()
        # No PASS/FAIL verdict emitted
        assert "VERDICT" not in captured.out, (
            "wrong-matrix run must not emit any VERDICT line in stdout"
        )
        assert "VERDICT = PASS" not in captured.out
        assert "VERDICT = FAIL" not in captured.out
        # The existing diagnosis text is preserved verbatim
        assert "eval_matrix_refinement" in captured.out, (
            "wrong-matrix diagnosis must still print the eval_matrix_refinement hint"
        )
        assert "may belong to a different matrix" in captured.out

    def test_in_matrix_run_still_grades(
        self, tmp_path: Path, script: ModuleType, capsys: pytest.CaptureFixture
    ) -> None:
        """(C) Negative control: an in-matrix scenario ('omakase_mission_open_ended')
        must still produce a real verdict (exit 0) with 'VERDICT' in stdout."""
        in_matrix_id = "omakase_mission_open_ended"
        assert in_matrix_id in script.expected_matrix_scenarios(), (
            f"'{in_matrix_id}' must be in eval_matrix.yaml for this test to be valid"
        )
        # gpt-5-mini 0.8 (>= 0.6 bar), anchor 1.0 (holds against committed baselines)
        summary = make_summary(
            {
                in_matrix_id: {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        rc = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
            ]
        )
        assert rc in {0, 1}, f"in-matrix run must return a real verdict (0 or 1), got exit {rc}"
        assert rc == 0, f"gpt-5-mini 0.8 >= 0.6 with anchor 1.0 must PASS, got exit {rc}"
        captured = capsys.readouterr()
        assert "VERDICT" in captured.out, "in-matrix run must emit a VERDICT line"

    def test_empty_expected_set_does_not_refuse(
        self, tmp_path: Path, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(D) When expected_matrix_scenarios() returns set() (unreadable/empty config),
        the guard must NOT fire — grading proceeds normally (exit in {0, 1}, not 2)."""
        monkeypatch.setattr(script, "expected_matrix_scenarios", lambda *a, **k: set())
        # Use any scenario id; it won't match the empty expected set
        summary = make_summary(
            {
                "any_scenario_id": {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = write_run_summary(tmp_path, summary)
        rc = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
            ]
        )
        assert rc in {0, 1}, (
            f"empty expected set must not trigger the zero-overlap guard; got exit {rc} "
            "(expected 0 or 1 — a real verdict, not an infra refusal)"
        )
        assert rc != 2, "guard must NOT fire when expected_scenarios is empty (best-effort path)"


# ---------------------------------------------------------------------------
# Smoke test: real configs/eval_baselines (COMMITTED TEST ARTIFACT)
# Satisfies feedback_test_layering: smoke + unit + functional coverage.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests: --matrix-config flag + commit_split_from_run_dir (Phase 13 / D-13-04)
# ---------------------------------------------------------------------------


class TestMatrixConfigFlag:
    """Tests for the --matrix-config argument and zero-overlap guard behavior."""

    def write_run_summary_with_scenario(self, tmp_path: Path, scenario_id: str) -> Path:
        """Write a summary.json containing only the given scenario id."""
        summary = make_summary(
            {
                scenario_id: {
                    GPT5_KEY: cell_with_cir(0.8, n=5),
                    ANCHOR_KEY: cell_with_cir(1.0, n=5),
                },
            }
        )
        run_dir = tmp_path / "2026-01-01T00-00-00Z"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary))
        return run_dir

    def test_matrix_config_flag_accepted(self, script: ModuleType) -> None:
        """--matrix-config is a recognized argument (does not raise SystemExit)."""
        # Parse --matrix-config without running (just test arg parsing)
        args = script.parse_args(
            [
                "--matrix-config",
                "configs/eval_matrix_arm.yaml",
            ]
        )
        assert args.matrix_config == "configs/eval_matrix_arm.yaml"

    def test_matrix_config_default_is_none(self, script: ModuleType) -> None:
        """--matrix-config defaults to None when not provided."""
        args = script.parse_args([])
        assert args.matrix_config is None

    def test_arm_run_dir_with_refinement_cheaper_passes_overlap_guard(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """With --matrix-config configs/eval_matrix_arm.yaml, a run dir containing
        only refinement_cheaper (not in default eval_matrix.yaml) should NOT exit 2."""
        # refinement_cheaper is NOT in default eval_matrix.yaml
        assert "refinement_cheaper" not in script.expected_matrix_scenarios()

        run_dir = self.write_run_summary_with_scenario(tmp_path, "refinement_cheaper")
        arm_config = str(REPO_ROOT / "configs" / "eval_matrix_arm.yaml")
        rc = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
                "--matrix-config",
                arm_config,
            ]
        )
        # Must not be exit 2 (zero-overlap guard must not fire)
        assert rc != 2, (
            "refinement_cheaper is in eval_matrix_arm.yaml — the zero-overlap guard "
            f"must accept this run dir; got exit {rc}"
        )

    def test_default_matrix_config_rejects_refinement_cheaper(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """Without --matrix-config, a run dir with only refinement_cheaper still exits 2."""
        run_dir = self.write_run_summary_with_scenario(tmp_path, "refinement_cheaper")
        rc = script.main(
            [
                "--run-dir",
                str(run_dir),
                "--baselines-dir",
                str(REPO_ROOT / "configs" / "eval_baselines"),
                # No --matrix-config → uses default eval_matrix.yaml
            ]
        )
        assert rc == 2, (
            "refinement_cheaper is NOT in default eval_matrix.yaml — "
            f"zero-overlap guard must fire and return exit 2; got {rc}"
        )

    def test_expected_matrix_scenarios_reads_arm_config(self, script: ModuleType) -> None:
        """expected_matrix_scenarios with arm config path returns both arm scenarios."""
        arm_path = REPO_ROOT / "configs" / "eval_matrix_arm.yaml"
        scenarios = script.expected_matrix_scenarios(arm_path)
        assert "omakase_mission_open_ended" in scenarios
        assert "refinement_cheaper" in scenarios


class TestCommitSplitFromRunDir:
    """Tests for commit_split_from_run_dir helper (D-13-04)."""

    def write_run_file(
        self,
        run_dir: Path,
        provider: str,
        model: str,
        scenario: str,
        run_n: int,
        commit_forced: bool,
        first_commit_call_step: int | None,
    ) -> Path:
        """Write an individual run JSON file following the REAL EvalRunReport shape.

        The payload mirrors scripts/eval_agent.py's report_to_dict(EvalRunReport) output
        (asdict of EvalRunReport dataclass): top-level keys are eval_queries_path,
        llm_provider, chat_model, query_count, aggregate, and queries (list).
        The deterministic block is nested under each queries[i], NOT at top level.

        This shape matches the actual run-dir JSON artifacts written by eval_agent.py.
        Fixture field set derived from EvalRunReport/QueryEvalResult/DeterministicEvalResult
        dataclasses in scripts/eval_agent.py (report_to_dict = asdict(report)).
        """
        from scripts.eval_agent import (  # noqa: PLC0415
            DETERMINISTIC_CHECKS,
            ActualEvalResult,
            CheckResult,
            DeterministicEvalResult,
            EvalRunReport,
            ExpectedEvalResult,
            QueryEvalResult,
            report_to_dict,
        )

        # Build a complete checks dict matching the real error-record pattern in make_error_record.
        # All scores are None — aggregate skips "error"-status records, so this is safe.
        empty_checks: dict[str, CheckResult] = {
            name: CheckResult(score=None, threshold=0.0, passed=False)
            for name in DETERMINISTIC_CHECKS
        }

        # Build a real DeterministicEvalResult with the relevant fields populated.
        # Field set matches DeterministicEvalResult dataclass definition exactly.
        det = DeterministicEvalResult(
            expected_results_met=(
                True if first_commit_call_step is not None or commit_forced else None
            ),
            checks=empty_checks,
            violations=[],
            tool_errors=[],
            first_tool_error=None,
            tool_calls=first_commit_call_step + 1 if first_commit_call_step is not None else 0,
            tool_names=[],
            revision_hints=0,
            revision_reasons=[],
            first_commit_call_step=first_commit_call_step,
            first_commit_mention_step=None,
            viable_candidates_per_step=[],
            rule8_met_per_step=[],
            rule8_met_but_kept_searching_steps=[],
            step_telemetry=[],
            viability_threshold=0.55,
            commit_forced=commit_forced,
            forced_commit_step=None,
            arm_flags={},
        )
        committed = first_commit_call_step is not None or commit_forced
        query_result = QueryEvalResult(
            id=f"{scenario}--run-{run_n}",
            question="test query",
            answer="test answer",
            contexts=[],
            reference="",
            tags=[],
            expected=ExpectedEvalResult(
                min_stops=None, max_stops=None, expects_clarification_or_relaxation=False
            ),
            actual=ActualEvalResult(
                result_count=1 if committed else 0,
                committed_stop_count=1 if committed else 0,
                place_ids=[],
                place_names=[],
                sources=[],
                answer_place_names=[],
            ),
            deterministic=det,
            final_reply="",
            latency_seconds=1.0,
            # Use status="error" so aggregate_results skips scorer means — no KeyError
            # from missing check scores.  This is the minimal-friction pattern for fixture
            # data that only needs the queries[i].deterministic block to be readable.
            status="error",
            error={"stage": "fixture", "type": "FixtureError", "message": "synthetic fixture"},
        )
        report = EvalRunReport(
            eval_queries_path="configs/eval_queries.yaml",
            llm_provider=provider,
            chat_model=f"{provider}/{model}",
            query_count=1,
            # Provide a minimal aggregate dict — the split reader never reads aggregate.
            aggregate={"committed_itinerary_rate": 1.0 if committed else 0.0},
            queries=[query_result],
        )
        provider_slug = f"{provider}--{model}"
        filename = f"{provider_slug}--{scenario}--run-{run_n}.json"
        (run_dir / filename).write_text(json.dumps(report_to_dict(report), indent=2))
        return run_dir / filename

    def write_run_file_old_shape(
        self,
        run_dir: Path,
        provider: str,
        model: str,
        scenario: str,
        run_n: int,
        commit_forced: bool,
        first_commit_call_step: int | None,
    ) -> Path:
        """Write a run JSON file with the OLD (buggy) top-level deterministic shape.

        This is the pre-fix fixture shape: deterministic at top level (NOT under queries[i]).
        Used by the CR-02 regression test to prove the fixed reader rejects this shape.
        """
        provider_slug = f"{provider}--{model}"
        filename = f"{provider_slug}--{scenario}--run-{run_n}-old.json"
        payload = {
            "provider": provider,
            "model": model,
            "scenario_id": scenario,
            "run_index": run_n,
            # OLD (buggy) shape: deterministic at TOP LEVEL — the fixed reader ignores this
            "deterministic": {
                "commit_forced": commit_forced,
                "first_commit_call_step": first_commit_call_step,
                "committed_itinerary_rate": 1.0
                if first_commit_call_step is not None or commit_forced
                else 0.0,
            },
            # No "queries" key at all
        }
        (run_dir / filename).write_text(json.dumps(payload))
        return run_dir / filename

    def make_run_dir(self, tmp_path: Path) -> Path:
        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()
        return run_dir

    def test_split_counts_forced_and_model_initiated(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """2 forced + 2 model-initiated + 1 never-committed → returns (2, 2)."""
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"
        provider, model = "openai", "gpt-5-mini"

        # 2 forced commits
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 0, True, None)
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 1, True, None)
        # 2 model-initiated commits
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 2, False, 3)
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 3, False, 5)
        # 1 never committed
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 4, False, None)

        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert mi == 2, f"expected 2 model-initiated, got {mi}"
        assert forced == 2, f"expected 2 forced, got {forced}"

    def test_split_all_forced(self, tmp_path: Path, script: ModuleType) -> None:
        """All forced commits → returns (0, n)."""
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"
        provider, model = "openai", "gpt-5-mini"

        for i in range(3):
            self.write_run_file(
                run_dir, provider, model, "omakase_mission_open_ended", i, True, None
            )

        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert mi == 0
        assert forced == 3

    def test_split_none_committed(self, tmp_path: Path, script: ModuleType) -> None:
        """No commits at all → returns (0, 0)."""
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"
        provider, model = "openai", "gpt-5-mini"

        for i in range(3):
            self.write_run_file(
                run_dir, provider, model, "omakase_mission_open_ended", i, False, None
            )

        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert mi == 0
        assert forced == 0

    def test_split_ignores_other_providers(self, tmp_path: Path, script: ModuleType) -> None:
        """Files from other providers are not counted."""
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"

        # Write files for gpt-5-mini (2 forced) and gpt-4o-mini (3 model-initiated)
        self.write_run_file(
            run_dir, "openai", "gpt-5-mini", "omakase_mission_open_ended", 0, True, None
        )
        self.write_run_file(
            run_dir, "openai", "gpt-5-mini", "omakase_mission_open_ended", 1, True, None
        )
        self.write_run_file(
            run_dir, "openai", "gpt-4o-mini", "omakase_mission_open_ended", 0, False, 2
        )
        self.write_run_file(
            run_dir, "openai", "gpt-4o-mini", "omakase_mission_open_ended", 1, False, 3
        )
        self.write_run_file(
            run_dir, "openai", "gpt-4o-mini", "omakase_mission_open_ended", 2, False, 4
        )

        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert mi == 0
        assert forced == 2  # only gpt-5-mini files

    def test_split_skips_malformed_json(self, tmp_path: Path, script: ModuleType) -> None:
        """Malformed JSON files are silently skipped (T-13-05-01)."""
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"

        # Write one valid file and one malformed file
        self.write_run_file(
            run_dir, "openai", "gpt-5-mini", "omakase_mission_open_ended", 0, True, None
        )
        (run_dir / "openai--gpt-5-mini--omakase_mission_open_ended--run-1.json").write_text(
            "not valid json {"
        )

        # Should not raise, should count the one valid forced file
        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert forced == 1
        assert mi == 0

    def test_split_skips_summary_json(self, tmp_path: Path, script: ModuleType) -> None:
        """summary.json is not counted as an individual run file."""
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"

        # Write a summary.json (should be ignored) and a real run file
        (run_dir / "summary.json").write_text(json.dumps({"scenarios": {}}))
        self.write_run_file(
            run_dir, "openai", "gpt-5-mini", "omakase_mission_open_ended", 0, False, 2
        )

        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert mi == 1
        assert forced == 0

    def test_split_empty_dir_returns_zeros(self, tmp_path: Path, script: ModuleType) -> None:
        """Empty run dir returns (0, 0) without raising."""
        run_dir = self.make_run_dir(tmp_path)
        mi, forced = script.commit_split_from_run_dir(run_dir, "openai/gpt-5-mini")
        assert mi == 0
        assert forced == 0

    # --- CR-02 regression tests: real EvalRunReport shape (queries[i].deterministic) ---

    def test_cr02_real_shape_returns_nonzero_counts(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """CR-02 regression: real EvalRunReport shape (queries[i].deterministic) returns correct counts.

        Tests behavior requirement 1 from the plan: 2 forced + 2 model-initiated + 1
        never-committed query returns (2, 2).

        This test FAILS against the pre-fix reader (data.get("deterministic") at top level
        returns None on real EvalRunReport shape, yielding (0, 0) instead of (2, 2)).
        """
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"
        provider, model = "openai", "gpt-5-mini"

        # 2 forced commits (commit_forced=True)
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 0, True, None)
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 1, True, None)
        # 2 model-initiated commits
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 2, False, 3)
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 3, False, 5)
        # 1 never committed
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 4, False, None)

        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert mi == 2, (
            f"CR-02 regression: expected 2 model-initiated on real EvalRunReport shape, got {mi}. "
            "Old top-level reader returns 0 — this test must fail on pre-fix code."
        )
        assert forced == 2, (
            f"CR-02 regression: expected 2 forced on real EvalRunReport shape, got {forced}. "
            "Old top-level reader returns 0 — this test must fail on pre-fix code."
        )

    def test_cr02_old_top_level_shape_returns_zeros(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """CR-02 regression: old top-level-only deterministic shape returns (0, 0).

        Tests behavior requirement 2 from the plan: a file with ONLY a top-level
        deterministic block (the OLD buggy shape) returns (0, 0) under the FIXED reader.

        This pins the fixture-shape regression: the fixed reader no longer accepts the old
        top-level shape, so tests using the old fixture cannot accidentally pass through the
        wrong code path.
        """
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"
        provider, model = "openai", "gpt-5-mini"

        # Write files in the OLD (buggy) shape — top-level deterministic, no queries list
        self.write_run_file_old_shape(
            run_dir, provider, model, "omakase_mission_open_ended", 0, True, None
        )
        self.write_run_file_old_shape(
            run_dir, provider, model, "omakase_mission_open_ended", 1, False, 3
        )

        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert mi == 0, (
            f"Fixed reader must return 0 model-initiated on old top-level shape, got {mi}. "
            "The old shape has no queries[] list; the fixed reader skips it."
        )
        assert forced == 0, (
            f"Fixed reader must return 0 forced on old top-level shape, got {forced}. "
            "The old shape has no queries[] list; the fixed reader skips it."
        )

    def test_cr02_commit_forced_and_model_initiated_classification(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """CR-02 behavior req 3: commit_forced vs first_commit_call_step classification per query.

        commit_forced=True → forced
        commit_forced=False/falsey + first_commit_call_step not None → model-initiated
        both falsey/None → neither (uncounted)
        """
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"
        provider, model = "openai", "gpt-5-mini"

        # forced: commit_forced=True, first_commit_call_step=None
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 0, True, None)
        # model-initiated: commit_forced=False, first_commit_call_step=4
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 1, False, 4)
        # neither: both falsey
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 2, False, None)

        mi, forced = script.commit_split_from_run_dir(run_dir, provider_key)
        assert forced == 1, f"Expected 1 forced (commit_forced=True), got {forced}"
        assert mi == 1, f"Expected 1 model-initiated (first_commit_call_step not None), got {mi}"

    def test_cr02_scenario_filtering_reads_from_query_scenario_id(
        self, tmp_path: Path, script: ModuleType
    ) -> None:
        """CR-02 behavior req 4: scenario filtering reads scenario_id from query record.

        When scenario_ids is provided, only queries[i] whose scenario matches are counted.
        The scenario comes from the query record's scenario_id field (set in the EvalRunReport
        queries list), not from filename parsing.
        """
        run_dir = self.make_run_dir(tmp_path)
        provider_key = "openai/gpt-5-mini"
        provider, model = "openai", "gpt-5-mini"

        # omakase: 2 model-initiated
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 0, False, 3)
        self.write_run_file(run_dir, provider, model, "omakase_mission_open_ended", 1, False, 4)
        # refinement_cheaper: 1 forced — should be excluded when filtering to omakase only
        self.write_run_file(run_dir, provider, model, "refinement_cheaper", 0, True, None)

        mi, forced = script.commit_split_from_run_dir(
            run_dir, provider_key, scenario_ids={"omakase_mission_open_ended"}
        )
        assert mi == 2, f"Expected 2 model-initiated for omakase only, got {mi}"
        assert forced == 0, f"Expected 0 forced for omakase only, got {forced}"


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
