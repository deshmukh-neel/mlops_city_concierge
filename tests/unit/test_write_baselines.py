"""Unit tests for scripts/write_baselines.py (D-11-07 / BASE-01).

Covers:
  - Eligible cell with n_scored == n_requested: writes baseline JSON, exit 0
  - Cell with n_scored < n_requested: REFUSED, exit 1 (D-10-03)
  - Scenario with baseline_eligible=False: all cells refused with D-10-09, exit 1
  - Prior _observations carried forward on rewrite
  - Missing/malformed summary.json: exit 2 (infra failure, distinct from refusal)
  - Module imports with all provider API keys unset (stdlib-only, no live services)
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "write_baselines.py"


def _load_script():
    """Import scripts.write_baselines and return the module."""
    import scripts.write_baselines as wb  # noqa: PLC0415

    return wb


def _make_summary(
    scenario_id: str,
    provider_key: str,
    n_scored: int,
    scorers: dict | None = None,
    baseline_eligible: bool = True,
) -> dict:
    """Build a minimal summary.json payload for tests."""
    if scorers is None:
        scorers = {
            "category_compliance": {
                "median": 1.0,
                "min": 1.0,
                "max": 1.0,
                "stdev": 0.0,
                "n": n_scored,
            }
        }
    return {
        "scenarios": {
            scenario_id: {
                "baseline_eligible": baseline_eligible,
                "providers": {
                    provider_key: {
                        "scorers": scorers,
                        "n_scored": n_scored,
                        "n_errored": 0,
                        "cell_valid": n_scored > 0,
                    }
                },
            }
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteBaselines:
    def test_eligible_cell_writes_baseline_and_exits_zero(self, tmp_path: Path) -> None:
        """Eligible cell with n_scored == n_requested writes a baseline JSON; exit 0."""
        wb = _load_script()

        summary_data = _make_summary(
            scenario_id="test_scenario",
            provider_key="openai/gpt-4o-mini",
            n_scored=5,
        )
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data), encoding="utf-8")

        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        rc = wb.main(
            [str(summary_path), "--n-requested", "5", "--baselines-dir", str(baselines_dir)]
        )

        assert rc == 0, f"Expected exit 0 for eligible cell; got {rc}"

        # Verify baseline file was written
        baseline_path = baselines_dir / "test_scenario.json"
        assert baseline_path.exists(), "Baseline JSON should be written"

        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert payload["scenario_id"] == "test_scenario"
        assert "generated_at" in payload
        assert payload["generated_by"] == "scripts/write_baselines.py"
        assert "openai/gpt-4o-mini" in payload["providers"]
        provider_cell = payload["providers"]["openai/gpt-4o-mini"]
        assert "scorers" in provider_cell
        assert "category_compliance" in provider_cell["scorers"]

    def test_partial_cell_refused_and_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Cell with n_scored < n_requested: REFUSED, D-10-03 message on stderr, exit 1."""
        wb = _load_script()

        # n_scored=3 < n_requested=5 → refusal
        summary_data = _make_summary(
            scenario_id="omakase_test",
            provider_key="openai/gpt-4o-mini",
            n_scored=3,
        )
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data), encoding="utf-8")

        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        rc = wb.main(
            [str(summary_path), "--n-requested", "5", "--baselines-dir", str(baselines_dir)]
        )

        assert rc == 1, f"Expected exit 1 for partial cell refusal; got {rc}"

        # Verify nothing was written
        baseline_path = baselines_dir / "omakase_test.json"
        assert not baseline_path.exists(), "No baseline should be written on refusal"

        # Verify D-10-03 message on stderr
        captured = capsys.readouterr()
        assert "D-10-03" in captured.err, "D-10-03 refusal code must appear in stderr"
        assert "REFUSED" in captured.err, "REFUSED keyword must appear in stderr"

    def test_all_cells_refused_with_prior_file_leaves_file_byte_identical(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """WR-02: prior baseline exists + ALL summary cells refused → the file
        must stay byte-identical (no provenance restamp over stale data); exit 1.

        Before the fix, `if any_written or updated_providers:` rewrote the file
        whenever a prior baseline existed: top-level generated_at/generated_by
        were bumped while every provider cell carried old data — a lying
        provenance stamp that also satisfied the check_baselines_fresh.py
        staleness gate without any actual data refresh.
        """
        wb = _load_script()

        scenario_id = "omakase_test"
        provider_key = "openai/gpt-4o-mini"

        # Prior baseline on disk (makes updated_providers non-empty).
        prior_baseline = {
            "scenario_id": scenario_id,
            "generated_at": "2026-01-01T00-00-00Z",
            "generated_by": "scripts/write_baselines.py",
            "providers": {
                provider_key: {
                    "scorers": {
                        "category_compliance": {
                            "median": 1.0,
                            "min": 1.0,
                            "max": 1.0,
                            "stdev": 0.0,
                            "n": 5,
                        }
                    }
                }
            },
        }
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        baseline_path = baselines_dir / f"{scenario_id}.json"
        baseline_path.write_text(json.dumps(prior_baseline, indent=2), encoding="utf-8")
        prior_bytes = baseline_path.read_bytes()

        # New summary: n_scored=3 < n_requested=5 → ALL cells refused.
        summary_data = _make_summary(
            scenario_id=scenario_id,
            provider_key=provider_key,
            n_scored=3,
        )
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data), encoding="utf-8")

        rc = wb.main(
            [str(summary_path), "--n-requested", "5", "--baselines-dir", str(baselines_dir)]
        )

        assert rc == 1, f"Expected exit 1 when all cells refused; got {rc}"
        assert baseline_path.read_bytes() == prior_bytes, (
            "prior baseline file must remain byte-identical when every cell was refused"
        )

        captured = capsys.readouterr()
        assert "REFUSED" in captured.err

    def test_quarantined_scenario_refused_and_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Scenario with baseline_eligible=False: all cells refused with D-10-09; exit 1."""
        wb = _load_script()

        summary_data = _make_summary(
            scenario_id="late_night_closure_cascade",
            provider_key="openai/gpt-4o-mini",
            n_scored=5,
            baseline_eligible=False,
        )
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data), encoding="utf-8")

        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        rc = wb.main(
            [str(summary_path), "--n-requested", "5", "--baselines-dir", str(baselines_dir)]
        )

        assert rc == 1, f"Expected exit 1 for quarantined scenario; got {rc}"

        # Nothing should be written for a quarantined scenario
        baseline_path = baselines_dir / "late_night_closure_cascade.json"
        assert not baseline_path.exists(), "Quarantined scenario must not be written"

        captured = capsys.readouterr()
        assert "D-10-09" in captured.err, "D-10-09 quarantine code must appear in stderr"
        assert "REFUSED" in captured.err

    def test_observations_carried_forward_on_rewrite(self, tmp_path: Path) -> None:
        """Prior _observations in existing baseline cell are carried forward on rewrite."""
        wb = _load_script()

        scenario_id = "refinement_cheaper"
        provider_key = "anthropic/claude-sonnet-4-6"

        # Write a prior baseline with _observations
        prior_obs = "Phase 9 PROV-03 SHIPPED-WITH-GAP. n=1 single-cell post-fix observation."
        prior_baseline = {
            "scenario_id": scenario_id,
            "generated_at": "2026-01-01T00-00-00Z",
            "generated_by": "scripts/write_baselines.py",
            "providers": {
                provider_key: {
                    "scorers": {
                        "category_compliance": {
                            "median": 1.0,
                            "min": 1.0,
                            "max": 1.0,
                            "stdev": 0.0,
                            "n": 1,
                        }
                    },
                    "_observations": prior_obs,
                }
            },
        }
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        baseline_path = baselines_dir / f"{scenario_id}.json"
        baseline_path.write_text(json.dumps(prior_baseline, indent=2), encoding="utf-8")

        # New summary with n_scored=5 (eligible)
        new_scorers = {
            "category_compliance": {
                "median": 1.0,
                "min": 0.8,
                "max": 1.0,
                "stdev": 0.1,
                "n": 5,
            }
        }
        summary_data = _make_summary(
            scenario_id=scenario_id,
            provider_key=provider_key,
            n_scored=5,
            scorers=new_scorers,
        )
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data), encoding="utf-8")

        rc = wb.main(
            [str(summary_path), "--n-requested", "5", "--baselines-dir", str(baselines_dir)]
        )

        assert rc == 0
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        cell = payload["providers"][provider_key]
        assert "_observations" in cell, "_observations must be carried forward"
        assert cell["_observations"] == prior_obs, "Carried _observations must match prior value"

    def test_missing_summary_exits_two(self, tmp_path: Path) -> None:
        """Missing summary.json → exit 2 (infra failure), distinct from refusal (1)."""
        wb = _load_script()

        missing_path = tmp_path / "nonexistent_summary.json"
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        rc = wb.main(
            [str(missing_path), "--n-requested", "5", "--baselines-dir", str(baselines_dir)]
        )

        assert rc == 2, f"Expected exit 2 for missing summary.json; got {rc}"

    def test_malformed_summary_exits_two(self, tmp_path: Path) -> None:
        """Malformed (non-JSON) summary.json → exit 2 (infra failure)."""
        wb = _load_script()

        bad_path = tmp_path / "bad_summary.json"
        bad_path.write_text("not valid json {{{", encoding="utf-8")

        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        rc = wb.main([str(bad_path), "--n-requested", "5", "--baselines-dir", str(baselines_dir)])

        assert rc == 2, f"Expected exit 2 for malformed summary.json; got {rc}"

    def test_module_imports_without_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """write_baselines.py is stdlib-only; imports cleanly with no provider keys set."""
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        import scripts.write_baselines as wb  # noqa: PLC0415

        importlib.reload(wb)
        # If we reach here, the import did not crash
        assert hasattr(wb, "main"), "write_baselines module must expose main()"
