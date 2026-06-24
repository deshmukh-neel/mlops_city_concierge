"""Content validator for configs/eval_baselines/*.json (Plan 03-13 / CR-01).

Phase 3 ships three baseline JSON stubs with ``"generated_at": null`` and
every per-scorer ``{median, min, max, stdev}`` set to ``null``. The
``scripts/check_baselines_fresh.py`` lint enforces only that *some* file under
``configs/eval_baselines/*.json`` was touched in a PR — it does not parse
contents or reject nulls. That gate is structurally incomplete: a Phase 4-6 PR
that whitespace-touches a baseline file passes the lint and merges with
nulls in place, then crashes downstream baseline-diff tooling on
``None < x`` comparisons (or, worse, silently treats ``None`` as 0).

This test is the **content gate**. It loads each baseline JSON and asserts
that ``generated_at`` is non-null and every ``{median, min, max}`` triple
under ``providers.{provider/model}.scorers.{scorer}`` is non-null too.

Activation:
    The current Phase 3 baselines are intentional PENDING_USER_RUN stubs;
    populating them requires a real ~15-min API-spend matrix run (see
    plan-03-13 / VERIFICATION.md handoff section). This test is therefore
    gated on the ``BASELINES_POPULATED=1`` env var so the test suite stays
    green during Phase 3 while still locking the content contract.

    Once the user runs ``APP_ENV=eval make eval-matrix RUNS=3`` and
    post-processes ``eval_reports/{ts}/summary.json`` into the three
    baseline files, the operator should set ``BASELINES_POPULATED=1`` in
    CI and the gate becomes hard. The skip-message below carries that
    activation instruction so a future operator does not have to dig.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = REPO_ROOT / "configs" / "eval_baselines"

BASELINES_POPULATED = os.environ.get("BASELINES_POPULATED") == "1"

pytestmark = pytest.mark.skipif(
    not BASELINES_POPULATED,
    reason=(
        "BASELINES_POPULATED != '1' — Phase 3 ships PENDING_USER_RUN stubs. "
        "After running `APP_ENV=eval make eval-matrix RUNS=3` and post-"
        "processing summary.json into configs/eval_baselines/*.json, set "
        "BASELINES_POPULATED=1 in CI to enable this content gate."
    ),
)


def baseline_paths() -> list[Path]:
    """Return every configs/eval_baselines/*.json file (sorted, deterministic)."""
    return sorted(BASELINES_DIR.glob("*.json"))


def test_baseline_directory_has_expected_files() -> None:
    """The three Phase-3 scenarios each ship one baseline JSON file."""
    names = {p.name for p in baseline_paths()}
    expected = {
        "omakase_mission_open_ended.json",
        "refinement_cheaper.json",
        "late_night_closure_cascade.json",
    }
    assert expected.issubset(names), (
        f"missing one or more Phase-3 baseline files: expected {expected}, found {names}"
    )


@pytest.mark.parametrize("baseline_path", baseline_paths(), ids=lambda p: p.name)
def test_baseline_generated_at_is_populated(baseline_path: Path) -> None:
    """`generated_at` must carry a real ISO timestamp, not the PENDING stub null.

    Catches the failure mode where a PR whitespace-touches a baseline file
    (satisfying the stale-baseline lint) but leaves contents as the
    PENDING_USER_RUN stubs.
    """
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert payload.get("generated_at") is not None, (
        f"{baseline_path.name} still has generated_at=null (PENDING_USER_RUN). "
        "Run `APP_ENV=eval make eval-matrix RUNS=3` and post-process "
        "eval_reports/{ts}/summary.json into this file."
    )


@pytest.mark.parametrize("baseline_path", baseline_paths(), ids=lambda p: p.name)
def test_baseline_scorer_stats_are_populated(baseline_path: Path) -> None:
    """Every {median, min, max} triple under providers.*.scorers.* must be non-null.

    The Phase 4-6 baseline-diff tooling consumes these values directly; a
    None median makes ``None < x`` raise TypeError or, worse, silently
    becomes 0 and locks in a spurious regression baseline.
    """
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    providers = payload.get("providers")
    assert isinstance(providers, dict) and providers, (
        f"{baseline_path.name}: 'providers' must be a non-empty mapping"
    )
    for provider_key, provider_block in providers.items():
        scorers = (provider_block or {}).get("scorers")
        assert isinstance(scorers, dict) and scorers, (
            f"{baseline_path.name}: providers.{provider_key}.scorers must be a non-empty mapping"
        )
        for scorer_name, stats in scorers.items():
            for field in ("median", "min", "max"):
                value = (stats or {}).get(field)
                assert value is not None, (
                    f"{baseline_path.name}: providers.{provider_key}.scorers."
                    f"{scorer_name}.{field} is null — refresh required"
                )
