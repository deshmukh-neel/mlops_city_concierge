---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: "06"
subsystem: eval-gates
tags: [ci, gates, baselines, advisory, conformance]
dependency_graph:
  requires: ["11-03", "11-05"]
  provides: ["BASE-03-ci-enforcement"]
  affects: [".github/workflows/ci.yml", "scripts/check_eval_gates.py", "Makefile"]
tech_stack:
  added: []
  patterns:
    - "baselines-mode input-source swap (reuses _check_gate unchanged)"
    - "advisory metric-name alias resolution (refinement_minimal_edit_median)"
    - "filename-stem fallback for pre-write_baselines legacy baseline JSONs"
key_files:
  created: []
  modified:
    - scripts/check_eval_gates.py
    - tests/unit/test_check_eval_gates.py
    - Makefile
    - .github/workflows/ci.yml
decisions:
  - "D-11-15: baselines-mode input-source swap reuses _check_gate unchanged; summary positional becomes nargs='?'"
  - "D-11-17: advisory entries report-only WARN; refinement_minimal_edit_median resolved to refinement_minimal_edit scorer"
  - "D-11-19: reasoning_conformance marker promoted to required CI (no continue-on-error)"
  - "D-11-20: aspirational misses (gpt-5-mini) remain non-blocking in baselines mode"
  - "Rule 1 fix: _build_summary_from_baselines falls back to baseline_path.stem when scenario_id absent (pre-write_baselines legacy compat)"
metrics:
  duration: "25m"
  completed: "2026-06-11"
  tasks_completed: 2
  files_changed: 4
---

# Phase 11 Plan 06: Baselines Gate CI — Summary

**One-liner:** Live-key-free BASE-03 CI enforcement via --baselines-mode flag in check_eval_gates.py, advisory WR-05 implementation, and reasoning conformance promotion.

## What Was Built

### Task 1: --baselines-mode synthesis + WR-05 advisory (TDD)

Added `_build_summary_from_baselines(baselines_dir: Path) -> dict` to `scripts/check_eval_gates.py`. This function:
- Iterates `sorted(baselines_dir.glob("*.json"))`, skipping `_snapshots/` subdirectory entries
- Derives `scenario_id` from the JSON `scenario_id` field, falling back to filename stem for legacy baseline JSONs that predate `write_baselines.py`
- Synthesises the exact `aggregate_cell_jsons` summary shape (`scenarios → scenario_id → baseline_eligible/providers → provider_key → scorers/n_scored/n_errored/cell_valid`)
- Derives `n_scored` from `scorers.<any_metric>.n`

Extended `_parse_args`: `summary` positional is now `nargs="?"` (optional when `--baselines-mode` set); added `--baselines-mode` (store_true) and `--baselines-dir` (default `configs/eval_baselines`).

In `main()`: when `--baselines-mode` is set, calls `_build_summary_from_baselines` instead of `_load_summary`; otherwise requires the positional summary argument.

Implemented WR-05 advisory gate evaluation inside `_check_gate`: after the hard-gate result is determined, iterates `gate.get("advisory") or []`; for each entry, resolves `refinement_minimal_edit_median` → `refinement_minimal_edit` (D-11-17); evaluates against every eligible cell and prints `ADVISORY miss ... [non-blocking]` if missed. Advisory results never enter the `violations` list and never change the exit code.

Added 7 new TDD tests covering:
- `_build_summary_from_baselines` shape correctness
- `_snapshots/` subdir is skipped
- Hard-gate synthetic regression (gpt-4o-mini below 0.8 → exit 1)
- Aspirational miss is non-blocking (gpt-5-mini below 0.6 → exit 0)
- `--baselines-mode` without positional summary does not crash
- Advisory miss prints ADVISORY line but exit code stays 0
- Advisory metric name resolution (`refinement_minimal_edit_median` resolves correctly)

### Task 2: Makefile target + CI steps

Added `.PHONY: eval-gates-check-baselines` Makefile target invoking:
```
$(POETRY_RUN) python scripts/check_eval_gates.py --baselines-mode --baselines-dir configs/eval_baselines --gates-config configs/eval_gates.yaml
```

Added two steps to the `eval-matrix` CI job (which installs with `poetry install --with dev --no-interaction` — no `--no-root` per `project_ci_no_root_eval_matrix`):
1. **Check gates against committed baselines (D-11-15)** — runs `make eval-gates-check-baselines`; hard gate, no `continue-on-error`
2. **Run reasoning conformance tests (required — D-11-19)** — runs `make test-reasoning-conformance`; promoted from quarantined to required; mock-driven, no live keys

## Verification

- `poetry run pytest tests/unit/test_check_eval_gates.py -v`: **31 passed**
- `poetry run pytest tests/unit/ -v`: **1192 passed, 11 skipped**
- `make eval-gates-check-baselines`: exits 0 (3× NOT-EVALUABLE for metrics not yet wired — committed_itinerary_rate absent from legacy baselines; correct behavior per T-10-04-01)
- `poetry run ruff check scripts/check_eval_gates.py`: clean
- YAML lint: `.github/workflows/ci.yml` parses cleanly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] _build_summary_from_baselines crashed on legacy baseline JSONs**
- **Found during:** Task 2 — running `make eval-gates-check-baselines` against real committed baselines
- **Issue:** `refinement_cheaper.json` and `late_night_closure_cascade.json` predate the `write_baselines.py` tool and lack the top-level `scenario_id` key. The initial implementation raised `ValueError: missing 'scenario_id' key`, exiting 2.
- **Fix:** Added filename-stem fallback: `scenario_id = payload.get("scenario_id") or baseline_path.stem`. This matches the convention that baseline JSONs are named `<scenario_id>.json`.
- **Files modified:** `scripts/check_eval_gates.py`
- **Commit:** f070aa2

## Known Stubs

None. The baselines-mode gate reports NOT-EVALUABLE for families whose `committed_itinerary_rate` metric is absent from current baseline JSONs (legacy baselines predate Phase 11 BASE-01 threading). This is correct behavior per T-10-04-01 — the gate will flip from NOT-EVALUABLE to enforced once Wave 2 live regen runs and new baselines are committed.

## Threat Flags

None. New CI steps use committed data (no live keys). The advisory entry mechanism is explicitly report-only and cannot be confused with a hard-gate pass (T-11-16 mitigated).

## Self-Check: PASSED

Files exist:
- scripts/check_eval_gates.py: FOUND
- tests/unit/test_check_eval_gates.py: FOUND
- Makefile: FOUND
- .github/workflows/ci.yml: FOUND

Commits exist:
- fd2ef3b (Task 1): FOUND
- f070aa2 (Task 2): FOUND
