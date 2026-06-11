---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 05
type: execute
wave: 3
depends_on: ["11-03"]
files_modified:
  - scripts/write_baselines.py
  - tests/unit/test_write_baselines.py
  - Makefile
autonomous: true
requirements: [BASE-01]
must_haves:
  truths:
    - "scripts/write_baselines.py reads a summary.json and writes per-scenario baseline JSON cells, refusing cells with n_scored < n_requested and scenarios with baseline_eligible=False"
    - "the writer stamps generated_at and generated_by and carries forward prior _observations"
    - "make write-baselines and make snapshot-baselines targets exist and invoke the tool / copy the pre-phase11 snapshots"
  artifacts:
    - path: "scripts/write_baselines.py"
      provides: "Baseline writer tool with D-10-03 refusal rule + 0/1/2 exit contract (D-11-07)"
    - path: "tests/unit/test_write_baselines.py"
      provides: "Refusal, write, snapshot, and importability coverage"
    - path: "Makefile"
      provides: "write-baselines + snapshot-baselines targets"
  key_links:
    - from: "scripts/write_baselines.py refusal check"
      to: "summary.json scenarios[sid].providers[fam].n_scored + scenarios[sid].baseline_eligible"
      via: "D-10-03 refusal conditions"
      pattern: "n_scored"
    - from: "scripts/write_baselines.py write path"
      to: "configs/eval_baselines/<scenario>.json"
      via: "atomic write with generated_at / generated_by stamp"
      pattern: "generated_by"
---

<objective>
Build `scripts/write_baselines.py` (D-11-07): a discrete, machine-enforced tool that reads a completed eval-matrix `summary.json` and writes/updates `configs/eval_baselines/*.json` cells. It REFUSES partial cells (`n_scored < n_requested`, the D-10-03 rule) and quarantined scenarios (`baseline_eligible: false`), stamps `generated_at` / `generated_by`, carries forward prior `_observations`, and follows the 0/1/2 exit-code contract. Add `make write-baselines` and `make snapshot-baselines` targets. This is the tool Wave-2 regen invokes — baselines are never hand-rolled again.

Purpose: BASE-01 requires baselines written by a tool that mechanically enforces the refusal rule so a partial or quarantined run can never become a committed baseline. Building it in Wave 1 (no live calls) means Wave 2 only runs it.
Output: The writer tool, its unit tests, and the Makefile targets.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: scripts/write_baselines.py with D-10-03 refusal + exit contract</name>
  <files>scripts/write_baselines.py, tests/unit/test_write_baselines.py</files>
  <read_first>
    - scripts/check_eval_gates.py — the analog. Read the imports block (lines 1-15), `_load_summary` (line 88), `_parse_args` (line 242), and `main()` (line 263) for the entry-point shape, argparse idiom, and the 0/1/2 exit-code contract (0=success, 1=refusal/content, 2=infra) with `except (OSError, ValueError): return 2`.
    - scripts/check_baselines_fresh.py — read `main()` (line 215) and its docstring exit-code table (lines 31-38) as the canonical 0/1/2 contract to imitate.
    - configs/eval_baselines/omakase_mission_open_ended.json AND configs/eval_baselines/refinement_cheaper.json — read both fully to learn the baseline JSON shape: top-level `scenario_id`, `generated_at`, `generated_by`, `providers.<provider/model>.scorers.<scorer>.{median,min,max,stdev,n}`, and the anthropic cell's `_observations` annotation (in refinement_cheaper.json) that must be carried forward.
    - scripts/eval_matrix.py — read the summary shape from `aggregate_cell_jsons` (the docstring at lines 226-244): `summary["scenarios"][sid]["baseline_eligible"]` and `summary["scenarios"][sid]["providers"][fam]` with `scorers`, `n_scored`, `n_errored`, `cell_valid`.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`scripts/write_baselines.py (NEW)` shows imports, argparse, refusal logic, baseline-write pattern, and the `_observations` carry-forward. §"Script Importability Without API Keys" shared pattern for the import-without-keys test.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md — Open Question 1 (`--n-requested` CLI arg is the safer, explicit choice for the refusal check).
  </read_first>
  <behavior>
    - Test: a summary with one eligible scenario whose provider cell has `n_scored == n_requested` writes a baseline JSON with that provider's `scorers` verbatim plus `generated_at` and `generated_by == "scripts/write_baselines.py"`; exit 0.
    - Test: a cell with `n_scored < n_requested` is REFUSED (not written), a REFUSED message is printed to stderr citing D-10-03, and the script exits 1.
    - Test: a scenario with `baseline_eligible == false` has all its cells refused with a D-10-09 quarantine message; exit 1.
    - Test: when a prior baseline cell carried `_observations`, the rewritten cell carries the same `_observations` forward.
    - Test: a missing/malformed summary.json exits 2 (infra), distinct from a refusal (1).
    - Test: the module imports with all provider API keys unset (`monkeypatch.delenv` for OPENAI/ANTHROPIC/DEEPSEEK/GOOGLE) — the writer is stdlib-only and never touches live services.
  </behavior>
  <action>
    Create `scripts/write_baselines.py` mirroring `check_eval_gates.py`'s structure: `#!/usr/bin/env python3`, `from __future__ import annotations`, stdlib-only imports (`argparse`, `json`, `sys`, `datetime`, `collections.abc.Sequence`, `pathlib.Path`). Add a `_load_json_file(path, label)` helper raising OSError on missing / ValueError on malformed JSON (shared JSON-load pattern). `_parse_args`: positional `summary` (path to summary.json), `--n-requested` (int, default 5), `--baselines-dir` (default `configs/eval_baselines`). `main()`: load summary (return 2 on OSError/ValueError). For each scenario in `summary["scenarios"]`: read `baseline_eligible` (default True); for each provider cell, apply the two refusal conditions — if `not baseline_eligible` refuse with a D-10-09 stderr line; if `cell["n_scored"] < n_requested` refuse with a D-10-03 stderr line — appending refused keys to a list and `continue`. For eligible+complete cells, build a baseline cell `{ "scorers": cell["scorers"], "generated_at": <utc YYYY-MM-DDTHH-MM-SSZ>, "generated_by": "scripts/write_baselines.py" }`, carrying forward `_observations` from the prior on-disk baseline if present. Write the per-scenario baseline JSON to `<baselines-dir>/<scenario_id>.json` (load the prior file to merge provider cells and preserve _observations; write with `json.dumps(..., indent=2, sort_keys=True)`). Return 1 if any cell was refused, else 0; 2 only on infra failure. Add `if __name__ == "__main__": raise SystemExit(main())`. Add all six behavior tests in `tests/unit/test_write_baselines.py` using `tmp_path` synthetic summaries.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_write_baselines.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `scripts/write_baselines.py` exists and `poetry run python scripts/write_baselines.py --help` exits 0
    - `grep -n "baseline_eligible\|n_scored" scripts/write_baselines.py` shows BOTH refusal conditions implemented
    - `grep -c "scripts/write_baselines.py" scripts/write_baselines.py` shows the `generated_by` stamp string present
    - The refusal test asserts exit 1 and the infra test asserts exit 2 (distinct)
    - `poetry run pytest tests/unit/test_write_baselines.py` exits 0
  </acceptance_criteria>
  <done>The writer enforces D-10-03 refusal mechanically, stamps provenance, carries _observations, and honors the 0/1/2 contract; tests pass.</done>
</task>

<task type="auto">
  <name>Task 2: Makefile write-baselines + snapshot-baselines targets</name>
  <files>Makefile</files>
  <read_first>
    - Makefile — read the eval-related targets block (lines 120-174): `eval-matrix` (120-125), `eval-matrix-refinement` (132-137), `probe-providers` (158-163), `eval-gates-check` (170-174) for the `.PHONY` + `## description` + `$(POETRY_RUN) python ...` target shape and the `RUNS ?= 1` / `SUMMARY` variable conventions.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`Makefile (MODIFY)` shows the exact `write-baselines` and `snapshot-baselines` target bodies (the snapshot target copies the three canonical baselines to `_snapshots/*.pre-phase11.json`).
    - configs/eval_baselines/_snapshots/ — confirm the directory exists and its existing snapshot filenames (e.g. `refinement_cheaper.pre-phase7.json`) so the `.pre-phase11.json` naming matches convention.
  </read_first>
  <action>
    Add two `.PHONY` targets after `eval-gates-check`. `write-baselines`: `## Write baseline JSONs from a completed summary.json (D-11-07; SUMMARY= required, RUNS= run count)` invoking `$(POETRY_RUN) python scripts/write_baselines.py $(SUMMARY) --n-requested $(RUNS) --baselines-dir configs/eval_baselines`. `snapshot-baselines`: `## Snapshot current canonical baselines to _snapshots/ as pre-phase11 (D-11-09)` running three `cp` commands copying `omakase_mission_open_ended.json`, `refinement_cheaper.json`, and `late_night_closure_cascade.json` from `configs/eval_baselines/` to `configs/eval_baselines/_snapshots/<name>.pre-phase11.json`. Keep the existing `RUNS ?= 1` default; the runbook overrides with `RUNS=5`.
  </action>
  <verify>
    <automated>make -n write-baselines SUMMARY=/tmp/x.json RUNS=5 2>&1 | grep -q "scripts/write_baselines.py" && make -n snapshot-baselines 2>&1 | grep -q "pre-phase11" && echo TARGETS_OK</automated>
  </verify>
  <acceptance_criteria>
    - `make -n write-baselines SUMMARY=/tmp/x.json RUNS=5` prints a `scripts/write_baselines.py` invocation with `--n-requested 5`
    - `make -n snapshot-baselines` prints three `cp` commands targeting `_snapshots/*.pre-phase11.json`
    - `grep -c "^\.PHONY: write-baselines\|^\.PHONY: snapshot-baselines" Makefile` returns 2
  </acceptance_criteria>
  <done>Both Makefile targets exist and dry-run to the expected commands.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| summary.json → committed baseline JSON | the writer is the only gate preventing a partial/quarantined run from becoming committed empirical record |
| writer output → git-committed configs | baseline JSON must carry no provider secrets (it carries only numeric scorer stats) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-11 | Tampering | partial-cell write | mitigate | D-10-03 refusal: `n_scored < n_requested` cells are refused with exit 1, never written |
| T-11-12 | Tampering | quarantined-scenario write | mitigate | `baseline_eligible: false` scenarios refused (D-10-09); late_night can never be regenerated by the tool |
| T-11-13 | Information disclosure | baseline JSON contents | mitigate | writer copies only the `scorers` numeric stats + provenance stamps; no env vars, prompts, or raw model output are written |
| T-11-05-SC | Tampering | npm/pip/cargo installs | mitigate | no new packages installed (RESEARCH §Package Legitimacy Audit: none); writer is stdlib-only |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_write_baselines.py -v` passes.
- `poetry run ruff check scripts/write_baselines.py` clean.
- `poetry run python scripts/write_baselines.py --help` exits 0.
- `make test` full suite passes.
</verification>

<success_criteria>
- BASE-01 writer tool exists with machine-enforced refusal, provenance stamping, and _observations carry-forward; Makefile targets wired; no hand-rolled baseline path remains.
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-05-SUMMARY.md` when done.
</output>
