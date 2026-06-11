---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 06
type: execute
wave: 4
depends_on: ["11-03", "11-05"]
files_modified:
  - scripts/check_eval_gates.py
  - tests/unit/test_check_eval_gates.py
  - Makefile
  - .github/workflows/ci.yml
autonomous: true
requirements: [BASE-03]
must_haves:
  truths:
    - "check_eval_gates.py --baselines-mode reads committed configs/eval_baselines/*.json, synthesizes a summary-shaped dict, and runs the existing gate-check loop with no live keys"
    - "a synthetic committed regression below a hard gate exits non-zero; an aspirational miss (gpt-5-mini) stays non-blocking (exit 0)"
    - "advisory gate entries (WR-05) are evaluated as report-only WARN with the refinement_minimal_edit_median metric name resolved"
    - "a CI step runs make eval-gates-check-baselines and the reasoning_conformance marker runs as a required (non-continue-on-error) CI step"
  artifacts:
    - path: "scripts/check_eval_gates.py"
      provides: "--baselines-mode flag + _build_summary_from_baselines + advisory implementation (D-11-15 / D-11-17)"
    - path: "tests/unit/test_check_eval_gates.py"
      provides: "Synthetic-regression test + advisory-report test + baselines-mode synthesis test"
    - path: "Makefile"
      provides: "eval-gates-check-baselines target"
    - path: ".github/workflows/ci.yml"
      provides: "baselines-mode gate CI step + reasoning_conformance required step (D-11-15c / D-11-19)"
  key_links:
    - from: "scripts/check_eval_gates.py --baselines-mode"
      to: "_check_gate (existing, unchanged)"
      via: "_build_summary_from_baselines produces the exact summary shape _check_gate consumes"
      pattern: "_build_summary_from_baselines"
    - from: ".github/workflows/ci.yml"
      to: "make eval-gates-check-baselines + make test-reasoning-conformance"
      via: "required CI steps (no continue-on-error)"
      pattern: "eval-gates-check-baselines"
---

<objective>
Wire BASE-03 CI enforcement, all live-key-free (D-09-10 / D-11-15): (a) add `--baselines-mode` to `check_eval_gates.py` that reads committed `configs/eval_baselines/*.json` and synthesizes a summary-shaped dict so a committed regression below a hard gate fails CI; (b) implement the WR-05 advisory gate reporting (report-only WARN) with the `refinement_minimal_edit_median` metric-name resolution (D-11-17); (c) add a synthetic-regression unit test proving the gate fires; (d) add the `eval-gates-check-baselines` Makefile target and a CI step; (e) promote the `reasoning_conformance` pytest marker to a required CI step (D-11-19).

Purpose: BASE-03 requires per-family merge gates enforced in CI without live keys. The baselines-mode reuses the existing `_check_gate` logic via an input-source swap; the conformance-marker promotion resolves the D-08-14 deferred decision.
Output: Gate checker with baselines mode + advisory, Makefile target, two CI steps, and tests.
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
  <name>Task 1: --baselines-mode synthesis + WR-05 advisory implementation</name>
  <files>scripts/check_eval_gates.py, tests/unit/test_check_eval_gates.py</files>
  <read_first>
    - scripts/check_eval_gates.py — read the whole file. Key: `_load_summary` (line 88), `_get_metric_value` (line 105, reads `cell["scorers"][metric]["median"]`), `_evaluate_op` (line 130), `_check_gate` (line 145, walks `summary["scenarios"][*]["providers"][family]`, returns "violation"/"aspirational_miss"/None, reports NOT-EVALUABLE when metric absent), `_parse_args` (line 242, positional `summary` + `--gates-config`), `main()` (line 263, the gate loop at lines 288-296 and the 0/1/2 exit logic). Note `_check_gate` does NOT currently evaluate the `advisory:` list — that is the WR-05 gap.
    - configs/eval_gates.yaml — read the full gate schema: each gate has `family`, `status`, `rationale`, `hard` (null or {metric,op,value}), `advisory` (list). The `openai/gpt-4o-mini` entry has an advisory `refinement_minimal_edit_median >= 0.0`; the `openai/gpt-5-mini` entry is `status: aspirational` with a hard `committed_itinerary_rate >= 0.6`.
    - configs/eval_baselines/omakase_mission_open_ended.json + refinement_cheaper.json — the on-disk baseline shape: `scenario_id`, `providers.<fam>.scorers.<metric>.{median,...,n}`. The synthesized summary must use these `scorers` blocks verbatim.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`scripts/check_eval_gates.py (MODIFY)` for `_build_summary_from_baselines`, the argparse extension (summary becomes `nargs="?"`), and the advisory-resolution pattern (`refinement_minimal_edit_median` → `refinement_minimal_edit`).
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md — Pitfall 5 (synthesis must match summary shape exactly — use the baseline scorers block verbatim) and Open Question 2 (`n_scored` derived from `scorers.<any_metric>.n`).
  </read_first>
  <behavior>
    - Test: `_build_summary_from_baselines(dir)` over a dir with one baseline JSON produces `{"scenarios": {sid: {"baseline_eligible": True, "providers": {fam: {"scorers": <verbatim>, "n_scored": <from scorers.*.n>, "n_errored": 0, "cell_valid": True}}}}}`; the `_snapshots/` subdir is skipped.
    - Test: `main(["--baselines-mode", "--baselines-dir", <dir>, "--gates-config", <real gates>])` with a committed `openai/gpt-4o-mini` baseline whose `committed_itinerary_rate.median` is below 0.8 (active hard gate) exits 1.
    - Test: the same with `openai/gpt-5-mini` `committed_itinerary_rate.median == 0.0` (below its aspirational 0.6) exits 0 — aspirational misses are non-blocking (D-11-20).
    - Test: an advisory entry whose metric is `refinement_minimal_edit_median` resolves to the `refinement_minimal_edit` scorer median and prints an ADVISORY/WARN line when missed, but never changes the exit code.
    - Test: `--baselines-mode` with no positional `summary` does not error on the missing positional.
  </behavior>
  <action>
    Add `_build_summary_from_baselines(baselines_dir: Path) -> dict` synthesizing the summary shape from committed baseline JSONs: iterate `sorted(baselines_dir.glob("*.json"))`, skip any path whose parent dir is `_snapshots`, read `scenario_id`, and for each provider cell emit `{"scorers": cell.get("scorers", {}), "n_scored": <derived from first scorers.*.n or cell.get("n_scored", 0)>, "n_errored": 0, "cell_valid": True}`; set `baseline_eligible: True` per scenario. Extend `_parse_args`: add `--baselines-mode` (store_true) and `--baselines-dir` (default `configs/eval_baselines`), and make `summary` `nargs="?"` default None. In `main()`, when `--baselines-mode` is set, call `_build_summary_from_baselines(Path(args.baselines_dir))` instead of `_load_summary(args.summary)` (still inside the try/except returning 2 on OSError/ValueError); otherwise keep the current path. Implement WR-05 advisory (D-11-17) inside `_check_gate` (or a helper it calls): after the hard-gate check, iterate `gate.get("advisory") or []`; for each, resolve the metric name (`refinement_minimal_edit_median` → `refinement_minimal_edit`), read the value via `_get_metric_value`, and if present and the op fails, print an `ADVISORY miss ... [non-blocking]` line — advisory results MUST NOT be added to `violations` or change the exit code. Add the five behavior tests.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_check_eval_gates.py -v -k "baselines_mode or synthes or advisory or aspirational"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "_build_summary_from_baselines\|--baselines-mode" scripts/check_eval_gates.py` shows both the synthesis fn and the flag
    - `scripts/check_eval_gates.py` contains the advisory metric-name resolution mapping `refinement_minimal_edit_median` to `refinement_minimal_edit`
    - The synthetic-regression test (gpt-4o-mini below 0.8) exits 1 and the aspirational test (gpt-5-mini below 0.6) exits 0
    - `poetry run pytest tests/unit/test_check_eval_gates.py -k "baselines_mode or advisory"` exits 0
  </acceptance_criteria>
  <done>Baselines-mode synthesizes the summary shape and reuses _check_gate unchanged; advisory entries report-only WARN; synthetic regression fires; aspirational misses stay non-blocking.</done>
</task>

<task type="auto">
  <name>Task 2: Makefile target + CI steps (baselines gate + conformance marker)</name>
  <files>Makefile, .github/workflows/ci.yml</files>
  <read_first>
    - Makefile — read `eval-gates-check` (lines 170-174) for the target shape and `test-reasoning-conformance` (lines 189-191, runs `pytest -m reasoning_conformance`).
    - .github/workflows/ci.yml — read the `lint-baselines` job (lines 107-164) and the `eval-matrix` job (lines 166-234), especially the install step at line 204 (`poetry install --with dev --no-interaction` — NOT `--no-root`, per the comment at lines 195-204 and memory `ci_no_root_eval_matrix`). Read the unit-tests job (lines 74-104) for where a conformance step could attach.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`.github/workflows/ci.yml (MODIFY)` and §`Makefile (MODIFY)` for the exact step/target bodies; D-11-19 conformance promotion (no continue-on-error) and D-11-15c baselines-mode CI step.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md — D-11-15 (CI live-key-free), D-11-19 (reasoning_conformance is mock-driven; fixtures SKIP gracefully when absent per D-10-12).
  </read_first>
  <action>
    Add a `.PHONY: eval-gates-check-baselines` Makefile target: `## Check gates against committed baselines (D-11-15; no live keys)` invoking `$(POETRY_RUN) python scripts/check_eval_gates.py --baselines-mode --baselines-dir configs/eval_baselines --gates-config configs/eval_gates.yaml`. In `ci.yml`, add a step that runs `make eval-gates-check-baselines` — attach it to a job that installs dev deps WITHOUT `--no-root` (the eval-matrix job already does this, or replicate its install) since check_eval_gates imports `app` transitively via the gates path; the step is a hard gate (no `continue-on-error`). Add a second CI step running `make test-reasoning-conformance` as a REQUIRED step (no `continue-on-error`), promoting the `reasoning_conformance` marker from quarantined to required per D-11-19 — annotate it as mock-driven (no live keys) with graceful fixture SKIP. Preserve the existing `lint-baselines`, `eval-matrix`, and structural-check steps unchanged.
  </action>
  <verify>
    <automated>make -n eval-gates-check-baselines 2>&1 | grep -q -- "--baselines-mode" && grep -q "eval-gates-check-baselines" .github/workflows/ci.yml && grep -q "test-reasoning-conformance" .github/workflows/ci.yml && echo CI_WIRED</automated>
  </verify>
  <acceptance_criteria>
    - `make -n eval-gates-check-baselines` prints a `check_eval_gates.py --baselines-mode` invocation
    - `.github/workflows/ci.yml` contains a step running `make eval-gates-check-baselines` with no `continue-on-error: true` on that step
    - `.github/workflows/ci.yml` contains a step running `make test-reasoning-conformance` with no `continue-on-error: true`
    - The job hosting the baselines-gate step installs with `poetry install --with dev --no-interaction` (no `--no-root`)
    - `make eval-gates-check-baselines` exits 0 against the current committed baselines (gate-evaluable or NOT-EVALUABLE, not a crash)
  </acceptance_criteria>
  <done>Makefile target + two required CI steps wired; baselines-gate runs live-key-free; conformance marker promoted to required.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| committed baseline JSON → CI gate verdict | a committed regression must fail CI; baselines mode is the enforcement point |
| CI runner → provider APIs | CI must never use live keys (D-09-10); baselines mode and conformance tests are mock/committed-data only |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-14 | Tampering | committed baseline regression | mitigate | D-11-15 baselines-mode gate fails CI when a committed hard-gate metric regresses; synthetic-regression test proves it fires |
| T-11-15 | Elevation of privilege | CI live-key usage | mitigate | D-11-15 / D-09-10: baselines mode reads committed JSON, conformance tests are mock-driven; no provider keys referenced in the new CI steps |
| T-11-16 | Spoofing | advisory false-pass | mitigate | WR-05 advisory entries are explicitly report-only and never alter the exit code, so an advisory miss cannot be confused with a hard-gate pass |
| T-11-06-SC | Tampering | npm/pip/cargo installs | mitigate | no new packages installed (RESEARCH §Package Legitimacy Audit: none); CI install uses pinned poetry lock |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_check_eval_gates.py -v` passes.
- `make eval-gates-check-baselines` runs against the current committed baselines without crashing (exit 0/1 by gate state, not 2).
- `poetry run ruff check scripts/check_eval_gates.py` clean.
- YAML lint: `.github/workflows/ci.yml` parses (e.g. `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"`).
- `make test` full suite passes.
</verification>

<success_criteria>
- BASE-03 CI enforcement live-key-free: baselines-mode gate fires on synthetic regression, advisory implemented, conformance marker required; Makefile + CI wired.
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-06-SUMMARY.md` when done.
</output>
