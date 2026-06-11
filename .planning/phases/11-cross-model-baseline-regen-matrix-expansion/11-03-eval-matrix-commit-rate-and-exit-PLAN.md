---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 03
type: execute
wave: 2
depends_on: ["11-01"]
files_modified:
  - scripts/eval_matrix.py
  - tests/unit/test_eval_matrix.py
autonomous: true
requirements: [BASE-01, BASE-03]
must_haves:
  truths:
    - "committed_itinerary_rate appears in summary.json under scenarios[sid].providers[fam].scorers with a {median,min,max,stdev,n} block"
    - "run_matrix classifies a cell that exits 1 as a violation-cell and a cell that exits >=2 as an error-cell, distinctly"
    - "the eval-matrix structural-check Check 6 builds a real make_error_record and validates the real record schema, not a tautology"
  artifacts:
    - path: "scripts/eval_matrix.py"
      provides: "D-11-02 commit-rate threading, D-11-16 run_matrix exit classification, WR-11 structural-check fix"
    - path: "tests/unit/test_eval_matrix.py"
      provides: "Commit-rate threading test, exit-classification test, WR-11 real-record structural test"
  key_links:
    - from: "scripts/eval_matrix.py aggregate_cell_jsons"
      to: "summary.json scenarios[sid].providers[fam].scorers.committed_itinerary_rate.median"
      via: "thread aggregate.committed_itinerary_rate into the scorers values list"
      pattern: "committed_itinerary_rate"
    - from: "scripts/eval_matrix.py run_matrix"
      to: "eval_agent exit codes 1 (violation) vs >=2 (error)"
      via: "returncode classification into violation_cells / error_cells"
      pattern: "returncode"
---

<objective>
Thread `committed_itinerary_rate` into the summary.json scorers block (D-11-02 — the keystone metric every BASE-03 hard gate rides on), update `run_matrix` to consume the new 0/1/2 exit codes from `eval_agent` and classify violation-cells vs error-cells distinctly (WR-07 / D-11-16), and fix the structural-check "Check 6" tautology to exercise the real `make_error_record` schema (WR-11 / D-11-18). All in `scripts/eval_matrix.py`. Depends on 11-01 (the eval_agent exit-code contract must exist before run_matrix can classify against it).

Purpose: Until `committed_itinerary_rate` reaches the gate-checker-readable `scorers` block, every hard gate reports NOT-EVALUABLE. This plan flips them to enforced. The exit-classification and structural-check fixes make the matrix's failure reporting honest before regen.
Output: A corrected `eval_matrix.py` plus unit coverage threading the metric end-to-end and proving exit classification.
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
@.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-01-SUMMARY.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Thread committed_itinerary_rate into summary.json scorers block</name>
  <files>scripts/eval_matrix.py, tests/unit/test_eval_matrix.py</files>
  <read_first>
    - scripts/eval_matrix.py — read `aggregate_cell_jsons` (line 212), especially the scorer-accumulation loop at lines 290-293 (`for scorer_name, value in _scorer_means_from_cell(payload).items(): provider_block.setdefault(scorer_name, []).append(value)`) and the cell-aggregate read at line 299 (`cell_aggregate = payload.get("aggregate") or {}`). Read `_scorer_means_from_cell` (line 162) — it whitelists against `CRITIQUE_THRESHOLDS` (line 192), which is WHY `committed_itinerary_rate` is excluded today. Read `_stats_for_values` (line 197) — it produces the `{median,min,max,stdev,n}` block automatically for any accumulated values list.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md — §"Data Flow: Commit-Rate Metric Threading (D-11-02)" and Pitfall 1 (threading gap between cell JSON and summary; warning sign is NOT-EVALUABLE persisting). Note: the metric is computed per-cell in eval_agent.py:1092 inside `aggregate.committed_itinerary_rate`; only summary.json needs it in `scorers`, NOT the per-cell JSON.
    - tests/unit/test_eval_matrix.py — read the existing `_write_cell` / `_write_cell_scored` helpers (around lines 1558, 1642) used by aggregator tests to synthesize cell JSONs and feed `aggregate_cell_jsons`.
  </read_first>
  <behavior>
    - Test: feed a synthetic cell JSON whose `aggregate` block has `committed_itinerary_rate: 1.0` (and n_scored>0) through `aggregate_cell_jsons`; assert `summary["scenarios"][sid]["providers"][fam]["scorers"]["committed_itinerary_rate"]["median"] == 1.0` and that the block also has `n`.
    - Test: multiple runs of the same cell with rates [1.0, 0.0, 1.0] aggregate to a `committed_itinerary_rate` median of 1.0 (or the correct median) in the scorers block.
    - Test: a cell whose `aggregate` lacks `committed_itinerary_rate` (legacy/back-compat) does NOT add a `committed_itinerary_rate` scorer key and does not raise.
  </behavior>
  <action>
    In `aggregate_cell_jsons`, immediately after the existing `_scorer_means_from_cell` accumulation loop (lines 290-293) and using the `cell_aggregate = payload.get("aggregate") or {}` already read at line 299 (read it earlier if ordering requires), thread `committed_itinerary_rate` as a supplemental scalar (D-11-02): read `commit_rate = cell_aggregate.get("committed_itinerary_rate")`, and if it is not None and not a bool, `provider_block.setdefault("committed_itinerary_rate", []).append(float(commit_rate))`. This bypasses the `CRITIQUE_THRESHOLDS` whitelist deliberately — add an inline comment noting committed_itinerary_rate is not a critique threshold but is the hard-gate metric per D-10-07/D-11-02. `_stats_for_values` will produce the `{median,min,max,stdev,n}` block for it automatically downstream. Do NOT add it to the per-cell JSON — only the summary needs it in scorers. Add the three behavior tests.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_matrix.py -v -k "committed_itinerary_rate or commit_rate"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "committed_itinerary_rate" scripts/eval_matrix.py` shows the metric threaded inside `aggregate_cell_jsons` (not only in a comment)
    - The new test asserting `summary[...]["scorers"]["committed_itinerary_rate"]["median"]` exists and passes
    - `poetry run pytest tests/unit/test_eval_matrix.py -k "committed_itinerary_rate"` exits 0
  </acceptance_criteria>
  <done>committed_itinerary_rate reaches the summary.json scorers block as a stats block; gate-checker reads become evaluable; tests pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: run_matrix exit classification + WR-11 structural-check fix</name>
  <files>scripts/eval_matrix.py, tests/unit/test_eval_matrix.py</files>
  <read_first>
    - scripts/eval_matrix.py — read `run_matrix` (line 457), signature returns `tuple[int, list[dict[str, Any]]]`; the subprocess-result handling at lines 523-530 currently records ANY `returncode != 0` into a single `failures` list and computes `rc = 0 if not failures else 1` at line 537. Read `_structural_check` (lines starting ~614, the Check blocks; current Check 6 area ~728-752 per RESEARCH — confirm by reading the structural-check function body) and the `make_error_record` import / availability.
    - scripts/eval_agent.py — confirm (from 11-01) the new exit codes: 1 = model-behavior violations, 2 = infra failure. The `make_error_record(case, stage, exc)` signature and `QueryEvalResult.status == "error"` / `.error["stage"]` shape.
    - app/eval/config.py — read `EvalQuery` constructor signature so the synthetic structural-check case is built with valid required fields.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`scripts/eval_matrix.py (MODIFY)` D-11-16 run_matrix split-list pattern and WR-11 real-record structural pattern.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md — Pitfall 6 (run_matrix cell-failure counting must distinguish rc==1 from rc>=2).
  </read_first>
  <behavior>
    - Test: `run_matrix` (or a focused helper) classifies a subprocess result with returncode 1 as a violation-cell and one with returncode 2 as an error-cell; a returncode-0 cell is neither. Matrix overall rc is 0 when only violations and no errors AND no hard-gate concern at this layer (violations are non-blocking here), 2 when any error-cell present. Mirror the exact rc contract the structural-check / CI consume — assert the documented behavior.
    - Test: the structural-check "Check 6" builds a real `make_error_record` from a synthetic `EvalQuery` and asserts `status == "error"` and `error["stage"] in {"setup","turn0","turnN"}`; it fails (returns nonzero / raises the documented signal) if the real record schema regresses.
  </behavior>
  <action>
    Update `run_matrix` to replace the single `failures` list with two lists: `violation_cells` (returncode == 1) and `error_cells` (returncode >= 2), each recording `{cell, stderr, returncode}` (D-11-16 / WR-07). Compute the matrix return code so error-cells dominate: rc 2 if any error-cell, rc 1 if any violation-cell and no error-cell, else 0 — match the exact precedence documented in 11-PATTERNS.md and keep it consistent with how the structural-check and CI step read it. Update the return signature to `tuple[int, list[dict[str, Any]], list[dict[str, Any]]]` (violations, errors) and update the single caller of `run_matrix` in `main()`/the matrix entrypoint to unpack three values and report violation vs error counts distinctly. For WR-11 / D-11-18: rewrite structural-check "Check 6" to import `EvalQuery` (already imported at module top) and `make_error_record`, construct a synthetic case, call `make_error_record(_synthetic_case, "turn0", RuntimeError("quota"))`, and validate `_err_record.status == "error"` and the error stage is in `{"setup","turn0","turnN"}` — emit the existing structural-check failure signal (stderr + nonzero) if either assertion fails. Add both behavior tests.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_matrix.py -v -k "run_matrix or exit_class or structural or check_6 or check6 or error_cell or violation_cell"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "error_cells\|violation_cells" scripts/eval_matrix.py` shows both lists in `run_matrix`
    - scripts/eval_matrix.py `run_matrix` return annotation is a 3-tuple and its caller unpacks three values
    - scripts/eval_matrix.py structural-check contains `make_error_record(` invoked with a real EvalQuery and asserts `status` / stage membership (not a tautology)
    - `poetry run pytest tests/unit/test_eval_matrix.py -k "run_matrix or structural"` exits 0
  </acceptance_criteria>
  <done>run_matrix distinguishes violation-cells from error-cells; structural-check exercises the real make_error_record schema; tests pass.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| matrix aggregation → committed baseline + gate check | committed_itinerary_rate routing decides whether hard gates are enforced or silently not-evaluable |
| eval_agent exit code → run_matrix classification | a misclassified infra failure could pass a regen as clean |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-06 | Tampering | committed_itinerary_rate routing | mitigate | D-11-02 threading + unit test asserting the metric lands in summary scorers block prevents silent NOT-EVALUABLE gates |
| T-11-07 | Repudiation | run_matrix returncode classification | mitigate | D-11-16 error-cell vs violation-cell split makes infra failures non-maskable in the matrix rc |
| T-11-08 | Tampering | structural-check Check 6 tautology | mitigate | WR-11 fix validates the real make_error_record schema so an error-record regression is caught in CI |
| T-11-03-SC | Tampering | npm/pip/cargo installs | mitigate | no new packages installed (RESEARCH §Package Legitimacy Audit: none) |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_matrix.py -v` passes.
- `make eval-matrix-refinement-structural-check` still exits 0 (CI hard gate, no subprocess) — confirm the run_matrix signature change did not break the structural-check entrypoint.
- `poetry run ruff check scripts/eval_matrix.py` clean.
- `make test` full suite passes.
</verification>

<success_criteria>
- D-11-02 keystone metric threaded; D-11-16 exit classification in place; WR-11 structural-check honest; no agent-behavior change.
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-03-SUMMARY.md` when done.
</output>
