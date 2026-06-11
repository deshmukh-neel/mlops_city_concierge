---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/eval_agent.py
  - tests/unit/test_eval_agent.py
autonomous: true
requirements: [BASE-01]
must_haves:
  truths:
    - "A single-turn eval case that raises during graph.ainvoke produces an error record (status='error', stage='turn0') instead of aborting the run"
    - "prior_committed_stops and prior_stops_obj scratch keys are never counted as tool calls in tool_calls_mean or tool_names"
    - "An all-errored cell (n_scored == 0) publishes None for deterministic_pass_rate / tool_success_rate / the other derived rates, never 1.0"
    - "eval_agent exits 2 on infra failure, 1 on model-behavior violations, 0 when clean"
  artifacts:
    - path: "scripts/eval_agent.py"
      provides: "WR-06 single-turn error capture, WR-08 phantom-key exclusion, WR-09 zero-n rate guard, D-11-16 0/1/2 exit-code contract"
    - path: "tests/unit/test_eval_agent.py"
      provides: "Unit coverage for each Wave-0 measurement fix"
  key_links:
    - from: "scripts/eval_agent.py count_tool_calls / tool_names_from_state"
      to: "_NON_TOOL_SCRATCH_KEYS frozenset"
      via: "membership exclusion"
      pattern: "_NON_TOOL_SCRATCH_KEYS"
    - from: "scripts/eval_agent.py main()"
      to: "report_has_errors / report_has_violations"
      via: "0/1/2 exit-code branch"
      pattern: "return 2"
---

<objective>
Land the Wave-0 measurement-semantics fixes that live in `scripts/eval_agent.py`: single-turn error capture (WR-06 / D-11-06), phantom-scratch-key exclusion from tool counting (WR-08 / D-11-05), zero-n derived-rate guards (WR-09 / D-11-04), and the 0/1/2 exit-code contract (WR-07 / D-11-16). These change the numbers BASE-01's live regen will bake into committed baselines, so they MUST land before any regen (D-11-01).

Purpose: Each fix corrects a measurement bug that would otherwise be permanently committed into the regenerated baselines. None of these change agent behavior — they change only what the eval harness records and how it reports failure.
Output: A corrected `eval_agent.py` plus unit coverage for each fix.
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
  <name>Task 1: Single-turn error capture + phantom-key exclusion</name>
  <files>scripts/eval_agent.py, tests/unit/test_eval_agent.py</files>
  <read_first>
    - scripts/eval_agent.py — read `make_error_record` (line 170), `count_tool_calls` (line 394), `tool_names_from_state` (line 399), `evaluate_case` (line 660). Match the existing multi-turn error-capture call shape `make_error_record(case, stage, exc)` and the existing scratch-iteration idiom in the two tool helpers.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`scripts/eval_agent.py (MODIFY)` shows the exact target shapes for `evaluate_case`, `count_tool_calls`, `tool_names_from_state`.
    - tests/unit/test_eval_agent.py — read existing test classes covering `count_tool_calls`, `evaluate_case`, and `make_error_record` to mirror fixture/monkeypatch conventions.
  </read_first>
  <behavior>
    - Test: a single-turn EvalQuery (case.turns is empty/falsy) where `graph.ainvoke` raises returns a record with `status == "error"` and `error["stage"] == "turn0"`; scorers are never reached.
    - Test: `count_tool_calls` on a state whose scratch contains both a real tool key (e.g. "semantic_search" with a list) and "prior_committed_stops" / "prior_stops_obj" lists counts only the real tool entries, excluding both phantom keys.
    - Test: `tool_names_from_state` returns the real tool names only, never "prior_committed_stops" or "prior_stops_obj".
    - Test: a normal single-turn case (no exception) still returns a scored result via `query_result_from_state`.
  </behavior>
  <action>
    Add module-level constant `_NON_TOOL_SCRATCH_KEYS = frozenset({"prior_committed_stops", "prior_stops_obj"})` near the tool-counting helpers (WR-08 / D-11-05). In `count_tool_calls` and `tool_names_from_state`, add `and key not in _NON_TOOL_SCRATCH_KEYS` (use the loop variable name each helper already uses) to the existing comprehension filter so the two prod-threading scratch keys are excluded from tool-call counting and tool-name listing. In `evaluate_case` (the `if case.turns:` early-return guards the multi-turn path), wrap the single-turn `graph.ainvoke(...)` call in `try/except Exception as exc:` and `return make_error_record(case, "turn0", exc)` from the except branch, mirroring the multi-turn error capture (WR-06 / D-11-06). Keep latency measurement intact via a `finally` block. Add `# noqa: BLE001` on the broad except per repo convention. Add the four behavior tests above to `tests/unit/test_eval_agent.py`.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_agent.py -v -k "tool_call or phantom or single_turn_error or scratch" </automated>
  </verify>
  <acceptance_criteria>
    - scripts/eval_agent.py contains `_NON_TOOL_SCRATCH_KEYS = frozenset({"prior_committed_stops", "prior_stops_obj"})`
    - `grep -n "_NON_TOOL_SCRATCH_KEYS" scripts/eval_agent.py` shows the constant referenced in BOTH `count_tool_calls` and `tool_names_from_state`
    - scripts/eval_agent.py `evaluate_case` single-turn branch contains `return make_error_record(case, "turn0", exc)` inside an `except Exception` block
    - `poetry run pytest tests/unit/test_eval_agent.py -k "phantom or single_turn_error"` exits 0
  </acceptance_criteria>
  <done>Single-turn exceptions yield error records; the two phantom scratch keys are excluded from tool counts; tests pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Zero-n derived-rate guards + 0/1/2 exit-code contract</name>
  <files>scripts/eval_agent.py, tests/unit/test_eval_agent.py</files>
  <read_first>
    - scripts/eval_agent.py — read `aggregate_results` (line 1011), specifically the derived-rate block at lines 1063-1072 (`deterministic_pass_rate`, `deterministic_violation_rate`, `expected_results_mismatch_rate`, `tool_error_rate`, `tool_success_rate`) and the `n_scored` local already computed in that function. Read `report_has_errors` (line 1132), `report_has_violations` (line 1147), and `main()` (line 1188, current line 1201 returns `1 if report_has_errors(report) or report_has_violations(report) else 0`).
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`scripts/eval_agent.py (MODIFY)` D-11-04 and D-11-16 blocks show the exact target expressions and the guarded build_report except-branch returning 2.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md — Pitfall 4 (WR-09 must guard ALL derived rate fields) and the D-11-16 exit-code pattern.
  </read_first>
  <behavior>
    - Test: `aggregate_results([])` (or a results list where `n_scored == 0`) returns `deterministic_pass_rate is None`, `tool_success_rate is None`, and the other three derived rates None; `cell_valid` stays False.
    - Test: `aggregate_results` over a normal scored list (n_scored > 0) still returns float derived rates (no None regression).
    - Test: `main()` returns 2 when `report_has_errors` is True (synthesize a report with `n_errored > 0`); returns 1 when only `report_has_violations` is True; returns 0 when clean.
    - Test: `main()` returns 2 (not 1) when `build_report` raises.
  </behavior>
  <action>
    In `aggregate_results`, guard all five derived rate fields (`deterministic_pass_rate`, `deterministic_violation_rate`, `expected_results_mismatch_rate`, `tool_error_rate`, `tool_success_rate`) with `... if n_scored > 0 else None` so an all-errored cell publishes None, never the `rate()`-on-zero fail-open 1.0 (WR-09 / D-11-04). Use the `n_scored` value already in scope. In `main()`, replace the single-line return with the 0/1/2 contract (D-11-16 / WR-07): wrap the `build_report` call in `try/except Exception` returning 2 with a stderr message; after rendering, `if report_has_errors(report): return 2`; `if report_has_violations(report): return 1`; `return 0`. Update the `main()` docstring exit-code table to read 0=clean, 1=model-behavior violations, 2=infra failure. Add the four behavior tests above.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_agent.py -v -k "zero_n or derived_rate or exit_code or rc_"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "if n_scored > 0 else None" scripts/eval_agent.py` returns at least 5 (all derived rate fields guarded)
    - scripts/eval_agent.py `main()` contains both `return 2` (infra) and `return 1` (violations) on distinct branches reading `report_has_errors` / `report_has_violations`
    - `poetry run pytest tests/unit/test_eval_agent.py -k "zero_n or exit_code"` exits 0
  </acceptance_criteria>
  <done>All five derived rates abstain to None on zero-n cells; main() honors the 0/1/2 exit-code contract; tests pass.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| eval harness → committed baseline JSON | Recorded numbers from this script become committed empirical record; a measurement bug bakes in permanently |
| CI ← eval_agent exit code | run_matrix and gate wiring consume the exit code; a wrong code mislabels infra failures as model failures |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-01 | Tampering | derived rates on all-errored cell | mitigate | D-11-04 None-guard prevents a fail-open 1.0 from being written into a baseline |
| T-11-02 | Repudiation | eval_agent exit code | mitigate | D-11-16 0/1/2 contract makes infra failures (2) distinguishable from violations (1) in CI logs |
| T-11-03 | Information disclosure | error records / stderr | accept | error records carry exception type/message only; no provider keys are interpolated into the record (existing make_error_record behavior unchanged) |
| T-11-01-SC | Tampering | npm/pip/cargo installs | mitigate | no new packages installed in this plan (RESEARCH §Package Legitimacy Audit: none) |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_agent.py -v` passes.
- `poetry run ruff check scripts/eval_agent.py` clean.
- `make test` (full suite — per project memory `full_suite_db_pool_contamination`, run the whole suite, not just the changed file).
</verification>

<success_criteria>
- WR-06, WR-08, WR-09, WR-07 (D-11-16) fixes present in `scripts/eval_agent.py` with passing unit coverage.
- No agent-behavior change (only measurement/exit-code semantics).
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-01-SUMMARY.md` when done.
</output>
