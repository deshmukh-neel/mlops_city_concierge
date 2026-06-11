---
phase: 10-eval-harness-honesty
plan: 07
type: execute
wave: 1
gap_closure: true
closes_crs: [CR-01]
depends_on: []
files_modified:
  - scripts/check_eval_gates.py
  - tests/unit/test_check_eval_gates.py
autonomous: true
requirements: [EVAL-03]
must_haves:
  truths:
    - "check_eval_gates exits 1 when a hard-gated family's cell drops below its gate value in a real aggregate_cell_jsons-shaped summary (nested scenarios.<id>.providers.<family>)"
    - "check_eval_gates reports NOT-EVALUABLE (not silent pass) when a hard-gated family has no cell anywhere in the real nested summary"
    - "The unit-test summary helper produces the real {'scenarios': {'<id>': {'providers': {...}}}} shape, not the flat {'providers': {...}} shape"
    - "An integration-level test feeds an actual aggregate_cell_jsons() output to the checker and proves a hard gate fires"
  artifacts:
    - path: "scripts/check_eval_gates.py"
      provides: "gate checker that walks the nested scenarios->providers shape"
      contains: "scenarios"
    - path: "tests/unit/test_check_eval_gates.py"
      provides: "tests using the real nested summary shape plus an aggregate_cell_jsons integration test"
      contains: "aggregate_cell_jsons"
  key_links:
    - from: "scripts/check_eval_gates.py"
      to: "summary['scenarios'][<id>]['providers'][<family>]"
      via: "_check_gate walks scenario blocks and looks up the family under each providers map"
      pattern: "scenarios"
    - from: "tests/unit/test_check_eval_gates.py"
      to: "scripts.eval_matrix.aggregate_cell_jsons"
      via: "integration test feeds a real aggregator output to script.main"
      pattern: "aggregate_cell_jsons"
---

<objective>
Close CR-01: `scripts/check_eval_gates.py` reads `summary.get('providers', {})` — a top-level
`providers` key that `aggregate_cell_jsons` never writes. The real summary shape is
`{'generated_at': ..., 'scenarios': {'<scenario_id>': {'providers': {'<family>': cell}, 'baseline_eligible': bool}}}`.
Because the checker looks at the wrong key, every hard-gated family reports NOT-EVALUABLE and the
script exits 0 against real output — `make eval-gates-check` is a structural no-op. The unit tests
bake in the same flat shape so CI stays green while the integration is broken.

Purpose: EVAL-03's executable, satisfiable gate enforcement must actually fire against the
summary.json the matrix runner produces, so Phase 11 BASE-03 can promote gates to CI on top of a
checker that works.
Output: A checker that walks the nested `scenarios -> providers` shape; tests rewritten to the real
shape; an integration test that wires `aggregate_cell_jsons()` output straight into the checker.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/10-eval-harness-honesty/10-VERIFICATION.md
@.planning/phases/10-eval-harness-honesty/10-04-SUMMARY.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Walk the nested scenarios->providers shape in _check_gate</name>
  <files>scripts/check_eval_gates.py</files>
  <read_first>
    - scripts/check_eval_gates.py (current `_check_gate` at lines 121-189; the broken lookup is `providers = summary.get("providers", {})` / `cell = providers.get(family)` at lines 155-156)
    - scripts/eval_matrix.py lines 209-384 — SOURCE OF TRUTH for the summary shape. `aggregate_cell_jsons` builds `result = {"generated_at": ..., "scenarios": scenarios_out}` where each `scenarios_out[scenario_id]` is `{"providers": {<provider_key>: cell}, "baseline_eligible": <bool>}` (the `baseline_eligible` key only present when `eval_queries_config` was passed). `<provider_key>` is the `openai/gpt-4o-mini`-style family string built by `_provider_label`.
  </read_first>
  <behavior>
    - Given a summary with `{"scenarios": {"refinement_cheaper": {"providers": {"openai/gpt-4o-mini": cell_rate_0.4}}}}` and an active gate `committed_itinerary_rate >= 0.8` on family `openai/gpt-4o-mini` → `_check_gate` returns "violation".
    - Given the same gate but a cell at rate 1.0 → returns "pass".
    - Given a summary where no scenario's providers map contains the gated family → returns "not_evaluable" and prints a NOT-EVALUABLE line naming the family.
    - Given a cell that lacks `committed_itinerary_rate` in its scorers block → returns "not_evaluable" (existing `_get_metric_value` None path, unchanged).
    - logged / quarantined-legacy-threading statuses still short-circuit to "skip" before any cell lookup (unchanged).
  </behavior>
  <action>
    Rewrite ONLY the cell-location block in `_check_gate` (currently lines 154-162: `providers = summary.get("providers", {})` then `cell = providers.get(family)`). Replace it with a walk over `summary.get("scenarios", {})`: iterate `for scenario_id, scenario_block in summary.get("scenarios", {}).items()`, read `candidate = scenario_block.get("providers", {}).get(family)`, and take the first non-None `candidate` as `cell`. Skip scenario blocks whose `baseline_eligible` is explicitly False (`if scenario_block.get("baseline_eligible", True) is False: continue`) so a quarantined scenario's cell never satisfies or violates a gate — this matches the D-10-09 quarantine intent and the verifier's documented fix. If no candidate is found after the loop, keep `cell = None` and fall through to the existing NOT-EVALUABLE branch (the print + `return "not_evaluable"` at lines 158-162 is unchanged). Do NOT touch `_get_metric_value`, `_evaluate_op`, `_HARD_STATUSES`, `_SKIP_STATUSES`, `main`, or the exit-code routing. Update the module docstring's "Not-evaluable condition" paragraph (lines 20-25) to say the cell is looked up under each scenario's `providers` map, not a top-level `providers` key.
  </action>
  <verify>
    <automated>poetry run python -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('c','scripts/check_eval_gates.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); g={'family':'openai/gpt-4o-mini','status':'active','hard':{'metric':'committed_itinerary_rate','op':'>=','value':0.8}}; summ={'scenarios':{'refinement_cheaper':{'providers':{'openai/gpt-4o-mini':{'scorers':{'committed_itinerary_rate':{'median':0.4}}}}}}}; r=m._check_gate(g,summ); print(r); sys.exit(0 if r=='violation' else 1)"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "summary.get(\"providers\"" scripts/check_eval_gates.py` returns 0 (the flat top-level lookup is gone).
    - `grep -c "scenarios" scripts/check_eval_gates.py` returns >= 1 (the nested walk is present).
    - The verify command prints `violation` and exits 0 (the gate fires against a real nested-shaped summary with a below-gate cell).
    - Feeding a nested summary whose providers map omits `openai/gpt-4o-mini` makes `_check_gate` return `not_evaluable` and print a line containing `NOT-EVALUABLE` and `openai/gpt-4o-mini`.
  </acceptance_criteria>
  <done>_check_gate locates cells under summary['scenarios'][*]['providers'][family]; a below-gate cell in the real shape returns "violation"; an absent family returns "not_evaluable"; quarantined scenarios are skipped during cell lookup.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Rewrite test summary helper to the real shape and add an aggregate_cell_jsons integration test</name>
  <files>tests/unit/test_check_eval_gates.py</files>
  <read_first>
    - tests/unit/test_check_eval_gates.py (current `_make_summary` at lines 84-99 builds `{"providers": providers, "errors": []}` — the flat shape; helper `_cell_with_rate` at 102-112 and `_cell_no_rate` at 114-123; the exit-code tests at 131-411 all call `_make_summary({...})`)
    - scripts/eval_matrix.py lines 209-384 (`aggregate_cell_jsons` output shape) and 411-440 area for cell-filename format; tests/unit/test_eval_matrix.py lines 312-440 for the `_write_cell` / `_write_cell_with_aggregate` helpers that write per-cell JSONs an aggregator can consume (mirror their filename + payload shape, or import and reuse them).
  </read_first>
  <behavior>
    - `_make_summary({"openai/gpt-4o-mini": cell})` now returns `{"scenarios": {"<scenario>": {"providers": {"openai/gpt-4o-mini": cell}}}, "errors": []}` so every existing exit-code test exercises the real nested path.
    - All eight existing tests (all-pass exit 0, active violation exit 1, aspirational miss exit 0, missing-yaml/summary exit 2, malformed exit 2, logged/quarantined never block, not-evaluable reported) still assert the same exit codes and stderr/stdout signals — only the input shape changes.
    - A new integration test writes per-cell JSONs into tmp_path, calls `scripts.eval_matrix.aggregate_cell_jsons(tmp_path)`, writes the result to summary.json, and asserts `script.main([...])` returns 1 when the aggregated gpt-4o-mini cell sits below the gate (this is the test that would have caught CR-01).
  </behavior>
  <action>
    Rewrite `_make_summary` (lines 84-99) so the `providers` mapping is nested under a single synthetic scenario id (e.g. `"refinement_cheaper"`) inside a top-level `scenarios` key: return `{"scenarios": {scenario_id: {"providers": providers}}, "errors": []}`. Keep the same callable signature `(providers, extra_top_level=None)` and the `extra_top_level` merge so the eight existing tests need no edits. Add a new test `test_integration_real_aggregate_output_fires_hard_gate` that: (1) writes the `_MINIMAL_GATES_YAML` to tmp_path; (2) writes one or more per-cell JSON files into a tmp dir using the eval_matrix cell-filename + `{"aggregate": {"committed_itinerary_rate_mean": 0.4, "n_scored": 5, "n_errored": 0}}` payload shape (reuse `_write_cell_with_aggregate` from test_eval_matrix.py by importing it, or replicate its filename format `{provider}--{model}--{scenario}--run{n}.json`); (3) imports and calls `from scripts.eval_matrix import aggregate_cell_jsons` then `summary = aggregate_cell_jsons(cell_dir)`; (4) writes `summary` to a summary.json file; (5) asserts `script.main([str(summary_file), "--gates-config", str(gates_file)]) == 1`. Verify the scorer name that the aggregator surfaces — `aggregate_cell_jsons` reads `aggregate.{scorer}_mean`, so the per-cell aggregate key must be `committed_itinerary_rate_mean` for the summary's scorer block to carry `committed_itinerary_rate`. Read the `_scorer_means_from_cell` helper in eval_matrix.py to confirm the `_mean` suffix stripping before finalizing the key name.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_check_eval_gates.py -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "\"providers\": providers" tests/unit/test_check_eval_gates.py` returns 0 (the flat-shape helper is gone).
    - `grep -c "\"scenarios\"" tests/unit/test_check_eval_gates.py` returns >= 1.
    - `grep -c "aggregate_cell_jsons" tests/unit/test_check_eval_gates.py` returns >= 1 (the integration test exists).
    - `poetry run pytest tests/unit/test_check_eval_gates.py -q` exits 0 with all prior tests plus the new integration test passing.
    - The new integration test asserts `script.main(...) == 1` against a summary produced by the real `aggregate_cell_jsons`, proving the hard gate fires end-to-end.
  </acceptance_criteria>
  <done>_make_summary emits the nested scenarios->providers shape; all eight original exit-code tests pass unchanged; a new integration test feeds real aggregate_cell_jsons output to the checker and asserts a hard-gate exit-1.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| summary.json -> check_eval_gates | Locally-produced, trusted CI artifact; no untrusted external input crosses here |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-07-01 | Tampering | check_eval_gates exit code | mitigate | The integration test in Task 2 pins the real summary shape so a future aggregator-shape drift makes the checker test go red instead of silently passing (this is the failure mode CR-01 itself was) |
| T-10-07-02 | Repudiation | gate NOT-EVALUABLE vs pass | accept | NOT-EVALUABLE is printed to stdout and is non-blocking by design (Phase 11 wires the metric); low risk, internal tooling only |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_check_eval_gates.py -q` exits 0.
- Manual: `grep -n "scenarios" scripts/check_eval_gates.py` shows the nested walk; `grep -n "summary.get(\"providers\"" scripts/check_eval_gates.py` returns nothing.
- Full suite (per project memory — real-graph tests leak a DB pool unless run together): `make test` exits 0.
</verification>

<success_criteria>
- check_eval_gates fires a hard-gate exit-1 against the real `aggregate_cell_jsons` summary shape (CR-01 closed).
- Tests use the real nested shape and include an aggregate_cell_jsons-fed integration test.
- The verification truth "make eval-gates-check exits non-zero when a summary's hard-gated cell drops below its gate value (EVAL-03)" flips to VERIFIED.
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-07-SUMMARY.md` when done.
</output>
