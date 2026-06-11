---
phase: 10-eval-harness-honesty
plan: 03
type: execute
wave: 3
depends_on: ["10-02"]
files_modified:
  - app/eval/config.py
  - configs/eval_queries.yaml
  - configs/eval_matrix.yaml
  - configs/eval_baselines/late_night_closure_cascade.json
  - scripts/eval_matrix.py
  - tests/unit/test_eval_matrix.py
autonomous: true
requirements: [EVAL-02, EVAL-04]
must_haves:
  truths:
    - "late_night_closure_cascade carries baseline_eligible: false and is excluded from baseline aggregation and gates"
    - "The quarantine decision is recorded in the scenario config, the matrix YAML comment, and the late_night baseline JSON annotation"
    - "The late_night baseline JSON is annotated as legacy-threading-shaped, NOT regenerated and NOT deleted"
    - "The PR #104 baseline<->matrix parity test passes and is verified to cover all current matrix files"
    - "omakase_mission_open_ended (single-turn) is unaffected by the quarantine"
  artifacts:
    - path: "app/eval/config.py"
      provides: "baseline_eligible field on EvalQuery (default True), honored by the matrix runner"
      contains: "baseline_eligible"
    - path: "configs/eval_baselines/late_night_closure_cascade.json"
      provides: "_observations annotation marking legacy-threading-shaped measurement"
      contains: "_observations"
    - path: "tests/unit/test_eval_matrix.py"
      provides: "quarantine-flag parse test + verified parity test coverage"
      contains: "baseline_eligible"
  key_links:
    - from: "configs/eval_queries.yaml late_night case"
      to: "EvalQuery.baseline_eligible"
      via: "YAML field parsed into the Pydantic model"
      pattern: "baseline_eligible"
    - from: "EvalQuery.baseline_eligible == False"
      to: "matrix aggregation / baseline inclusion"
      via: "quarantined cells marked so baseline tooling skips them"
      pattern: "baseline_eligible"
---

<objective>
Quarantine `late_night_closure_cascade` from baselines and gates (EVAL-02) and verify the
PR #104 baseline<->matrix parity test (EVAL-04). The closure-cascade turn-2 scorers were
designed against the full-tool-history shape (memory `project_eval_multi_turn_threading_bug`);
migrating it to prod threading would redesign the scenario — out of scope for a harness-honesty
phase (D-10-09). So it stays runnable as a diagnostic but is explicitly excluded: a
`baseline_eligible: false` flag on the scenario (D-10-09), honored by the matrix runner, with
the decision recorded in three places (scenario config, matrix YAML comment, baseline JSON
annotation; the gates-doc entry is added in plan 10-04). The late_night baseline JSON is
annotated with `_observations` (D-10-10) but NOT regenerated and NOT deleted.

EVAL-04: the parity test `test_baseline_provider_cells_match_matrix_entries` already shipped in
PR #104; this plan verifies it passes and that `_DEFERRED_BASELINE_CELLS` covers exactly the
current matrix files, extending only if a gap exists.

Purpose: Phase 11's regen must skip the quarantined scenario automatically; a future matrix-add
must not silently drift from its baseline.
Output: a parsed, runner-honored quarantine flag; an annotated (not regenerated) baseline; a
verified parity test.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/10-eval-harness-honesty/10-CONTEXT.md
@.planning/phases/10-eval-harness-honesty/10-PATTERNS.md
@.planning/phases/10-eval-harness-honesty/10-02-SUMMARY.md
@app/eval/config.py
@configs/eval_queries.yaml
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add baseline_eligible to EvalQuery, quarantine late_night in config, honor it in the matrix runner</name>
  <files>app/eval/config.py, configs/eval_queries.yaml, configs/eval_matrix.yaml, scripts/eval_matrix.py</files>
  <read_first>
    - app/eval/config.py (read EvalQuery :145-177 — model_config extra="forbid" plus threading_mode/expected_refinement fields; a new optional field with a default keeps the 30 legacy cases valid)
    - configs/eval_queries.yaml (read the late_night_closure_cascade case at :410-421 — it has NO threading_mode so defaults to legacy; add baseline_eligible: false here)
    - configs/eval_matrix.yaml (read the scenarios list :29-31 and the comment style at :16; add a D-10-09/10 comment block adjacent to late_night)
    - scripts/eval_matrix.py (read aggregate_cell_jsons and iter_cells to find where a baseline_eligible=false scenario is marked in the summary; respect the 10-02 error-threading code already present)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-09 quarantine now/migrate later; D-10-10 three-place recording)
  </read_first>
  <behavior>
    - EvalQuery parses a new optional boolean field baseline_eligible defaulting to True; the 30 existing legacy cases remain valid (extra="forbid" satisfied because the field is declared).
    - The late_night_closure_cascade case in eval_queries.yaml sets baseline_eligible: false and parses cleanly.
    - A cell whose scenario has baseline_eligible=False is marked in summary.json so baseline tooling can skip it; omakase_mission_open_ended (defaulting True) is unaffected.
  </behavior>
  <action>
    Add `baseline_eligible: bool = True` to the EvalQuery model in app/eval/config.py (default True preserves all 30 legacy cases and omakase; the field is declared so extra="forbid" accepts it when present). Add a docstring line citing D-10-09. In configs/eval_queries.yaml, add `baseline_eligible: false` to the late_night_closure_cascade case (the case at :410-421) with an inline comment citing D-10-09 (legacy threading shape; closure-cascade turn-2 scorers designed for full-tool-history; migration deferred). In configs/eval_matrix.yaml, add a D-10-09/10 comment block adjacent to the late_night_closure_cascade scenario entry (decision in comment as breadcrumb only). In scripts/eval_matrix.py, make aggregate_cell_jsons honor the flag: resolve each scenario's baseline_eligible from the eval-queries config and surface a `baseline_eligible: false` marker in that scenario's summary block so any baseline writer (Phase 11) skips it. The cell still RUNS as a diagnostic — only skip baseline-eligibility, never execution. Do NOT change what the scenario measures.
  </action>
  <verify>
    <automated>poetry run python -c "from app.eval.config import load_eval_queries; cfg=load_eval_queries('configs/eval_queries.yaml'); ln=[c for c in cfg.hand_written if c.id=='late_night_closure_cascade'][0]; om=[c for c in cfg.hand_written if c.id=='omakase_mission_open_ended'][0]; assert ln.baseline_eligible is False and om.baseline_eligible is True"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "baseline_eligible" app/eval/config.py` shows the field declared with default True on EvalQuery.
    - `grep -n "baseline_eligible" configs/eval_queries.yaml` shows `baseline_eligible: false` on the late_night case.
    - The verify one-liner exits 0 (late_night parses False, omakase parses True).
    - `grep -n "baseline_eligible\|D-10-09" configs/eval_matrix.yaml` shows the quarantine comment adjacent to late_night.
    - `poetry run pytest tests/unit/ -q -k "eval_config or eval_matrix"` exits 0 (the 30 legacy cases still validate).
  </acceptance_criteria>
  <done>The quarantine flag is a first-class parsed field, set false on late_night, recorded in matrix YAML, and honored as a baseline-skip marker by the aggregator; the scenario still runs as a diagnostic.</done>
</task>

<task type="auto">
  <name>Task 2: Annotate the late_night baseline JSON and verify/extend the EVAL-04 parity test</name>
  <files>configs/eval_baselines/late_night_closure_cascade.json, tests/unit/test_eval_matrix.py</files>
  <read_first>
    - configs/eval_baselines/late_night_closure_cascade.json (read the top-level keys: scenario_id, generated_at, generated_by, closure_check_confirmed, providers — add _observations WITHOUT touching providers or any score numbers)
    - tests/unit/test_eval_matrix.py (read _DEFERRED_BASELINE_CELLS :101-104 and test_baseline_provider_cells_match_matrix_entries :116-148 — the EVAL-04 anchor from PR #104; this is verification, not rewrite)
    - .planning/phases/10-eval-harness-honesty/10-CONTEXT.md (D-10-10 annotate-not-regenerate; EVAL-04 verify-only unless a gap appears)
  </read_first>
  <action>
    Add a top-level `"_observations"` string key to configs/eval_baselines/late_night_closure_cascade.json per D-10-10: state it is a legacy-threading-shaped measurement, not comparable to prod, NOT regenerated in Phase 10 (annotated only), and reference the quarantine decision. Do NOT modify `providers`, score values, `scenario_id`, or any other existing key — annotation only. In tests/unit/test_eval_matrix.py, add `test_late_night_scenario_is_baseline_ineligible` asserting `load_eval_queries(...)` yields baseline_eligible False for late_night and that the baseline JSON contains the `_observations` annotation. Run the existing `test_baseline_provider_cells_match_matrix_entries` and confirm it passes; verify `_DEFERRED_BASELINE_CELLS` keys cover every matrix file currently in configs/ (eval_matrix.yaml, eval_matrix_refinement.yaml). If a matrix file exists with no `_DEFERRED_BASELINE_CELLS` entry, add an empty-set entry (quarantine is NOT a deferral — late_night stays in the parity check). Do NOT add late_night to _DEFERRED_BASELINE_CELLS.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_matrix.py -q -k "parity or baseline or late_night or provider_cells"</automated>
  </verify>
  <acceptance_criteria>
    - `python -c "import json; assert '_observations' in json.load(open('configs/eval_baselines/late_night_closure_cascade.json'))"` exits 0.
    - The late_night baseline JSON `providers` block byte-content is unchanged (only `_observations` added) — verifiable by `git diff --stat` showing additions only, no score-line deletions.
    - `tests/unit/test_eval_matrix.py::test_baseline_provider_cells_match_matrix_entries` passes for every matrix file.
    - `tests/unit/test_eval_matrix.py::test_late_night_scenario_is_baseline_ineligible` exists and passes.
  </acceptance_criteria>
  <done>The late_night baseline is annotated (not regenerated/deleted); EVAL-04 parity is verified across all current matrix files with quarantine kept distinct from deferral.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| eval-queries YAML → EvalQuery model | the baseline_eligible flag governs whether a scenario can seed a prod baseline; a missing flag must default to eligible (fail-safe toward inclusion in parity, not silent exclusion) |
| baseline JSON → Phase 11 regen tooling | the _observations annotation tells future tooling this measurement is not prod-comparable |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-10-03-01 | Tampering | late_night baseline JSON edit | mitigate | Annotation-only change; acceptance criterion asserts providers/score lines are byte-unchanged (git diff additions only) so the quarantine annotation cannot silently alter recorded scores |
| T-10-03-02 | Repudiation | quarantine decision provenance | mitigate | Decision recorded in three places (scenario config inline comment, matrix YAML comment, baseline _observations) all citing D-10-09/10 — auditable trail |
| T-10-03-03 | Elevation of privilege | baseline_eligible default | accept | Defaulting True means a new scenario is parity-checked by default (fail toward enforcement, not toward silent skip); quarantine is opt-in and explicit |
| T-10-03-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_matrix.py -q` exits 0.
- `python -c "import json; json.load(open('configs/eval_baselines/late_night_closure_cascade.json'))"` parses (valid JSON after annotation).
- `poetry run ruff check app/eval/config.py scripts/eval_matrix.py tests/unit/test_eval_matrix.py` passes.
</verification>

<success_criteria>
- late_night_closure_cascade is quarantined via a parsed baseline_eligible flag, recorded in three places, runnable but baseline-ineligible (EVAL-02).
- The PR #104 parity test is verified to cover all current matrix files in both directions modulo documented deferrals (EVAL-04).
</success_criteria>

<output>
Create `.planning/phases/10-eval-harness-honesty/10-03-SUMMARY.md` when done.
</output>
