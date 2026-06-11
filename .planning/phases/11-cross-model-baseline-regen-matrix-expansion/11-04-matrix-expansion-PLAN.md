---
phase: 11-cross-model-baseline-regen-matrix-expansion
plan: 04
type: execute
wave: 3
depends_on: ["11-03"]
files_modified:
  - configs/eval_matrix.yaml
  - tests/unit/test_eval_matrix.py
autonomous: true
requirements: [BASE-02]
must_haves:
  truths:
    - "configs/eval_matrix.yaml contains openai/gpt-5-mini, anthropic/claude-sonnet-4-6, deepseek/deepseek-reasoner alongside the existing openai/gpt-4o-mini and deepseek/deepseek-chat, all flag-OFF (no env override)"
    - "late_night_closure_cascade is removed from the eval_matrix.yaml scenarios list but stays runnable via the SCENARIOS param"
    - "the baseline-vs-matrix parity test passes with the three new entries listed as documented Wave-1 deferrals"
  artifacts:
    - path: "configs/eval_matrix.yaml"
      provides: "3 new cross-model entries + late_night scenario removal (D-11-12 / D-11-13)"
    - path: "tests/unit/test_eval_matrix.py"
      provides: "Updated entry/scenario counts + _DEFERRED_BASELINE_CELLS for eval_matrix.yaml + _MATRIX_TO_BASELINES late_night removal"
  key_links:
    - from: "configs/eval_matrix.yaml entries"
      to: "tests/unit/test_eval_matrix.py _DEFERRED_BASELINE_CELLS / count assertions"
      via: "atomic co-update in the same commit"
      pattern: "_DEFERRED_BASELINE_CELLS"
---

<objective>
Expand `configs/eval_matrix.yaml` with the three new flag-OFF cross-model entries (`openai/gpt-5-mini`, `anthropic/claude-sonnet-4-6`, `deepseek/deepseek-reasoner`) per D-11-12, and remove `late_night_closure_cascade` from the default scenarios list per D-11-13 (it stays runnable via the explicit SCENARIOS param). Atomically update the `test_eval_matrix.py` parity test and count assertions in the SAME commit so CI does not break (the parity test was designed to catch exactly the Phase-9 atomicity gap).

Purpose: BASE-02 requires the four cross-model providers in the matrix before the Wave-2 regen produces their baseline cells. During Wave 1 the new entries have no baseline cells yet, so they are listed as documented deferrals; Wave 2 removes them from the deferral set after the live cells land.
Output: Expanded matrix YAML + atomically-updated parity test.
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

<task type="auto">
  <name>Task 1: Add 3 cross-model entries + remove late_night scenario from eval_matrix.yaml</name>
  <files>configs/eval_matrix.yaml</files>
  <read_first>
    - configs/eval_matrix.yaml — read the full file. Current `entries` (lines 23-27) are `openai/gpt-4o-mini` and `deepseek/deepseek-chat`; current `scenarios` (lines 29-40) are `omakase_mission_open_ended` and `late_night_closure_cascade` (the latter under a long D-10-09/10 comment block).
    - configs/eval_matrix_refinement.yaml — read its `entries` block to mirror the exact entry schema (`- provider: X` / `model: Y`). Flag-OFF means NO `env:` block on the new entries (the refinement YAML has `env:` overrides; this one must not).
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`configs/eval_matrix.yaml (MODIFY)` shows the exact new-entries block and the late_night removal comment.
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-CONTEXT.md — D-11-12 (gemini stays OUT of eval_matrix.yaml — gemini lives only in eval_matrix_refinement.yaml) and D-11-13 (late_night removal + cost rationale).
  </read_first>
  <action>
    Add three entries to the `entries` list, each flag-OFF (provider/model only, no `env:` block): `provider: openai / model: gpt-5-mini`, `provider: anthropic / model: claude-sonnet-4-6`, `provider: deepseek / model: deepseek-reasoner`. Add an inline comment `# Phase 11 / D-11-12 / BASE-02: cross-model entries (flag-OFF)`. Do NOT add gemini — it stays exclusively in eval_matrix_refinement.yaml per D-11-12 / PROV-04. Remove `late_night_closure_cascade` from the `scenarios` list (D-11-13), leaving only `omakase_mission_open_ended`. Replace the removed entry with a comment block documenting: the D-10-09 quarantine, that it stays runnable via `make eval-matrix SCENARIOS=late_night_closure_cascade`, and the cost rationale (5 providers × 5 runs of a baseline-ineligible scenario is pure burn). Preserve the existing top-of-file D-06 comment header.
  </action>
  <verify>
    <automated>poetry run python -c "from app.eval.config import load_eval_matrix; from pathlib import Path; m = load_eval_matrix(Path('configs/eval_matrix.yaml')); ps = {(e.provider, e.model) for e in m.entries}; assert len(m.entries) == 5, m.entries; assert ('openai','gpt-5-mini') in ps; assert ('anthropic','claude-sonnet-4-6') in ps; assert ('deepseek','deepseek-reasoner') in ps; assert ('gemini' not in {p for p,_ in ps}); assert m.scenarios == ['omakase_mission_open_ended'], m.scenarios; print('OK')"</automated>
  </verify>
  <acceptance_criteria>
    - `configs/eval_matrix.yaml` parses via `load_eval_matrix` with exactly 5 entries and exactly 1 scenario
    - The five entries include `openai/gpt-5-mini`, `anthropic/claude-sonnet-4-6`, `deepseek/deepseek-reasoner`; no gemini entry present
    - `scenarios` is exactly `[omakase_mission_open_ended]`; the file contains a comment referencing `SCENARIOS=late_night_closure_cascade`
    - None of the three new entries carry an `env:` block (flag-OFF)
  </acceptance_criteria>
  <done>Matrix YAML has 5 flag-OFF entries and 1 scenario; late_night documented as removed-but-runnable.</done>
</task>

<task type="auto">
  <name>Task 2: Atomically update parity test counts + deferred-cells map</name>
  <files>tests/unit/test_eval_matrix.py</files>
  <read_first>
    - tests/unit/test_eval_matrix.py — read `test_repo_eval_matrix_yaml_loads_via_load_eval_matrix` (lines 33-52, currently asserts `len(matrix.entries) == 2`, `len(matrix.scenarios) == 2`, and `"late_night_closure_cascade" in matrix.scenarios`). Read `_DEFERRED_BASELINE_CELLS` (lines 101-104) and `_MATRIX_TO_BASELINES` (lines 106-112) and `test_baseline_provider_cells_match_matrix_entries` (lines 115-149) to understand the both-directions parity logic (orphans = baseline keys without a matrix entry; missing = matrix entries without a baseline cell, which must exactly equal the deferred set).
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-RESEARCH.md — Pitfall 2 (parity test fails after YAML expansion unless updated atomically) and Open Question 3 (gpt-5-mini Wave-1 deferral handling).
    - .planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-PATTERNS.md — §`tests/unit/test_eval_matrix.py (MODIFY)` Wave-1 update pattern (new providers added to deferred set; late_night removed from _MATRIX_TO_BASELINES so its baseline JSON is no longer parity-checked against the matrix).
  </read_first>
  <action>
    Update `test_repo_eval_matrix_yaml_loads_via_load_eval_matrix`: change `len(matrix.entries)` assertion from 2 to 5; change `len(matrix.scenarios)` from 2 to 1; add `assert ("openai","gpt-5-mini") in providers`, `assert ("anthropic","claude-sonnet-4-6") in providers`, `assert ("deepseek","deepseek-reasoner") in providers`; replace `assert "late_night_closure_cascade" in matrix.scenarios` with `assert "late_night_closure_cascade" not in matrix.scenarios` (D-11-13 invariant); keep `assert "omakase_mission_open_ended" in matrix.scenarios`. In `_DEFERRED_BASELINE_CELLS`, set the `"eval_matrix.yaml"` value to `{"openai/gpt-5-mini", "anthropic/claude-sonnet-4-6", "deepseek/deepseek-reasoner"}` with a comment that these are Wave-1 deferrals removed after Wave-2 regen lands the live baseline cells. In `_MATRIX_TO_BASELINES`, remove `"late_night_closure_cascade.json"` from the `"eval_matrix.yaml"` list (it is no longer a default-matrix scenario, so its baseline must not be parity-checked against the matrix entries) — leave only `"omakase_mission_open_ended.json"`. Do NOT delete the late_night baseline JSON file itself (D-10-10 annotate-not-delete standing).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_matrix.py -v -k "yaml_loads or baseline_provider_cells_match"</automated>
  </verify>
  <acceptance_criteria>
    - `tests/unit/test_eval_matrix.py` `_DEFERRED_BASELINE_CELLS["eval_matrix.yaml"]` equals the three-provider set
    - `_MATRIX_TO_BASELINES["eval_matrix.yaml"]` no longer contains `late_night_closure_cascade.json`
    - `poetry run pytest tests/unit/test_eval_matrix.py -k "yaml_loads or baseline_provider_cells_match"` exits 0
    - `configs/eval_baselines/late_night_closure_cascade.json` still exists on disk (not deleted)
  </acceptance_criteria>
  <done>Parity test and count assertions match the expanded matrix; the three new entries are documented deferrals; CI is green on the same commit as the YAML change.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| matrix YAML ↔ parity test | drift between matrix entries and baseline cells silently weakens the regen audit; the parity test is the only guard |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-11-09 | Tampering | matrix/test atomicity | mitigate | parity test + count assertions updated in the same commit as the YAML (D-11-12/13; closes the PROV-02 chore-3800737 atomicity-gap class) |
| T-11-10 | Information disclosure | new provider entries | accept | YAML carries provider/model names only, no keys; gemini intentionally excluded per PROV-04 |
| T-11-04-SC | Tampering | npm/pip/cargo installs | mitigate | no new packages installed (RESEARCH §Package Legitimacy Audit: none) |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_matrix.py -v` passes.
- `make eval-matrix-refinement-structural-check` still exits 0 (unchanged matrix file).
- `make test` full suite passes.
</verification>

<success_criteria>
- BASE-02 matrix config present (4 cross-model providers, gemini excluded); late_night quarantined from default run; parity test atomically consistent.
</success_criteria>

<output>
Create `.planning/phases/11-cross-model-baseline-regen-matrix-expansion/11-04-SUMMARY.md` when done.
</output>
