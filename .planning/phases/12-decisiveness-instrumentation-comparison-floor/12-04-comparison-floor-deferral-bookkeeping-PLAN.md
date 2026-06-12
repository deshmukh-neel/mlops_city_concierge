---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: 04
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/unit/test_eval_matrix.py
  - docs/baseline_regen.md
  - docs/eval_gates.md
  - .planning/ROADMAP.md
  - .planning/REQUIREMENTS.md
autonomous: true
requirements: [ANCH-02, ANCH-03]
must_haves:
  truths:
    - "gemini/gemini-3.1-pro-preview is recorded as a deferred comparison-floor cell with a v2.2 user-decision (D-12-09) deferral note, alongside anthropic"
    - "Every non-deferred matrix cell is confirmed honest n=5 — the baseline↔matrix parity test passes with exactly {anthropic, gemini} as the documented deferrals"
    - "docs/baseline_regen.md and docs/eval_gates.md both document the gemini deferral as a v2.2 budget decision (D-12-09), not just a transient Phase-11 quota error"
    - "ROADMAP success criterion 4 and REQUIREMENTS ANCH-02/ANCH-03 wording are amended to match D-12-09 (gemini deferred; comparison floor = matrix minus anthropic AND gemini)"
  artifacts:
    - path: "tests/unit/test_eval_matrix.py"
      provides: "_DEFERRED_BASELINE_CELLS retains gemini + anthropic with D-12-09 rationale"
      contains: "D-12-09"
    - path: "docs/eval_gates.md"
      provides: "Gemini deferral section (D-12-09) parallel to the Anthropic deferral section"
      contains: "Gemini deferral"
  key_links:
    - from: "tests/unit/test_eval_matrix.py _DEFERRED_BASELINE_CELLS"
      to: "configs/eval_matrix.yaml + eval_matrix_refinement.yaml cells"
      via: "test_baseline_provider_cells_match_matrix_entries parity assertion"
      pattern: "_DEFERRED_BASELINE_CELLS"
---

<objective>
Complete the v2.2 comparison floor as bookkeeping (ANCH-02, ANCH-03), honoring the
discuss-time user decision D-12-09: the gemini n=5 baseline is DEFERRED (no quota/billing
top-up) and joins anthropic in deferred-cell status. This plan ships NO gemini-baseline
generation — per the Phase 12 CONTEXT, the planner must NOT create a gemini-baseline plan
as a phase-completion requirement.

The work is to make the deferral explicit and consistent across the codebase and planning
docs, and to verify (via the existing parity test) that every NON-deferred matrix cell is
already honest n=5 (true since Phase 11 for the openai pair + deepseek pair):

- ANCH-02 is satisfied as "gemini measured-or-deferred with a recorded note" — here, the
  deferred-with-note branch (D-12-09).
- ANCH-03 is reinterpreted per D-12-09: all matrix cells EXCEPT the two deferred cells
  (anthropic, gemini) are honest n=5; both deferred cells retain their `_DEFERRED_BASELINE_CELLS`
  entries with deferral notes.

Purpose: every Phase 13 arm is judged against the same comparison floor; this records its
exact shape (matrix minus anthropic and gemini) so there is no ambiguity downstream.
Output: updated deferred-cell rationale + symmetric gemini docs + amended ROADMAP/REQUIREMENTS.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-CONTEXT.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Record the gemini deferral as a v2.2 user decision (D-12-09) and verify parity</name>
  <files>tests/unit/test_eval_matrix.py</files>
  <read_first>
    - tests/unit/test_eval_matrix.py (lines 103-180: the deferred-cell block — the comment
      at 105-117 sanctioning the gemini (D-11-11) and anthropic (D-11-20) deferrals, the
      `_DEFERRED_BASELINE_CELLS: dict[str, set[str]]` at line 118 with gemini under
      `eval_matrix_refinement.yaml` (line 121) and anthropic under `eval_matrix.yaml` (line
      127), and the `test_baseline_provider_cells_match_matrix_entries` parity test at lines
      142-180 that asserts `missing == deferred`)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-CONTEXT.md
      (D-12-09: gemini n=5 DEFERRED at user decision 2026-06-11; joins anthropic; both stay
      logged-not-gated; ANCH-03 reinterpreted as "all cells except the two deferred cells
      are honest n=5"; planner ships no gemini-baseline plan)
  </read_first>
  <action>
    Do NOT remove gemini or anthropic from `_DEFERRED_BASELINE_CELLS` — both stay deferred
    per D-12-09. Update the gemini deferral comment (line 120, currently "D-11-11: gemini
    deferred — errored during regen; retry when GEMINI_API_KEY quota permits") to also cite
    the milestone-level decision: keep the D-11-11 reference and append a D-12-09 note that
    the gemini n=5 baseline is now deferred out of v2.2 scope at user decision (2026-06-11,
    no quota/billing top-up — same treatment as anthropic ANCH-01), revisit when budget
    allows. Update the sanctioning comment block (lines 105-117) so the gemini bullet
    reflects the v2.2 user-decision deferral (not only the transient Phase-11 regen error).
    Leave the anthropic entry and all parity logic unchanged. Then run the parity test to
    confirm the comparison floor is intact: every non-deferred matrix cell still maps to a
    baseline cell (`missing == deferred` holds with deferred == {gemini} for the refinement
    matrix and {anthropic} for the default matrix) — this is the ANCH-03 verification.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_matrix.py -x -q -k "parity or baseline_provider_cells or deferred"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "D-12-09" tests/unit/test_eval_matrix.py` matches in the gemini deferral comment
    - `_DEFERRED_BASELINE_CELLS` still contains `"gemini/gemini-3.1-pro-preview"` (under eval_matrix_refinement.yaml) and `"anthropic/claude-sonnet-4-6"` (under eval_matrix.yaml) — neither removed
    - `poetry run pytest tests/unit/test_eval_matrix.py -q -k "baseline_provider_cells"` passes (every non-deferred cell maps to a baseline cell — ANCH-03 honest-n=5 floor confirmed)
    - The gemini deferral comment text references the user decision / no quota top-up (not only "errored during regen")
  </acceptance_criteria>
  <done>gemini deferral cites D-12-09 as a v2.2 user decision; parity test confirms the non-deferred floor is honest n=5; both deferred cells retained.</done>
</task>

<task type="auto">
  <name>Task 2: Document the gemini deferral symmetrically in the runbook and gates docs</name>
  <files>docs/baseline_regen.md, docs/eval_gates.md</files>
  <read_first>
    - docs/baseline_regen.md (the gemini deferral references at lines 131-132, 202-220, 233,
      288 — currently framed as a Phase-11 transient quota error / "retry when quota permits";
      the "Anthropic deferral (D-11-20)" section at line 233 as the symmetric model to follow)
    - docs/eval_gates.md (the "Anthropic deferral (2026-06-11)" section at line 70 — the
      exact heading/structure to mirror for a parallel Gemini section)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-CONTEXT.md
      (D-12-09 wording + the Deferred Ideas note that ROADMAP/REQUIREMENTS bookkeeping is a
      docs-only amendment)
  </read_first>
  <action>
    In `docs/baseline_regen.md`: add or update a Gemini deferral note (parallel to the
    "Anthropic deferral (D-11-20)" section) stating that as of v2.2 (D-12-09, user decision
    2026-06-11) the gemini n=5 baseline is deferred — no quota/billing top-up — and gemini
    stays logged-not-gated with its `_DEFERRED_BASELINE_CELLS` entry intact. Note the single
    scored gemini run already hit commit-rate 1.0 (measurement debt, not unknown risk), and
    keep the existing promotion path (re-run matrix → write baselines → promote) for when
    budget allows. In `docs/eval_gates.md`: add a "## Gemini deferral (2026-06-11)" section
    mirroring the existing "## Anthropic deferral (2026-06-11)" section at line 70 —
    cite D-12-09, the logged-not-gated status (configs/eval_gates.yaml gemini family), and
    the promotion path. Do not alter the Anthropic sections. These are docs-only edits.
  </action>
  <verify>
    <automated>grep -n "D-12-09" docs/baseline_regen.md docs/eval_gates.md && grep -n "Gemini deferral" docs/eval_gates.md</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "Gemini deferral" docs/eval_gates.md` matches a new section heading parallel to the Anthropic deferral section
    - `grep -n "D-12-09" docs/baseline_regen.md docs/eval_gates.md` matches in both files
    - The gemini deferral text in both docs frames it as a v2.2 user budget decision (no quota/billing top-up), not only a transient regen error
    - The existing Anthropic deferral sections in both files are unchanged
  </acceptance_criteria>
  <done>Both docs document the gemini deferral as a D-12-09 v2.2 decision, symmetric with the anthropic deferral, with the promotion path preserved.</done>
</task>

<task type="auto">
  <name>Task 3: Amend ROADMAP and REQUIREMENTS to match D-12-09</name>
  <files>.planning/ROADMAP.md, .planning/REQUIREMENTS.md</files>
  <read_first>
    - .planning/ROADMAP.md (Phase 12 "Success Criteria" item 4 at line 66 and the
      "External dependency" note at line 68 — both predate D-12-09 and say gemini's
      _DEFERRED_BASELINE_CELLS entry is CLEARED and the gemini baseline is written; D-12-09
      supersedes this)
    - .planning/REQUIREMENTS.md (the ANCH section at lines 21-26: ANCH-02 "gemini ... measured
      and written via write_baselines.py" and ANCH-03 "gemini's _DEFERRED_BASELINE_CELLS
      entry cleared, anthropic's retained" — both predate D-12-09)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-CONTEXT.md
      (the <domain> scope-change note and Deferred Ideas: ROADMAP/REQUIREMENTS need a
      docs-only amendment to reflect the gemini deferral; bookkeeping commits go to main)
  </read_first>
  <action>
    In `.planning/ROADMAP.md` Phase 12: amend Success Criterion 4 so it reads that gemini's
    n=5 baseline is DEFERRED at user decision (D-12-09) and joins anthropic in deferred-cell
    status; the comparison floor is the matrix minus BOTH deferred cells (anthropic AND
    gemini); every other (non-deferred) matrix cell is honest n=5; both deferred cells retain
    their `_DEFERRED_BASELINE_CELLS` entries with deferral notes. Update the "External
    dependency" note so it no longer treats ANCH-02 as a phase-completion requirement (gemini
    deferred, not blocking). In `.planning/REQUIREMENTS.md`: amend ANCH-02 to record the
    gemini n=5 baseline as deferred (D-12-09, user declined quota/billing — same treatment as
    ANCH-01 anthropic) with the single scored run's commit-rate 1.0 noted as measurement debt;
    amend ANCH-03 so it reads "all matrix cells except the two deferred cells (anthropic,
    gemini) are honest n=5; both deferred cells retain `_DEFERRED_BASELINE_CELLS` entries with
    deferral notes." Keep the traceability table accurate (ANCH-02/03 remain Phase 12, now
    satisfied as deferred-with-note). Do not touch INST or DEC/REPLAY/PROMO requirements.
  </action>
  <verify>
    <automated>grep -n "D-12-09\|deferred" .planning/ROADMAP.md | grep -i "gemini\|D-12-09" && grep -n "deferred\|D-12-09" .planning/REQUIREMENTS.md | grep -i "gemini\|ANCH"</automated>
  </verify>
  <acceptance_criteria>
    - ROADMAP Phase 12 Success Criterion 4 states the gemini baseline is deferred (D-12-09) and the comparison floor excludes BOTH anthropic and gemini
    - REQUIREMENTS ANCH-02 records gemini as deferred (D-12-09, user decision) with measurement-debt framing
    - REQUIREMENTS ANCH-03 reads "all cells except anthropic AND gemini are honest n=5; both deferred cells retain entries with notes"
    - `grep -c "D-12-09" .planning/ROADMAP.md .planning/REQUIREMENTS.md` is >= 1 in each file
    - No INST/DEC/REPLAY/PROMO requirement text is modified
  </acceptance_criteria>
  <done>ROADMAP success criterion 4 and REQUIREMENTS ANCH-02/03 now match D-12-09: gemini deferred, comparison floor = matrix minus anthropic and gemini.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| planning docs ↔ code deferred-cell guard | ROADMAP/REQUIREMENTS wording and `_DEFERRED_BASELINE_CELLS` must agree on which cells are deferred, or Phase 13 judges against the wrong floor |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-11 | Tampering | accidental clearing of a deferred cell | mitigate | Task 1 acceptance gate asserts gemini + anthropic remain in _DEFERRED_BASELINE_CELLS; parity test fails closed if the floor drifts |
| T-12-12 | Repudiation | docs/roadmap claim a baseline that does not exist | mitigate | docs and ROADMAP amended to deferred-with-note; parity test confirms non-deferred cells are genuinely honest n=5 |
| T-12-SC | Tampering | npm/pip/cargo installs | mitigate | no package installs; docs/config/test-comment edits only; no slopcheck needed |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_matrix.py -q` passes (parity floor intact, both deferrals documented)
- `grep -n "D-12-09" tests/unit/test_eval_matrix.py docs/baseline_regen.md docs/eval_gates.md .planning/ROADMAP.md .planning/REQUIREMENTS.md` matches across the bookkeeping surface
- `_DEFERRED_BASELINE_CELLS` still lists gemini and anthropic; no gemini baseline JSON was generated
- ROADMAP success criterion 4 and REQUIREMENTS ANCH-02/03 describe the comparison floor as matrix minus anthropic AND gemini
</verification>

<success_criteria>
- ANCH-02 satisfied as deferred-with-note (D-12-09): gemini joins anthropic in deferred-cell status, logged-not-gated, with the deferral recorded in code + docs + planning
- ANCH-03 satisfied (reinterpreted per D-12-09): every non-deferred matrix cell is honest n=5 (parity test green); both deferred cells retain their entries with notes
- No gemini-baseline generation shipped (CONTEXT prohibition honored); promotion path preserved for when budget allows
</success_criteria>

<output>
Create `.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-04-SUMMARY.md` when done
</output>
