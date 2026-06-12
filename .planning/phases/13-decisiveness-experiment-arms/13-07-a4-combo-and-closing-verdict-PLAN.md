---
phase: 13-decisiveness-experiment-arms
plan: 07
type: execute
wave: 4
depends_on: ["13-06"]
files_modified:
  - docs/decisiveness_arm_verdicts.md
  - .planning/REQUIREMENTS.md
  - .planning/ROADMAP.md
autonomous: false
requirements: [DEC-05]
must_haves:
  truths:
    - "The A4 (A1+A2 combo) decision is made against the D-13-01 rule — run ONLY if neither A1 nor A2 cleared the falsifier alone but both showed positive signal — and the decision rationale is recorded in the verdict doc either way"
    - "If A4 runs, it follows the same protocol (smoke n=1, full n=5, falsifier-arm grade, honest recording) and stays within the hard cap of 4 total full live matrix runs"
    - "The closing verdict explicitly states which arm (if any) cleared the INST-05 falsifier bar — or records an honest null result — with per-arm n=5 numbers for gpt-5-mini, deepseek-reasoner, and the gpt-4o-mini anchor"
    - "The closing verdict states the Phase-14 consequence: all arms plateaued below the bar → Phase 14 entry gate OPEN; any arm cleared → Phase 14 SKIPPED, proceed to Phase 15"
  artifacts:
    - path: "docs/decisiveness_arm_verdicts.md"
      provides: "A4 decision + (optional) A4 section + closing verdict line"
      contains: "Closing verdict"
    - path: ".planning/REQUIREMENTS.md"
      provides: "DEC-01..05 traceability flipped to Complete"
      contains: "DEC-05"
  key_links:
    - from: "docs/decisiveness_arm_verdicts.md closing verdict"
      to: "Phase 14 conditional entry gate (ROADMAP.md)"
      via: "explicit cleared-or-null statement the Phase-14 gate reads"
      pattern: "INST-05"
---

<objective>
Close the phase: decide the A4 conditional combo per D-13-01 (run only if neither
A1 nor A2 cleared alone but both showed positive signal), execute it within the
4-run hard cap if sanctioned, and write the DEC-05 closing verdict — the single
explicit line naming the arm that cleared the INST-05 bar or recording the honest
null result. Then flip the phase bookkeeping (REQUIREMENTS traceability,
ROADMAP plan checkboxes).

Purpose: roadmap criterion 5 requires the verdict doc to record per-arm n=5
numbers and explicitly state which arm cleared — or the null result. Phase 14's
conditional entry gate and Phase 15's promotion both read this closing verdict
(D-13-09). The A4 decision needs human judgment on what counts as "positive
signal", so this plan is not autonomous.

Output: completed docs/decisiveness_arm_verdicts.md (A4 decision + closing
verdict), updated traceability. Nothing ships enabled-by-default — promotion is
Phase 15.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md
@.planning/phases/13-decisiveness-experiment-arms/13-06-SUMMARY.md
@docs/decisiveness_arm_verdicts.md
</context>

<tasks>

<task type="checkpoint:decision" gate="blocking">
  <name>Task 1: Decide the A4 conditional combo</name>
  <decision>Run the A4 combo arm (A1+A2: VIABILITY_CONTRACT_ENABLED=1 + FORCED_COMMIT_STEP=6) or skip it and proceed to the closing verdict.</decision>
  <context>D-13-01 sanctions A4 ONLY if neither A1 nor A2 alone cleared the INST-05 falsifier (gpt-5-mini pooled >= 0.6, anchor non-regression) but BOTH show positive signal (e.g. commit-rate improvement over the Phase-12 floor without anchor or scorer regression). The hard cap is 4 full live matrix runs total — 3 are consumed; A4 would be the 4th and last. The A1/A2/A3 sections of docs/decisiveness_arm_verdicts.md carry the numbers needed for this call. Stacking beyond A1+A2 (e.g. +A3) is deferred to Phase 15 per the phase's Deferred Ideas — do not propose it here.</context>
  <options>
    <option id="run-a4">
      <name>Run A4 (A1+A2 combo)</name>
      <pros>Tests whether the two positive-but-insufficient levers compound; uses the sanctioned 4th run; a clearing combo skips Phase 14 entirely.</pros>
      <cons>Additional live spend; only valid if the D-13-01 precondition actually holds (neither cleared alone, both positive).</cons>
    </option>
    <option id="skip-a4">
      <name>Skip A4</name>
      <pros>Correct when an arm already cleared (combo unnecessary) or when A1/A2 showed no positive signal (combo unjustified); preserves budget.</pros>
      <cons>If both arms were positive-but-short, skipping leaves the sanctioned combo untested and weakens the Phase-14 plateau claim.</cons>
    </option>
  </options>
  <action>Summarize the A1/A2 falsifier outcomes against the D-13-01 precondition (neither cleared alone AND both positive), present the two options, then PAUSE for the operator's selection. Record the chosen option + rationale for Task 2.</action>
  <resume-signal>Select: run-a4 or skip-a4 (with one sentence of rationale to record in the verdict doc).</resume-signal>
</task>

<task type="auto">
  <name>Task 2: Execute A4 if sanctioned, then write the closing verdict</name>
  <files>docs/decisiveness_arm_verdicts.md</files>
  <read_first>
    - docs/decisiveness_arm_verdicts.md (A1/A2/A3 filled sections + the A4 placeholder and empty Closing verdict section from 13-06)
    - .planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md D-13-04 (A2/A4 verdict lines must carry the model-initiated vs forced split; forced commits count toward commit rate only if quality scorers hold >= baseline and the anchor is behaviorally unchanged) and D-13-09 (closing-line requirement)
    - Makefile eval-matrix-arm + eval-falsifier-arm targets (if A4 runs)
  </read_first>
  <action>
    Record the Task 1 decision + rationale in the A4 section. IF run-a4: execute the same protocol as 13-06 Task 3 with BOTH flags exported together (`VIABILITY_CONTRACT_ENABLED=1 FORCED_COMMIT_STEP=6`) — smoke n=1 with arm_flags self-description check (both flags must appear in the run JSON's arm_flags), then `make eval-matrix-arm RUNS=5`, then `make eval-falsifier-arm RUN_DIR=<dir>`; fill the A4 section with run dirs, per-scenario breakdown, pooled per-model rates, the model-initiated vs forced split (D-13-04 applies to A4 exactly as to A2, including the anchor red-flag rule), and the falsifier exit code. IF skip-a4: mark the A4 section "NOT RUN" with the recorded rationale. THEN write the Closing verdict section: a per-arm summary table (arm, flag config, gpt-5-mini pooled rate, deepseek-reasoner pooled rate, gpt-4o-mini anchor rate vs baseline, falsifier exit code) followed by ONE explicit closing line naming which arm (if any) cleared the INST-05 bar — or the honest null result ("No arm cleared the INST-05 falsifier bar; all arms plateaued below gpt-5-mini >= 0.6") — and the consequence: null result → Phase 14 (Richer State Replay) entry gate OPEN; any arm cleared → Phase 14 SKIPPED, Phase 15 proceeds with the winning arm. For A2/A4, the closing line must state whether the rate is split-qualified per D-13-04(c) (quality scorers held AND anchor unchanged) — a forced-inflated rate that fails those conditions does NOT count as clearing.
  </action>
  <verify>
    <automated>grep -qi "closing verdict" docs/decisiveness_arm_verdicts.md && grep -qi "INST-05" docs/decisiveness_arm_verdicts.md && echo OK</automated>
    <human-check>Read the closing verdict: the cleared-or-null statement is unambiguous, every number traces to a falsifier print, and the Phase-14 consequence is stated.</human-check>
  </verify>
  <acceptance_criteria>
    - The A4 section carries either a full run record (run dirs, numbers, split, exit code) or "NOT RUN" + rationale — never left as a placeholder.
    - The Closing verdict section has the per-arm summary table covering gpt-5-mini, deepseek-reasoner, and gpt-4o-mini.
    - Exactly one explicit closing line states which arm cleared the INST-05 bar or the honest null result.
    - The Phase-14 consequence (gate OPEN vs SKIPPED) is stated.
    - Total full live matrix runs across the phase <= 4.
    - No baselines written (`git status configs/eval_baselines/` clean) and no flags enabled by default anywhere (promotion is Phase 15).
  </acceptance_criteria>
  <done>The DEC-05 verdict document is complete: per-arm numbers, A4 decision honored, one explicit cleared-or-null closing line that Phase 14's entry gate can read mechanically.</done>
</task>

<task type="auto">
  <name>Task 3: Flip phase bookkeeping (REQUIREMENTS traceability + ROADMAP)</name>
  <files>.planning/REQUIREMENTS.md, .planning/ROADMAP.md</files>
  <read_first>
    - .planning/REQUIREMENTS.md (DEC-01..05 checkboxes + the Traceability table rows for Phase 13)
    - .planning/ROADMAP.md Phase 13 section (plan checkboxes + the Progress table row)
  </read_first>
  <action>
    In `.planning/REQUIREMENTS.md`: check off DEC-01 through DEC-05 in the requirements list and flip their Traceability rows from Pending to Complete, with a one-line note on DEC-05 pointing at docs/decisiveness_arm_verdicts.md and its closing result. In `.planning/ROADMAP.md`: check off the Phase 13 plan list entries and update the Progress table row (plans complete count, status, date). Per the project convention (memory: docs-progress commits go to main directly), these bookkeeping edits are committed straight to main as a docs commit — but ONLY after the phase branch carrying the code/verdict work has merged; if the phase work is still on an unmerged branch, record the flips in the branch instead and note that in the SUMMARY.
  </action>
  <verify>
    <automated>grep -q "\[x\] \*\*DEC-05\*\*" .planning/REQUIREMENTS.md && grep -c "DEC-0[1-5] | Phase 13 | Complete" .planning/REQUIREMENTS.md | awk '{exit ($1==5)?0:1}' && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - All five DEC requirement checkboxes are checked and their traceability rows read Complete.
    - The ROADMAP Phase 13 entry shows all plans checked and the Progress table row updated.
    - The DEC-05 traceability note references docs/decisiveness_arm_verdicts.md.
  </acceptance_criteria>
  <done>Phase 13 requirement and roadmap bookkeeping reflects the completed arm experiments and verdict.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| human A4 decision → live spend | The 4th capped run is gated on an explicit human decision against the D-13-01 rule |
| closing verdict → Phase 14 entry gate | Downstream phase selection depends on this document's honesty |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-07-01 | Repudiation | ambiguous or missing cleared-or-null statement breaking the Phase-14 gate | mitigate | acceptance criterion requires exactly one explicit closing line; human-check verifies unambiguity |
| T-13-07-02 | Elevation of Privilege | forced-inflated A2/A4 rate counted as clearing the bar | mitigate | D-13-04(c) split-qualification rule applied in the closing line: quality scorers must hold and the anchor must be unchanged or the rate does not count |
| T-13-07-03 | Tampering | A4 run outside its D-13-01 precondition or beyond the 4-run cap | mitigate | blocking checkpoint:decision with the precondition in-context; acceptance criterion asserts total runs <= 4 |
| T-13-07-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan; slopcheck N/A |
</threat_model>

<verification>
- docs/decisiveness_arm_verdicts.md is complete: four arm sections resolved (run or NOT RUN + rationale) and a closing verdict with the explicit cleared-or-null line and Phase-14 consequence.
- REQUIREMENTS.md and ROADMAP.md bookkeeping flipped.
- Total live matrix runs <= 4; no default-on flags; no baselines written.
</verification>

<success_criteria>
- DEC-05 satisfied: arm verdicts documented per the INST-05 falsifier with per-arm n=5 numbers for gpt-5-mini, deepseek-reasoner, and the gpt-4o-mini anchor, and an explicit winning-arm-or-null statement (roadmap criterion 5).
- A4 conditionality honored per D-13-01 within the 4-run cap.
- Phase-14 conditional entry gate has its required input.
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-07-SUMMARY.md` when done.
</output>
