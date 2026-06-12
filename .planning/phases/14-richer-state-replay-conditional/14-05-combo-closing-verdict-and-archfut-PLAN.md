---
phase: 14-richer-state-replay-conditional
plan: 05
type: execute
wave: 4
depends_on: ["14-04"]
files_modified:
  - docs/replay_arm_verdicts.md
  - .planning/ROADMAP.md
  - .planning/REQUIREMENTS.md
autonomous: false
requirements: [REPLAY-01, REPLAY-02]
must_haves:
  truths:
    - "The R3 conditional combo (R1+R2) decision is recorded with its D-14-01 precondition check (run only if neither R1 nor R2 clears alone but both show positive signal)"
    - "The discretionary best-replay+FORCED_COMMIT_STEP=6 stack valve decision is recorded (run only if best replay arm AND A2 each independently showed positive signal; ≤4-run cap respected)"
    - "On plateau, the verdict doc contains an ARCH-FUT-01 Evaluation with cumulative evidence chain, contingency restatement, and a written recommendation bounded by Decision 3 — and Phase 15 scope is an explicit USER CHECKPOINT"
    - "If a replay arm CLEARS the INST-05 bar, the winning flag config + run-dir path are recorded as Phase-15 promotion inputs"
    - "The stale ROADMAP line-53 [x]/(completed 2026-06-12) marker on Phase 14 is corrected to reflect actual completion state"
  artifacts:
    - path: "docs/replay_arm_verdicts.md"
      provides: "filled R3/valve decision + Closing Verdict + ARCH-FUT-01 Evaluation + Phase-15 consequence + explicit closing line"
      contains: "Explicit Closing Line"
    - path: ".planning/ROADMAP.md"
      provides: "corrected Phase 14 status (stale line-53 checkbox fixed) + finalized plan list"
      contains: "Phase 14"
    - path: ".planning/REQUIREMENTS.md"
      provides: "REPLAY-01/REPLAY-02 traceability updated to completed with verdict-doc reference"
      contains: "REPLAY-01"
  key_links:
    - from: "docs/replay_arm_verdicts.md Closing Verdict"
      to: "Phase 15 scope decision (user checkpoint)"
      via: "ARCH-FUT-01 recommendation OR winning-arm promotion input"
      pattern: "Phase-15"
---

<objective>
Close Phase 14: decide and record the conditional R3 combo and the discretionary stack valve under the ≤4-run cap, write the Closing Verdict + per-arm summary table + ARCH-FUT-01 Evaluation (on plateau) or winning-arm promotion inputs (on clear), surface the explicit USER CHECKPOINT that gates Phase 15 scope (D-14-08), and fix the stale ROADMAP line-53 bookkeeping marker.

Purpose: Criterion 3 requires the combined REPLAY result to either clear the INST-05 bar (Phase 15 begins) or be documented as a plateau triggering an explicit ARCH-FUT-01 evaluation before Phase 15 scope is finalized. The plateau-vs-clear decision is a user checkpoint, not Claude's call. This plan also performs phase-close bookkeeping.
Output: A complete docs/replay_arm_verdicts.md, corrected ROADMAP/REQUIREMENTS bookkeeping, and a user checkpoint on Phase 15 scope.

RUN BUDGET: R1 + R2 already consumed 2 of ≤4. R3 (if it qualifies) is run 3; the discretionary stack valve (if it qualifies) is run 4. Either or both may be NOT RUN per their preconditions — record the decision and precondition check honestly either way. No billing top-ups (D-11-14).
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
@.planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md
@.planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md
@docs/decisiveness_arm_verdicts.md
@docs/replay_arm_verdicts.md
</context>

<tasks>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 1: Decide + record R3 conditional combo and the discretionary stack valve</name>
  <files>docs/replay_arm_verdicts.md</files>
  <read_first>
    - docs/replay_arm_verdicts.md (the filled R1 + R2 sections from Plan 14-04 — the inputs to the R3/valve precondition checks; the R3 + valve scaffold sections to fill)
    - docs/decisiveness_arm_verdicts.md (the A4 conditional-combo decision block format to mirror, and the A2 = 0.500 number)
    - .planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md (D-14-01: R3 condition + discretionary stack valve condition + ≤4-run hard cap)
    - Makefile lines 220-250 (eval-matrix-arm + eval-falsifier-arm targets for any qualifying run)
  </read_first>
  <action>
    Evaluate the two remaining run slots against their D-14-01 preconditions, run only those that qualify (real API spend, smoke-first with arm_flags verification), and record each decision + precondition check in docs/replay_arm_verdicts.md, then PAUSE for human verification.

    R3 conditional combo (R1+R2, flag config REPLAY_MULTI_MESSAGE_ENABLED=1 REPLAY_CONTENT_BLOCKS_ENABLED=1, all DEC flags unset): run ONLY if neither R1 nor R2 cleared the falsifier alone AND both showed positive signal vs the flag-off floor. Record the precondition check mirroring the A4 decision block. If it qualifies: smoke `make eval-matrix-arm RUNS=1` with both replay flags set, verify the smoke arm_flags shows both replay flags True and the three DEC flags off, full `RUNS=5`, grade with `make eval-falsifier-arm RUN_DIR=<dir>`, fill the R3 section (run dirs, smoke arm_flags, three-delta table, pasted falsifier output, closing verdict). If it does not qualify, record NOT RUN with the failing precondition.

    Discretionary stack valve (best-replay-arm flag + FORCED_COMMIT_STEP=6): may run ONLY if R1/R2/R3 all plateau AND the best replay arm AND A2 each independently showed positive signal. This is deliberately NOT a pure arm (it stacks one replay flag on A2's forced-commit) and is the escalation valve, run only as run 4 of ≤4 and only if it qualifies. Record the decision + precondition check either way (RUN with run dirs + smoke arm_flags, or NOT RUN with rationale).

    Enforce the hard cap: at most 4 full live matrix runs total this phase (R1 + R2 + at most R3 + at most the valve). Then PAUSE for human verification.
  </action>
  <what-built>
    Based on the R1 + R2 results from Plan 14-04, the executor evaluates R3 and the discretionary stack valve against their D-14-01 preconditions, runs only those that qualify (smoke-first with arm_flags verification, ≤4-run cap), records each decision + precondition check, then pauses for human verification of any spend.
  </what-built>
  <how-to-verify>
    1. Open docs/replay_arm_verdicts.md: confirm the R3 section records either a filled run (run dirs, smoke arm_flags showing BOTH replay flags True, three-delta table, pasted falsifier output) OR a NOT RUN decision with an explicit precondition check.
    2. Confirm the discretionary stack-valve decision is recorded with its precondition check (RUN with run dirs, or NOT RUN with rationale).
    3. Confirm the total live-run count across the phase is ≤4 (R1, R2, and at most two more).
    4. If any run was spent, confirm its smoke arm_flags matched the intended config before the full spend.
  </how-to-verify>
  <verify>
    <automated>grep -Eq "R3|conditional combo" docs/replay_arm_verdicts.md && grep -Eq "stack valve|FORCED_COMMIT_STEP" docs/replay_arm_verdicts.md</automated>
  </verify>
  <acceptance_criteria>
    - The R3 section records either a filled qualifying run (run dirs + smoke arm_flags with both replay flags True + three-delta table + pasted falsifier output) OR a NOT RUN decision with an explicit precondition check
    - The discretionary stack-valve decision is recorded with its precondition check (RUN-with-dirs or NOT-RUN-with-rationale)
    - The total live-run count across the phase is ≤4 (source assertion against the verdict doc Run Budget)
    - Any run that was spent has a recorded smoke arm_flags verification preceding the full spend
  </acceptance_criteria>
  <resume-signal>Type "approved" to proceed to the closing verdict + ARCH-FUT-01 section, or describe issues.</resume-signal>
  <done>The R3 conditional combo and the discretionary stack valve are each recorded as RUN-with-dirs or NOT-RUN-with-precondition-check, the ≤4-run cap is respected, and the human has verified any spend.</done>
</task>

<task type="auto">
  <name>Task 2: Write the Closing Verdict, summary table, and ARCH-FUT-01 Evaluation (or winning-arm promotion inputs)</name>
  <files>docs/replay_arm_verdicts.md</files>
  <read_first>
    - docs/replay_arm_verdicts.md (the filled R1/R2/R3/valve sections from 14-04 + Task 1 — the data the closing verdict summarizes)
    - docs/decisiveness_arm_verdicts.md (the Closing Verdict + Per-Arm Summary Table + Explicit Closing Line + Phase-14 Consequence format to mirror)
    - .planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md (D-14-07 three-delta columns, D-14-08 ARCH-FUT-01 three-part structure + Decision 3 bound + user checkpoint)
    - .planning/REQUIREMENTS.md (Out of Scope row "Replacing LangGraph / custom imperative loop ... ARCH-FUT-01 stays a contingency triggered only if all DEC arms fail" — the contingency to restate)
  </read_first>
  <action>
    Complete the Closing Verdict of docs/replay_arm_verdicts.md (D-14-07, D-14-08). Write the Per-Arm Summary Table with the three delta-related columns per arm (pooled rate, Delta vs flag-off floor, Delta vs A2 0.500) for R1, R2, R3, and the valve (NOT RUN rows where applicable). Then branch on the result.

    On PLATEAU (no replay arm cleared the INST-05 bar): write the "ARCH-FUT-01 Evaluation" section with the three required parts — (a) the cumulative evidence chain (v2.1 proved byte-correct reasoning-state round-trip through all four adapters; Phase 13 arms null; Phase 14 replay deltas as measured), (b) a restatement of what the ARCH-FUT-01 contingency would entail (LangGraph replacement / custom imperative loop — the Out-of-Scope contingency), and (c) a written recommendation bounded by Decision 3 (gpt-4o-mini stays anchor, ~30s/turn budget — the likely shape is "ratify anchor, defer ARCH-FUT-01 to a future milestone", but the data speaks first). Mark Phase 15 scope finalization as an explicit USER CHECKPOINT.

    On CLEAR (a replay arm cleared the bar): instead record the winning flag config + run-dir path as Phase-15 promotion inputs in the same DEC-05 format (no ARCH-FUT-01 section needed beyond a note that the contingency is not triggered).

    Write the Explicit Closing Line (one sentence stating the verdict) and the Phase-15 Consequence (winning config + run dir OR documented plateau + user checkpoint before Phase-15 scope). Do NOT modify docs/decisiveness_arm_verdicts.md (it stays immutable; cross-link only).
  </action>
  <verify>
    <automated>grep -q "Explicit Closing Line" docs/replay_arm_verdicts.md && grep -q "Per-Arm Summary Table" docs/replay_arm_verdicts.md && grep -Eq "ARCH-FUT-01 Evaluation|Phase-15 promotion" docs/replay_arm_verdicts.md && grep -q "USER CHECKPOINT" docs/replay_arm_verdicts.md</automated>
  </verify>
  <acceptance_criteria>
    - The Per-Arm Summary Table lists R1, R2, R3, and valve rows (NOT RUN where applicable) with the three delta-related columns (D-14-07)
    - On plateau: the ARCH-FUT-01 Evaluation section contains all three parts (evidence chain, contingency restatement, recommendation bounded by Decision 3) AND a USER CHECKPOINT marker for Phase 15 scope
    - On clear: the winning flag config + run-dir path are recorded as Phase-15 promotion inputs
    - The Explicit Closing Line is a single sentence stating whether any replay arm cleared the INST-05 bar
    - `docs/decisiveness_arm_verdicts.md` is unchanged by this plan (git diff shows no modification)
  </acceptance_criteria>
  <done>The verdict doc is complete with a per-arm summary table, the ARCH-FUT-01 Evaluation (or winning-arm promotion inputs), an explicit closing line, and a Phase-15 user checkpoint — the Phase-13 record untouched.</done>
</task>

<task type="auto">
  <name>Task 3: Phase-close bookkeeping — fix stale ROADMAP line-53 marker + update traceability</name>
  <files>.planning/ROADMAP.md, .planning/REQUIREMENTS.md</files>
  <read_first>
    - .planning/ROADMAP.md line 53 (the stale `[x] **Phase 14: Richer State Replay** ... (completed 2026-06-12)` marker — a Phase-13-close artifact; the progress table line 166 correctly says Not started) and the Phase 14 Details block (lines 122-134, "Plans": TBD to finalize) and the progress table (line 166)
    - .planning/REQUIREMENTS.md lines 42-43 and 94-95 (REPLAY-01 / REPLAY-02 definitions + traceability table rows showing Pending)
    - .planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md (specifics: "Fix the stale ROADMAP.md line-53 marker during planning bookkeeping")
    - docs/replay_arm_verdicts.md (the canonical verdict to reference in traceability)
  </read_first>
  <action>
    Phase-close bookkeeping (these doc edits land on main per project convention; in-plan execution just edits the files). (1) Correct the stale ROADMAP line-53 marker: the premature `[x] ... (completed 2026-06-12)` on the Phase 14 milestone bullet must match TRUE state at execution time — if the phase is actually complete now, keep a legitimate checkbox + completion date; if not yet, restore `[ ]` and remove the false completion date. The marker must match reality, not the Phase-13-close copy-paste artifact, and stay consistent with the progress-table Phase 14 row. (2) In the Phase 14 Details block, replace `**Plans**: TBD` with the finalized plan list (14-01 through 14-05 with one-line objectives) and update the progress table row for Phase 14 to the correct plan count + status. (3) In REQUIREMENTS.md, update the REPLAY-01 and REPLAY-02 traceability rows from Pending to Complete with a reference to docs/replay_arm_verdicts.md as the canonical verdict (mirroring how DEC-05 references docs/decisiveness_arm_verdicts.md), and check the `[ ]` boxes on the REPLAY-01/REPLAY-02 requirement bullets if the phase is complete. Keep all edits factual to the actual run outcome — do not assert "completed" for runs that did not happen.
  </action>
  <verify>
    <automated>grep -n "Phase 14" .planning/ROADMAP.md | head; grep -n "REPLAY-01\|REPLAY-02" .planning/REQUIREMENTS.md</automated>
  </verify>
  <acceptance_criteria>
    - ROADMAP line-53 Phase 14 marker matches actual state (no premature `(completed 2026-06-12)` if the phase is not actually done; legitimate checkbox+date if it is) and is consistent with the progress-table row
    - The Phase 14 Details block lists the finalized 14-01..14-05 plans (no `TBD`) and the progress table Phase 14 row shows the correct plan count
    - REQUIREMENTS.md REPLAY-01 and REPLAY-02 traceability rows reference docs/replay_arm_verdicts.md and reflect the true completion state
    - No fabricated completion claims (bookkeeping reflects actual run outcomes)
  </acceptance_criteria>
  <done>The stale ROADMAP marker is corrected, the Phase 14 plan list is finalized in the roadmap, and REPLAY-01/REPLAY-02 traceability points at the verdict doc with accurate status.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| env flags → R3 / valve live runs | Operator-exported flags drive the combo + stack runs; smoke arm_flags verification is the integrity check |
| verdict + roadmap docs → human Phase-15 decision | Documentation feeding the user checkpoint; no executable surface |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-14-11 | Denial of service | exceeding the run cap via R3 + valve | mitigate | ≤4-run hard cap enforced in the verdict doc Run Budget; R3 + valve only run if their preconditions hold; each is recorded as RUN-with-dirs or NOT-RUN-with-rationale |
| T-14-12 | Repudiation | premature/false phase-completion bookkeeping | mitigate | Task 3 acceptance criteria forbid asserting completion for runs that did not happen; the stale line-53 marker is corrected to match reality, removing the Phase-13-copy artifact |
| T-14-13 | Tampering | editing the immutable Phase-13 verdict record | mitigate | Cross-link only; acceptance criteria assert docs/decisiveness_arm_verdicts.md is unchanged (git diff) |
| T-14-SC | Tampering | npm/pip/cargo installs | mitigate | No new package installs (existing make targets + doc edits); slopcheck N/A |
</threat_model>

<verification>
- docs/replay_arm_verdicts.md is complete: R3/valve decisions recorded, Per-Arm Summary Table with three-delta columns, ARCH-FUT-01 Evaluation (or promotion inputs), explicit closing line, Phase-15 user checkpoint
- ≤4 total live matrix runs across the phase
- ROADMAP line-53 marker corrected; Phase 14 plan list finalized; REPLAY traceability updated to the verdict doc
- docs/decisiveness_arm_verdicts.md unchanged
</verification>

<success_criteria>
- The combined REPLAY result is documented: either a replay arm clears the INST-05 bar (Phase 15 promotion inputs recorded) OR a plateau is documented triggering the ARCH-FUT-01 Evaluation (criterion 3)
- The plateau-vs-clear Phase-15 scope decision is surfaced as an explicit USER CHECKPOINT (D-14-08)
- Phase-close bookkeeping is accurate (stale ROADMAP marker fixed, traceability updated, no false completion claims)
</success_criteria>

<output>
Create `.planning/phases/14-richer-state-replay-conditional/14-05-SUMMARY.md` when done
</output>
