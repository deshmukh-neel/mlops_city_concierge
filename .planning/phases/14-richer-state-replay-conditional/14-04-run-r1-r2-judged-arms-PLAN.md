---
phase: 14-richer-state-replay-conditional
plan: 04
type: execute
wave: 3
depends_on: ["14-01", "14-02", "14-03"]
files_modified:
  - docs/replay_arm_verdicts.md
autonomous: false
requirements: [REPLAY-01, REPLAY-02]
must_haves:
  truths:
    - "Before any full n=5 spend, a flag-off byte-identity check confirms the control run reproduces the Phase-13 plateau with arm_flags replay keys both false"
    - "R1 (REPLAY_MULTI_MESSAGE_ENABLED=1, all DEC flags unset) is measured at n=5 temp=1.0 with a preceding n=1 smoke whose arm_flags dict is verified"
    - "R2 (REPLAY_CONTENT_BLOCKS_ENABLED=1, all DEC flags unset) is measured at n=5 temp=1.0 with a preceding n=1 smoke whose arm_flags dict is verified"
    - "Per-arm tables report pooled commit rate plus delta vs flag-off floor plus delta vs A2 (0.500) for gpt-5-mini, gpt-4o-mini anchor, deepseek-reasoner"
    - "Anchor non-regression (gpt-4o-mini holds its floor) is checked for both judged arms; any regression is flagged explicitly (A3 precedent)"
  artifacts:
    - path: "docs/replay_arm_verdicts.md"
      provides: "filled R1 + R2 sections (run dirs, smoke arm_flags, per-model three-delta tables, pasted falsifier output, closing verdicts)"
      contains: "REPLAY_MULTI_MESSAGE_ENABLED"
  key_links:
    - from: "make eval-matrix-arm (env flags)"
      to: "docs/replay_arm_verdicts.md R1/R2 sections"
      via: "run dir + eval-falsifier-arm output pasted verbatim"
      pattern: "eval_reports"
---

<objective>
Run the two judged replay arms PURE (R1 multi-message replay, R2 content-block preservation), each with the three DEC arm flags UNSET, at n=5 temp=1.0 against the Phase-13 plateau, smoke-first with arm_flags verification, and record per-arm three-delta tables + pasted falsifier output + closing verdicts in docs/replay_arm_verdicts.md. Verify anchor non-regression for both.

Purpose: This is the evidence-production core of Phase 14. REPLAY-01 and REPLAY-02 deltas vs the plateau (criteria 1 and 2) are measured here, not assumed. The runs spend real API money, so each full run is preceded by an n=1 smoke whose arm_flags dict is human-verified to match the intended pure-arm config, and the ≤4-run hard cap is respected (R1 + R2 consume 2 of 4).
Output: Filled R1 and R2 sections in the verdict doc.

RUN BUDGET: R1 + R2 = 2 of the ≤4 total live matrix runs (D-14-01/D-14-02). The flag-off control run is a verification baseline, not a judged arm — reuse the recorded Phase-13 plateau numbers as the floor rather than spending a 5th judged run; only run a fresh control if no usable plateau control run exists. No billing top-ups: a partial is recorded honestly and never written as a baseline (D-11-14).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md
@.planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md
@docs/decisiveness_arm_verdicts.md
@docs/replay_arm_verdicts.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Flag-off byte-identity verification + flag-off floor selection</name>
  <files>docs/replay_arm_verdicts.md</files>
  <read_first>
    - .planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md (section "Flag-off byte-identity verification": run the full matrix with both flags unset; arm_flags must show replay keys false; only then [skip-baseline])
    - app/agent/graph.py (the flag-off paths from Plan 14-01 — confirm both default OFF)
    - docs/decisiveness_arm_verdicts.md (the recorded Phase-13 plateau control run dirs + per-model floor numbers to reuse as the flag-off floor)
    - Makefile lines 220-250 (eval-matrix-arm + eval-falsifier-arm targets — env-flag driven, reused as-is, no Makefile change)
  </read_first>
  <action>
    Establish the flag-off floor and verify byte-identity (D-14-02 / Phase-13 [skip-baseline] practice). Run a CI-mode smoke `make eval-matrix-arm RUNS=1 LLM_OVERRIDE=scripted` (no live keys) with NEITHER replay flag set and confirm the produced run JSON `deterministic.arm_flags` shows `replay_multi_message: false` and `replay_content_blocks: false` alongside the four Phase-13 keys (viability_contract, forced_commit_step, parallel_tool, viability_threshold_override). Then fix the flag-off floor: reuse the recorded Phase-13 plateau comparison-floor numbers already in docs/decisiveness_arm_verdicts.md as the "Delta vs flag-off floor" denominator rather than spending a fresh judged run. Record in the verdict doc Run Budget section which recorded run dir / floor numbers serve as the flag-off floor, and record the byte-identity check result (PASS/FAIL). Do NOT spend a live n=5 control unless no usable plateau floor exists.
  </action>
  <verify>
    <automated>make eval-matrix-arm RUNS=1 LLM_OVERRIDE=scripted 2>&1 | tail -5; ls -dt eval_reports/*/ | head -1</automated>
  </verify>
  <acceptance_criteria>
    - A scripted-mode smoke run with no replay flags set produces a run JSON whose `deterministic.arm_flags` shows `replay_multi_message: false` AND `replay_content_blocks: false` AND the four Phase-13 keys present (verified by inspecting the latest run dir JSON)
    - The verdict doc Run Budget section names the flag-off floor source (recorded Phase-13 plateau numbers / run dir) used as the delta denominator
    - The byte-identity verification result is recorded (PASS/FAIL) in the verdict doc before any live spend
  </acceptance_criteria>
  <done>Flag-off byte-identity is confirmed in scripted mode, the flag-off floor source is fixed and recorded, and no unnecessary live control run is spent.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 2: Run R1 (multi-message replay) smoke + full n=5, record verdict</name>
  <files>docs/replay_arm_verdicts.md</files>
  <read_first>
    - docs/replay_arm_verdicts.md (the R1 scaffold section to fill + the Run Budget / flag-off floor recorded in Task 1)
    - docs/decisiveness_arm_verdicts.md (the A2 = 0.500 best-DEC-arm number for the "Delta vs A2" column, and the per-arm section format to mirror)
    - Makefile lines 220-250 (eval-matrix-arm + eval-falsifier-arm targets and their RUNS / RUN_DIR / env-flag usage)
    - scripts/eval_agent.py lines 920-938 (the arm_flags dict that the smoke verification inspects)
  </read_first>
  <action>
    Run R1 PURE (REPLAY-01): export REPLAY_MULTI_MESSAGE_ENABLED=1 with ALL three DEC flags UNSET in the environment (FORCED_COMMIT_STEP unset/0, VIABILITY_CONTRACT_ENABLED unset, PARALLEL_TOOL_EXECUTION_ENABLED unset). Autonomous steps before the human-verify pause:
    1. Smoke: `REPLAY_MULTI_MESSAGE_ENABLED=1 make eval-matrix-arm RUNS=1`. Inspect the smoke run JSON `deterministic.arm_flags`; it MUST read replay_multi_message=True, replay_content_blocks=False, viability_contract=False, forced_commit_step=0, parallel_tool=False. If it does NOT match, STOP — do not spend the full run (this is the confounded-run guard).
    2. Full: `REPLAY_MULTI_MESSAGE_ENABLED=1 make eval-matrix-arm RUNS=5`.
    3. Grade: `make eval-falsifier-arm RUN_DIR=<full run dir>`; capture exit code + per-scenario breakdown.
    4. Fill the R1 section of docs/replay_arm_verdicts.md: smoke + full run dirs, smoke arm_flags verification line, per-model table with pooled commit rate + Delta vs flag-off floor + Delta vs A2 (0.500) for gpt-5-mini / gpt-4o-mini anchor / deepseek-reasoner, falsifier exit code, pasted falsifier per-scenario breakdown verbatim, and a closing verdict. Check anchor non-regression explicitly (gpt-4o-mini must hold its floor; flag any regression per the A3 precedent). Respect the ≤4-run cap (R1 = run 1 of ≤4). Then PAUSE for human verification of the spend + numbers.
  </action>
  <what-built>
    Plan 14-01 wired REPLAY_MULTI_MESSAGE_ENABLED to route plan() through adapter.replay_reasoning_state_multi. This task runs R1 PURE (DEC flags unset) at n=5 temp=1.0, grades it with the falsifier, and fills the R1 section of docs/replay_arm_verdicts.md, then pauses for human verification of the real API spend.
  </what-built>
  <how-to-verify>
    1. Open docs/replay_arm_verdicts.md and confirm the R1 section is filled with real run dirs (under eval_reports/), the smoke arm_flags line showing replay_multi_message: True + the other flags off, and a per-model table with all three delta columns populated.
    2. Confirm the pasted falsifier breakdown and exit code are present and match the recorded table numbers.
    3. Confirm the anchor (gpt-4o-mini) row shows non-regression vs its floor, or an explicit regression flag if it moved.
    4. Sanity-check the API spend is within expectation for one n=1 smoke + one n=5 matrix run (3 models x 2 scenarios), and that the ≤4-run cap is respected (R1 = run 1 of ≤4).
  </how-to-verify>
  <verify>
    <automated>grep -q "REPLAY_MULTI_MESSAGE_ENABLED" docs/replay_arm_verdicts.md && grep -q "eval_reports" docs/replay_arm_verdicts.md</automated>
  </verify>
  <acceptance_criteria>
    - The R1 section is filled with real eval_reports/ run dirs (smoke + full)
    - The smoke arm_flags line shows replay_multi_message: True with the three DEC flags off (confounded-run guard satisfied)
    - The per-model table has pooled commit rate + Delta vs flag-off floor + Delta vs A2 (0.500) for all three run models
    - The falsifier exit code + pasted per-scenario breakdown are present and reconcile with the table numbers
    - The anchor (gpt-4o-mini) row records non-regression OR an explicit regression flag
  </acceptance_criteria>
  <resume-signal>Type "approved" to proceed to R2, or describe issues (e.g. arm_flags mismatch, partial run, anchor regression to investigate).</resume-signal>
  <done>R1 is measured at n=5 PURE with verified smoke arm_flags, the R1 verdict section is filled with three-delta tables + pasted falsifier output, anchor non-regression is checked, and the human has verified the spend.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Run R2 (content-block preservation) smoke + full n=5, record verdict</name>
  <files>docs/replay_arm_verdicts.md</files>
  <read_first>
    - docs/replay_arm_verdicts.md (the R2 scaffold section + the R2 Evidence Audit prediction filled by Plan 14-02 — the EXPECTED-NULL conclusion to reconcile against)
    - docs/decisiveness_arm_verdicts.md (A2 = 0.500 for the Delta vs A2 column)
    - Makefile lines 220-250 (eval-matrix-arm + eval-falsifier-arm targets)
    - scripts/eval_agent.py lines 920-938 (arm_flags dict for smoke verification)
  </read_first>
  <action>
    Run R2 PURE (REPLAY-02): export REPLAY_CONTENT_BLOCKS_ENABLED=1 with all three DEC flags UNSET. Plan 14-02's evidence audit predicted R2 is EXPECTED-NULL on the three RUN models (they emit string-content AIMessages; str() collapse was a no-op for them); R2 runs anyway because criterion 2 requires a measured delta, not an assumption. Autonomous steps before the human-verify pause:
    1. Smoke: `REPLAY_CONTENT_BLOCKS_ENABLED=1 make eval-matrix-arm RUNS=1`. Inspect arm_flags; it MUST read replay_multi_message=False, replay_content_blocks=True, and the three DEC flags off. If mismatch, STOP.
    2. Full: `REPLAY_CONTENT_BLOCKS_ENABLED=1 make eval-matrix-arm RUNS=5`. (Per D-14-05 the R2 smoke MAY be downscoped given the expected-null prediction, but the full n=5 still runs to produce the measured delta.)
    3. Grade: `make eval-falsifier-arm RUN_DIR=<full run dir>`; capture exit code + breakdown.
    4. Fill the R2 section of docs/replay_arm_verdicts.md: run dirs, smoke arm_flags line, per-model three-delta table (pooled rate + Delta vs flag-off floor + Delta vs A2), falsifier exit code, pasted breakdown verbatim, and a closing verdict that EXPLICITLY ties the measured delta back to the R2 Evidence Audit prediction (was the delta null as predicted? — this is the criterion-2 explanation of whether str() collapse caused observable loss). Check anchor non-regression. Respect the ≤4-run cap (R2 = run 2 of ≤4). Then PAUSE for human verification.
  </action>
  <what-built>
    Plan 14-01 wired REPLAY_CONTENT_BLOCKS_ENABLED to preserve pre-cutoff AIMessage content shape through _prune_for_llm. This task runs R2 PURE at n=5, grades it, fills the R2 verdict section reconciling the measured delta against the R2 Evidence Audit's EXPECTED-NULL prediction, then pauses for human verification of the real API spend.
  </what-built>
  <how-to-verify>
    1. Open docs/replay_arm_verdicts.md R2 section: confirm real run dirs, smoke arm_flags showing replay_content_blocks: True + others off, and a filled per-model three-delta table.
    2. Confirm the closing verdict explicitly states whether the measured R2 delta matched the EXPECTED-NULL prediction from the R2 Evidence Audit (criterion 2: explanation of whether str() collapse caused observable loss).
    3. Confirm anchor non-regression (or explicit flag).
    4. Confirm the ≤4-run cap is respected (R1 + R2 = runs 1-2 of ≤4).
  </how-to-verify>
  <verify>
    <automated>grep -q "REPLAY_CONTENT_BLOCKS_ENABLED" docs/replay_arm_verdicts.md && grep -q "R2 Evidence Audit" docs/replay_arm_verdicts.md</automated>
  </verify>
  <acceptance_criteria>
    - The R2 section is filled with real eval_reports/ run dirs (smoke + full)
    - The smoke arm_flags line shows replay_content_blocks: True with replay_multi_message=False and the three DEC flags off
    - The per-model table has all three delta columns for the three run models
    - The closing verdict explicitly reconciles the measured R2 delta against the EXPECTED-NULL prediction (criterion-2 explanation)
    - The anchor row records non-regression OR an explicit regression flag
  </acceptance_criteria>
  <resume-signal>Type "approved" to proceed to the combo/closing-verdict plan (14-05), or describe issues.</resume-signal>
  <done>R2 is measured at n=5 PURE with verified smoke arm_flags, the R2 verdict section is filled and reconciled against the evidence-audit prediction, anchor non-regression is checked, and the human has verified the spend.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| env flags → eval matrix run | Operator-exported arm flags drive the live run config; the smoke arm_flags verification is the integrity check that the intended pure-arm config actually took effect |
| live LLM provider APIs → run JSON | Provider responses recorded into eval_reports; real billable network calls |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-14-08 | Tampering | wrong arm flags silently set | mitigate | Mandatory n=1 smoke + arm_flags dict verification before every full n=5 spend; STOP if arm_flags does not match the intended pure-arm config (prevents attributing a confounded run) |
| T-14-09 | Denial of service | runaway API spend / billing exhaustion | mitigate | Hard cap of <=4 full live matrix runs enforced in the verdict doc Run Budget; no billing top-ups; partials recorded honestly and never written as baselines (D-11-14) |
| T-14-10 | Repudiation | fabricated or misrecorded numbers | mitigate | Falsifier output pasted verbatim from the tool; per-model table numbers must reconcile with the pasted breakdown; human-verify checkpoint reviews both |
| T-14-SC | Tampering | npm/pip/cargo installs | mitigate | No new package installs (uses existing make targets + installed deps); slopcheck N/A |
</threat_model>

<verification>
- R1 and R2 sections of docs/replay_arm_verdicts.md are filled with real run dirs, verified smoke arm_flags, three-delta per-model tables, pasted falsifier output, and closing verdicts
- Each full run was preceded by a smoke whose arm_flags matched the intended pure-arm config (DEC flags off)
- Anchor non-regression checked for both arms; any regression flagged
- ≤4-run cap respected (R1 + R2 = 2 runs)
</verification>

<success_criteria>
- REPLAY-01 (R1) measured at n=5 vs the plateau with the delta vs A2 reported (criterion 1)
- REPLAY-02 (R2) measured at n=5 vs the plateau with the delta reported and tied to the R2 Evidence Audit explanation (criterion 2)
- Pure arms (all DEC flags unset) confirmed by smoke arm_flags verification before each spend
- Anchor held its floor or any regression is explicitly flagged
</success_criteria>

<output>
Create `.planning/phases/14-richer-state-replay-conditional/14-04-SUMMARY.md` when done
</output>
