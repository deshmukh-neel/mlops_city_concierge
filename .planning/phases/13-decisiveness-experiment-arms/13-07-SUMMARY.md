---
phase: 13-decisiveness-experiment-arms
plan: "07"
subsystem: docs/eval
tags: [dec-05, arm-verdicts, closing-verdict, a4-skip, phase-14-gate, null-result]
dependency_graph:
  requires: [13-06-run-judged-arms]
  provides: [dec-05-verdict, phase-14-entry-gate-signal, phase-13-bookkeeping]
  affects:
    - docs/decisiveness_arm_verdicts.md
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
tech_stack:
  added: []
  patterns: [d-13-01-a4-conditionality, d-13-04-split-qualification, d-13-09-closing-line]
key_files:
  created: []
  modified:
    - docs/decisiveness_arm_verdicts.md
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
decisions:
  - "skip-a4: A1 showed zero signal (0.000 pooled); D-13-01 precondition (both A1 AND A2 positive signal) not satisfied; 4th run slot unused"
  - "Closing verdict: honest null result — no arm cleared INST-05 bar (gpt-5-mini >= 0.6); Phase 14 entry gate OPEN"
  - "A2 split-qualified per D-13-04(c): 0.500 rate is entirely model-initiated (forced=0); quality scorers held on committed episodes; anchor unchanged; does not clear 0.6 bar"
  - "Phase 14 (Richer State Replay) proceeds: all DEC arms plateaued below INST-05 bar; conditional entry gate is satisfied"
metrics:
  duration: "~15 minutes"
  completed: "2026-06-12T10:30:00Z"
  tasks: 3
  files: 3
---

# Phase 13 Plan 07: A4 Combo and Closing Verdict Summary

DEC-05 closed with an honest null result: no arm cleared the INST-05 bar (gpt-5-mini >= 0.6
at n=5 with no anchor regression). A4 skip decision recorded per D-13-01. Phase 14 entry
gate is OPEN.

## What Was Built

**Task 1 (checkpoint:decision — resolved before this executor):** Human decided skip-a4.
Rationale: A1 = 0.000 (zero signal); D-13-01 precondition "both arms show positive signal"
not satisfied. No code ran; the 4th run slot is unused. Run budget = 3/4 slots.

**Task 2: A4 decision + closing verdict written in docs/decisiveness_arm_verdicts.md (commit b544e10)**

- A4 section: "NOT RUN" with D-13-01 precondition check spelled out in full.
- Closing verdict: per-arm summary table (gpt-5-mini / deepseek-reasoner / gpt-4o-mini,
  all four arms) with pooled rates, model-initiated vs forced split, and falsifier exit codes.
- Explicit closing line: "No arm cleared the INST-05 falsifier bar. All arms plateaued below
  gpt-5-mini >= 0.6."
- Phase-14 consequence stated: entry gate OPEN; Phase 14 proceeds, Phase 15 does not skip.
- A2 split-qualification note per D-13-04(c): 0.500 entirely model-initiated, quality scorers
  held, anchor unchanged — split-qualified but still below 0.6 bar.

**Task 3: Phase 13 bookkeeping flipped (commit cf7eb31)**

- REQUIREMENTS.md: DEC-05 traceability row updated with closing verdict note (honest null
  result; Phase 14 entry gate OPEN; references docs/decisiveness_arm_verdicts.md). All five
  DEC checkboxes [x] and traceability rows Complete were already in place from prior plans.
- ROADMAP.md: plan 13-07 checked; Phase 13 progress row updated to 7/7 Complete 2026-06-12;
  Phase 13 phase-list entry checked with closing result noted in annotation.

## Deviations from Plan

None — plan executed exactly as written. The skip-a4 path was anticipated by the plan;
the A4 section's "NOT RUN + rationale" format was the specified output for this case.

## Per-Arm Result Summary

| Arm | Flag | gpt-5-mini pooled | gpt-4o-mini anchor | Falsifier | Signal |
|-----|------|-------------------|--------------------|-----------|--------|
| A1 | VIABILITY_CONTRACT_ENABLED=1 | 0.000 | 1.000 PASS | 1 (FAIL) | None |
| A2 | FORCED_COMMIT_STEP=6 | 0.500 (model-initiated, split-qualified) | 1.000 PASS | 1 (FAIL) | Positive |
| A3 | PARALLEL_TOOL_EXECUTION_ENABLED=1 | 0.500 | 0.500 REGRESSION | 1 (FAIL) | Regression |
| A4 | NOT RUN (D-13-01 precondition: A1 no signal) | — | — | — | — |

**Run budget:** 3/4 slots consumed. **Total live matrix runs across Phase 13: 3.**

## Commits

| Hash | Message |
|------|---------|
| b544e10 | docs(13-07): record skip-a4 decision and write closing verdict |
| cf7eb31 | docs(13-07): flip Phase 13 bookkeeping — REQUIREMENTS + ROADMAP |

## Self-Check: PASSED

- [x] `docs/decisiveness_arm_verdicts.md` contains "Closing Verdict" section
- [x] `docs/decisiveness_arm_verdicts.md` contains "INST-05"
- [x] A4 section contains "NOT RUN" + D-13-01 rationale
- [x] Closing verdict has per-arm summary table (4 arms)
- [x] Closing line is unambiguous: "No arm cleared the INST-05 falsifier bar"
- [x] Phase-14 consequence stated explicitly
- [x] `.planning/REQUIREMENTS.md` DEC-05 traceability note references docs/decisiveness_arm_verdicts.md
- [x] `.planning/ROADMAP.md` Phase 13 progress row: 7/7 Complete 2026-06-12
- [x] Total live matrix runs <= 4 (= 3)
- [x] No baselines written; no flags enabled by default
- [x] Commits b544e10 and cf7eb31 exist in git log

## Known Stubs

None — all sections filled with real data or explicit "NOT RUN" with rationale.

## Threat Flags

None — verdict doc records only run-dir paths and flag names; no API keys or credentials
committed. No new network endpoints, auth paths, or schema changes introduced.
