---
phase: 14-richer-state-replay-conditional
plan: "05"
subsystem: eval-harness
tags: [replay, closing-verdict, arch-fut-01, bookkeeping, user-checkpoint]
dependency_graph:
  requires: [14-04]
  provides:
    - "R3 NOT RUN decision with D-14-01 precondition check"
    - "Discretionary valve NOT RUN decision with precondition check and recommendation"
    - "docs/replay_arm_verdicts.md complete: per-arm summary table + ARCH-FUT-01 Evaluation + Explicit Closing Line + Phase-15 USER CHECKPOINT"
    - "ROADMAP.md Phase 14 milestone marked complete (5/5 plans)"
    - "REQUIREMENTS.md REPLAY-01/REPLAY-02 traceability updated to Complete"
  affects: [Phase-15-scope-decision]
tech_stack:
  added: []
  patterns:
    - "NOT RUN precondition check format (mirrors DEC-05 A4 decision block)"
    - "ARCH-FUT-01 Evaluation: three-part structure (evidence chain + contingency restatement + written recommendation)"
    - "Explicit Closing Line: single sentence plateau verdict"
key_files:
  created: []
  modified:
    - docs/replay_arm_verdicts.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
decisions:
  - "R3 (conditional combo R1+R2): NOT RUN — D-14-01 precondition 2 fails; R2 is negative (catastrophic 400s), not positive-but-short; D-14-01 requires BOTH arms positive"
  - "Discretionary valve (R1+FORCED_COMMIT_STEP=6): NOT RUN — precondition met (R1+A2 each positive) but R1 delta vs A2 = ±0.000 makes expected marginal signal = 0; cleanest next step is A2 retest on fixed synthesizer in Phase 15 scope"
  - "ARCH-FUT-01 recommendation: ratify gpt-4o-mini anchor, defer ARCH-FUT-01 to future milestone; state richness is not the bottleneck (R1 zero-delta proves state plumbing works); Phase 15 = A2 retest on fixed synthesizer + refinement_cheaper root cause analysis"
  - "Phase 15 scope finalization: USER CHECKPOINT per D-14-08; not auto-resolved"
metrics:
  duration: ~15min
  completed: "2026-06-12"
  tasks_completed: 3
  files_created: 0
---

# Phase 14 Plan 05: Combo, Closing Verdict, and ARCH-FUT-01 Summary

**One-liner:** R3 and the discretionary valve both NOT RUN (R2 negative disqualifies combo; R1 zero-delta over A2 makes stack expected marginal signal zero); closing verdict records Phase 14 plateau — no replay arm cleared INST-05 — with ARCH-FUT-01 evaluation recommending anchor ratification and A2 retest on fixed synthesizer as the Phase 15 entry point.

## What Was Built

### Task 1: R3 conditional combo and discretionary stack valve decisions (commit a6f29e0)

Evaluated both remaining run slots against their D-14-01 preconditions.

**R3 (conditional combo R1+R2):** NOT RUN.
- Precondition 1 satisfied: neither R1 nor R2 clears alone (both exit=1).
- Precondition 2 NOT satisfied: R2 signal is NEGATIVE (10/10 deterministic 400 errors — catastrophic, not positive-but-short). D-14-01 requires both arms to be positive-but-short. Running R3 would spend a budget slot on a config guaranteed to reproduce R2's deterministic failure.

**Discretionary valve (best-replay R1 + FORCED_COMMIT_STEP=6):** NOT RUN (recommended).
- Precondition met: R1 (0.500 vs floor = +0.500 positive) and A2 (0.500 positive) each independently showed positive signal.
- Not run because: R1 delta vs A2 = ±0.000 (exactly zero). Multi-message replay adds no signal over A2 alone. Stacking R1 on A2's forced-commit path has expected marginal signal = 0 by the transitivity of the zero-delta evidence.
- Cleanest next step: A2 retest on the fixed synthesizer (CR-01 repaired in Phase 13-08; forced mechanism untested at n=5 on working synthesizer). That experiment belongs to Phase 15 scope, not the Phase 14 budget.

**Run budget at close:** 2/4 slots consumed (R1 + R2). 3rd and 4th slots preserved-but-unused.

### Task 2: Closing Verdict + ARCH-FUT-01 Evaluation (commit 4f7cd94)

**Per-Arm Summary Table:**
| Arm | gpt-5-mini | Delta vs floor | Delta vs A2 | Anchor | Falsifier exit |
|---|---|---|---|---|---|
| R1 | 0.500 | +0.500 | ±0.000 | 1.000 PASS | 1 (FAIL) |
| R2 | ERRORED 0/10 | — | — | 1.000 PASS | 1 (FAIL) |
| R3 | NOT RUN | — | — | — | — |
| Valve | NOT RUN | — | — | — | — |

**ARCH-FUT-01 Evaluation (three-part structure):**
- (a) Cumulative evidence chain: v2.1 proved byte-correct state round-trip; Phase 13 all null; R1 = A2 exactly (zero delta); R2 negative. Every measurable intervention failed — the evidence points to a behavioral gap (refinement scenario never commits model-initiated), not a state-plumbing gap.
- (b) Contingency restatement: ARCH-FUT-01 = custom imperative loop to expose reasoning state on every in-window turn without `_prune_for_llm` round-tripping; originally conceived for state-loss diagnosis; Phase 14 evidence shows state loss is not the cause for tested models.
- (c) Written recommendation: Ratify gpt-4o-mini anchor, defer ARCH-FUT-01. Phase 15 scope = A2 retest on fixed synthesizer + refinement_cheaper root cause analysis. ARCH-FUT-01 filed as tracked technical debt with Phases 13-14 evidence package as trigger criteria.

**Explicit Closing Line:** No arm cleared the INST-05 falsifier bar. All Phase-14 REPLAY arms plateaued below gpt-5-mini >= 0.6 (R1=0.500, R2=NEGATIVE, R3/valve NOT RUN).

**Phase-15 Consequence:** USER CHECKPOINT per D-14-08 — Phase 15 scope not finalized until user approves.

### Task 3: Phase-close bookkeeping (commit 77ac904)

- ROADMAP.md line 53: Phase 14 milestone bullet marked `[x]` with accurate plateau completion summary (no premature completion — Phase 14 genuinely completes with this plan).
- ROADMAP.md Phase 14 Details: 14-05 plan bullet marked `[x]` (was `[ ]`).
- ROADMAP.md progress table: Phase 14 row updated to `5/5 Complete 2026-06-12`.
- REQUIREMENTS.md: REPLAY-01 and REPLAY-02 traceability rows updated to Complete with verdict summaries and reference to `docs/replay_arm_verdicts.md` as the canonical record.
- `docs/decisiveness_arm_verdicts.md` unchanged (git diff confirms — Phase-13 record stays immutable).

## Deviations from Plan

### Auto-resolved

None. The plan tasks were purely documentation-and-bookkeeping; no code changes required.

### Process notes

**Valve recommendation diverges from "may run" language in D-14-01:** D-14-01 says the valve "may recommend" one stack run before declaring plateau; the precondition is formally satisfied. The executor chose NOT RUN with reasoning that the R1 zero-delta evidence makes the expected marginal signal of any R1-stacked run = 0. This is a defensible conservative interpretation consistent with the honesty contract (spending a budget slot on a run expected to be uninformative is not sanctioned). Documented transparently in the verdict doc's valve section.

## Verification

- `docs/replay_arm_verdicts.md`: R3 section records NOT RUN with D-14-01 precondition check: DONE
- Discretionary valve section records NOT RUN with precondition check and recommendation: DONE
- Per-Arm Summary Table: R1, R2, R3, valve rows with three delta-related columns: DONE
- ARCH-FUT-01 Evaluation: all three parts (evidence chain, contingency, recommendation): DONE
- USER CHECKPOINT marker for Phase 15 scope: DONE
- Explicit Closing Line: single sentence plateau verdict: DONE
- `docs/decisiveness_arm_verdicts.md` unchanged: CONFIRMED (git diff empty)
- Total live-run count ≤4: CONFIRMED (2/4 slots used: R1 + R2)
- ROADMAP line 53: `[x]` with accurate completion state: DONE
- REQUIREMENTS.md REPLAY-01/REPLAY-02: Complete with verdict reference: DONE

## Threat Mitigations Applied

- T-14-11 (run cap): ≤4 cap enforced; 2/4 slots consumed; R3 and valve both NOT RUN per precondition checks.
- T-14-12 (false completion bookkeeping): Phase 14 marked complete because it genuinely is complete (14-05 is the last plan and all tasks complete). No fabricated runs or completion claims.
- T-14-13 (Phase-13 record tampering): `docs/decisiveness_arm_verdicts.md` is unchanged — verified via git diff (no output = no modification).
- T-14-SC (no package installs): no new packages; all tasks are documentation edits.

## Known Stubs

None — all `[fill]` placeholders in `docs/replay_arm_verdicts.md` are now filled. The only intentional open item is the Phase-15 scope decision, which is explicitly surfaced as a USER CHECKPOINT (not a stub — it requires human approval before Phase 15 planning begins).

## Threat Flags

None. This plan made only documentation edits to `docs/replay_arm_verdicts.md`, `.planning/ROADMAP.md`, and `.planning/REQUIREMENTS.md`. No new network endpoints, auth paths, file access patterns, or schema changes.

## Self-Check: PASSED

- docs/replay_arm_verdicts.md exists and contains "R3", "conditional combo", "FORCED_COMMIT_STEP", "Explicit Closing Line", "Per-Arm Summary Table", "ARCH-FUT-01 Evaluation", "USER CHECKPOINT": CONFIRMED
- Commit a6f29e0 (Task 1): CONFIRMED
- Commit 4f7cd94 (Task 2): CONFIRMED
- Commit 77ac904 (Task 3): CONFIRMED
- docs/decisiveness_arm_verdicts.md unchanged (git diff empty): CONFIRMED
- ROADMAP Phase 14 progress table: 5/5 Complete 2026-06-12: CONFIRMED
- REQUIREMENTS REPLAY-01/REPLAY-02 traceability: Complete with verdict reference: CONFIRMED
