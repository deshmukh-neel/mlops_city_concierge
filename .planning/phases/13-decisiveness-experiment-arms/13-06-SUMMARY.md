---
phase: 13-decisiveness-experiment-arms
plan: "06"
subsystem: docs/eval
tags: [dec-05, arm-verdicts, arm-runs, a1, a2, a3, live-matrix, verdict-record]
dependency_graph:
  requires: [13-03-critique-scoping, 13-04-graph-arms, 13-05-arm-matrix-config]
  provides: [arm-verdict-record, a1-result, a2-result, a3-result]
  affects: [docs/decisiveness_arm_verdicts.md]
tech_stack:
  added: []
  patterns: [smoke-first-protocol, d-13-09-field-set, d-13-02-arm-protocol, d-13-04-honesty-contract]
key_files:
  created:
    - docs/decisiveness_arm_verdicts.md
  modified:
    - docs/decisiveness_arm_verdicts.md
decisions:
  - "A1 (VIABILITY_CONTRACT_ENABLED=1): gpt-5-mini 0.000 pooled, zero signal — viability contract had no effect on decisiveness"
  - "A2 (FORCED_COMMIT_STEP=6): gpt-5-mini 0.500 pooled (positive signal but below 0.6); forced mechanism never fired (forced=0 for all models); improvement is entirely model-initiated on omakase"
  - "A3 (PARALLEL_TOOL_EXECUTION_ENABLED=1): anchor regression on refinement_cheaper (0.000 vs baseline 1.000); latency comparison unmeasurable (Phase-12 floor has no step_telemetry); A3 FAIL on both criteria"
  - "Phase-12 comparison-floor run dirs predate INST-04 step_telemetry instrumentation; tool_exec_seconds=None for all Phase-12 runs; A3 latency comparison gap documented honestly"
  - "A4 conditional assessment: A1 shows no signal (0.0=floor), A2 shows positive signal (0.5>floor); A4 qualification is marginal — decision deferred to plan 13-07"
metrics:
  duration: "~110 minutes (Task 3: 3 arm runs x 30+ min each)"
  completed: "2026-06-12T09:45:00Z"
  tasks: 3
  files: 1
---

# Phase 13 Plan 06: Run Judged Arms Summary

Three live arm runs (A1 + A2 + A3) executed at n=5, temp=1.0, smoke-first protocol per
D-13-02. All three arms FAIL the INST-05 bar. A1/A2 show a progression (0.0 → 0.5 on
gpt-5-mini). A3 introduces anchor regression. Verdicts recorded verbatim per D-11-14.

## What Was Built

**Task 1: docs/decisiveness_arm_verdicts.md scaffold (commit 47ae44b)**

Created 247-line verdict doc with D-13-09 field set: INST-05 bar definition, run budget
contract, four arm sections (A1/A2/A3/A4) with placeholders, closing verdict placeholder.

**Task 2: Human-verify checkpoint (API spend approval)**

Gate required operator approval before live API spend. Approved.

**Task 3: Live arm runs A1 + A2 + A3, verdicts recorded (commit 4df4e5c)**

Three full n=5 live arm runs executed with smoke-first verification. All verdicts recorded
verbatim in `docs/decisiveness_arm_verdicts.md`.

**A1 — VIABILITY_CONTRACT_ENABLED=1:**
- Smoke dir: `eval_reports/2026-06-12T06-15-29Z` — arm_flags verified correct
- Full run dir: `eval_reports/2026-06-12T06-25-52Z` — 31 files (30 runs + summary.json)
- gpt-5-mini pooled: 0.000 (omakase=0.0, refinement=0.0); model-initiated=1/10, forced=0/10
- gpt-4o-mini anchor: 1.000; model-initiated=8/10, forced=0/10 — non-regression confirmed
- deepseek: 0.000 (informational)
- Falsifier exit code: 1 (FAIL)
- Verdict: viability contract + critique recalibration had ZERO effect on gpt-5-mini
  decisiveness. No positive signal — the 0.45 override variant is not warranted.

**A2 — FORCED_COMMIT_STEP=6:**
- Smoke dir: `eval_reports/2026-06-12T07-16-04Z` — arm_flags verified correct
- Full run dir: `eval_reports/2026-06-12T07-27-03Z` — 31 files (30 runs + summary.json)
- gpt-5-mini pooled: 0.500 (omakase=1.0, refinement=0.0); model-initiated=4/10, forced=0/10
- gpt-4o-mini anchor: 1.000; model-initiated=9/10, forced=0/10 — NO RED FLAG
- deepseek: 0.000 (informational)
- Falsifier exit code: 1 (FAIL)
- Key finding: FORCED_COMMIT_STEP=6 mechanism NEVER FIRED — forced=0 for all models across
  all 10 episodes each. The viability gate (all_slots_viable at step 6) was not satisfied.
  All A2 improvement is model-initiated. A2 shows POSITIVE SIGNAL (0.5 > 0.0 = A1).

**A3 — PARALLEL_TOOL_EXECUTION_ENABLED=1:**
- Smoke dir: `eval_reports/2026-06-12T08-21-16Z` — arm_flags verified correct; one DeepSeek
  error cell (known recurring pattern, informational only)
- Full run dir: `eval_reports/2026-06-12T08-30-52Z` — 31 files (30 runs + summary.json)
- gpt-5-mini pooled: 0.500 (omakase=1.0, refinement=0.0); model-initiated=5/10, forced=0/10
- gpt-4o-mini anchor: 0.500 (omakase=1.0, refinement=0.000); model-initiated=5/10, forced=0/10
  — ANCHOR REGRESSION (refinement_cheaper 0.000 vs baseline 1.000)
- Falsifier exit code: 1 (FAIL — anchor regression)
- Latency comparison: UNMEASURABLE — Phase-12 floor run dirs have no step_telemetry
  instrumentation (field added in Phase 13 plan 13-01). Phase-12 runs show tool_exec_seconds=None.
- A3 raw tool_exec_seconds (gpt-4o-mini): omakase mean=5.927s, refinement mean=6.471s
- Verdict: A3 FAIL on both pass criteria — scorer regression (anchor) AND latency unmeasurable.
  Parallel tool execution introduced real behavioral regression on refinement_cheaper.

## Deviations from Plan

### Auto-documented Issues

**1. [Rule 1 - Discovery] Phase-12 comparison floor has no step_telemetry**
- **Found during:** Task 3 A3 latency analysis
- **Issue:** Phase-12 comparison-floor run dirs (`2026-06-11T*`) have `step_telemetry: None`
  because the INST-04 instrumentation was added in Phase 13 (plan 13-01). The plan's A3
  latency table required filling Phase-12 baseline values, but no valid data exists.
- **Fix:** Documented honestly as "UNMEASURABLE" in the latency table and closing verdict.
  Raw A3 arm values recorded for future reference. No fabrication.
- **Files modified:** `docs/decisiveness_arm_verdicts.md`

**2. [Rule 1 - Discovery] A3 anchor regression on refinement_cheaper**
- **Found during:** Task 3 A3 falsifier grading
- **Issue:** gpt-4o-mini dropped from 1.000 (baseline) to 0.000 (median) on refinement_cheaper
  under PARALLEL_TOOL_EXECUTION_ENABLED=1. 3/5 runs had committed_itinerary_rate=0.0.
- **Fix:** Documented verbatim as anchor regression in verdict and closing verdict. A3 FAIL
  verdict recorded on scorer regression grounds, independently of latency.
- **Files modified:** `docs/decisiveness_arm_verdicts.md`

## Arm Run Summary

| Arm | Flag | gpt-5-mini pooled | gpt-4o-mini (anchor) | Falsifier | Signal |
|-----|------|-------------------|----------------------|-----------|--------|
| A1 | VIABILITY_CONTRACT_ENABLED=1 | 0.000 | 1.000 (PASS) | 1 (FAIL) | None |
| A2 | FORCED_COMMIT_STEP=6 | 0.500 | 1.000 (PASS) | 1 (FAIL) | Positive |
| A3 | PARALLEL_TOOL_EXECUTION_ENABLED=1 | 0.500 | 0.500 (REGRESSION) | 1 (FAIL) | Regression |

A4 qualification: A1 no signal (0.0), A2 positive signal (0.5) → ambiguous; deferred to 13-07.

## Commits

| Hash | Message |
|------|---------|
| 47ae44b | docs(13-06): scaffold decisiveness_arm_verdicts.md with D-13-09 field set |
| 4df4e5c | feat(13-06): run judged arms A1/A2/A3 and record verdicts |

## Self-Check: PASSED

- [x] `docs/decisiveness_arm_verdicts.md` exists and all arm sections filled
- [x] A1 smoke dir `eval_reports/2026-06-12T06-15-29Z` exists (confirmed 6 files)
- [x] A1 full run dir `eval_reports/2026-06-12T06-25-52Z` exists (confirmed 31 files)
- [x] A2 smoke dir `eval_reports/2026-06-12T07-16-04Z` exists (confirmed 7 files)
- [x] A2 full run dir `eval_reports/2026-06-12T07-27-03Z` exists (confirmed 31 files)
- [x] A3 smoke dir `eval_reports/2026-06-12T08-21-16Z` exists (confirmed 7 files)
- [x] A3 full run dir `eval_reports/2026-06-12T08-30-52Z` exists (confirmed 31 files)
- [x] Falsifier output pasted verbatim for all 3 arms (no fabrication)
- [x] Forced commits documented: forced=0 for ALL models in ALL arms
- [x] Anchor regression documented for A3
- [x] Phase-12 floor telemetry gap documented honestly
- [x] Commits 47ae44b and 4df4e5c verified in git log
- [x] Run budget: 3/4 slots consumed (A1+A2+A3); A4 slot preserved for 13-07 decision

## Known Stubs

None — all arm sections filled with real verbatim data. A4 section and closing verdict
"decision" line intentionally defer to plan 13-07 per the verdicts.md spec.

## Threat Flags

None — verdict doc records only run-dir paths and flag names; no API keys or credentials
committed. eval_reports/ dirs contain only local run output, not baselines (D-11-14 honored).
