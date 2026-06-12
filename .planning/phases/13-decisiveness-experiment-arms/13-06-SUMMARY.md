---
phase: 13-decisiveness-experiment-arms
plan: "06"
subsystem: docs/eval
tags: [dec-05, arm-verdicts, verdict-scaffold, checkpoint]
dependency_graph:
  requires: [13-03-critique-scoping, 13-04-graph-arms, 13-05-arm-matrix-config]
  provides: [arm-verdict-skeleton, live-run-checkpoint]
  affects: [docs/decisiveness_arm_verdicts.md]
tech_stack:
  added: []
  patterns: [doc-as-contract, d-13-09-field-set, smoke-first-protocol]
key_files:
  created:
    - docs/decisiveness_arm_verdicts.md
  modified: []
decisions:
  - "docs/decisiveness_arm_verdicts.md scaffolded with D-13-09 field set: INST-05 bar, run budget contract, four arm sections (A1/A2/A3/A4), and closing verdict placeholder"
  - "A2 model-initiated vs forced split format pinned per D-13-04; anchor red-flag rule explicit"
  - "A3 judged on latency reduction + zero scorer regression, NOT commit rate per D-13-01"
  - "A4 conditional on neither A1/A2 clearing alone but both showing positive signal; consumes 4th (final) live run slot"
metrics:
  duration: "~5 minutes"
  completed: "2026-06-12"
  tasks: 1
  files: 1
---

# Phase 13 Plan 06: Run Judged Arms Summary

Verdict doc scaffold created with D-13-09 field set; live arm runs gated on human-verify checkpoint (Task 2 — operator approval for API spend).

## What Was Built

**Task 1: docs/decisiveness_arm_verdicts.md scaffold**

Created `docs/decisiveness_arm_verdicts.md` with 247 lines covering:

- **Header:** Role as DEC-05 record, Phase-14 conditional entry gate input, and Phase-15
  promotion input. INST-05 falsifier definition: gpt-5-mini pooled committed_itinerary_rate
  >= 0.6 AND gpt-4o-mini anchor non-regression AND falsifier exit code 0.
- **Run Budget Contract:** Hard cap 4 full live matrix runs per D-13-01; smoke n=1 before
  each full spend; partial results labeled PARTIAL and never written to baselines (D-11-14).
- **A1 section:** Flag config `VIABILITY_CONTRACT_ENABLED=1`, override unset per
  `docs/decisiveness_dec03_decision.md` (first run isolates scoping effect). D-13-09 placeholders
  for smoke + full run dirs, arm_flags verification, per-model results table, falsifier output,
  and closing verdict.
- **A2 section:** Flag config `FORCED_COMMIT_STEP=6`. Explicit model-initiated vs forced split
  format (`commit_rate X.X (model-initiated M/N, forced F/N)`). Anchor red-flag rule per
  D-13-04(c): gpt-4o-mini commits before step 6 on its own — ANY anchor behavior change is a
  red flag. D-13-09 placeholders for all fields.
- **A3 section:** Flag config `PARALLEL_TOOL_EXECUTION_ENABLED=1`. Explicitly states A3 judged
  on latency reduction + zero scorer regression, NOT commit rate (D-13-01). Latency analysis
  table with INST-04 step_telemetry source documented.
- **A4 section:** Marked CONDITIONAL — run only if neither A1 nor A2 clears alone but both show
  positive signal; consumes the 4th (final) run slot; decision deferred to plan 13-07.
- **Closing verdict section:** Placeholder for plan 13-07 to fill.

## Deviations from Plan

None — Task 1 executed exactly as written.

## Checkpoint Status

Task 2 is a `checkpoint:human-verify` gate — awaiting operator approval for live API spend
before proceeding to Task 3 (running A1/A2/A3 arms at n=5).

## Commits

| Hash | Message |
|------|---------|
| 47ae44b | docs(13-06): scaffold decisiveness_arm_verdicts.md with D-13-09 field set |

## Self-Check: PASSED

- [x] `docs/decisiveness_arm_verdicts.md` exists and is 247 lines (>= 60 required)
- [x] Contains `VIABILITY_CONTRACT_ENABLED` (A1 flag)
- [x] Contains `FORCED_COMMIT_STEP` (A2 flag)
- [x] Contains `PARALLEL_TOOL_EXECUTION_ENABLED` (A3 flag)
- [x] Contains `model-initiated` (A2 split format)
- [x] Contains `CONDITIONAL` (A4 conditionality)
- [x] Contains `>= 0.6` INST-05 bar
- [x] Contains anchor red-flag rule
- [x] Contains `latency reduction` (A3 judging criterion)
- [x] Contains `write_baselines` (no-partial-baselines rule reference)
- [x] Automated verify check passed: `OK`
- [x] Commit 47ae44b verified in git log

## Known Stubs

All arm sections are intentional placeholders — they will be filled by Task 3 after human
approval in Task 2. The doc skeleton is complete and structurally correct per D-13-09.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary crossings. The verdict doc
records only run-dir paths and flag names, not API keys (T-13-06-04 mitigated by design).
