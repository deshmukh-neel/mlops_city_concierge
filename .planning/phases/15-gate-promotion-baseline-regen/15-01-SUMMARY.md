---
phase: 15-gate-promotion-baseline-regen
plan: "01"
subsystem: eval-diagnostic
tags: [diagnostic, refinement, root-cause, zero-spend, phase-15]
dependency_graph:
  requires: [eval_reports/2026-06-12T07-27-03Z]
  provides: [docs/promotion_decision.md]
  affects: [15-02-PLAN.md (A2 retest), 15-03-PLAN.md (baseline regen)]
tech_stack:
  added: []
  patterns: [read-existing-json-telemetry, zero-spend-diagnostic]
key_files:
  created: [docs/promotion_decision.md]
  modified: []
decisions:
  - "D-15-08 DEFERRED: typed-slot viability gap on refinement_cheaper is structural (all_slots_viable never True for 3-type request), not a one-line code bug; ARCH-FUT-01 not re-opened"
metrics:
  duration: 30min
  completed: 2026-06-14
  tasks: 2
  files_changed: 1
---

# Phase 15 Plan 01: Root-Cause Diagnostic + D-15-08 Disposition Summary

**One-liner:** Typed-slot viability gap on refinement_cheaper — Hayes Valley 3-type search (Restaurant + Cocktail Bar + Dessert Shop) never accumulates viable candidates for all slots (max viable_candidates_per_step = 1 across all 5 runs), so all_slots_viable is always False and the forced-commit gate cannot fire; DEFERRED.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Diagnose refinement_cheaper=0.000 from existing run JSONs | ec4343c | docs/promotion_decision.md (created) |
| 2 | Decide and record D-15-08 fix disposition | ec4343c | docs/promotion_decision.md (disposition section) |

## What Was Done

### Task 1: Root-Cause Diagnostic

Read all 10 gpt-5-mini run JSONs from `eval_reports/2026-06-12T07-27-03Z` (5 omakase + 5
refinement_cheaper). Key telemetry extracted:

**gpt-5-mini refinement_cheaper actual values:**
- `viable_candidates_per_step`: max=1 across all 5 runs (representing only 1 typed slot covered)
- `rule8_met_per_step`: all-False in all 5 runs (cumulative typed coverage never satisfied)
- `first_commit_call_step`: None in 4/5 runs; step=4 in run-3 only
- `forced_commit_step` (actual): None in all 5 runs (forced path never fired)
- `commit_forced`: False in all 5 runs
- `committed_itinerary_rate` median: 0.000 ([0,0,0,1,0])

**gpt-5-mini omakase actual values (contrast):**
- `viable_candidates_per_step`: reaches 2-4 per step (untyped path, no type constraints)
- `rule8_met_per_step`: flips to True at step 5 in run-0 ([F,F,F,F,F,T,T,T])
- `committed_itinerary_rate` median: 1.000 ([0,0,1,1,1])
- Pooled (median-of-medians): 0.500 — matches `docs/decisiveness_arm_verdicts.md` exactly

**Root cause — two structural reasons:**
1. The refinement_cheaper scenario uses `requested_primary_types: ["Restaurant", "Cocktail Bar",
   "Dessert Shop"]`, triggering the TYPED path in `app/agent/viability.py::all_slots_viable`.
   This requires at least one distinct viable place_id (cosine >= 0.55 from semantic_search)
   per type. The cumulative viable count maxes at 1 in all 5 runs — only one type is ever
   covered. The `nearby` tool hardcodes similarity=0.0 and cannot contribute (WR-01).
2. The omakase scenario has no `requested_primary_types` — uses the UNTYPED path needing only
   N=3 distinct viable place_ids cumulatively, which is achievable in the Mission corpus.

**CR-01 over-determination confirmed:** In this run dir, the forced-commit synthesizer was also
broken (Plan 13-08 fixed it). But even with the fix, `all_slots_viable=False` for ALL
refinement_cheaper runs at step 6 — the gate would have blocked the forced path regardless.

**Zero-spend verification:** eval_reports/ before=43 dirs, after=43 dirs (diff empty).

### Task 2: D-15-08 Fix Disposition

Applied the D-15-08 trivial-fix-or-defer boundary from `15-CONTEXT.md`.

**Disposition: DEFERRED**

The finding is a retrieval/scenario-coverage property (Hayes Valley typed-slot candidates
scarce at cosine >= 0.55 in semantic_search scratch), NOT a one-line gate bug. Possible fixes
would require loosening the viability gate (changes behavior for all scenarios, not
one-line) or expanding the corpus/tuning search queries (multi-file/non-trivial). None meets
the one-flag/one-line/low-risk bar.

No changes to `app/agent/graph.py` or `tests/unit/test_graph_forced_commit.py`.
`git diff --stat app/agent/graph.py tests/unit/test_graph_forced_commit.py` is empty.

ARCH-FUT-01 is not re-opened. Prod default unchanged (FORCED_COMMIT_STEP=0).

Baseline regen remains sequenced last per D-15-05 (no app code changes in this plan, so
`check_baselines_fresh.py` watch-set not tripped).

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — `docs/promotion_decision.md` has explicit "Sections Pending (Plans 02 and 03)"
clearly labeled as incomplete and gated on future plan execution. The core root-cause section
is complete with actual telemetry values.

## Threat Flags

None — this plan reads existing JSON files and writes a docs file only. No new network
endpoints, auth paths, file access patterns, or schema changes.

## Self-Check: PASSED

- `docs/promotion_decision.md` exists: confirmed (295 lines)
- Contains "rule8_met_per_step": confirmed
- Contains "viable_candidates_per_step": confirmed
- Contains "Fix Disposition": confirmed
- Contains "DEFERRED": confirmed
- `git diff --stat app/agent/graph.py`: empty (no changes)
- No new eval_reports/ run dir: confirmed (before/after listing diff is empty)
- Task commit ec4343c exists: confirmed
