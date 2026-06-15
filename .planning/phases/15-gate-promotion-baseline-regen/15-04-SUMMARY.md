---
phase: 15-gate-promotion-baseline-regen
plan: "04"
subsystem: docs/planning-bookkeeping
tags: [milestone-close, promotion-decision, promo-01, promo-02, promo-03, inst-05, v2.2]
dependency_graph:
  requires: [15-03-SUMMARY.md]
  provides: [docs/promotion_decision.md finalized, PROMO-01/02/03 Complete, v2.2 milestone closed]
  affects: [.planning/REQUIREMENTS.md, .planning/ROADMAP.md, .planning/STATE.md]
tech_stack:
  added: []
  patterns: [milestone-audit-anchor, cross-link-immutable-records, honest-null-verdict]
key_files:
  created: []
  modified:
    - docs/promotion_decision.md
    - .planning/REQUIREMENTS.md (gitignored — not committed)
    - .planning/ROADMAP.md (gitignored — not committed)
    - .planning/STATE.md (gitignored — not committed)
decisions:
  - "INST-05 honest null result (case a): no arm cleared 0.600 bar across Phases 13/14/15"
  - "gpt-4o-mini anchor ratified: omakase flag-off median=1.000, gate >= 0.8 holds"
  - "ARCH-FUT-01 deferred as tracked technical debt with Phase 13-14 evidence chain as trigger criteria"
  - "Prod-default FORCED_COMMIT_STEP=6 flip flagged, NOT implemented (D-15-07)"
  - "Anchor provenance correction: prior refinement_cheaper 1.000 baseline was REFINEMENT_STRUCTURED_PLAN_ENABLED=true arm artifact; re-baselined to honest 0.000 flag-off"
metrics:
  duration: "~10 minutes"
  completed: "2026-06-15"
  tasks_completed: 2
  tasks_total: 2
---

# Phase 15 Plan 04: Milestone Close — Promotion Decision Finalization Summary

**One-liner:** Finalized docs/promotion_decision.md as the v2.2 milestone-audit anchor (D-15-03) with INST-05 honest-null verdict (case a), cross-links to both immutable verdict docs, anchor-provenance correction, and PROMO-01/02/03 bookkeeping flipped to Complete.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Finalize + commit docs/promotion_decision.md | 57c5f4c | docs/promotion_decision.md |
| 2 | Milestone-close bookkeeping (ROADMAP/REQUIREMENTS/STATE) | skipped (.planning gitignored) | .planning/REQUIREMENTS.md, .planning/ROADMAP.md, .planning/STATE.md |

## What Was Built

**Task 1 — docs/promotion_decision.md finalized:**

The document was restructured to serve as the D-15-03 milestone-audit anchor:
- Added role header identifying the document's function (single record a reviewer reads to understand how v2.2 closed)
- Replaced the ad-hoc "Cross-References" block with a formal `## Inputs (immutable cross-links)` section markdown-linking both `docs/decisiveness_arm_verdicts.md` (Phase-13 DEC-05 record) and `docs/replay_arm_verdicts.md` (Phase-14 REPLAY-05 record) with explicit one-line notes that neither is appended to or modified
- Added gate-threshold clarification: enforced values live in `configs/eval_gates.yaml` as source of truth; measured rates in this doc are observational
- Updated status from "In progress" to "COMPLETE — all four Phase-15 plans executed; milestone closed 2026-06-15"
- Added `## Milestone Closing Summary` with:
  - **INST-05 Verdict: CASE (a) — Honest Null Result** (data-dependent, driven by measured A2 retest 0.500 pooled < 0.600 floor)
  - Locked-regardless statements: anchor ratified, ARCH-FUT-01 deferred, prod-default flip flagged-not-implemented, enforced vs logged table
  - Anchor provenance correction (D-15-07): prior 1.000 baseline was flag-ON arm artifact
  - Phase-15 outcome summary table

**Task 2 — Milestone-close bookkeeping:**

- **REQUIREMENTS.md:** Three PROMO requirement checkboxes flipped from `[ ]` to `[x]`; traceability table rows flipped from Pending to Complete with one-line outcome notes per requirement; last-updated footer updated to 2026-06-15 noting v2.2 closed.
- **ROADMAP.md:** v2.2 milestone entry updated from 🚧 to ✅ with closure note; Phase 15 section updated with outcome paragraph and Phase 15 checkbox marked; Progress table row updated to 4/4 Complete 2026-06-15; last-updated footer updated.
- **STATE.md:** Frontmatter status changed from `executing` to `complete`; completed_phases 3→4; completed_plans 23→24; percent 75→100; Current Position updated to COMPLETE; Decisions-log entry appended summarizing Phase-15 measured outcome.

**Preflight confirmed:** `git status --short` showed only the three intended `.planning/` files modified (NOT docs/promotion_decision.md, already committed in Task 1). Both closed verdict docs unchanged (git diff --quiet exits 0). Bookkeeping commit skipped because `.planning/` is gitignored per project convention — the SDK `skipped_gitignored` intentional path, per plan protocol.

## Deviations from Plan

None — plan executed exactly as written. The `.planning/` gitignore is the expected production behavior per the established convention (bookkeeping files tracked internally, not in git history).

## Verification Results

- `grep -q "decisiveness_arm_verdicts.md" docs/promotion_decision.md` — PASS
- `grep -q "replay_arm_verdicts.md" docs/promotion_decision.md` — PASS
- `grep -qi "INST-05" docs/promotion_decision.md` — PASS
- `git diff --quiet -- docs/decisiveness_arm_verdicts.md docs/replay_arm_verdicts.md` — PASS (exit 0; both immutable)
- `wc -l docs/promotion_decision.md` — 868 lines (>= 80 minimum)
- `grep -Eq '\|\s*PROMO-01\s*\|\s*Phase 15\s*\|\s*Complete\s*' .planning/REQUIREMENTS.md` — PASS
- `grep -Eq '\|\s*PROMO-02\s*\|\s*Phase 15\s*\|\s*Complete\s*' .planning/REQUIREMENTS.md` — PASS
- `grep -Eq '\|\s*PROMO-03\s*\|\s*Phase 15\s*\|\s*Complete\s*' .planning/REQUIREMENTS.md` — PASS

## Self-Check: PASSED

- docs/promotion_decision.md exists and is committed at 57c5f4c (confirmed via `git log --oneline -1`)
- Both verdict docs unchanged (git diff --quiet exits 0)
- PROMO-01/02/03 Complete in REQUIREMENTS.md (regex verified)
- ROADMAP Phase 15 row shows 4/4 Complete 2026-06-15
- STATE.md status=complete, 100%
- Task 2 bookkeeping skipped `.planning gitignored` — intentional per project convention
- Zero new eval_reports/ run dirs created
