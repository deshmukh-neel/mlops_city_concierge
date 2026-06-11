---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Reasoning-Model Compat
current_phase: 10
status: discussed
last_updated: "2026-06-11T00:56:34.740Z"
last_activity: 2026-06-10
progress:
  total_phases: 7
  completed_phases: 3
  total_plans: 17
  completed_plans: 17
  percent: 50
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** v2.1 Reasoning-Model Compat (started 2026-06-03)
**Current phase:** 10

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03 for v2.1 milestone start)
See: .planning/MILESTONES.md for historical record (v1.0, v2.0)
See: .planning/milestones/v2.0-{ROADMAP,REQUIREMENTS}.md for v2.0 archive

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Phase 10 — eval harness honesty (context gathered; ready to plan)

## Status

- [x] v1.0 Knowledge Graph shipped 2026-05-14
- [x] v2.0 Production Readiness shipped 2026-06-03 via PR #100 (main `14e01dd`)
- [x] v2.1 milestone formalized via `/gsd-new-milestone v2.1` (2026-06-03)
- [x] v2.1 requirements defined (REQUIREMENTS.md — 20 requirements, 4 categories)
- [x] v2.1 roadmap created (phases 7-10, 2026-06-04)
- [x] Phase 7: Prompt/rubric decoupling — completed 2026-06-04
- [x] Phase 8: Reasoning-state thread-through (contract + harness) — completed 2026-06-04
- [x] Phase 9: Per-provider state preservation impls (gpt-5 → DeepSeek → Claude → Gemini 3) — completed 2026-06-05, merged PR #103
- [ ] Phase 10: Eval harness honesty (EVAL-01..06; re-scoped 2026-06-10, original BASE scope moved to Phase 11)
- [ ] Phase 11: Cross-model baseline regen + matrix expansion (BASE-01..04)

## Notes

- **Empirical anchor gate for v2.1:** gpt-5-mini × `refinement_cheaper` × prod × flag-on commits 3 stops in median 5/5 runs at temp=1.0 (currently 0/1). Gates on PROV-01 in Phase 9.
- **Phase 7 falsifier:** if `gpt-5-mini × refinement_cheaper` is still 0/5 after Phase 7 ships, prompt-coupling was not the root cause — state-loss dominates and Phase 9 scope stays at full. If > 0, prompt-coupling contributed and Phase 9 scope may shrink.
- **Phase 8 harness-swap decision gate (REASON-05):** if conformance tests pass in isolation but fail through `graph.invoke`, Phase 8 surfaces this as an explicit blocker and v2.1 replans around a custom imperative loop before Phase 9 starts. This is a real architectural branch point.
- v2.0 closed with one accepted-with-notes gate (D-06-09 part 2 baseline regen — pre-Phase-6 1.0 baselines were Phase-4 fail-open false positives). Real fix is the reasoning-content thread-through landing in Phase 8/9 and honest regen in Phase 10.
- Agent driver remains locked to `openai/gpt-4o-mini` for prod until Phase 9 sub-phases ship per-provider.
- Flagged for separate hotfix (carried from v2.0): CLO-01 — over-aggressive closure detection on Mission queries.

## Resume

Next step: `/gsd-plan-phase 10` to plan Phase 10 (Eval Harness Honesty). Context is in
`.planning/phases/10-eval-harness-honesty/10-CONTEXT.md` (D-10-01..17, all areas decided).
Working branch: `gsd/phase-10-eval-harness-honesty` (off main @ e3dc6c2, post-PR #104).

## Current Position

Phase: 10 (eval-harness-honesty) — CONTEXT GATHERED
Plan: 0 of TBD. Phase 9 COMPLETE (merged PR #103); pre-phase harness fixes merged PR #104.
Status: discussed
Last activity: 2026-06-10

### Blockers

None active for Phase 9 completion. PROV-05 atomicity audit completed (`.planning/phases/09-per-provider-state-preservation-implementations/09-05-AUDIT.md`). Phase 9 PR-ready: all 5 plans shipped, atomicity audit done, gates documented as SHIPPED-WITH-GAP / SHIPPED-STRUCTURAL / PASS-WITH-FINDINGS per Wave 1/2/3 + D-06-09 precedent. Per `feedback_user_merges_prs`: do NOT run `gh pr merge` once CI is green.

**Pre-Phase-11 prerequisite (was pre-Phase-10 before the 2026-06-10 re-scope):** OpenAI embeddings quota topped up 2026-06-10 (repo secret rotated, CI green). Cloud SQL must be reachable before Phase 11 BASE-01 (re-measure anthropic n=5 + first-time gemini n=5). Phase 10 itself needs no live infra beyond ~$0.05 of probes.

**PROV-05 audit findings carried into PATTERNS.md / Phase 10:**

- Phase 9's additive-overlay pattern (matrix YAML + baseline JSON + cell-count test extended by every sub-phase) makes mid-stack single-PROV revert non-mechanical; cumulative reverse-pop is the realistic developer workflow.
- PROV-02 chore commit 3800737 has a latent test-vs-data atomicity gap (added YAML entry without updating co-tracked `test_eval_matrix.py` assertion); masked at commit time by PROV-03's later bump. Future phases adopt convention: when a sub-phase appends to a shared additive data file with a co-tracked cell-count test, the same commit updates both.

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 9 P09-05 | 60m | 2 tasks | 1 files |

## Decisions

- [Phase ?]: Phase 9 PROV-05 atomicity audit: PASS-WITH-FINDINGS — import isolation PASS; cumulative reverse-pop revert preserves v2.0 anchor; PROV-02 chore 3800737 latent test-vs-data atomicity gap documented as note (D-06-09 precedent)

## Accumulated Context

### Roadmap Evolution

- Phase 10 edited: re-scoped to Eval Harness Honesty (EVAL-01..06) after post-Phase-9 harness analysis; BASE scope moved to Phase 11
- Phase 11 added: Cross-Model Baseline Regen + Matrix Expansion (carries BASE-01..04 from original Phase 10)
