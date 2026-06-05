---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Reasoning-Model Compat
current_phase: 9
status: "Phase 9 — Plan 2 of 5 SHIPPED-WITH-GAP; Wave 3 unblocked"
last_updated: "2026-06-05T19:08:17.347Z"
last_activity: "2026-06-05 -- Plan 09-02 SHIPPED-WITH-GAP per Wave 1 D-06-09 precedent; PROV-02 complete; Wave 3 (09-03 Anthropic Claude) unblocked"
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 17
  completed_plans: 15
  percent: 44
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** v2.1 Reasoning-Model Compat (started 2026-06-03)
**Current phase:** 9

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03 for v2.1 milestone start)
See: .planning/MILESTONES.md for historical record (v1.0, v2.0)
See: .planning/milestones/v2.0-{ROADMAP,REQUIREMENTS}.md for v2.0 archive

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Phase 9 — per provider state preservation implementations

## Status

- [x] v1.0 Knowledge Graph shipped 2026-05-14
- [x] v2.0 Production Readiness shipped 2026-06-03 via PR #100 (main `14e01dd`)
- [x] v2.1 milestone formalized via `/gsd-new-milestone v2.1` (2026-06-03)
- [x] v2.1 requirements defined (REQUIREMENTS.md — 20 requirements, 4 categories)
- [x] v2.1 roadmap created (phases 7-10, 2026-06-04)
- [ ] Phase 7: Prompt/rubric decoupling
- [ ] Phase 8: Reasoning-state thread-through (contract + harness)
- [ ] Phase 9: Per-provider state preservation impls (gpt-5 → DeepSeek → Claude → Gemini 3)
- [ ] Phase 10: Cross-model baseline regen + matrix expansion

## Notes

- **Empirical anchor gate for v2.1:** gpt-5-mini × `refinement_cheaper` × prod × flag-on commits 3 stops in median 5/5 runs at temp=1.0 (currently 0/1). Gates on PROV-01 in Phase 9.
- **Phase 7 falsifier:** if `gpt-5-mini × refinement_cheaper` is still 0/5 after Phase 7 ships, prompt-coupling was not the root cause — state-loss dominates and Phase 9 scope stays at full. If > 0, prompt-coupling contributed and Phase 9 scope may shrink.
- **Phase 8 harness-swap decision gate (REASON-05):** if conformance tests pass in isolation but fail through `graph.invoke`, Phase 8 surfaces this as an explicit blocker and v2.1 replans around a custom imperative loop before Phase 9 starts. This is a real architectural branch point.
- v2.0 closed with one accepted-with-notes gate (D-06-09 part 2 baseline regen — pre-Phase-6 1.0 baselines were Phase-4 fail-open false positives). Real fix is the reasoning-content thread-through landing in Phase 8/9 and honest regen in Phase 10.
- Agent driver remains locked to `openai/gpt-4o-mini` for prod until Phase 9 sub-phases ship per-provider.
- Flagged for separate hotfix (carried from v2.0): CLO-01 — over-aggressive closure detection on Mission queries.

## Resume

Next step: `/gsd-plan-phase 7` to plan Phase 7 (Prompt/Rubric Decoupling).

## Current Position

Phase: 09 (per-provider-state-preservation-implementations) — EXECUTING
Plan: 2 of 5 complete (09-01 SHIPPED-WITH-GAP, 09-02 SHIPPED-WITH-GAP). Next: Plan 09-03 (anthropic-claude-wiring), Wave 3.
Status: Phase 09 — Plan 2 of 5 complete (SHIPPED-WITH-GAP). Next: Plan 09-03 (anthropic-claude-wiring), Wave 3.
Last activity: 2026-06-05 -- Plan 09-02 SHIPPED-WITH-GAP per Wave 1 D-06-09 precedent; PROV-02 complete; Wave 3 (09-03 Anthropic Claude) unblocked

### Blockers

None active. Plan 09-02's PROV-02 lower-bar gate (`deepseek-reasoner × refinement_minimal_edit` median ≥ 0.6) fails at 0.0 — accepted-with-notes per Wave 1 D-06-09 precedent: adapter charter delivered (5 unit + 1 conformance test pass; reasoning_content round-trips through `graph.ainvoke`); 5/5 runs hit the documented `low_similarity` critique-loop signature; decisiveness gap is downstream of state preservation per `project_deepseek_decisiveness_gap` + `project_reasoning_models_break_agent_loop` memories and carried forward to v2.1 phases 2-4. See `.planning/phases/09-per-provider-state-preservation-implementations/09-02-SUMMARY.md` "Ship rationale" section. Plan 09-01's D-09-02 BLOCKER stays CLOSED: SHIPPED-WITH-GAP per Option 3.
