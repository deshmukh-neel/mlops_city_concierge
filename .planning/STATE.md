---
gsd_state_version: 1.0
milestone: none
milestone_name: between milestones
current_phase: null
status: ready_for_next_milestone
last_updated: "2026-06-03T18:50:00.000Z"
last_activity: 2026-06-03 -- v2.0 milestone shipped (PR #100 -> main 14e01dd); MILESTONES.md created; v2.1 reasoning-model compat drafted in ROADMAP and PROJECT
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** none (v2.0 shipped; v2.1 drafted in ROADMAP but not started)
**Current phase:** none

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03 after v2.0 milestone)
See: .planning/MILESTONES.md for historical record
See: .planning/milestones/v2.0-ROADMAP.md + v2.0-REQUIREMENTS.md for v2.0 archive

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Between milestones. v2.1 Reasoning-Model Compat is drafted; run `/gsd-new-milestone v2.1` to formalize when ready.

## Status

- [x] v1.0 Knowledge Graph shipped 2026-05-14
- [x] v2.0 Production Readiness shipped 2026-06-03 via PR #100 (main `14e01dd`)
- [x] v2.0 archives written to `.planning/milestones/v2.0-{ROADMAP,REQUIREMENTS}.md`
- [x] ROADMAP.md collapsed; PROJECT.md evolved; MILESTONES.md created
- [ ] v2.1 milestone formalized (`/gsd-new-milestone v2.1`)

## Notes

- v2.0 closed with one accepted-with-notes gate (D-06-09 part 2 baseline regen — pre-Phase-6 1.0 baselines were Phase-4 fail-open false positives now surfaced by an actually-committing agent). Real fix is reasoning-content thread-through, scoped to v2.1.
- Agent driver locked to `openai/gpt-4o-mini` for prod until v2.1 ships. gpt-5-mini / gpt-5.4-mini / DeepSeek reasoner / Claude / Gemini 3 all fail on tool-loop tasks because `_prune_for_llm` drops `reasoning_content` across turns.
- Flagged for v2.1 or separate hotfix: CLO-01 — over-aggressive closure detection on Mission queries (likely hours-data drift or `_per_stop_closure_status` overly cautious; pre-Phase-3 D-07 sanity check confirmed `place_is_open` is correct).

## Resume

When ready to start v2.1:
1. `/gsd-new-milestone v2.1` to scope reasoning-model compat (4 phases drafted in ROADMAP).
2. Empirical anchor gate for the milestone: gpt-5-mini × `refinement_cheaper` commits 3 stops in median 5/5 runs at temp=1.0.
