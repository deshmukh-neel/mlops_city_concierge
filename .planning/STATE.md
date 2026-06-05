---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Reasoning-Model Compat
current_phase: 9
status: executing
last_updated: "2026-06-05T22:06:14.970Z"
last_activity: 2026-06-05
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 17
  completed_plans: 17
  percent: 47
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
Plan: 4 of 5 complete (09-01 SHIPPED-WITH-GAP, 09-02 SHIPPED-WITH-GAP, 09-03 SHIPPED-WITH-GAP, 09-04 SHIPPED-STRUCTURAL). Next: Plan 09-05 (revertability-audit), Wave 5.
Status: Ready to execute
Last activity: 2026-06-05

### Blockers

None active for next-plan dispatch. Plan 09-04's PROV-04 charter (state-preservation) shipped structurally per D-09-08 no-merge-gate framing — GeminiAdapter handles BOTH the Phase 8 synthetic fixture shape AND the live-probe-confirmed lcgg 4.x wire shape (`additional_kwargs["__gemini_function_call_thought_signatures__"]: dict[tc_id, base64_str]`); 13 unit tests + 1 conformance sibling pass. ADAPTERS registry consolidated to explicit literal at one site (D-09-07 Option B) — all 4 reasoning providers (openai/deepseek/anthropic/gemini) wired off NoOp; kimi (PROV-FUT-02 library-blocked) + scripted (CI-only) stay on NoOp. Empirical n=5 baseline measurement deferred to Phase 10 BASE-01 per user-approved Option B — same OpenAI embeddings quota blocker from Plan 09-03 still in effect (`semantic_search` 429s on every matrix cell regardless of LLM provider). See `.planning/phases/09-per-provider-state-preservation-implementations/09-04-SUMMARY.md` for live-probe finding + asymmetry visualization + Phase 10 BASE-01 carry-forward (re-measure anthropic n=5 + first-time gemini n=5). **Pre-Phase-10 prerequisite:** OpenAI quota top-up before Phase 10 BASE-01.
