---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: Reasoning-Model Decisiveness
status: planning
last_updated: "2026-06-11"
last_activity: 2026-06-11
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** v2.2 Reasoning-Model Decisiveness (started 2026-06-11)
**Current phase:** 12 (not started)

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-11 for v2.2 milestone start)
See: .planning/MILESTONES.md for historical record (v1.0, v2.0, v2.1)

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Phase 12 — Decisiveness Instrumentation + Comparison Floor

## Current Position

Phase: 12 of 15 (Decisiveness Instrumentation + Comparison Floor)
Plan: — (roadmap created, planning not yet started)
Status: Ready to plan
Last activity: 2026-06-11 — v2.2 roadmap created (4 phases, 18 requirements mapped)

Progress: [░░░░░░░░░░] 0%

## Blockers / Readiness Notes

- **ANCH-01 (anthropic n=5):** requires billing top-up before the Anthropic baseline plan in Phase 12 can run. INST plans in Phase 12 have no external dependency and execute first.
- **ANCH-02 (gemini n=5):** requires Gemini quota resolution. Same pattern — INST first, ANCH when credentials available.
- **Phase 14 is CONDITIONAL:** only entered if all Phase 13 DEC arms plateau below the INST-05 falsifier bar (gpt-5-mini < 0.6 at n=5). If any arm clears, Phase 14 is skipped.
- **DEC-01 constraint:** viability-contract prompt change must not touch any text covered by the Phase-7 CI grep gate. Verify gate stays green before merging.
- **All arm judging:** Phase-10 honest gates, n=5, temp=1.0 — no exceptions.
- **Baselines only via `scripts/write_baselines.py`** (D-11-14 locked).

## Accumulated Context

### Key Decisions (v2.2)

- Decision 3 (resolved at milestone start): prod latency budget ~30s/turn; gpt-4o-mini stays anchor; reasoning models are documented alternates. Bounds investment in Phase 13/14.
- Anti-scope locked: no LangGraph replacement, no multi-agent split, no new scorers, no provider-shopping.
- Falsifier definition (INST-05): "works" iff gpt-5-mini commit rate ≥ 0.6 at n=5 AND gpt-4o-mini holds ≥ its honest baseline.
- Phase 14 conditional entry gate: all DEC arms must plateau below the INST-05 bar to enter REPLAY work.

### Deferred Items (carried from v2.1)

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Baselines | anthropic n=5 (billing exhaustion) | Blocked on top-up | Phase 11 close |
| Baselines | gemini n=5 (quota) | Blocked on quota | Phase 11 close |
| Gates | gpt-5-mini aspirational gate (≥0.6) | Logged-not-enforced | Phase 11 close |
| Decisiveness | gpt-5-mini 2/5 + deepseek 0/5 commit rate | v2.2 scope | Phase 9/11 close |

## Session Continuity

Last session: 2026-06-11
Stopped at: Roadmap created. No plans written yet.
Resume file: None
Next step: `/gsd-plan-phase 12`
