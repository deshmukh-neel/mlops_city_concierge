---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: Reasoning-Model Decisiveness
current_phase: 14
status: "Phase 13 shipped — PR #108"
last_updated: "2026-06-12T18:21:02.481Z"
last_activity: 2026-06-12
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 15
  completed_plans: 15
  percent: 50
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** v2.2 Reasoning-Model Decisiveness (started 2026-06-11)
**Current phase:** 14

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-11 for v2.2 milestone start)
See: .planning/MILESTONES.md for historical record (v1.0, v2.0, v2.1)

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Phase 14 — richer state replay (conditional)

## Current Position

Phase: 13 (decisiveness-experiment-arms) — EXECUTING
Plan: Not started
Status: Phase 13 shipped — PR #108
Last activity: 2026-06-12

Progress: [██████████] 100%

## Blockers / Readiness Notes

- **ANCH-01 (anthropic n=5): DEFERRED out of v2.2 scope** (2026-06-11, user declined billing top-up). Anthropic stays logged-not-gated with its `_DEFERRED_BASELINE_CELLS` entry intact; revisit when budget allows.
- **ANCH-02 (gemini n=5):** requires Gemini quota resolution. INST plans in Phase 12 have no external dependency and execute first; the gemini baseline plan runs when quota is available.
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
| Baselines | anthropic n=5 (billing exhaustion) | Deferred out of v2.2 (no top-up — user decision) | Phase 11 close; extended 2026-06-11 |
| Baselines | gemini n=5 (quota) | Blocked on quota | Phase 11 close |
| Gates | gpt-5-mini aspirational gate (≥0.6) | Logged-not-enforced | Phase 11 close |
| Decisiveness | gpt-5-mini 2/5 + deepseek 0/5 commit rate | v2.2 scope | Phase 9/11 close |

## Session Continuity

Last session: 2026-06-12T17:43:49.527Z
Stopped at: Completed 13-06-run-judged-arms-PLAN.md
Resume file: None
Next step: `/gsd-plan-phase 12`

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 12 P01 | 3min | 3 tasks | 3 files |
| Phase 12 P03 | 5min | 3 tasks | 3 files |
| Phase 12 P04 | 3min | 3 tasks | 5 files |
| Phase 12 P05 | 5m | 2 tasks | 2 files |
| Phase 13 P01 | 10min | 3 tasks | 5 files |
| Phase 13 P02 | 2min | 2 tasks | 2 files |
| Phase 13 P03 | 4min | 2 tasks | 3 files |
| Phase 13 P04 | 25min | 3 tasks | 3 files |
| Phase 13 P05 | 7min | 3 tasks | 4 files |
| Phase Phase 13 P06 P110min | 3 tasks | 1 files tasks | - files |
| Phase 13 P10 | 5m | 3 tasks | 7 files |

## Decisions

- [Phase ?]: D-12-09 honored: gemini n=5 baseline deferred at user budget decision (2026-06-11); joins anthropic as second deferred cell; comparison floor = matrix minus BOTH anthropic AND gemini
- [Phase ?]: DEC-03 threshold code default stays 0.55; LOW_SIMILARITY_THRESHOLD_OVERRIDE env var is the A1 experiment knob; first A1 run keeps it unset
- [Phase ?]: low_similarity hint suppressed once all_slots_viable=True (DEC-03); only fires in pre-candidate steps; co-tuned with DEC-01 via shared VIABILITY_CONTRACT_ENABLED flag
- [Phase ?]: Arm config scenario IDs match baseline stems; split reader uses provider_slug prefix stripping (13-05)
- [Phase ?]: A1 VIABILITY_CONTRACT_ENABLED: gpt-5-mini 0.000 pooled — zero signal
- [Phase ?]: A2 FORCED_COMMIT_STEP=6: gpt-5-mini 0.500 pooled (positive signal); forced mechanism never fired
- [Phase ?]: A3 PARALLEL_TOOL_EXECUTION: anchor regression refinement_cheaper 0.000 vs 1.000 baseline; latency unmeasurable (Phase-12 no telemetry)
- [Phase ?]: A4 conditional: A1 no signal (0.0), A2 positive (0.5); A4 qualification deferred to plan 13-07
