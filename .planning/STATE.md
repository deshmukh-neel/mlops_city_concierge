---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: Reasoning-Model Decisiveness
current_phase: 15
status: Awaiting next milestone
last_updated: "2026-06-15T06:22:04.812Z"
last_activity: 2026-06-15 — Milestone v2.2 completed and archived
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 24
  completed_plans: 24
  percent: 100
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** None — v2.2 shipped 2026-06-15; awaiting next milestone
**Last shipped:** v2.2 Reasoning-Model Decisiveness (Phases 12-15)

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-15 after v2.2 milestone)
See: .planning/MILESTONES.md for historical record (v1.0, v2.0, v2.1, v2.2)

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Planning next milestone (`/gsd-new-milestone`)

## Current Position

Phase: Milestone v2.2 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-06-15 — Milestone v2.2 completed and archived

## Blockers / Readiness Notes

No active blockers — v2.2 is closed. Open threads to weigh when scoping the next milestone:

- **Reasoning-model decisiveness is confirmed architectural.** v2.2 ran six interventions (four DEC arms + two REPLAY arms); none cleared the INST-05 falsifier bar (best 0.500 vs 0.6). The deferred contingency is ARCH-FUT-01 (replace the LangGraph loop with a custom imperative loop); trigger = the Phases 13-14 evidence chain. Don't re-derive — the verdict docs settle it.
- **gpt-4o-mini is the ratified prod anchor** (omakase median 1.000 flag-off, gate ≥ 0.8 enforced). gpt-5-mini stays logged-not-gated.
- **Standing invariants (locked, carry forward):** baselines only via `scripts/write_baselines.py` (D-11-14); all eval judging at n=5, temp=1.0 against Phase-10 honest gates.

## Accumulated Context

### Key Decisions (v2.2 — full log in PROJECT.md Key Decisions)

- Decision 3: prod latency budget ~30s/turn; gpt-4o-mini stays anchor; reasoning models are documented alternates.
- INST-05 falsifier: "works" iff gpt-5-mini commit rate ≥ 0.6 at n=5 AND gpt-4o-mini holds ≥ its honest baseline. No goalpost-moving — every arm judged against it.
- v2.2 closed as an honest null: no arm cleared the bar; anchor re-ratified; ARCH-FUT-01 deferred as tracked debt (user-ratified D-14-08). Canonical record: `docs/promotion_decision.md`.

### Deferred Items (open at v2.2 close)

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Baselines | anthropic n=5 (billing) | Deferred — no top-up (user decision); logged-not-gated, promotion path in `docs/baseline_regen.md` | Phase 11 close; extended 2026-06-11 |
| Baselines | gemini n=5 (billing/quota) | Deferred (D-12-09) — single scored run already 1.0; logged-not-gated | Phase 12 (D-12-09) |
| Baselines | `refinement_cheaper` gpt-4o-mini (committed 0.0) | Stale vs post-retrieval-fix ~0.8 rate; clean follow-up regen | Phase 15 close |
| Architecture | ARCH-FUT-01 (custom imperative loop) | Deferred as tracked debt; trigger = Phases 13-14 evidence chain | Phase 14 (D-14-08) |
| Config | Prod-default `FORCED_COMMIT_STEP=6` flip | Flagged, NOT implemented (D-15-07) | Phase 15 close |

## Session Continuity

Last session: 2026-06-15 — v2.2 milestone completed and archived
Stopped at: Milestone close (archived, tagged v2.2)
Resume file: None
Next step: `/gsd-new-milestone`

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
| Phase 14 P14-01 | 15min | 3 tasks | 5 files |
| Phase 14 P14-02 | 15min | 2 tasks | 3 files |
| Phase 14 P03 | 10min | 2 tasks | 1 files |
| Phase 14 P14-04 | 95min | 3 tasks | 1 files |
| Phase 14 P14-05 | 15min | 3 tasks | 3 files |
| Phase 15 P02 | 6960 | 2 tasks | 1 files |
| Phase 15 P03 | 513 | 3 tasks | 8 files |

## Decisions

- [Phase ?]: D-12-09 honored: gemini n=5 baseline deferred at user budget decision (2026-06-11); joins anthropic as second deferred cell; comparison floor = matrix minus BOTH anthropic AND gemini
- [Phase ?]: DEC-03 threshold code default stays 0.55; LOW_SIMILARITY_THRESHOLD_OVERRIDE env var is the A1 experiment knob; first A1 run keeps it unset
- [Phase ?]: low_similarity hint suppressed once all_slots_viable=True (DEC-03); only fires in pre-candidate steps; co-tuned with DEC-01 via shared VIABILITY_CONTRACT_ENABLED flag
- [Phase ?]: Arm config scenario IDs match baseline stems; split reader uses provider_slug prefix stripping (13-05)
- [Phase ?]: A1 VIABILITY_CONTRACT_ENABLED: gpt-5-mini 0.000 pooled — zero signal
- [Phase ?]: A2 FORCED_COMMIT_STEP=6: gpt-5-mini 0.500 pooled (positive signal); forced mechanism never fired
- [Phase ?]: A3 PARALLEL_TOOL_EXECUTION: anchor regression refinement_cheaper 0.000 vs 1.000 baseline; latency unmeasurable (Phase-12 no telemetry)
- [Phase ?]: A4 conditional: A1 no signal (0.0), A2 positive (0.5); A4 qualification deferred to plan 13-07
- [Phase ?]: D-14-05: R2 EXPECTED-NULL on tested cells — str() collapse is NO-OP for all three RUN models; only AnthropicAdapter uses list-content and is deferred (D-12-09)
- [Phase ?]: R1 REPLAY_MULTI_MESSAGE: gpt-5-mini 0.500 — identical to A2, delta vs A2 = 0.000; positive vs floor but below 0.6 bar; anchor held 1.000
- [Phase ?]: R2 REPLAY_CONTENT_BLOCKS: EXPECTED-NULL REFUTED for gpt-5-mini — 10/10 deterministic 400s (Responses-API list content embeds function_call items; str() collapse was load-bearing); NEGATIVE signal, R3 combo precondition fails; anchor held median 1.000
- [Phase ?]: R3 NOT RUN: D-14-01 precondition 2 fails — R2 negative (catastrophic 400s), not positive-but-short
- [Phase ?]: Valve NOT RUN: precondition met but R1 zero-delta vs A2 makes expected marginal signal = 0; A2 retest is Phase 15 scope
- [Phase ?]: ARCH-FUT-01: ratify gpt-4o-mini anchor; defer ARCH-FUT-01; Phase 15 = A2 retest on fixed synthesizer + refinement_cheaper root cause analysis
- [Phase 15]: v2.2 CLOSED 2026-06-15: A2 retest (FORCED_COMMIT_STEP=6) confirmed gpt-5-mini 0.500 pooled — INST-05 honest null (no arm cleared 0.600 bar across Phases 13/14/15). Root cause: refinement_cheaper typed-slot viability gate never satisfied (structural, not code bug). gpt-4o-mini anchor RATIFIED (omakase 1.000 flag-off, gate >= 0.8). Baseline provenance corrected (prior refinement_cheaper 1.000 was flag-ON arm artifact; re-baselined to honest 0.000 flag-off). 6 runnable cells written. ARCH-FUT-01 DEFERRED as tracked debt. Prod-default FORCED_COMMIT_STEP=6 flip flagged but NOT implemented (D-15-07). Canonical record: docs/promotion_decision.md.

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
