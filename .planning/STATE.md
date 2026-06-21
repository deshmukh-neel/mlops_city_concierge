---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: Adaptive Data Loop
status: executing
last_updated: "2026-06-21T00:37:15.937Z"
last_activity: 2026-06-21
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 13
  completed_plans: 12
  percent: 75
---

# Project State

**Project:** City Concierge
**Initialized:** 2026-05-14
**Active milestone:** v2.3 Adaptive Data Loop (Phases 16-19)
**Last shipped:** v2.2 Reasoning-Model Decisiveness (Phases 12-15, 2026-06-15)

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-15 after v2.3 scoping)
See: .planning/MILESTONES.md for historical record (v1.0, v2.0, v2.1, v2.2)

**Core value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Current focus:** Phase 19 — productionized-loop-metric-loop

## Current Position

Phase: 19 (productionized-loop-metric-loop) — EXECUTING
Plan: 4 of 4
Status: Ready to execute
Last activity: 2026-06-21

## Blockers / Readiness Notes

- **FALSIFY-01 PASSED (the v2.3 hard gate).** Phase 16 proved the loop can add places that weren't there and make them retrievable: hit@5 0/5 → 5/5, delta +1.000, exit 0, against an isolated `city_concierge_sandbox` DB; prod untouched. The milestone is cleared to proceed to Phases 17-19. Canonical evidence: `.planning/phases/16-loop-falsifier/16-VERIFICATION.md` + MLflow `coverage_agent` experiment.
- **Phases 17-19 are NOT yet planned and NOT in ROADMAP.** The v2.3 milestone was started in STATE with Phases 16-19 (LOG=17, GAP=18, LOOP-01..03+METRIC=19 per 16-CONTEXT.md) but only Phase 16 was ever scoped/rendered. ROADMAP.md/MILESTONES.md/PROJECT.md still describe v2.2 as the last milestone — **v2.3 roadmap rendering is outstanding doc work** before Phase 17 can be discussed/planned.
- **`phase.complete` mis-marked the milestone "complete" (corrected).** Because the ROADMAP had no Phase 17 row, the SDK treated Phase 16 as the final phase and reverted STATE to a v2.2-milestone-complete snapshot. This STATE.md was hand-corrected to reflect v2.3 active / Phase 16 of 4 complete. The phase WORK is genuinely done (verified); only the bookkeeping was wrong.
- **gpt-4o-mini remains the ratified prod anchor** — no model changes in v2.3 (carried from v2.2).
- **Sandbox DB invariant:** `SANDBOX_DATABASE_URL` is a SEPARATE Postgres DB (`city_concierge_sandbox`), never the shared prod `places_raw`. The falsifier coerces `DATABASE_URL` at runtime and MUST `get_settings.cache_clear()` + `close_db_pool()` after (lru_cache footgun — see 16-03).

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

Last session: 2026-06-21T00:37:15.928Z
Stopped at: 19-04-PLAN.md Task 2 (checkpoint:human-action gate=blocking — operator floor-calibration run)
Resume file: 19-04-PLAN.md Task 2 (checkpoint:human-action gate=blocking)
Next step: Operator runs `make loop` with live keys + sandbox; provides before/after hit@k, recall@k, delta, exit code, chosen FLOOR, MLflow run id, then types "approved". Then /gsd-execute-phase 19 resumes at Task 3.

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
| Phase 16-loop-falsifier P01 | 12 | 2 tasks | 4 files |
| Phase 17-query-logging-log P01 | 117 | 2 tasks | 1 files |
| Phase 18-gap-mining-gap P01 | 20 | 4 tasks | 6 files |
| Phase 18-gap-mining-gap P02 | 287 | 2 tasks | 2 files |
| Phase 18-gap-mining-gap P03 | 421 | 2 tasks | 2 files |
| Phase 19 P01 | 166 | 2 tasks | 2 files |
| Phase 19-productionized-loop-metric-loop P02 | 216 | 2 tasks | 3 files |
| Phase 19-productionized-loop-metric-loop P03 | 301 | 2 tasks | 1 files |

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
- [Phase 16]: v2.3 FALSIFY-01 gate PASSED 2026-06-15: loop falsifier proved before→after hit@5 delta +1.000 (0/5 → 5/5) against isolated sandbox; exit 0; 1 paid Google Places call (seed-isolation), 20 places ingested to sandbox only, prod untouched. Gap = ("Outer Sunset","vietnamese"). Two live bugs found+fixed during execution: (1) settings lru_cache prod-leak (cache_clear+close_db_pool after DATABASE_URL coercion — 3101577/250b93b); (2) code-review CR-01 before-snapshot was 0.0-by-construction, now asserts sandbox emptiness in-process. All 13 code-review findings fixed. Verified 7/7. MLflow under coverage_agent exp 6. Milestone cleared to proceed to Phases 17-19 (NOT yet planned/roadmapped).
- [Phase 15]: v2.2 CLOSED 2026-06-15: A2 retest (FORCED_COMMIT_STEP=6) confirmed gpt-5-mini 0.500 pooled — INST-05 honest null (no arm cleared 0.600 bar across Phases 13/14/15). Root cause: refinement_cheaper typed-slot viability gate never satisfied (structural, not code bug). gpt-4o-mini anchor RATIFIED (omakase 1.000 flag-off, gate >= 0.8). Baseline provenance corrected (prior refinement_cheaper 1.000 was flag-ON arm artifact; re-baselined to honest 0.000 flag-off). 6 runnable cells written. ARCH-FUT-01 DEFERRED as tracked debt. Prod-default FORCED_COMMIT_STEP=6 flip flagged but NOT implemented (D-15-07). Canonical record: docs/promotion_decision.md.
- [Phase ?]: D-02/D-03/D-04: user_query_log 7-column demand-signal table created; raw-message verbatim store documented; chained to head e0cd7069bc8f; write path deferred to 17-02
- [Phase ?]: Shared guard module path and H3 pass condition
- [Phase 18-03]: TRUE pair-level supply (gather_pair_supply) supersedes RESEARCH Open Question #1 per-cuisine resolution; ingested_query_texts filters checkpoints to status='completed' AND normalizes FIELD_MODE:: prefix; sandbox guard runs on SAME conn as insert_pending; cold-start keyed on empty demand (not judge absence)
- [Phase ?]: D-02: --populated IS the idempotent populated reset (DROP+re-provision); --reset is schema-only; gap-bucket exclusion covers per-neighborhood + citywide + eatery-overlap queries via LOOP_GAP_NEIGHBORHOOD/LOOP_GAP_CUISINE

## Operator Next Steps

- Phase 16 (Loop Falsifier) is COMPLETE and verified — FALSIFY-01 gate PASSED. Branch: `gsd/phase-16-loop-falsifier` (13 source commits; ready to PR/merge per your usual flow).
- Render the v2.3 milestone (Phases 16-19) into ROADMAP.md + MILESTONES.md so tracking tools see Phase 17 as next (the gap that caused phase.complete to mis-mark the milestone).
- Then `/gsd-discuss-phase 17` (LOG — user-query logging to Cloud SQL, the foundational v2.3 requirement).
