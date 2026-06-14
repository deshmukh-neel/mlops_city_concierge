# Phase 14: Richer State Replay (CONDITIONAL) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 14-richer-state-replay-conditional
**Areas discussed:** Arm structure & comparison point, Multi-message replay mechanics, Content-block preservation semantics, Plateau verdict & ARCH-FUT-01 handoff (all delegated)

---

## Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Arm structure & comparison point | R1/R2 separate vs combined vs sequenced; pure vs stacked on A2; run-budget cap | ✓ (delegated) |
| Multi-message replay mechanics | Adapter interface change vs graph-side loop; provider coverage | ✓ (delegated) |
| Content-block preservation semantics | What survives the prune; token-cost guardrail; str()-collapse loss analysis | ✓ (delegated) |
| Plateau verdict & ARCH-FUT-01 handoff | Verdict doc location; ARCH-FUT-01 artifact; user checkpoint | ✓ (delegated) |

**User's choice:** "u got it fable" — full delegation of all four areas to Claude,
matching the Phase-13 delegation pattern.
**Notes:** No per-area question loops ran; Claude resolved each area directly from the
codebase, the DEC-05 verdict document, and the Phase-12/13 locked decisions. Resulting
decisions D-14-01..08 recorded in 14-CONTEXT.md.

---

## Claude's Discretion

All areas. Key calls made on the user's behalf:
- Pure (non-stacked) judged arms R1/R2 + conditional R3 combo + one discretionary
  best-replay+A2 escalation run; ≤4 full live matrix runs (mirrors D-13-01/02)
- Additive adapter interface (existing single-replay signature and 9-test conformance
  harness untouched); generic multi-replay default in the ABC
- R2 evidence audit of existing run JSONs before any live spend; minimal-diff verbatim
  content preservation, tool_calls still stripped
- New `docs/replay_arm_verdicts.md` (Phase-13 record stays immutable); ARCH-FUT-01
  section is recommendation-only with a user checkpoint before Phase 15

## Deferred Ideas

- Replay+DEC stacking matrices beyond the single escalation run — Phase 15
- ARCH-FUT-01 execution (architectural rethink) — future milestone, user decision
- Cross-request reasoning-state persistence (`io.py:32-40` text-only rebuild) — verify-only note
- Anthropic/gemini n=5 baselines — remain deferred (D-12-09)
