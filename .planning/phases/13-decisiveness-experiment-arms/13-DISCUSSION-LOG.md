# Phase 13: Decisiveness Experiment Arms - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-12
**Phase:** 13-decisiveness-experiment-arms
**Areas discussed:** Run budget & matrix shape, Forced-commit honesty & semantics, Arm toggling & prod shipping, Arm combination & verdict structure

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Run budget & matrix shape | Which scenarios in scope, staged vs full runs, spend ceiling | ✓ (delegated) |
| Forced-commit honesty & semantics | How forced commits are scored/flagged; default N; relation to short_circuit_max_steps | ✓ (delegated) |
| Arm toggling & what ships to prod | Env vars vs config; what's enabled by default in Phase 13 | ✓ (delegated) |
| Arm combination & verdict structure | DEC-01+03 joint vs crossed; combo run; verdict doc location/contents | ✓ (delegated) |

**User's choice:** "u got it fable, do ur best king, make my shit work" — full
delegation of all four areas to Claude's discretion.

**Notes:** Same delegation pattern as Phase 12 ("u decide for me"). Claude resolved all
four areas as D-13-01..09 in CONTEXT.md:

- **Run budget:** 3 judged arms (+1 conditional combo, ≤4 live matrix runs cap), 3
  models × 2 scenarios (omakase + refinement_cheaper — omakase-only is vacuous per
  D-12-08) × n=5; smoke n=1 before each full spend; no billing top-ups, partial results
  recorded honestly.
- **Forced-commit:** synthetic commit from best-so-far viable candidates at step N
  (default 6) through the normal commit path; only fires when every slot has a viable
  candidate; `commit_forced`/`forced_commit_step` telemetry; verdict must report
  model-initiated vs forced split — forced commits count for the product metric but
  cannot silently game the falsifier.
- **Toggling:** all arms env-flagged, default OFF (`VIABILITY_CONTRACT_ENABLED`,
  `FORCED_COMMIT_STEP`, `PARALLEL_TOOL_EXECUTION_ENABLED`); prod untouched until
  Phase 15 promotion; one shared flag for DEC-01+03 mechanically enforces co-tuning.
- **Combination/verdict:** A1 = DEC-01+03 joint arm; A2 = DEC-02; A3 = DEC-04 (latency
  arm, judged on latency not commit rate); A4 = A1+A2 combo only if neither clears
  alone with positive signal; verdicts in `docs/decisiveness_arm_verdicts.md`.

## Claude's Discretion

All four areas (full delegation). Planner may refine env-var names, file names, and
default N, but must preserve: co-tuned A1, forced-commit transparency split,
flags-off-by-default prod surface, two-scenario pooling universe, ≤4 live matrix run
cap.

## Deferred Ideas

- Arm stacking beyond the single A1+A2 combo — Phase-15 material
- Richer state replay — Phase 14, conditional on all-arm plateau
- Anthropic/gemini n=5 baselines — remain deferred (D-12-09)
