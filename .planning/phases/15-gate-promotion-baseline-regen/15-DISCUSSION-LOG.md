# Phase 15: Gate Promotion + Baseline Regen - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-14
**Phase:** 15-Gate Promotion + Baseline Regen
**Areas discussed:** A2 retest gating, Baseline regen config, Gate promotion bar, Root-cause depth

**Framing note:** Phase 15 WHAT-scope was already ratified by the user at the D-14-08
checkpoint (`docs/replay_arm_verdicts.md`). This discussion clarified only HOW to execute
that ratified scope. The user selected the recommended option for every question.

---

## A2 retest gating

### Q1 — Is the A2 retest a hard gate or a measured input?

| Option | Description | Selected |
|--------|-------------|----------|
| Measured input, never blocks | Record delta honestly; promotion proceeds regardless (A2-if-clears, else flag-off). Phase 15 always completes. | ✓ |
| Hard gate — A2 must clear to promote | Below 0.6 blocks gpt-5-mini promotion; Phase 15 stops at "documented plateau". | |
| Skip A2 retest, promote flag-off only | Treat 0.500<0.6 as answered; skip the highest-ROI experiment. | |

**User's choice:** Measured input, never blocks → **D-15-01**
**Notes:** Preserves Phases 13/14 honest-null discipline; resolves the "winning arm" framing.

### Q2 — Run shape; can the A2 run feed baselines?

| Option | Description | Selected |
|--------|-------------|----------|
| Two separate runs: A2 arm + flag-off baseline | A2 (FORCED_COMMIT_STEP=6) feeds verdict; flag-off n=5 feeds baselines. Anchor non-regression on both. | ✓ |
| One A2 run, reuse for both | Cheaper but contaminates baselines with experiment flags. | |
| Flag-off baseline only, fold A2 into it | Doesn't work — flag-off can't fire the forced path. | |

**User's choice:** Two separate runs → **D-15-02**
**Notes:** Keeps experiment flags out of committed baselines (D-11-14); within ≤4-run cap.

### Q3 — Where is the A2 retest result recorded?

| Option | Description | Selected |
|--------|-------------|----------|
| New docs/promotion_decision.md | Single milestone-closing record; cross-links the two closed verdict docs. | ✓ |
| Append to docs/replay_arm_verdicts.md | Re-opens an immutable closed record. | |
| Just verdict tables, no narrative doc | Loses the milestone-closing synthesis. | |

**User's choice:** New docs/promotion_decision.md → **D-15-03**
**Notes:** Follows Phase-14 "closed records immutable, new phase = new doc".

---

## Baseline regen config

### Q1 — What is the honest baseline cell-set?

| Option | Description | Selected |
|--------|-------------|----------|
| Runnable cells only: gpt-4o-mini + gpt-5-mini + deepseek | "All matrix cells" = "all eligible cells"; matches write_baselines.py enforcement. | ✓ |
| Attempt all 5 providers, top up billing | Reopens the locked D-12-09 billing decision. | |
| Anchor cell only | Too narrow — loses the reasoning-model comparison floor. | |

**User's choice:** Runnable cells only → **D-15-04**
**Notes:** anthropic/gemini stay deferred-logged; late_night quarantined.

### Q2 — How does regen sequence against the staleness gate?

| Option | Description | Selected |
|--------|-------------|----------|
| Regen LAST, after all code changes land | Baselines reflect final merged behavior; staleness gate passes. | ✓ |
| Regen first, freeze code after | Inverts the dependency; a later fix makes baselines stale. | |
| Snapshot current, diff-only update | Unnecessary complexity; write_baselines.py handles carry-forward. | |

**User's choice:** Regen LAST → **D-15-05**
**Notes:** Standard docs/baseline_regen.md discipline.

---

## Gate promotion bar

### Q1 — Which models get an enforced hard gate?

| Option | Description | Selected |
|--------|-------------|----------|
| Anchor stays enforced; reasoning models stay logged unless A2 clears | Enforce only what's stable AND measured; gpt-5-mini enforced only if A2 clears 0.6. | ✓ |
| Enforce every measured model at its empirical floor | A 0.0 hard gate is meaningless; misrepresents failing configs. | |
| Only ever enforce the anchor | Pre-decides against gpt-5-mini even if A2 surprises. | |

**User's choice:** Anchor enforced; reasoning models logged unless A2 clears → **D-15-06**
**Notes:** Short-falling entries retain `logged`/`aspirational` with a Phase-15 rationale note.

### Q2 — What config does an enforced gate certify (A2 = experiment config)?

| Option | Description | Selected |
|--------|-------------|----------|
| Enforce only against the config CI runs; A2 = unlock rationale | Gate certifies flag-off prod default; A2 is the evidence, not the gate value. | ✓ |
| Promote on A2 data, switch prod default to FORCED_COMMIT_STEP=6 | Real prod-behavior change; bigger than gate promotion — flag + defer. | |
| Enforce on A2 data regardless of CI config | CI would fail spuriously / pass meaninglessly. | |

**User's choice:** Enforce against CI config; A2 = rationale → **D-15-07**
**Notes:** Prod-default flip is a separate, likely-deferred decision (anchor ratified as-is).

---

## Root-cause depth

### Q1 — What is the root-cause analysis allowed to produce?

| Option | Description | Selected |
|--------|-------------|----------|
| Diagnostic-only by default; ship a fix ONLY if one-flag/one-line low-risk | Written analysis from existing JSONs; trivial fix behind A2 retest; larger findings deferred. | ✓ |
| Diagnostic-only, never ship a fix | Cleanest boundary but may measure known-broken behavior. | |
| Root-cause AND fix whatever's found | Unbounded scope creep; re-litigates the ARCH-FUT-01 deferral. | |

**User's choice:** Diagnostic-only by default; trivial-fix-only → **D-15-08**
**Notes:** Zero new diagnostic runs — existing telemetry suffices; does not re-open ARCH-FUT-01.

### Q2 — Pipeline ordering (diagnose-before-retest vs baselines-last)?

| Option | Description | Selected |
|--------|-------------|----------|
| Diagnose → trivial fix → A2 + flag-off → promote → baselines last → latency → close | Single clean dependency chain; retest measures final code. | ✓ |
| Run A2 first, diagnose after from fresh data | Burns an extra live run if the diagnostic then finds a fix. | |
| Let the planner decide ordering | The dependency is a correctness constraint, not a preference. | |

**User's choice:** Fixed pipeline order → **D-15-09**
**Notes:** diagnose-before-retest is mandatory so the A2 retest measures FINAL code.

---

## Claude's Discretion

The user selected the recommended option for every question. Planner discretion (per CONTEXT.md
`### Claude's Discretion`): exact env-var / run-config / file names; latency-report aggregation
shape; smoke-verification shape; plan-splitting. Substance to preserve: two-separate-runs
(experiment vs baseline), mandatory anchor non-regression, the diagnose→fix→retest→promote→
baseline-last→latency ordering, the new (not-appended) promotion_decision.md, runnable-cells-only
baselines, gate-provenance honesty, and the diagnostic-only-unless-trivial fix boundary.

## Deferred Ideas

- gpt-5-mini refinement_cheaper decisiveness (real fix if non-trivial) — future milestone.
- ARCH-FUT-01 execution — ratified-deferred tracked debt (D-14-08).
- Prod-default flip to FORCED_COMMIT_STEP=6 — separate, likely-deferred decision (D-15-07).
- anthropic + gemini n=5 baselines and gate enforcement — deferred (D-12-09 billing/quota).
