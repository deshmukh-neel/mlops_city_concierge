# Phase 11: Cross-Model Baseline Regen + Matrix Expansion - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 11-cross-model-baseline-regen-matrix-expansion
**Areas discussed:** Pre-regen harness fixes, BASE-01 regen mechanics, BASE-02 matrix expansion, BASE-03/04 CI enforcement
**Mode:** Auto (user delegated: "ill let u decide on whats best, do what u gotta do") — Claude selected the recommended option for every question; all gray areas auto-selected.

---

## Pre-Regen Harness Fixes (sequencing + WR-12 semantics)

| Option | Description | Selected |
|--------|-------------|----------|
| Wave-0 pre-fixes before regen | WR-06/08/09/12 + commit-rate threading land first; live regen is the LAST wave | ✓ |
| Regen first, fix after | Get fresh numbers now, patch scorers later | |
| Defer all WRs to v2.2 | Regen on current semantics | |

**Choice rationale:** Each WR changes recorded numbers; fixing after regen forces a second full regen (~hours of live calls). ROADMAP explicitly says "fix before regenerating baselines or `tool_calls_mean` bakes in the skew".

| Option (WR-12 semantics) | Description | Selected |
|--------|-------------|----------|
| Abstain (None, excluded) | category_compliance unmeasurable with zero committed stops; decisiveness already hard-gated via committed_itinerary_rate | ✓ |
| Score 0.0 | Penalize non-commitment in the compliance scorer too | |
| Keep 1.0 | Status quo (contradicts docstring; fail-open) | |

**Choice rationale:** 0.0 double-penalizes (commit-rate gates already carry the signal) and would conflate "didn't commit" with "committed wrong categories" in advisory medians. Abstain matches the Phase-10 honesty principle and the retained Branch-1 abstain precedent (D-10-04).

---

## BASE-01 Regen Mechanics

| Option | Description | Selected |
|--------|-------------|----------|
| Discrete writer tool + runbook + snapshots | scripts/write_baselines.py enforcing D-10-03 refusal; docs/baseline_regen.md; _snapshots/*.pre-phase11.json | ✓ |
| Hand-rolled regen with documented steps | Runbook only, baselines edited by hand from summary.json | |

**Choice rationale:** Phase 10 already specified the refusal rule (D-10-03) and noted "baseline writer does not exist yet — lands wherever Phase 11 builds it". Hand-rolling is how generated_by blobs and n=1 cells snuck in.

**Notes:** Gemini regen failure does not block (logged-not-gated → documented deferral via `_DEFERRED_BASELINE_CELLS`); gated families erroring does block.

---

## BASE-02 Matrix Expansion

| Option | Description | Selected |
|--------|-------------|----------|
| 3 new entries, gemini excluded, late_night dropped from default scenarios, sequential | Matches BASE-02 text + PROV-04 standing + avoids 5×5 diagnostic burn | ✓ |
| Include gemini in cross-model matrix | 4 new entries | |
| Keep late_night in default matrix | Quarantined scenario still runs for every entry | |
| Add parallel execution | ProcessPoolExecutor over cells to cut wall-clock | |

**Choice rationale:** BASE-02 names exactly three new providers; PROV-04 explicitly keeps Gemini out of the prod matrix. late_night is baseline-ineligible — running it across 5 providers × 5 runs is cost without signal (stays runnable explicitly). Parallelism re-deferred: rate-limit storms are the 21-14-30Z failure mode and regen is not a hot path.

---

## BASE-03/04 CI Enforcement & Gate Promotion

| Option | Description | Selected |
|--------|-------------|----------|
| Gates vs committed baselines + synthetic-regression test + exit-code contract; conformance marker promoted; anthropic re-ratified; gpt-5-mini stays aspirational | CI enforces from committed artifacts, no live keys (D-09-10 honored) | ✓ |
| Live n=5 matrix in CI with secrets | Direct empirical enforcement per PR | |
| Keep gates local-only | CI unchanged; gates remain a human-checkpoint step | |

**Choice rationale:** D-09-10 (no live keys in CI) is standing and repeatedly re-affirmed; BASE-03's own success criterion is "fires on a synthetic regression", which committed-artifact checking + a synthetic test satisfies exactly. WR-05 advisory entries implemented as WARN (docs already promise them) rather than deleted. WR-07 exit codes folded in because gate wiring consumes them.

**BASE-04 note:** Verified `check_baselines_fresh.py` watches only `app/agent/` — `app/llm_factory.py` (provider branches, thinking/temp policies) is an actual coverage gap; watch-set extended there + `configs/eval_matrix*.yaml`.

---

## Claude's Discretion

- Script/tool/Make-target names; runbook structure; baselines-mode implementation shape.
- IN-01..IN-06 folded opportunistically only.
- Wave decomposition (suggested: Wave 0 pre-fixes → Wave 1 wiring → Wave 2 live regen last).

## Deferred Ideas

- Parallel matrix execution (re-deferred).
- late_night prod-threading redesign (v2.2).
- Advisory→hard median-gate promotion (v2.2).
- Gemini promotion to gated/prod matrix (PROV-FUT-01).
- gpt-5-mini aspirational gate passing (v2.2 decisiveness target).
