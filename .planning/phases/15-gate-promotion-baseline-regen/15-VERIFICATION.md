---
phase: 15-gate-promotion-baseline-regen
verified: 2026-06-15T00:00:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
resolved:
  - test: "REQUIREMENTS.md traceability row regex"
    resolution: "Resolved option (b) — strict format compliance. PROMO-01/02/03 traceability rows reformatted to exactly '| PROMO-XX | Phase 15 | Complete |'; outcome notes moved into the checkbox lines above (which carry no regex). The 15-04-PLAN.md acceptance regex (\\|\\s*PROMO-0N\\s*\\|\\s*Phase 15\\s*\\|\\s*Complete\\s*\\|) now passes for all three rows and the verdict-doc git diff --quiet check passes. Fixed in commit 2252532."
---

# Phase 15: Gate Promotion + Baseline Regen Verification Report

**Phase Goal:** The winning arm's honest baselines are written for all matrix cells, reasoning-model gates are promoted from logged-not-gated to enforced where the data earns it, and the prod latency budget analysis is documented — closing the milestone with a ratified prod-driver recommendation.

**Verified:** 2026-06-15
**Status:** passed (the lone human_needed item — PROMO traceability row format — was resolved via strict-format compliance in commit 2252532; see frontmatter `resolved`)
**Re-verification:** No — initial verification, single gap resolved inline

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Honest n=5 flag-off baselines written for all runnable matrix cells (gpt-4o-mini, gpt-5-mini, deepseek-reasoner x 2 live scenarios); anthropic/gemini absent; late_night quarantined | VERIFIED | `configs/eval_baselines/omakase_mission_open_ended.json` and `refinement_cheaper.json` both carry `generated_at: 2026-06-15T01-57-36Z` from `scripts/write_baselines.py`. 6 runnable cells written (3 providers x 2 scenarios). Anthropic absent from providers list. Deepseek-chat omakase cell retains prior stamp (correct — not in arm config). `make eval-gates-check-baselines` exits 0. `check_baselines_fresh.py origin/main` exits 0. |
| 2 | gpt-4o-mini gate re-ratified active >= 0.8 from flag-off Run #2 omakase median; not from the FORCED_COMMIT_STEP=6 experiment | VERIFIED | `configs/eval_gates.yaml` line 22: `status: active`, hard gate `committed_itinerary_rate >= 0.8` with `scenarios: [omakase_mission_open_ended]`. Rationale explicitly cites D-15-06 and Run #2 timestamp `2026-06-15T00-46-43Z`. Gate value sourced from flag-off omakase median=1.000. Commits a3e05d2, 4249f64. |
| 3 | gpt-5-mini stays logged (NOT promoted) because flag-off Run #2 pooled median = 0.500 < 0.600 floor; gate value never taken from the FORCED_COMMIT_STEP=6 experiment | VERIFIED | `configs/eval_gates.yaml` line 36: `status: logged`, `hard: null`. Rationale cites D-15-07, explicitly states pooled=0.500 < 0.600, states unlock mechanism is data-dependent (model-initiated, not forced-commit), and confirms FORCED_COMMIT_STEP=6 flip is deferred. |
| 4 | deepseek/anthropic/gemini stay logged with no 0.0-floor enforced gate | VERIFIED | All three families have `status: logged`, `hard: null` in `configs/eval_gates.yaml`. Phase-15 rationale notes added. No 0.0-floor enforced gate exists anywhere in the 7-gate file. |
| 5 | PROMO-03 latency report exists with honest ~46-47s omakase observation (over ~30s budget), LLM vs tool decomposition, overhead reconciliation, and ratify-gpt-4o-mini recommendation | VERIFIED | `docs/promotion_decision.md` "## PROMO-03 Latency Report" section (line 682): median latency=47.3s stated explicitly against the ~30s/turn budget; "4/5 runs EXCEED the 30s/turn budget" stated; LLM/tool/overhead decomposition table present; A2 corroboration at 45.9s cited; step count named as dominant lever; honest ratify-anchor recommendation with latency caveat present. |
| 6 | docs/promotion_decision.md is the single v2.2 milestone-closing record with cross-links to both immutable verdict docs and a data-dependent INST-05 verdict; neither verdict doc was modified | VERIFIED | File is 868 lines; contains "## Inputs (Immutable Cross-Links)" section with markdown links to both `docs/decisiveness_arm_verdicts.md` and `docs/replay_arm_verdicts.md` with explicit "Closed record — this document does NOT append to or modify it" notes. INST-05 verdict is CASE (a) — sourced from measured A2 pooled=0.500 < 0.600. `git diff --quiet -- docs/decisiveness_arm_verdicts.md docs/replay_arm_verdicts.md` exits 0 (both immutable, no commits since Phase 13/14). |
| 7 | REQUIREMENTS.md traceability rows show PROMO-01/02/03 as Complete in Phase 15 per the acceptance-criteria regex | WARNING | Rows 96-98 of `.planning/REQUIREMENTS.md` contain `| PROMO-01 | Phase 15 | Complete — <notes> |` etc. The content is correct (all three marked Complete in Phase 15). However, the 15-04-PLAN.md acceptance-criteria verify command `grep -Eq '\|\s*PROMO-01\s*\|\s*Phase 15\s*\|\s*Complete\s*\|'` requires the column to END at "Complete" with no trailing content — the actual rows have inline outcome notes appended. The SUMMARY claimed this check passed; it would not have (verified in this session). The milestone intent is satisfied; the strict regex format compliance is not. Human decision needed. |

**Score:** 6/7 truths verified (Truth 7 is a WARNING-level discrepancy, not a hard blocker on goal achievement)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/promotion_decision.md` | Milestone-closing record with all 5 content blocks + cross-links + data-dependent INST-05 verdict | VERIFIED | 868 lines, 11 sections. All 5 blocks present: Root-Cause, D-15-08 Disposition, A2 Retest Delta, Run #2 Provenance, Anchor Provenance Correction, Gate Promotion Decisions, Baseline Regen Provenance, PROMO-03 Latency Report, Milestone Closing Summary. Committed at 57c5f4c. |
| `configs/eval_gates.yaml` | Gate promotion decisions with D-15-06/07 provenance; scenarios filter on gpt-4o-mini | VERIFIED | 70-line file. gpt-4o-mini: `status: active`, `hard.scenarios: [omakase_mission_open_ended]`. gpt-5-mini: `status: logged`, `hard: null`. All others logged. 7 total gate entries. D-15 cited in rationale fields. |
| `configs/eval_baselines/refinement_cheaper.json` | Fresh n=5 flag-off provenance stamps; honest committed_itinerary_rate for runnable providers | VERIFIED | `generated_at: 2026-06-15T01-57-36Z`. gpt-4o-mini median=0.0 (honest flag-off; PROVENANCE CORRECTION note present). gpt-5-mini median=0.0. deepseek-reasoner median=0.0. Anthropic present but no `committed_itinerary_rate` (n=1 pre-existing cell, not regenerated — correct, was absent from arm config). |
| `configs/eval_baselines/omakase_mission_open_ended.json` | Fresh n=5 flag-off provenance stamps for gpt-4o-mini, gpt-5-mini, deepseek-reasoner | VERIFIED | `generated_at: 2026-06-15T01-57-36Z`. gpt-4o-mini median=1.0. gpt-5-mini median=1.0. deepseek-reasoner median=0.0. Anthropic correctly absent. Deepseek-chat retains `2026-06-11` stamp (not in arm config — correct). |
| `docs/decisiveness_arm_verdicts.md` | UNCHANGED by Phase 15 | VERIFIED | `git diff --quiet` exits 0. Last commit to this file: `edf7174` (Phase 13). Zero commits during Phase 15 execution window (2026-06-14 to 2026-06-15). |
| `docs/replay_arm_verdicts.md` | UNCHANGED by Phase 15 | VERIFIED | `git diff --quiet` exits 0. Last commit to this file: `4c44231` (Phase 14). Zero commits during Phase 15 execution window. |
| `scripts/check_eval_gates.py` | Optional `hard.scenarios` filter; gate not silently passable when gated scenario absent | VERIFIED | Lines 348-356: `hard_scenarios` filter applied. Line 356-359: when `scoped_cells` is empty after filtering, returns `not_evaluable` (not "pass"). Confirmed empirically: test with omakase absent from summary returns `not_evaluable`. |
| `.planning/REQUIREMENTS.md` | PROMO-01/02/03 rows = `| PROMO-XX | Phase 15 | Complete |` per strict regex | WARNING | Rows exist at lines 96-98 and say "Complete" but with trailing inline notes. Strict regex fails. Data intent is correct; format compliance is a SUMMARY accuracy issue. See Truth 7. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Run #2 flag-off summary.json (`eval_reports/2026-06-15T00-46-43Z/`) | `configs/eval_baselines/*.json` | `scripts/write_baselines.py` | VERIFIED | Commit 4249f64 includes both baseline JSONs. `generated_at: 2026-06-15T01-57-36Z` matches Run #2 date. SUMMARY records `make write-baselines SUMMARY=eval_reports/2026-06-15T00-46-43Z/summary.json RUNS=5` exit 0. |
| Run #2 run JSONs step_telemetry (llm_call_seconds + tool_exec_seconds) | `docs/promotion_decision.md` PROMO-03 Latency Report | Per-step aggregation with overhead reconciliation | VERIFIED | Table at lines 693-700 shows per-run decomposition with summed vs observed columns. Overhead column computed. A2 corroboration at 45.9s cited and cross-referenced against 47.3s Run #2 median. |
| `docs/promotion_decision.md` | `docs/decisiveness_arm_verdicts.md` + `docs/replay_arm_verdicts.md` | Markdown cross-links (immutable inputs, not appended) | VERIFIED | "## Inputs (Immutable Cross-Links)" section markdown-links both docs with explicit "Closed record" notes. Searched for `decisiveness_arm_verdicts.md` and `replay_arm_verdicts.md` — both found with correct link text. |
| `.planning/REQUIREMENTS.md` traceability | PROMO-01/02/03 rows = Complete | Status flip committed to main (50daeb0) | WARNING | Rows exist as Complete with notes; strict regex fails. See Truth 7. |
| `configs/eval_gates.yaml` gpt-4o-mini gate | omakase scenario scoped | `hard.scenarios` filter in check_eval_gates.py | VERIFIED | Gate scoping confirmed in yaml (line 29: `scenarios: [omakase_mission_open_ended]`) and in checker code (lines 348-359). Behavioral test: gate passes at omakase=1.0, is not silently bypassed when gated scenario absent. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `configs/eval_baselines/refinement_cheaper.json` | `committed_itinerary_rate.median` for gpt-4o-mini | Run #2 flag-off summary.json via `write_baselines.py` | YES — 0.000 from actual 5-run measurements | VERIFIED |
| `configs/eval_baselines/omakase_mission_open_ended.json` | `committed_itinerary_rate.median` for gpt-4o-mini | Run #2 flag-off summary.json via `write_baselines.py` | YES — 1.000 from actual 5-run measurements | VERIFIED |
| `docs/promotion_decision.md` latency table | `latency_seconds`, `llm_call_seconds`, `tool_exec_seconds` | Run #2 run JSONs `queries[i].deterministic.step_telemetry` and `queries[i].latency_seconds` | YES — per-run values tabulated | VERIFIED |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Gate checker passes with current baselines | `make eval-gates-check-baselines` | `check_eval_gates: OK — 7 gates checked` exit 0 | PASS |
| Baseline freshness check passes | `git fetch origin main && python scripts/check_baselines_fresh.py origin/main` | `check_baselines_fresh: OK — no watch-set changes vs origin/main (22 paths changed total)` exit 0 | PASS |
| Full test suite passes | `make test` | `1408 passed, 51 skipped, 9 deselected, 17 warnings in 33.99s` | PASS |
| Verdict docs unchanged | `git diff --quiet -- docs/decisiveness_arm_verdicts.md docs/replay_arm_verdicts.md` | exit 0 | PASS |
| FORCED_COMMIT_STEP prod default is 0 (flag off) | `grep "FORCED_COMMIT_STEP.*default.*0" app/agent/graph.py` | Line 326: `int(os.environ.get("FORCED_COMMIT_STEP", "0") or "0")` | PASS |
| graph.py unchanged in Phase 15 | `git diff ec4343c HEAD -- app/agent/graph.py \| wc -l` | 0 lines diff | PASS |
| Run #1 dir exists with 31 files | `ls eval_reports/2026-06-14T23-44-15Z/ \| wc -l` | 31 | PASS |
| Run #2 dir exists with 31 files | `ls eval_reports/2026-06-15T00-46-43Z/ \| wc -l` | 31 | PASS |
| Scenario-scoped gate returns not_evaluable (not silent pass) when gated scenario absent | Python test of `_check_gate` | `not_evaluable` | PASS |

---

### Probe Execution

Step 7c: SKIPPED — No `scripts/*/tests/probe-*.sh` files found. Phase 15 is an eval-harness / docs / gate-config phase; the mandatory pre-matrix provider probes were live runs during Phase 15 execution (not verifier-re-runnable without live API keys).

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PROMO-01 | 15-02-PLAN.md, 15-03-PLAN.md | Baselines regenerated honest n=5 via write_baselines.py for all runnable matrix cells | SATISFIED | `configs/eval_baselines/*.json` carry `generated_at: 2026-06-15T01-57-36Z`; 6 runnable cells; write_baselines exit 0 documented; check_baselines_fresh exits 0 confirmed in this session |
| PROMO-02 | 15-01-PLAN.md, 15-03-PLAN.md | Gates promoted where data earns it, explicitly logged where it doesn't | SATISFIED | `configs/eval_gates.yaml`: gpt-4o-mini `active` >= 0.8 (data earns it); gpt-5-mini `logged` (data doesn't, pooled=0.500 < 0.600); deepseek/anthropic/gemini `logged`; gate checker exits 0 |
| PROMO-03 | 15-03-PLAN.md | Latency report with honest anchor decomposition and prod-driver recommendation | SATISFIED | `docs/promotion_decision.md` PROMO-03 section: 47s median over 30s budget stated honestly; LLM/tool/overhead decomposition; decisiveness as dominant lever; ratify-anchor recommendation |

**Orphaned requirements:** None. All three PROMO requirements claimed by Phase 15 plans are covered.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `.planning/REQUIREMENTS.md` | 96-98 | Traceability rows have "Complete — <notes>" format, but 15-04-PLAN.md acceptance-criteria regex requires strict `\| Complete \|` (no trailing text) | Warning | SUMMARY claimed verify passed; it would not have. Data content is correct. Format discrepancy only. |
| `15-04-SUMMARY.md` | 78-80 | Claims `grep -Eq '\|\s*PROMO-01\s*\|\s*Phase 15\s*\|\s*Complete\s*\|' .planning/REQUIREMENTS.md` — PASS; this regex would actually return non-zero due to trailing notes in column | Warning | Self-check accuracy issue in SUMMARY. Milestone goal still achieved — rows are substantively Complete. |

No TBD/FIXME/XXX debt markers found in Phase 15 modified files. No stubs or placeholder returns in `scripts/check_eval_gates.py`, `configs/eval_gates.yaml`, or `docs/promotion_decision.md`.

---

### Human Verification Required

#### 1. REQUIREMENTS.md Traceability Row Format

**Test:** Read `.planning/REQUIREMENTS.md` lines 96-98. Run `grep -Eq '\|\s*PROMO-01\s*\|\s*Phase 15\s*\|\s*Complete\s*\|' .planning/REQUIREMENTS.md`.

**Expected:** Either (a) accept the current format `| PROMO-01 | Phase 15 | Complete — <notes> |` as sufficient for milestone traceability (the milestone intent is clearly satisfied), or (b) reformat the rows to match the strict regex by removing inline notes from the status column (moving them to a separate row or dropping them).

**Why human:** The SUMMARY claimed the acceptance-criteria regex passed during execution; this verifier confirmed it does not pass with the current row format. A human must decide whether to accept as-is (the data content is correct and the milestone goal is clearly achieved) or require format alignment. This is a bookkeeping accuracy question, not a goal-achievement blocker.

---

### Gaps Summary

No hard blockers found. All three PROMO requirements are substantively achieved and verified against the actual codebase:

- PROMO-01: 6 runnable baseline cells written with honest flag-off n=5 provenance stamps. Freshness check passes. Gate check passes.
- PROMO-02: gpt-4o-mini re-ratified at >= 0.8 (omakase-scoped); gpt-5-mini correctly stayed logged at pooled 0.500; no 0.0-floor gates; data-dependent rationale. Gate checker exits 0.
- PROMO-03: Honest 47s omakase latency (budget NOT met, explicitly stated); LLM/tool/overhead decomposition; prod-driver recommendation with honest latency caveat.

The `check_eval_gates.py` hard.scenarios deviation is sound: a family with no cell in gated scenarios returns `not_evaluable`, not a silent pass.

The single human item is a bookkeeping accuracy question (REQUIREMENTS.md row format) that does not block the phase goal.

---

### Locked-Decision Compliance

| Constraint | Status | Evidence |
|------------|--------|----------|
| ARCH-FUT-01 not executed | VERIFIED | `app/agent/graph.py` unchanged (0-line diff vs Phase 15 start). Word "ARCH-FUT-01" appears 9 times in promotion_decision.md — all in "DEFERRED" / "not executed" context. |
| Prod default not flipped to FORCED_COMMIT_STEP=6 | VERIFIED | `app/agent/graph.py` line 326: `default "0"`. Not in `.env`. |
| <= 4-run cap respected (2 full runs consumed) | VERIFIED | 4 eval_reports dirs from Phase 15: 2 smokes (`2026-06-14T23-33-58Z`, `2026-06-15T00-35-56Z`) + 2 full runs (`2026-06-14T23-44-15Z`, `2026-06-15T00-46-43Z`). Total full runs = 2 (smokes not counted per plan). |
| D-15-08 deferred, not re-opening ARCH-FUT-01 | VERIFIED | `docs/promotion_decision.md` "## D-15-08 Fix Disposition" section explicitly states "DEFERRED", explains why it exceeds one-line bar, and states "This DOES NOT re-open ARCH-FUT-01." graph.py unchanged. |

---

_Verified: 2026-06-15_
_Verifier: Claude (gsd-verifier)_
