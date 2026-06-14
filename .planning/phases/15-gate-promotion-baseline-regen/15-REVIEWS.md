---
phase: 15
reviewers: [codex]
reviewed_at: 2026-06-14T22:21:05Z
plans_reviewed: [15-01-PLAN.md, 15-02-PLAN.md, 15-03-PLAN.md, 15-04-PLAN.md]
---

# Cross-AI Plan Review — Phase 15

> Reviewer note: only Codex was requested (`--codex`). The `claude` CLI is skipped
> for independence because this review was orchestrated from within Claude Code.
> A first Codex invocation was quota-blocked; a retry succeeded and produced the
> review below.

## Codex Review

**Overall Findings**
The plan set is strong and mostly preserves the Phase 15 honesty constraints: diagnose before retest, separate experiment vs flag-off baseline runs, baseline regeneration last, and no gate promotion unless earned. The main risks are provenance leaks from inherited experiment env vars, a few places where expected results are pre-baked instead of computed, incomplete recovery paths for partial live runs, and final closure language that should remain data-dependent after the A2 retest.

## 15-01-PLAN.md

**Summary**  
Good zero-spend diagnostic plan with the right ordering and a disciplined trivial-fix boundary. It correctly makes `docs/promotion_decision.md` the new record and avoids appending to closed verdict docs. The main weakness is that the action text partially assumes the diagnostic conclusion before reading the data.

**Strengths**
- Correctly runs before live spend, preserving D-15-09.
- Uses existing telemetry fields instead of new diagnostic runs.
- Keeps the D-15-08 fix boundary narrow and explicit.
- Separates diagnostic outcome from ARCH-FUT-01, preventing scope creep.

**Concerns**
- **MEDIUM:** The plan says to “establish the contrast” with expected arrays. That risks confirmation bias if actual run JSONs differ.
- **MEDIUM:** `git status --porcelain eval_reports/` is not a reliable zero-spend check if there were pre-existing untracked or modified eval artifacts.
- **LOW:** `files_modified` lists `app/agent/graph.py` but not tests, even though a shipped fix requires a regression test.
- **LOW:** It focuses heavily on the A2 run dir; if the claim is “across every arm,” the doc should either inspect Phase 14 artifacts too or narrow the claim.

**Suggestions**
- Make the diagnostic explicitly data-driven: “tabulate all runs and report actual values, including contradictions.”
- Capture an eval_reports before/after snapshot for the zero-spend assertion.
- Add test files to optional `files_modified` if a graph fix ships.
- Require the doc to name exactly which run dirs were inspected and which were not.

**Risk Assessment: MEDIUM**  
The implementation risk is low, but provenance risk is medium because the plan currently nudges the executor toward a predetermined finding.

## 15-02-PLAN.md

**Summary**  
This is the core measurement plan and it correctly separates the A2 experiment from the flag-off baseline source. The human checkpoint, smoke-before-spend pattern, and anchor non-regression checks are well placed. The largest issue is that “all flags unset” is stated but not mechanically enforced in the commands.

**Strengths**
- Preserves the two-run discipline: experiment run is never used for baselines.
- Uses smoke runs with `arm_flags` verification before n=5 spend.
- Explicitly checks forced-path telemetry, not just commit rate.
- Treats A2 as measured input, not a hard phase gate.

**Concerns**
- **HIGH:** Run #2 can inherit `FORCED_COMMIT_STEP` or other experiment env vars from the shell. `make eval-matrix-arm RUNS=5` does not itself prove flags are unset.
- **MEDIUM:** Run-cap accounting is ambiguous. Two n=1 smokes plus two n=5 runs are four live matrix invocations, while the plan describes only two runs against the cap.
- **MEDIUM:** Partial run handling is under-specified. A partial cannot become a baseline source, but the retry/stop path within the <=4-run cap is not clear.
- **MEDIUM:** “A2 retest is only meaningful if the forced path fired” needs nuance: if `all_slots_viable` never becomes true on refinement, non-fire is itself a meaningful result.
- **LOW:** “Median holds >= prior baseline” should name the exact metric and threshold per scenario or family.

**Suggestions**
- Use explicit clean-env commands for Run #2, e.g. unset every experiment flag in the invocation.
- For Run #1, explicitly unset all non-A2 flags while setting only `FORCED_COMMIT_STEP=6`.
- Define cap accounting: whether smokes count as live matrix runs or only full n=5 runs.
- Add a partial-run decision tree: retry within cap, mark blocked, or record partial and skip regen.
- Define anchor non-regression using the same metric `check_eval_gates.py` enforces.

**Risk Assessment: HIGH**  
The measurement design is sound, but a single leaked env var would invalidate the baseline provenance and gate decisions.

## 15-03-PLAN.md

**Summary**  
This plan correctly translates measurements into durable artifacts and is the strongest on D-15-04/05/06/07 provenance. Baseline regeneration through `write_baselines.py` and the latency decomposition are well scoped. The main issue is some confusing gpt-5 gate rationale wording if A2 clears but flag-off does or does not support enforcement.

**Strengths**
- Correctly uses Run #2 flag-off `summary.json` as the only baseline source.
- Keeps experiment data out of baseline generation.
- Avoids meaningless 0.0 enforced gates.
- Requires both gate checks and baseline freshness checks.
- Latency reporting is honest and decomposed by LLM/tool time.

**Concerns**
- **MEDIUM:** The gpt-5 enforcement branch says the forced path is the mechanism even though the hard gate must be sourced from flag-off CI behavior. That can blur D-15-07 provenance.
- **MEDIUM:** `check_baselines_fresh.py origin/main` assumes `origin/main` is the right comparison point and locally available.
- **MEDIUM:** Latency aggregation should reconcile `latency_seconds` against summed `step_telemetry`; otherwise overhead may be silently lost.
- **LOW:** Gate editing happens before baseline regeneration; acceptable, but the final gate check after regen is the one that matters most.

**Suggestions**
- Rewrite gpt-5 rationale rules: if flag-off earns the floor, enforce on flag-off data; if only A2 earns it, do not enforce and document forced-commit as experimental.
- Include run count, median/p95 or min/max, and step count in the latency report.
- Report telemetry sum plus observed `latency_seconds` so orchestration overhead is visible.
- Confirm the exact baseline freshness base ref before running the check.

**Risk Assessment: MEDIUM**  
The core mechanics are solid; risk is mainly in wording/provenance clarity and final freshness assumptions.

## 15-04-PLAN.md

**Summary**  
The closure plan correctly makes `docs/promotion_decision.md` the milestone audit anchor and protects closed verdict docs. It also handles planning bookkeeping in the established style. The risky part is that it hardcodes “honest null result” language even though the A2 retest outcome is not known until Plan 02 completes.

**Strengths**
- Cross-links immutable verdict docs instead of modifying them.
- Consolidates all Phase 15 outputs into one reviewer-facing record.
- Updates ROADMAP, REQUIREMENTS, and STATE after artifacts are produced.
- Explicitly records anchor ratification, ARCH-FUT-01 deferral, and prod-default non-flip.

**Concerns**
- **HIGH:** Final summary language must be data-dependent. If the A2 retest clears, “no arm cleared INST-05” becomes false or at least needs qualification.
- **MEDIUM:** “Gate numbers are NOT duplicated” conflicts with prior sections that necessarily include measured rates and deltas. Clarify that `eval_gates.yaml` remains source of truth for enforced thresholds.
- **MEDIUM:** Committing straight to main needs a clean-worktree/focused-commit preflight so unrelated changes are not swept in.
- **LOW:** Verification greps are broad; they do not prove PROMO-01/02/03 were updated correctly.

**Suggestions**
- Make closing language conditional: “If A2 did not clear…” vs “If A2 cleared experimentally but not flag-off…”
- Add final preflight: `git status --short`, verify only intended files changed, and confirm verdict docs unchanged.
- Add a final consolidated verification step: gate check, baseline freshness check, and full test status if any code changed in Plan 01.
- Clarify that measured rates may appear in the doc, but enforced gate thresholds live in `configs/eval_gates.yaml`.

**Risk Assessment: MEDIUM**  
Mostly documentation/bookkeeping risk, but the hardcoded null-result language could undermine the phase’s provenance honesty.

---

## Consensus Summary

Only one reviewer (Codex) was invoked, so there is no cross-reviewer agreement to
compute. The synthesis below ranks Codex's findings by severity and impact so the
planner can act on the highest-priority items first.

### Highest-Priority Concerns (HIGH)

1. **Run #2 env-var provenance leak (15-02).** `make eval-matrix-arm RUNS=5` does not
   itself prove experiment flags are unset — an inherited `FORCED_COMMIT_STEP` (or other
   arm flag) from the shell would silently contaminate the flag-off baseline source and
   invalidate D-15-07 provenance. The smoke `arm_flags` check catches it *if read*, but
   the invocation should explicitly clear the environment. Likewise Run #1 should set
   ONLY `FORCED_COMMIT_STEP=6` and unset every other arm flag.
2. **Hardcoded null-result language (15-04).** The closure plan pre-writes "no arm cleared
   INST-05 / honest null result," but that statement is only true if the Plan 02 A2 retest
   does NOT clear 0.6. The closing language must be data-dependent on the Plan 02 outcome.

### Notable Medium Concerns

- **Run-cap accounting (15-02):** two n=1 smokes + two n=5 runs = four live matrix
  invocations, yet the plan frames it as "two runs against the ≤4-run cap." Define
  whether smokes count.
- **Partial-run handling (15-02):** the retry/stop/record-partial decision tree within
  the cap is under-specified.
- **Non-fire is meaningful (15-02):** "A2 only meaningful if forced path fired" needs
  nuance — if `all_slots_viable` never holds on refinement, a non-fire is itself the result.
- **gpt-5 gate-rationale wording (15-03):** the "forced path is the mechanism" rationale
  can blur D-15-07 (hard gate value must come from flag-off CI behavior, not the experiment).
- **`check_baselines_fresh.py origin/main` base ref (15-03):** assumes `origin/main` is the
  right, locally-available comparison point — confirm before running.
- **Latency reconciliation (15-03):** reconcile observed `latency_seconds` against summed
  `step_telemetry` so orchestration overhead isn't silently lost.
- **Straight-to-main preflight (15-04):** bookkeeping commits need a clean-worktree /
  focused-commit preflight so unrelated changes aren't swept in.

### Lower-Priority / Polish (LOW)

- Diagnostic should report ACTUAL telemetry values (including contradictions), not pre-baked
  expected arrays — confirmation-bias guard (15-01).
- Zero-spend check should use a before/after `eval_reports/` snapshot, not just
  `git status --porcelain` (15-01).
- Add test files to optional `files_modified` when a graph fix ships (15-01).
- Verification greps are broad; they don't prove PROMO-01/02/03 were updated correctly (15-04).

### Reviewer Risk Verdicts

| Plan  | Codex Risk |
|-------|------------|
| 15-01 | MEDIUM (provenance/confirmation-bias, not implementation) |
| 15-02 | **HIGH** (single leaked env var invalidates baseline provenance) |
| 15-03 | MEDIUM (wording/provenance clarity + freshness base ref) |
| 15-04 | MEDIUM (hardcoded null-result language is the sharp edge) |

### Divergent Views

None — single reviewer.
