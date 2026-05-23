---
phase: 04-category-compliance-fix
plan: 07
subsystem: testing
tags: [eval-baselines, merge-gate, regression-guard, docs, eval-matrix, openai, deepseek, rationale-stop-alignment, category-compliance-strict]

requires:
  - phase: 04-category-compliance-fix
    provides: 04-01 strict scorer + RevisionReason; 04-02 eval-config + runner bypass; 04-03 graph injection; 04-04 prompts; 04-05 revision dispatch; 04-06 intake pipeline. All of Phase 4 code must land before re-baselining so the new floor measures the post-Phase-4 behavior.
  - phase: 03-eval-harness-extension
    provides: configs/eval_baselines/ + scripts/check_baselines_fresh.py + scripts/eval_matrix.py post-process contract. 04-07 reuses these unchanged.
provides:
  - Post-Phase-4 eval baseline floor for the Phase 4-6 merge-gate diff (configs/eval_baselines/*.json)
  - Pre-Phase-4 snapshot preserved at .planning/phases/04-category-compliance-fix/04-pre-baselines-snapshot.json (T-04-07-01 supply-chain mitigation — old floor preserved as a diff-friendly anchor)
  - D-04-14 locked decision in 04-CONTEXT.md (absolute-floor reinterpretation of Gate 3 when pre-Phase-4 RAT-03 baseline was saturated via fail-open)
  - Reproducible gate-evaluation artifact (.planning/phases/04-category-compliance-fix/04-gate-evaluation.txt)
  - REQUIREMENTS.md + ROADMAP.md verified consistent with D-04-09 (RAT-01 + RAT-03 → Phase 4; RAT-02 → Phase 5)
affects: [phase-05-rationale-stop-alignment-fix, phase-04-merge-gate, future-baseline-rotations]

tech-stack:
  added: []
  patterns:
    - "Pre-/post-snapshot pattern for baseline-diff merge gates: snapshot the OLD floor as a checked-in JSON file BEFORE overwriting the source baselines, so the merge gate has unambiguous numbers to compare against (T-04-07-01 mitigation)."
    - "Missing-key gate-interpretation pattern: when a scorer is added in the phase being gated, a delta-vs-old-baseline is undefined; substitute a meaningful absolute floor and document the trade-off as a locked decision (D-04-13 for CAT-03 strict, D-04-14 for RAT-03 saturated baselines)."
    - "Per-scenario aggregate post-process: scripts/04-07-post-process.py (execution-time scratch under .planning/) reads summary.json from eval_matrix and writes one baseline JSON per scenario with the same shape downstream tooling already consumes."

key-files:
  created:
    - .planning/phases/04-category-compliance-fix/04-pre-baselines-snapshot.json (pre-Phase-4 floor snapshot, 355 lines)
    - .planning/phases/04-category-compliance-fix/04-07-post-process.py (matrix → per-scenario baseline helper)
    - .planning/phases/04-category-compliance-fix/04-gate-evaluation.txt (recorded gate verdict under D-04-14)
    - .planning/phases/04-category-compliance-fix/04-07-SUMMARY.md (this file)
  modified:
    - configs/eval_baselines/omakase_mission_open_ended.json (re-baselined post-Phase-4; category_compliance_strict scorer added)
    - configs/eval_baselines/refinement_cheaper.json (re-baselined; new scorer added)
    - configs/eval_baselines/late_night_closure_cascade.json (re-baselined for tracking per D-04-12)
    - .planning/phases/04-category-compliance-fix/04-CONTEXT.md (D-04-14 appended to <decisions> block alongside D-04-13)
    - scripts/eval_agent.py (Rule 2 deviation — wired category_compliance_strict into DETERMINISTIC_CHECKS)
    - tests/unit/test_eval_agent.py (fixture update for new scorer)
    - .planning/REQUIREMENTS.md (verified — RAT-01 + RAT-03 → Phase 4; D-04-09 scope note present)
    - .planning/ROADMAP.md (verified — Phase 5 narrowed to RAT-02 only; Phase 4 Plans 7/7 with full plan list; Coverage table updated)

key-decisions:
  - "D-04-14 locked: when the pre-Phase-4 baseline rationale_stop_alignment median is ≥ 0.95 (saturated via fail-open on non-convergence), Gate 3 is reinterpreted as an absolute floor of 0.8 instead of +0.2 delta. Same class of missing-key issue D-04-13 already resolved for CAT-03 strict. Result: 1.000 on both gated scenarios — Gate 3 PASSES."
  - "Re-baseline timing matters: D-04-13's delta gate is computed against POST-Phase-4 baselines, not the current ones. Step ordering: snapshot OLD floor → overwrite source baselines → compute gate vs snapshot."
  - "Side observation worth documenting: Phase 4's category compliance fix doubled as a geographic_coherence + walking_budget_respected improvement on omakase (0.000 → 1.000 on both). Categorically correct stops cluster geographically and pack into walking budgets — the slot-aware retrieval surfaces this for free."

patterns-established:
  - "Pre-Phase-4 baseline snapshot (force-added into gitignored .planning/) as the OLD-floor anchor for any future delta-vs-baseline merge gate."
  - "Locked decision protocol for missing-key gate problems: document the trade-off as a numbered decision (D-04-14 in this case) so reviewers see the rationale before PR merge."
  - "Two-tier gate reporting: print FULL math (all four gates × two scenarios × five guards) to the user BEFORE asking for the resume signal, never a summary line. The recorded gate-evaluation.txt is the durable artifact."

requirements-completed: [CAT-03, CAT-04, RAT-01, RAT-03]

duration: 55min
completed: 2026-05-22
---

# Phase 4 Plan 07: Re-Baseline + Docs Summary

**Post-Phase-4 eval baselines committed; Phase 4 merge gate verified PASS under D-04-14 (RAT-03 absolute floor 0.8); REQUIREMENTS.md + ROADMAP.md confirmed consistent with D-04-09 (RAT-01 folded into Phase 4).**

## Performance

- **Duration:** 55 min
- **Started:** 2026-05-22T22:41:18Z (pre-baseline snapshot)
- **Completed:** 2026-05-22T23:36:29Z (gate-evaluation commit)
- **Tasks:** 5 + 1 deviation (1 RED+GREEN: pre-snapshot → re-baseline matrix run → docs verify → checkpoint → D-04-14 lock → gate re-eval)
- **Files modified:** 9 (4 new under .planning/, 3 eval baselines, 2 eval_agent wiring files, 2 main-repo planning files verified)

## Accomplishments

- **Pre-Phase-4 floor preserved.** `04-pre-baselines-snapshot.json` captures the OLD baseline as a diff-friendly anchor BEFORE the baselines were overwritten by the re-run. This is the supply-chain mitigation for T-04-07-01 (tampered new baselines diverge visibly against the snapshot in PR review).
- **Eval matrix re-run committed.** `APP_ENV=eval make eval-matrix RUNS=3` against the post-Phase-4 codebase produced 18 cells (2 providers × 3 scenarios × 3 runs). The three baseline JSONs in `configs/eval_baselines/` are now overwritten with the post-Phase-4 floor including the new `category_compliance_strict` scorer.
- **Strict scorer wired into eval_agent.** Discovered mid-execution that plan 04-01 added `category_compliance_strict` to `app/agent/critique/checks.py` + `CRITIQUE_THRESHOLDS` + `itinerary_violations` but never wired it into `scripts/eval_agent.py::DETERMINISTIC_CHECKS`. Without this wiring, the eval matrix never computed the strict scorer and the D-04-13 merge gate couldn't be evaluated. Fixed as a Rule 2 deviation in commit `0876947` (see Deviations section).
- **Merge-gate verdict: PASSES.** Under D-04-14 (locked during this plan's checkpoint), all four gates pass on both gated scenarios for openai/gpt-4o-mini:
  - Gate 1 (CAT-03 strict absolute floor 0.3): omakase=1.000, refinement=1.000 — PASS
  - Gate 2 (family-level regression guard, Δ ≥ 0): omakase 1.000 vs 1.000, refinement 1.000 vs 1.000 — PASS
  - Gate 3 (RAT-03 absolute floor 0.8 per D-04-14): omakase=1.000, refinement=1.000 — PASS
  - Gate 4 (5 guard scorers, no regression): all PASS; **omakase showed +1.000 improvements on `geographic_coherence` and `walking_budget_respected` (0.000 → 1.000)** — Phase 4's category fix doubled as a geo+walking win.
- **D-04-14 locked.** Appended to `<decisions>` block in `04-CONTEXT.md` alongside D-04-13 (commit `9a75bc3`). Codifies the absolute-floor reinterpretation of Gate 3 for saturated-baseline cases.
- **REQUIREMENTS.md + ROADMAP.md verified.** Edits exist in the main repo's `.planning/` (gitignored, will be persisted by the orchestrator post-worktree). Confirmed:
  - REQUIREMENTS.md: RAT-01 → Phase 4 (line 115), RAT-02 → Phase 5 (line 116), RAT-03 → Phase 4 (line 117); D-04-09 scope note at line 47.
  - ROADMAP.md: Phase 5 Requirements line shows "RAT-02" only; D-04-09 narrowing note included; Phase 4 Plans 7/7 with full plan list; Coverage table maps "RAT-01, RAT-03 → Phase 4".

## Task Commits

The plan opened a worktree at `worktree-agent-ab32e60189bdee697`. All commits below are on that branch.

1. **Task 1: Snapshot pre-Phase-4 baselines** — `3aa7042` (`data(04-07): snapshot pre-Phase-4 baselines for D-04-13 merge-gate delta`)
2. **Rule 2 deviation: Wire strict scorer into eval_agent** — `0876947` (`fix(04-07): register category_compliance_strict in eval_agent DETERMINISTIC_CHECKS`)
3. **Task 2: Re-run eval-matrix RUNS=3 and overwrite baselines** — `d68f903` (`data(04-07): re-baseline configs/eval_baselines after Phase 4 code lands`)
4. **Task 4a: Lock D-04-14 in CONTEXT.md** — `9a75bc3` (`docs(04-07): lock D-04-14 absolute-floor RAT-03 gate reinterpretation`)
5. **Task 4b: Re-evaluate merge gate under D-04-14** — `da49f68` (`docs(04-07): re-evaluate merge gate under D-04-14 — PASSES`)

**Plan metadata commit:** to be appended after this SUMMARY.md lands (final step of the executor).

_Task 3 (REQUIREMENTS.md + ROADMAP.md) and Task 5 (verify) did not produce worktree-side commits — `.planning/REQUIREMENTS.md` and `.planning/ROADMAP.md` live in the main repo (gitignored from the worktree per `.gitignore:239`). Orchestrator handles persistence in the main repo after worktree cleanup._

## Files Created/Modified

### New (force-added under gitignored .planning/)
- `.planning/phases/04-category-compliance-fix/04-pre-baselines-snapshot.json` — 355-line snapshot of the OLD baseline floor with `snapshot_taken_at` + `git_sha` (`972e6fe`) + the 3 scenario blocks for both providers.
- `.planning/phases/04-category-compliance-fix/04-07-post-process.py` — Execution-time helper that reads `eval_reports/<ts>/summary.json` from eval_matrix and writes one baseline JSON per scenario with the post-Phase-4 `_observations` text.
- `.planning/phases/04-category-compliance-fix/04-gate-evaluation.txt` — Recorded gate-math output (the FULL four-gates × two-scenarios × five-guards math, not a summary line) under D-04-14. Final verdict: **GATE PASSES**.
- `.planning/phases/04-category-compliance-fix/04-07-SUMMARY.md` — This file.

### Modified
- `configs/eval_baselines/omakase_mission_open_ended.json` — Re-baselined; `category_compliance_strict` scorer present; `_observations` updated to reflect post-Phase-4 measurement.
- `configs/eval_baselines/refinement_cheaper.json` — Same.
- `configs/eval_baselines/late_night_closure_cascade.json` — Same (tracked but not gated per D-04-12).
- `.planning/phases/04-category-compliance-fix/04-CONTEXT.md` — D-04-14 appended to the `<decisions>` block under "Grading + gates" alongside D-04-13.
- `scripts/eval_agent.py` — Rule 2 deviation: added `category_compliance_strict` import and entry in `DETERMINISTIC_CHECKS`.
- `tests/unit/test_eval_agent.py` — Fixture updated so `aggregate_results` does not `KeyError` on the new scorer.

### Verified (no worktree-side write)
- `.planning/REQUIREMENTS.md` (main repo) — Confirmed RAT-01 + RAT-03 at Phase 4, RAT-02 at Phase 5; D-04-09 scope note present at line 47.
- `.planning/ROADMAP.md` (main repo) — Confirmed Phase 5 Requirements line "RAT-02" only, Phase 4 Plans 7/7 with full plan list, Coverage table updated.

## Decisions Made

- **D-04-14 (locked, 2026-05-22):** When the pre-Phase-4 baseline median for `rationale_stop_alignment` is ≥ 0.95 (saturated via fail-open on non-convergence), Gate 3 of D-04-13 is reinterpreted as an absolute floor of 0.8 instead of a +0.2 delta. Same class of missing-key issue D-04-13 already resolved for `category_compliance_strict`. The 0.8 floor preserves the spirit of "no regression on a meaningful threshold" without requiring impossible-by-construction headroom. Phase 4 measured 1.000 on both gated scenarios — Gate 3 PASSES under D-04-14. RAT-03 metric definition is flagged for revisit in a future phase: the saturated baseline reflects scorer abstention, not "rationale-alignment quality on committed itineraries".

- **Force-add convention for plan artifacts under .planning/.** Followed the convention established by `3aa7042` (pre-baselines-snapshot) and earlier Phase 4 SUMMARY commits: `.planning/` is gitignored globally but per-phase artifacts in `.planning/phases/04-*/` get `git add -f` so reviewers see the snapshot, gate-evaluation output, and SUMMARY in PR diffs.

- **Plan stopped short of pushing/PR.** Per the user's GSD workflow (`feedback_user_merges_prs.md`), the executor finishes at "branch is gate-ready"; the user opens the PR.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 — Missing Critical Functionality] `category_compliance_strict` not wired into `scripts/eval_agent.py::DETERMINISTIC_CHECKS`**
- **Found during:** Task 2 (eval-matrix re-run). The first 3 cell JSONs included `category_compliance_mean` but NOT `category_compliance_strict_mean`, which would have broken the task-2 acceptance grep `assert 'category_compliance_strict' in b['providers'][...]['scorers']`.
- **Issue:** Plan 04-01 added `category_compliance_strict` to `app/agent/critique/checks.py` + `CRITIQUE_THRESHOLDS` + `itinerary_violations`, but never wired it into `scripts/eval_agent.py`'s `DETERMINISTIC_CHECKS` dict. Without that wiring, `eval_matrix.py`'s post-process never computed `category_compliance_strict`, so the D-04-13 merge gate ("strict median ≥ 0.3 absolute floor") could not be evaluated against the new baselines.
- **Fix:** Added `category_compliance_strict` to the import block and the `DETERMINISTIC_CHECKS` dict in `scripts/eval_agent.py`; updated the query-result fixture in `tests/unit/test_eval_agent.py` so `aggregate_results` doesn't `KeyError` on the new scorer key.
- **Files modified:** `scripts/eval_agent.py`, `tests/unit/test_eval_agent.py`.
- **Verification:** After this fix, re-running `make eval-matrix RUNS=3` produced cell JSONs with `category_compliance_strict_mean` populated, and the task-2 acceptance grep passed.
- **Committed in:** `0876947` (`fix(04-07): register category_compliance_strict in eval_agent DETERMINISTIC_CHECKS`).
- **Why this was out-of-scope for plan 04-01 but in-scope for plan 04-07:** Plan 04-01 was scoped to checks.py changes; its plan author assumed `DETERMINISTIC_CHECKS` was auto-derived from `itinerary_violations`. It's not — eval_agent maintains its own explicit dict. The gap surfaced only when 04-07 ran eval-matrix and looked for the new key.

**2. [Rule 1 — Bug] REQUIREMENTS.md Traceability had RAT-02 → Phase 4 from out-of-band drift**
- **Found during:** Task 3 (REQUIREMENTS.md edit). Per the plan's `<read_first>` notes (line 142-147), the planner asserted RAT-01 and RAT-03 were ALREADY Phase 4 from an earlier reshuffle. On verification the rows were indeed at Phase 4 (lines 115 and 117). But a tangential row had drifted: an earlier branch-state had RAT-02 listed under "Phase 4" temporarily before the D-04-09 narrowing was finalized.
- **Issue:** Out-of-band edit drift: the row reassignment from RAT-02 → Phase 5 had not made it back into the file after D-04-09 was locked.
- **Fix:** Restored the canonical state: RAT-01 → Phase 4, RAT-02 → Phase 5, RAT-03 → Phase 4. Added the D-04-09 scope note at line 47 of the "Rationale-Stop Alignment Fix" section explaining the fold.
- **Files modified:** `.planning/REQUIREMENTS.md`.
- **Verification:** `grep -n "RAT-0[123]" .planning/REQUIREMENTS.md` shows the three rows at correct phases; `grep -n "D-04-09" .planning/REQUIREMENTS.md` shows the scope note present.
- **Committed in:** Orchestrator commits this in the main repo after worktree cleanup.

### Deferred Issues (out of scope, logged for future)

- **REQUIREMENTS.md uses "Phase 1..5" headers while ROADMAP.md uses "Phase 2..6".** The two files have a long-standing numbering drift from pre-v2.0 reshuffles. REQUIREMENTS.md numbers internal sections "Phase 1..5" (still using pre-v2.0 labels); ROADMAP.md is on the new "Phase 2..6" numbering. The Traceability column in REQUIREMENTS.md DOES use the new numbering (the rows say `Phase 4`, `Phase 5`, etc., consistent with ROADMAP.md), but the section headers above the table do not. Fixing this is out of scope for plan 04-07; it should be a separate doc-bookkeeping commit at the start of Phase 5.

---

**Total deviations:** 2 auto-fixed (1 Rule 2 missing critical, 1 Rule 1 bug); 1 deferred out-of-scope.
**Impact on plan:** Both auto-fixes were necessary for the plan's own acceptance criteria. The Rule 2 fix lives on the worktree branch as a commit; the Rule 1 fix is verified in the main repo and the orchestrator handles persistence. No scope creep — both fixes were needed to complete 04-07's stated work.

## Issues Encountered

- **Gate failed under literal D-04-13 reading.** Initial run of the gate-eval one-liner produced GATE FAILS because the +0.2 delta on `rationale_stop_alignment` was unsatisfiable (old baseline 1.000 saturated from fail-open; you can't be +0.2 above 1.000). User-decided option (c) of the Escalation Gate: lock D-04-14 with absolute floor 0.8. Gate now PASSES.

## Side Observations

- **omakase_mission_open_ended openai went from 0.000 → 1.000 on TWO guard scorers.** Pre-Phase-4: `geographic_coherence` median 0.000, `walking_budget_respected` median 0.000 (the model committed itineraries spread across SoMa/Bernal Heights/Outer Mission). Post-Phase-4: both 1.000. The slot-aware retrieval (`primary_type_family` injection in graph layer + `slot_index` kwarg) makes the model pull category-correct candidates from the local cluster around the user's neighborhood — they cluster geographically AND pack into walking budgets for free. Worth flagging as a v2.0 success story beyond the literal "category compliance" framing.
- **DeepSeek baselines: still saturated at 1.000 via fail-open.** Per D-04-11, DeepSeek is tracked but not gated; the decisiveness-gap memory `project_deepseek_decisiveness_gap.md` documents 0/9 commits on real-provider runs. Phase 4 doesn't change that — DeepSeek scorers continue to read 1.000 because no itineraries are committed, not because the category enforcement worked.
- **late_night_closure_cascade openai: `geographic_coherence` and `walking_budget_respected` went 1.000 → 1.000 (median) but stdev 0.577.** Some runs scored 0.0, some 1.0. Per D-04-12 this scenario is tracked-not-gated, and per memory `project_eval_multi_turn_threading_bug.md` the eval-harness threading mismatch makes the measurement non-prod-comparable. Phase 5 (RAT-02) will revisit.

## Next Phase Readiness

- **Phase 4 is gate-ready for PR.** The branch on this worktree (`worktree-agent-ab32e60189bdee697`) has all the Phase 4 commits + the merge-gate artifacts. The next step (out-of-scope for this plan) is: orchestrator merges worktree → user opens PR → user merges to main per `feedback_user_merges_prs.md`.
- **D-04-14 must be cited in the PR description.** Reviewers should see the absolute-floor reinterpretation explained alongside D-04-13's CAT-03 absolute-floor decision. Both decisions are now locked in CONTEXT.md.
- **Phase 5 narrowed.** RAT-02 only (closure-swap placeholder bleed). Per memory `project_v2_milestone_active.md`, the next planning step is `/gsd:plan-phase 5`.
- **RAT-03 metric flagged for revisit.** The fail-open saturated baseline means `rationale_stop_alignment` as currently defined doesn't actually measure rationale-alignment quality on committed itineraries — it measures "didn't abstain". Future phase should decide whether to (a) tighten the scorer to fail-closed on empty stops, (b) reformulate it to compare committed primary_type against rationale keywords, or (c) accept the current definition with a documented interpretation note.

## Self-Check: PASSED

- All 10 files referenced in this SUMMARY exist on disk (worktree + main repo verified).
- All 5 task commits exist in `git log --all`: `3aa7042`, `0876947`, `d68f903`, `9a75bc3`, `da49f68`.
- Gate-eval verdict (`04-gate-evaluation.txt`): GATE PASSES under D-04-14.

---
*Phase: 04-category-compliance-fix*
*Completed: 2026-05-22*
