---
phase: 04-category-compliance-fix
plan: 05
subsystem: agent
tags: [revision-loop, critique-dispatch, rationale-alignment, dry-helper, tdd]

requires:
  - phase: 03-eval-harness-extension
    provides: rationale_stop_alignment scorer registered in itinerary_violations
  - phase: 04-category-compliance-fix
    provides: 04-01's rationale_misaligned literal on RevisionReason
provides:
  - is_rationale_aligned public helper (single source of truth for the per-stop alignment rule)
  - _first_misaligned_stop_index helper that consumes the shared rule
  - rationale_misaligned dispatch branch in _hint_for_violation
  - In-loop self-correct: rationale_stop_alignment violations now produce RevisionHints instead of catch-all constraint_unmet_in_final
affects: [phase-04-category-compliance-fix, agent-revision-loop, critique-dispatch]

tech-stack:
  added: []
  patterns:
    - Public-helper DRY extraction shared by scorer + dispatcher
    - Parallel-branch addition to existing _hint_for_violation dispatcher
    - TDD RED/GREEN commits per task

key-files:
  created:
    - .planning/phases/04-category-compliance-fix/04-05-SUMMARY.md
  modified:
    - app/agent/critique/checks.py
    - app/agent/revision.py
    - tests/unit/test_critique_checks.py
    - tests/unit/test_agent_self_correct.py

key-decisions:
  - "is_rationale_aligned is the single source of truth for the per-stop name-or-family-keyword rule; both rationale_stop_alignment (scorer) and _first_misaligned_stop_index (dispatcher) call it."
  - "Budget gate keys by check name (rationale_stop_alignment), so the new reason's two-retry budget is independent of constraint_unmet_in_final."
  - "Existing RevisionAction='swap_stop' is reused because it matches the action vocabulary the model already understands; the reason key (rationale_misaligned) is the differentiator."
  - "RevisionHint.target carries 0-based stop_index for programmatic use; the human-readable detail string uses 1-indexed Stop N for the model and observers."

patterns-established:
  - "Per-stop scorer rules can be safely extracted into public helpers when a dispatcher needs the SAME rule (DRY); a regression test on a fixed fixture locks byte-identical scorer output."
  - "Closure-swap placeholder bleed scenarios serve as canonical fixtures for the rationale alignment branch in dispatcher tests."

requirements-completed: [CAT-02, RAT-01]

duration: 18 min
completed: 2026-05-22
---

# Phase 04 Plan 05: Revision Hint Dispatch Summary

**The agent self-corrects misaligned rationales in-loop: rationale_stop_alignment violations now produce rationale_misaligned RevisionHints (via a shared public per-stop helper) so the model gets two retries to rewrite before the agent ships with caveats.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-05-22T21:57:23Z
- **Completed:** 2026-05-22T22:15:02Z
- **Tasks:** 2 (each as TDD RED/GREEN pair)
- **Files modified:** 4

## Accomplishments

- Extracted `is_rationale_aligned(stop) -> bool` as a public helper in `app/agent/critique/checks.py` and refactored `rationale_stop_alignment` to call it inside its loop — observable scorer behavior is byte-identical (locked by `test_rationale_stop_alignment_behavior_unchanged_after_extraction` on a fixed 4-stop fixture).
- Added `_first_misaligned_stop_index(state) -> int` in `app/agent/revision.py` that consumes the shared public helper (no per-stop logic duplicated across modules).
- Added a new branch in `_hint_for_violation` for `reason == "rationale_stop_alignment"` that emits `RevisionHint(reason="rationale_misaligned", target={"stop_index": <0-based>}, suggested_action="swap_stop")` with a 1-indexed detail string.
- Verified the existing `critique_final_with_stops` driving loop (revision.py:266-283) picks up the new branch unchanged and feeds the HumanMessage with `CRITIQUE_ITINERARY` prefix into the next plan() step.
- Verified the existing two-retry budget contract is honored — `MAX_REVISIONS_PER_REASON` is unchanged, and the budget is keyed by check name (`rationale_stop_alignment`) so the new reason gets an independent budget naturally.

## Task Commits

1. **Task 1 RED: failing tests for is_rationale_aligned + regression fixture** - `aacc5e0` (test)
2. **Task 1 GREEN: extract is_rationale_aligned public helper** - `df4fc39` (feat)
3. **Task 2 RED: failing tests for rationale_misaligned dispatch** - `3ea038e` (test)
4. **Task 2 GREEN: wire rationale_misaligned dispatch + _first_misaligned_stop_index** - `d7d5750` (feat)

**Plan metadata:** committed separately with this summary.

## Files Created/Modified

- `app/agent/critique/checks.py` - Added `is_rationale_aligned(stop) -> bool` (public helper). Refactored `rationale_stop_alignment` to call it inside `sum(...)`. Added `Stop` to the `app.agent.state` import.
- `app/agent/revision.py` - Added `_first_misaligned_stop_index(state) -> int` helper between `_diagnose_last_tool_result` and `_hint_for_violation`. Added new `if reason == "rationale_stop_alignment":` branch in `_hint_for_violation` before the catch-all return. Added `is_rationale_aligned` to the `app.agent.critique.checks` import.
- `tests/unit/test_critique_checks.py` - Added `is_rationale_aligned` to the imports. Added 5 helper tests (name match, family-keyword match, neither match, None primary_type, empty rationale) and 1 regression test pinning `rationale_stop_alignment` output on a fixed 4-stop fixture.
- `tests/unit/test_agent_self_correct.py` - Extended `test_hint_for_violation_maps_each_check` parametrize with a new `("rationale_stop_alignment", "rationale_misaligned", "swap_stop")` tuple (and gave the fixture stops misaligned rationales so the dispatcher has something to point at). Added 3 helper tests for `_first_misaligned_stop_index` (first-offender index, None primary_type, empty stops). Added 2 functional tests for the dispatch path: closure-placeholder bleed emits the hint with the right shape/budget/HumanMessage, and a budget-exhausted state ships with caveats.

## Decisions Made

- **DRY refactor with a regression lock.** The plan's ADVISORY 3 said to extract a shared helper; locked the zero-behavior-change with `test_rationale_stop_alignment_behavior_unchanged_after_extraction` which hand-computes the expected score against the pre-extraction scorer body on a 4-stop fixture covering every branch (name-match, family-keyword-match, neither, None primary_type).
- **`is_rationale_aligned` is defensive on empty/None rationale.** The pre-extraction scorer body coerced `rationale.lower()` after the `if stop.rationale else ""` guard. The new helper preserves that — empty rationale returns False rather than raising. This is a tightening, not a behavior change, because `rationale` is required by Pydantic (`Stop.rationale: str`), so the empty-string path was only reachable when callers passed `""` explicitly.
- **The parametrized dispatcher test uses misaligned-rationale fixture stops.** Without misaligned rationales the dispatcher would still return a hint targeting `stop_index=0` (defensive fallback), but using realistic closure-swap placeholders makes the test more diagnostic — if the dispatcher's stop-finding logic regresses, the assertion `hint.target == {"stop_index": 0}` would still pass on the parametrized test but fail on the dedicated `test_hint_for_violation_rationale_misaligned_targets_stop_index` which expects `stop_index=1`.
- **Budget key = check name, NOT hint reason.** The existing `critique_final_with_stops` driving loop already does `_bumped_counts(state, actionable)` where `actionable` is the check name from `itinerary_violations`. The new branch produces a hint whose reason is `rationale_misaligned` but the budget key stays `rationale_stop_alignment` — independent of `constraint_unmet_in_final`, no code change needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Wrong-branch commit recovered via cherry-pick + reset**
- **Found during:** Task 1 RED commit
- **Issue:** The Bash tool reset cwd from the worktree back to the main repo between calls. The `git add` + `git commit` for the RED test landed on `feature/v2-category-compliance` (the orchestrator base branch) instead of the worktree's `worktree-agent-*` branch. The pre-commit HEAD assertion would normally have caught this, but it only fires when `.git` is a file (worktree marker) — the main repo's `.git` is a directory, so the assertion no-oped.
- **Fix:** From inside the worktree, `git cherry-pick d85aa93` brought the RED commit onto the worktree branch as `aacc5e0`. Then from the main repo, `git reset --hard 811eff3` rewound `feature/v2-category-compliance` to the orchestrator base. No protected branches were touched (not in main/master/develop/trunk/release/* deny-list), no concurrent commits were destroyed (the worktree was just spawned and the only commit was mine).
- **Files modified:** None (recovery only)
- **Verification:** `git log --oneline -3 feature/v2-category-compliance` shows `811eff3` at HEAD; `git log --oneline -5 worktree-agent-a849d966826e13694` shows the full plan history.
- **Committed in:** N/A (process fix)
- **Prevention going forward:** For every subsequent bash call I prefixed `cd "/Users/pnhek/.../worktree-path"` to keep the working dir inside the worktree, and ran the explicit `git rev-parse --abbrev-ref HEAD` namespace check before every commit.

---

**Total deviations:** 1 auto-fixed (Rule 3: 1)
**Impact on plan:** No plan/source impact. The cherry-pick + reset recovered cleanly; both branches are in the expected state. The remaining 3 commits (Task 1 GREEN, Task 2 RED, Task 2 GREEN) all landed directly on the worktree branch.

## Issues Encountered

- The Claude Code Bash tool resets cwd between calls. The pre-commit worktree HEAD assertion in `task_commit_protocol` only fires when `.git` is a file, so a Bash call running in the main repo wouldn't trip it; I added an explicit `cd "<worktree>"` prefix to every subsequent Bash call. Surfaced via the wrong-branch commit on Task 1 RED, recovered via cherry-pick.
- `poetry` is not on PATH in this shell. Used the pre-existing `.venv/` (set up during plan 04-01) at the repo root via `../../../.venv/bin/python -m pytest|mypy` for all verification commands. Pre-commit hooks (`ruff check`, `ruff format`) ran cleanly on every commit via their pinned environments.
- The worktree base commit (`811eff3`) does not have the Phase 4 plan files or `04-CONTEXT.md` checked in (only the SUMMARY files from plans 04-01 and 04-02 are tracked there). Reading the plan and context files succeeded because the Read tool walked outside the worktree to the orchestrator's main-repo copy at `/Users/pnhek/.../.planning/...`. The SUMMARY for 04-05 is written inside the worktree at the canonical location.

## Verification

- `.venv/bin/python -m pytest tests/unit/test_critique_checks.py -v -k "is_rationale_aligned or rationale_stop_alignment"` — 16 passed (5 new helper tests, 1 regression test, 10 pre-existing scorer tests, all green pre and post extraction).
- `.venv/bin/python -m pytest tests/unit/test_agent_self_correct.py -v -k "rationale_misaligned or first_misaligned or hint_for_violation_maps"` — 13 passed (7 parametrized dispatcher entries including the new one, 3 helper tests, 2 functional tests, 1 retry-budget test).
- `.venv/bin/python -m pytest tests/unit/test_agent_self_correct.py tests/unit/test_critique_checks.py -v` — 107 passed.
- `.venv/bin/python -m pytest tests/unit -v` (per memory `project_full_suite_db_pool_contamination.md`) — **777 passed, 7 skipped**.
- `.venv/bin/python -m mypy app/agent/revision.py app/agent/critique/checks.py` — clean.
- `grep -n "def is_rationale_aligned" app/agent/critique/checks.py` — 1 line (line 298).
- `grep -n "_first_misaligned_stop_index" app/agent/revision.py` — 2 lines (def + use).
- `grep -n "rationale_misaligned" app/agent/revision.py` — 1 line (the new branch's RevisionHint constructor; the detail string references the check name "rationale_stop_alignment" so this grep does not over-count).
- `grep -c "MAX_REVISIONS_PER_REASON\s*=" app/agent/revision.py` — 1 (budget constant unchanged).
- Python probe: `_hint_for_violation('rationale_stop_alignment', state_with_misaligned_first_stop)` returns `hint.reason == 'rationale_misaligned'` and `hint.target == {'stop_index': 0}`.
- Python probe: `_first_misaligned_stop_index(state_with_aligned_first_misaligned_second)` returns 1 (skips matching stop, finds the second).

## Known Stubs

None. The rationale_misaligned dispatch is fully wired end-to-end through the existing `critique_final_with_stops` driving loop. The plan's `<success_criteria>` says "the model gets two retries; after that the agent ships rather than infinite-loop" — both branches exercised by tests, no placeholder text or empty values introduced.

## User Setup Required

None. The change is internal to the agent revision loop. No environment variables, no external services, no migrations.

## Threat Flags

None. Every file modified is in the existing critique/revision trust boundary (state → dispatcher → model). No new network endpoints, auth paths, or schema changes at trust boundaries were introduced.

## Next Phase Readiness

- The dispatch is keyed by check name, so adding any new `rationale_*` scorer in checks.py will be picked up automatically by either an existing branch or the catch-all — no further dispatcher changes needed for variant scorers.
- Phase 5 (RAT-02 / closure-swap placeholder bleed) inherits the dispatch wiring: when the swap node's "Walking-distance alternative for X" placeholder reaches commit, `rationale_stop_alignment` fires (it's an EVAL-02 regression target with a dedicated test) and `rationale_misaligned` now drives the rewrite via this dispatcher.
- The shared `is_rationale_aligned` helper is the single source of truth for "is this stop's rationale aligned." Future scorers/dispatchers that want the same per-stop semantics should call it rather than re-implement the rule.

## Self-Check: PASSED

- Found SUMMARY file at `.planning/phases/04-category-compliance-fix/04-05-SUMMARY.md`.
- Found key modified files: `app/agent/critique/checks.py`, `app/agent/revision.py`, `tests/unit/test_critique_checks.py`, `tests/unit/test_agent_self_correct.py`.
- Found commits on `worktree-agent-a849d966826e13694`: `aacc5e0` (test 04-05 RED Task 1), `df4fc39` (feat 04-05 GREEN Task 1), `3ea038e` (test 04-05 RED Task 2), `d7d5750` (feat 04-05 GREEN Task 2).

---
*Phase: 04-category-compliance-fix*
*Completed: 2026-05-22*
