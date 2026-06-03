---
phase: 06-minimal-edit-refinement
plan: 06-03
subsystem: agent/critique
tags: [refinement, scorer, critique, deterministic, merge-gate, REF-01]
requires: []
provides:
  - "refinement_minimal_edit deterministic scorer"
  - "CRITIQUE_THRESHOLDS['refinement_minimal_edit'] strict 1.0 entry"
  - "DETERMINISTIC_CHECKS['refinement_minimal_edit'] eval-baseline registration"
  - "scratch contract: prior_committed_stops, refinement_target_slot, refinement_context"
affects:
  - "itinerary_violations gains a 9th _try wiring (refinement_minimal_edit)"
  - "every per-cell baseline JSON now carries refinement_minimal_edit_mean"
  - "tests/unit/test_eval_agent.py query_result helper now lists 9 checks"
tech-stack:
  added: []
  patterns:
    - "Pure-function-of-state scorer (analog of category_compliance)"
    - "Five-branch precedence (N-3): mutually exclusive abstain/fail/normal paths"
    - "Dual-table registration (CRITIQUE_THRESHOLDS + DETERMINISTIC_CHECKS) with regression-guard test (HIGH-1)"
key-files:
  created: []
  modified:
    - "app/agent/critique/checks.py (+122 lines: scorer + threshold + itinerary_violations wiring)"
    - "scripts/eval_agent.py (+2 lines: import + DETERMINISTIC_CHECKS entry)"
    - "tests/unit/test_critique_checks.py (+381 lines: import + 2 smoke + TestRefinementMinimalEdit 16 methods)"
    - "tests/unit/test_eval_agent.py (+28 lines: TestDeterministicChecksRegistration + query_result helper extension)"
decisions:
  - "Branch precedence ordering finalized at five branches (abstain / fail-loud × 3 / legitimate-zero-denom / normal); see TestRefinementMinimalEdit method names for the canonical taxonomy"
  - "Did NOT refactor DETERMINISTIC_CHECKS to derive from CRITIQUE_THRESHOLDS membership; coupling deferred to v2.1 to preserve Phase 4/5 baseline parity (per plan §Task 3 step 3)"
  - "Smoke tests retained alongside TestRefinementMinimalEdit class — additional regression guards at near-zero cost"
metrics:
  duration_minutes: 11
  completed_date: "2026-06-03"
  tasks: 3
  files_modified: 4
  files_created: 0
  net_lines_added: 533
  test_methods_added: 18  # 2 smoke + 16 TestRefinementMinimalEdit + 1 TestDeterministicChecksRegistration - 1 over-count
---

# Phase 06 Plan 03: refinement_minimal_edit Scorer + Eval Wiring Summary

## One-liner

Ships the `refinement_minimal_edit` deterministic scorer with five-branch precedence (N-3), HIGH-2 denominator fix, N-2 `refinement_context` fail-loud flag, and dual registration in both `CRITIQUE_THRESHOLDS` and `scripts/eval_agent.py:DETERMINISTIC_CHECKS` so the REF-01 merge gate is present in every per-cell baseline JSON.

## What Shipped

### Task 1 — `refinement_minimal_edit` scorer (commits `495917f` + `aad9316`)

- New pure-state scorer in `app/agent/critique/checks.py` implementing the D-06-08 math contract: fraction of PRIOR non-target stops that survive byte-equal in current `state.stops`.
- Registered in `CRITIQUE_THRESHOLDS` with strict 1.0 threshold (D-06-09; REF-01 is binary).
- Wired into `itinerary_violations._try(...)` adjacent to `rationale_stop_alignment`. Branch 1 abstain (refinement_context absent → 1.0) preserves the existing revision-loop fail-open behavior so the new scorer never produces a spurious violation in the standard `/chat` critique path.
- Five-branch precedence (mutually exclusive, evaluated in order):
  - Branch 1 (abstain → 1.0): `refinement_context` absent or False — non-refinement / ad-hoc invocation.
  - Branch 2 (fail-loud → 0.0): `refinement_context` True but `prior_committed_stops` is None / missing / empty list OR `refinement_target_slot` is missing.
  - Branch 3 (fail-loud → 0.0): `refinement_context` True, prior non-empty, but every entry malformed (`prior_by_slot` collapses to empty).
  - Branch 4 (legitimate zero-denom → 1.0): every surviving prior entry IS the target slot (lone-stop-target case).
  - Branch 5 (normal): `matches / len(prior_non_target_slots)`.
- HIGH-2 fix verified at source: `grep -cE "prior_non_target_slots|len\(prior_non_target_slots\)" app/agent/critique/checks.py` returns 6 (was 0 pre-fix).
- N-2 fix verified at source: `grep -cE "refinement_context" app/agent/critique/checks.py` returns 9 (was 0 pre-fix).

### Task 2 — Unit test coverage (commit `6e14338`)

- New `TestRefinementMinimalEdit` class in `tests/unit/test_critique_checks.py` with **16 test methods** (>= 13 required), each named after the branch or regression guard it covers:
  - `test_branch_1_no_refinement_context_returns_1_0_abstain`
  - `test_branch_1_refinement_context_explicit_false_returns_1_0_abstain`
  - `test_branch_2_refinement_context_true_empty_prior_returns_0_0_fail_loud` (N-2 regression guard)
  - `test_branch_2_refinement_context_true_missing_target_slot_returns_0_0`
  - `test_branch_2_refinement_context_true_none_prior_returns_0_0`
  - `test_branch_3_refinement_context_true_all_malformed_prior_returns_0_0_fail_loud`
  - `test_branch_4_single_stop_target_lone_stop_returns_1_0`
  - `test_branch_5_all_non_target_stops_preserved_returns_1_0`
  - `test_branch_5_dropped_non_target_slot_scores_below_1_0` (HIGH-2 regression guard — drops)
  - `test_branch_5_inserted_non_target_slot_does_not_help_score` (HIGH-2 regression guard — inserts)
  - `test_branch_5_one_of_two_non_target_stops_changed_returns_0_5`
  - `test_branch_5_all_non_target_stops_changed_returns_0_0`
  - `test_registered_in_thresholds`
  - `test_registered_in_itinerary_violations_when_refinement_context_present`
  - `test_not_in_itinerary_violations_when_no_refinement_context` (preserves revision-loop fail-open)
  - `test_pure_function_no_db_access` (`get_conn` sentinel)
- 2 smoke tests retained from Task 1 RED for the file-level scope (subsumed by the class but kept as additional regression guards).
- DB-pool-contamination guard per `project_full_suite_db_pool_contamination.md`: every `itinerary_violations` integration test patches `no_hallucinated_place_ids`, `temporal_coherence`, and `constraints_satisfied` so the full suite cannot leak a live DB pool from this class.
- Test fixtures use >= 20-char `place_id` values per the HIGH-4 residual fix from plan 06-01 Task 3 (defensive against future regex validation on `Stop.place_id`).

### Task 3 — DETERMINISTIC_CHECKS registration (commits `f4b42f7` + `783cbc3`)

- `scripts/eval_agent.py`: imported `refinement_minimal_edit` and added it to `DETERMINISTIC_CHECKS` (now 9 entries, was 8).
- New `TestDeterministicChecksRegistration` class in `tests/unit/test_eval_agent.py` with `test_refinement_minimal_edit_registered_in_deterministic_checks` asserting both **presence** AND **callable identity** — catches the silent-rename / shadow-by-wrapper failure mode that the HIGH-1 reviewer flagged.
- Rule 1 auto-fix: extended the existing `query_result()` test helper to include `refinement_minimal_edit` so `aggregate_results` does not raise `KeyError` (the helper's own docstring mandates this: "any scorer added to that dict must also appear here").
- Did NOT refactor `DETERMINISTIC_CHECKS` to derive from `CRITIQUE_THRESHOLDS` membership — that v2.1 ticket would silently re-baseline Phases 4/5; out of scope for Phase 6 per plan §Task 3 step 3.

## N-2 + N-3 Fixes Verified

| Fix | Mechanism | Source verification | Test verification |
|-----|-----------|---------------------|-------------------|
| N-2 (refinement_context flag) | New `state.scratch['refinement_context']: bool` flag distinguishes "non-refinement → abstain" from "refinement scenario where turn 0 failed → fail-loud". | `grep -cE "refinement_context" app/agent/critique/checks.py` → 9 (was 0). | `test_branch_2_refinement_context_true_empty_prior_returns_0_0_fail_loud` asserts the silent-pass path is closed. |
| N-3 (five-branch precedence) | Five mutually exclusive branches evaluated in order; explicit numbered list in the scorer docstring. | `grep -cE "Branch [1-5]" app/agent/critique/checks.py` returns 9 (5 docstring lines + 4 inline comments). | `grep -cE "def test_branch_[1-5]_" tests/unit/test_critique_checks.py` → 12 (>= 8 required); 16 test methods total. |
| HIGH-1 (DETERMINISTIC_CHECKS) | Explicit dual registration with identity-check regression guard. | `python -c "from scripts.eval_agent import DETERMINISTIC_CHECKS; assert len(DETERMINISTIC_CHECKS) == 9"` → ok. | `TestDeterministicChecksRegistration::test_refinement_minimal_edit_registered_in_deterministic_checks` passes. |
| HIGH-2 (PRIOR-based denominator) | Build `prior_by_slot` and iterate `prior_non_target_slots`; current state only feeds the matches computation. | `grep -cE "prior_non_target_slots" app/agent/critique/checks.py` → 6. | `test_branch_5_dropped_non_target_slot_scores_below_1_0` returns 0.5 (would return 1.0 pre-fix); `test_branch_5_inserted_non_target_slot_does_not_help_score` returns 1.0 (insertions are neutral). |

## Branch 1 Preserves Revision-Loop Fail-Open

The `itinerary_violations` revision-loop invocation passes states without `refinement_context` set in scratch. Branch 1 (abstain → 1.0) means the new scorer returns 1.0 every time in this path, so it never adds a spurious violation to the revision loop. This is asserted by `test_not_in_itinerary_violations_when_no_refinement_context` — without that guard, the new scorer would break every existing revision turn by producing a phantom "refinement_minimal_edit" violation.

## Registration Sites (All Three Confirmed)

1. `app/agent/critique/checks.py:CRITIQUE_THRESHOLDS["refinement_minimal_edit"]` = 1.0 — line 33.
2. `app/agent/critique/checks.py:itinerary_violations._try("refinement_minimal_edit", refinement_minimal_edit)` — line 509.
3. `scripts/eval_agent.py:DETERMINISTIC_CHECKS["refinement_minimal_edit"]` = `refinement_minimal_edit` — line 53.

Each site has a dedicated regression-guard test:
- (1) `TestRefinementMinimalEdit::test_registered_in_thresholds`
- (2) `TestRefinementMinimalEdit::test_registered_in_itinerary_violations_when_refinement_context_present` + `test_not_in_itinerary_violations_when_no_refinement_context`
- (3) `TestDeterministicChecksRegistration::test_refinement_minimal_edit_registered_in_deterministic_checks` (HIGH-1 guard)

## Tests Run

- `poetry run pytest tests/unit/test_critique_checks.py -v` → **80 passed** (was 62 pre-plan)
- `poetry run pytest tests/unit/test_eval_agent.py -v` → **62 passed** (was 61 pre-plan)
- `poetry run pytest tests/unit/test_eval_agent.py::TestDeterministicChecksRegistration -v` → **1 passed**
- `poetry run pytest tests/unit/ -q -x` (full unit suite) → **852 passed, 7 skipped, 0 failed**

## Commits

| # | Hash | Type | Description |
|---|------|------|-------------|
| 1 | `495917f` | test | RED: smoke for refinement_minimal_edit (ImportError driver) |
| 2 | `aad9316` | feat | GREEN: implement scorer + CRITIQUE_THRESHOLDS + itinerary_violations wiring |
| 3 | `6e14338` | test | full TestRefinementMinimalEdit (16 methods, N-3 + N-2 + HIGH-2 coverage) |
| 4 | `f4b42f7` | test | RED: TestDeterministicChecksRegistration (HIGH-1 driver) |
| 5 | `783cbc3` | feat | GREEN: register in DETERMINISTIC_CHECKS + query_result helper |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Extended `query_result()` test helper in `tests/unit/test_eval_agent.py` to include `refinement_minimal_edit`**

- **Found during:** Task 3
- **Issue:** The plan only directed Task 3 to add the scorer to `DETERMINISTIC_CHECKS` and add a regression-guard test. But the existing `query_result()` helper in `tests/unit/test_eval_agent.py` has an explicit docstring contract: "any scorer added to that dict must also appear here, otherwise `aggregate_results` raises KeyError when iterating per-scorer means". Without this fix, `test_aggregate_results_flattens_mean_metrics` would have failed with `KeyError: 'refinement_minimal_edit'`.
- **Fix:** Added `"refinement_minimal_edit": CheckResult(score=1.0, threshold=1.0, passed=True)` to the helper's `checks` dict.
- **Files modified:** `tests/unit/test_eval_agent.py`
- **Commit:** `783cbc3`

**2. [Worktree-path-safety recovery] Initial Edit calls accidentally modified the main-repo file instead of the worktree file**

- **Found during:** Task 1 RED phase
- **Issue:** First Edit calls used the absolute path under `/Users/.../mlops_city_concierge/tests/...` (main repo) instead of `/Users/.../mlops_city_concierge/.claude/worktrees/agent-a547946101bd43254/tests/...` (worktree). The Read tool initially found the plan only at the main repo path (since the plan file had not yet been committed to git), and subsequent Read/Edit calls on source files used that same prefix.
- **Fix:** Restored the contaminated main-repo file with `git checkout -- tests/unit/test_critique_checks.py` (targeted single-file restore, no destructive blanket reset), then re-applied the same edit to the worktree path. The worktree HEAD never advanced past the contamination; no rewrite required.
- **Files modified:** none in final commit (the main repo was already clean before any commit was made)
- **Commit:** (none — pre-commit recovery)
- **Followup:** This is the worktree-path-safety hazard documented in the executor instructions. Logged in `deferred-items.md` (if it exists) so future worktree executors can audit for similar drift before staging.

### Architectural Deviations

None. All implementation followed the plan's `<action>` step-by-step.

## Known Stubs

None. The scorer is fully wired into both the runtime critique loop and the eval baseline.

## Threat Flags

None. The plan's `<threat_model>` register (T-06-03-01 through T-06-03-05) was honored without introducing new surfaces:

- T-06-03-01 (malformed prior tampering) → mitigated by defensive `.get()` reads in `prior_by_slot` builder + Branch 3 fail-loud + `_try(...)` wrapper.
- T-06-03-02 (eval YAML scratch injection) → accepted; pre-existing eval-runner surface.
- T-06-03-03 (DoS via DB exhaustion) → mitigated by pure-state design; `test_pure_function_no_db_access` asserts no `get_conn` call.
- T-06-03-04 (registration drift) → mitigated by `TestDeterministicChecksRegistration::test_refinement_minimal_edit_registered_in_deterministic_checks` identity check.
- T-06-03-05 (turn-0 silent-pass) → mitigated by N-2 `refinement_context` flag + Branch 2 fail-loud; `test_branch_2_refinement_context_true_empty_prior_returns_0_0_fail_loud` regression guard.

## Followups for Downstream Plans

- **Plan 06-06 must populate the three scratch keys for every refinement scenario:** `prior_committed_stops` (list[dict] of `{slot, place_id}`), `refinement_target_slot` (int, 1-indexed), and `refinement_context` (bool — set to True regardless of turn-0 commit outcome so Branch 2 fail-loud can surface failures).
- **Plan 06-07 baseline regeneration:** `refinement_minimal_edit_mean` will appear as a new key in every per-cell baseline JSON. Diff against pre-Phase-6 baselines is expected to introduce the new column.
- **v2.1 ticket (out of scope):** Unify `CRITIQUE_THRESHOLDS` and `DETERMINISTIC_CHECKS` into a single source of truth. The dual-registration guard (T-06-03-04) is correct but maintenance overhead grows linearly with scorer count.

## Self-Check: PASSED

- [x] `app/agent/critique/checks.py` exists, contains `def refinement_minimal_edit(state: ItineraryState) -> float`
- [x] `CRITIQUE_THRESHOLDS["refinement_minimal_edit"] == 1.0` (line 33)
- [x] `itinerary_violations` wires the scorer (line 509)
- [x] `scripts/eval_agent.py:DETERMINISTIC_CHECKS["refinement_minimal_edit"]` registered (line 53)
- [x] `tests/unit/test_critique_checks.py::TestRefinementMinimalEdit` exists with 16 methods
- [x] `tests/unit/test_eval_agent.py::TestDeterministicChecksRegistration` exists with 1 method
- [x] Commit `495917f` exists (Task 1 RED)
- [x] Commit `aad9316` exists (Task 1 GREEN)
- [x] Commit `6e14338` exists (Task 2)
- [x] Commit `f4b42f7` exists (Task 3 RED)
- [x] Commit `783cbc3` exists (Task 3 GREEN)
- [x] All 852 unit tests pass (`pytest tests/unit/ -q -x`)
