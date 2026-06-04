---
phase: 07-prompt-rubric-decoupling
plan: 04
subsystem: critique-scorer

tags:
  - critique
  - scorer
  - refinement
  - merge-gate
  - primary_type

# Dependency graph
requires:
  - phase: 07-prompt-rubric-decoupling
    plan: 02
    provides: "prior_committed_stops scratch entries extended to {slot, place_id, primary_type}; ExpectedRefinement docstring documenting the scratch-shape extension"
provides:
  - "refinement_minimal_edit Branch 5 enforces same-target-primary_type per D-07-07 four-cell matrix (abstain / fail-loud-current / mismatch / match)"
  - "Behavioral rule prompt rule 10 used to prescribe ('SAME primary_type / Google Place category as the original') now lives in the scorer where the binary 1.0 merge gate enforces it deterministically"
  - "Five-branch precedence docstring extended (Branches 1-4 explicitly category-blind; Branch 5 documents the new sub-check)"
affects:
  - 07-05-scorer-tests-and-grep-gate
  - 07-07-rebaseline-and-falsifier

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Scorer extension in-place: single scorer / single merge-gate / single dispatcher entry invariant preserved per D-07-05 (no new threshold key, no new itinerary_violations wiring)"
    - "Parallel dict (prior_primary_type_by_slot) alongside existing prior_by_slot: keeps Branch 5 byte-equality logic byte-identical to Phase 6, localizes the diff, makes the new lookup self-documenting"
    - "D-07-07 abstain-first precedence: prior-None abstain takes precedence over current-None fail-loud so legacy 06-06 scratch payloads cannot be penalized for a current-side defect they cannot observe"
    - "EXACT STRING EQUALITY comparison (D-07-05) — no family_of(); the prompt rule being moved was 'SAME primary_type', not 'same family'"

key-files:
  created:
    - ".planning/phases/07-prompt-rubric-decoupling/07-04-SUMMARY.md"
  modified:
    - "app/agent/critique/checks.py"

key-decisions:
  - "D-07-05 honored: extension is in-place in Branch 5 only — no new scorer, no new CRITIQUE_THRESHOLDS key, no new itinerary_violations dispatcher entry. CRITIQUE_THRESHOLDS['refinement_minimal_edit'] == 1.0 unchanged"
  - "D-07-07 four-cell matrix implemented with abstain-first precedence: prior None → abstain (byte_fraction); current None → fail-loud (0.0); mismatch → 0.0; match → byte_fraction unchanged"
  - "Implementation choice (Claude's Discretion item 3 in CONTEXT.md): parallel prior_primary_type_by_slot rather than upgrading prior_by_slot to carry the full entry — preserves Phase 6 byte-equality logic exactly"
  - "Defensive coercion: non-string primary_type values coerced to None so the abstain branch fires rather than an EQ comparison crashing on a wrong-typed value sneaking through the scratch contract (tightens the eval-runner contract beyond plan ask)"
  - "Defensive bounds-check on state.stops[target_slot - 1] for refinement turns that lost the target slot entirely — abstain (return byte_fraction) rather than IndexError"
  - "Branch 4 (lone-stop target) explicitly preserves Phase 6 abstain semantics per PATTERNS.md ('Preserve abstain semantics on Branch 4') — the category check does NOT fire when the byte-equality denominator is already zero"

patterns-established:
  - "Pattern: extend scorers in-place (not by adding sibling scorers) when the new check shares the same merge-gate semantic and operates on the same data slice. Folds cleanly into existing branch precedence without growing the dispatcher surface."
  - "Pattern: when a Pydantic-validated state field can also be optionally absent on the scratch payload, the abstain branch reads it with a defensive type-coerce-to-None — protects against shape violations downstream of contract changes."

requirements-completed:
  - PROMPT-03

# Metrics
duration: ~25min
completed: 2026-06-04
---

# Phase 7 Plan 04: Scorer Category Extend Summary

**`refinement_minimal_edit` Branch 5 now enforces same-`primary_type` on the target (replacement) slot per the D-07-07 four-cell matrix — the behavioral rule prompt rule 10 used to prescribe ("SAME `primary_type` / Google Place category as the original") now lives in the scorer where the binary 1.0 merge gate can enforce it deterministically. Branches 1-4 are byte-identical to Phase 6 (category-blind), `CRITIQUE_THRESHOLDS` and the `itinerary_violations` dispatcher are unchanged, and all 16 existing `TestRefinementMinimalEdit` tests continue to pass under the extension.**

## Performance

- **Duration:** ~25 min (Poetry venv bootstrap on fresh worktree dominated; actual implementation was the parallel-dict build + matrix dispatch)
- **Tasks:** 1
- **Files modified:** 1 (`app/agent/critique/checks.py`)

## Accomplishments

- `app/agent/critique/checks.py::refinement_minimal_edit` Branch 5 extended in-place with the target-slot `primary_type` sub-check per D-07-05 + D-07-07.
- New parallel `prior_primary_type_by_slot: dict[int, str | None]` built alongside the existing `prior_by_slot` defensive build (entries that pass `slot` + `place_id` validation additionally have their `primary_type` read; non-string values coerced to `None`).
- Four-cell D-07-07 matrix dispatched after computing `byte_fraction`:
  - `prior_target_pt is None` → abstain (return `byte_fraction`; migration path for legacy 06-06 scratch payloads).
  - `current_target_pt is None` → fail-loud (`0.0`; the commit dropped a real field, this is a defect).
  - `prior_target_pt != current_target_pt` → category mismatch (`0.0`; binary merge-gate semantic, no fractional penalty).
  - Both present and equal → return `byte_fraction` unchanged.
- Defensive bounds-check on `state.stops[target_slot - 1]`: if the refinement turn lost the target slot entirely, the byte-fraction already reflects the loss and the category sub-check abstains (returns `byte_fraction` rather than raising `IndexError`).
- Function docstring extended:
  - New "Phase 7 / D-07-05 extension (PROMPT-03)" paragraph explaining the in-place extension, scratch contract reference (plan 07-02 / D-07-06), and the prompt-rule-moved provenance.
  - New "D-07-07 four-cell `primary_type` matrix" table inside the extension paragraph.
  - Branch 5 description updated to mention the category sub-check and the EXACT-STRING-EQUALITY decision (D-07-05).
  - Branches 1-4 description explicitly annotated "Category check does NOT fire" (per PATTERNS.md "Preserve abstain semantics on Branch 4" + analogous semantic for Branches 1-3).
  - New "Current-state field read (Phase 7 / D-07-06)" docstring section pointing at `state.stops[target_slot - 1].primary_type` and the commit-time wiring (`commit_itinerary` populates from `places_raw` via CAT-01..CAT-04).
  - `prior_committed_stops` scratch contract description updated to include the new `primary_type: str | None` field with a backward-compat note about legacy 06-06 payloads triggering the abstain branch.
- `CRITIQUE_THRESHOLDS['refinement_minimal_edit'] == 1.0` unchanged.
- `itinerary_violations` dispatcher byte-identical (no new `_try(...)` entry; verified by `git diff` showing all hunks bounded within `refinement_minimal_edit`).

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend Branch 5 of `refinement_minimal_edit` with target-slot `primary_type` check** — `ba5afc0` (feat)

## Files Created/Modified

- `app/agent/critique/checks.py` — `refinement_minimal_edit` extended in-place per D-07-05 + D-07-07. `+123 -13` lines (mostly docstring; the new logic is ~25 lines of dispatch). `CRITIQUE_THRESHOLDS` (line 33) untouched; `itinerary_violations` (lines 470-506 pre-edit) untouched.

## Decisions Made

- **Parallel `prior_primary_type_by_slot` dict** instead of upgrading `prior_by_slot` to store the full entry. CONTEXT.md leaves this to planner/implementer discretion ("Claude's Discretion item 3"). Picking the parallel-dict approach keeps the Phase 6 byte-equality logic (the `current_by_slot.get(slot) == prior_by_slot[slot]` comparison at the heart of Branch 5) byte-identical — the change is purely additive at the data-build layer and the new lookup is self-documenting in the dispatch logic.
- **Abstain-first precedence in the D-07-07 dispatch** (prior-None abstain checked before current-None fail-loud). This is what the plan specifies; the rationale documented inline is that legacy 06-06 scratch payloads cannot be penalized for a current-side defect they cannot observe — the prior-None branch is also the migration path.
- **EXACT STRING EQUALITY comparison** (not `family_of()`). D-07-05 specifies this explicitly; the prompt rule being moved was "SAME `primary_type` / Google Place category", not "same family". If a future scorer wants family-level resilience it would be a sibling check.
- **Defensive non-string coercion to `None`** at the build site (`pt if isinstance(pt, str) else None`). Tightens the eval-runner contract beyond what the plan explicitly asks for — protects against shape violations sneaking through the scratch contract without crashing the scorer. The cost is one extra `isinstance` check per prior entry; the benefit is that future contract drift surfaces as an abstain (no scorer change in score) rather than a `TypeError` on `==`.
- **Defensive bounds-check on `state.stops[target_slot - 1]`** mirrors the plan's explicit instruction. Refinement turns that lost the target slot entirely abstain on the category sub-check (the byte-fraction denominator already accounted for the loss).
- **Branch 4 stays category-blind** per PATTERNS.md ("Preserve abstain semantics on Branch 4"). The lone-stop case is a degenerate refinement shape (`prior_non_target_slots` empty by definition) and treating it as a category abstain keeps the scorer's no-data semantics consistent. Plan 07-05 will pin this with an explicit test.

## Deviations from Plan

None — the plan executed exactly as written. The deferred refinements ("Implementer may instead change `prior_by_slot` to store the full dict per slot") were considered and explicitly rejected in favor of the parallel-dict shape for the reason in Decisions Made above. The defensive non-string coercion and the bounds-check both match the spirit of the plan's "Add a defensive guard" instructions for `current_target_idx` bounds and the implicit expectation that contract violations should not crash the scorer.

## Issues Encountered

- **Poetry virtualenv bootstrap on fresh worktree** — the new `worktree-agent-abb98553c6f62c912` worktree did not yet have a Poetry venv; first `poetry run python` invocation created the venv (`~/Library/Caches/pypoetry/virtualenvs/mlops-city-concierge-qWGC2-pf-py3.13`) but had no packages installed → `ModuleNotFoundError: psycopg2`. Resolved by running `poetry install --quiet` once (~3 min); subsequent runs reused the venv.

  Not a Rule 1-3 deviation; it's expected first-time-in-worktree environment setup. Documented here so future agents in fresh worktrees know to allow ~3 min for the install before the inline smoke assertions can run.

## User Setup Required

None — no environment variables, dashboards, or external services need configuration. The change is purely internal to the offline-eval scorer.

## Next Phase Readiness

- **Plan 07-05 unblocked:** the new branch-specific unit tests pin each cell of the D-07-07 matrix (match / mismatch / prior-None abstain / current-None fail-loud), pin the bounds-check abstain path (target slot lost from `state.stops`), pin Branch 4's explicit category-blindness, and land the PROMPT-02 grep gate per D-07-04. The scorer-extension code is in place; the tests can read it directly.
- **Plan 07-07 partially unblocked:** the re-baseline step per D-07-10 now has a deterministic scorer to re-measure against. The pre-Phase-7 `eval_matrix_refinement.json` baseline (n=5 from commit `61aee1b`) will move under the new scorer — that's expected and exactly why D-07-10 pre-authorizes a re-baseline. PROMPT-04 (no-regression) is evaluated against the post-Phase-7 re-baseline, not the pre-Phase-7 one.
- **Plan 07-06 (PROMPT-01 `/chat` integration test) unaffected:** the integration test threads a real 2-turn conversation through the LangGraph agent and asserts the response has the same stop count + byte-equal `place_id`s on non-target slots + a different `place_id` on the target slot. None of those assertions touch the new category sub-check (the integration test is a behavior-level check, not a scorer-level check), so the new branch logic is transparent to it.

## Self-Check: PASSED

- `app/agent/critique/checks.py` extension present at commit `ba5afc0` (`git show ba5afc0 --stat` reports `1 file changed, 123 insertions(+), 13 deletions(-)`).
- All 6 inline smoke assertions from the PLAN `<automated>` verify pass (category mismatch zeros byte-fraction, category match preserves byte-fraction, prior-None abstain returns byte-fraction unchanged, current-None fails-loud at `0.0`, Branch 1 abstain still returns `1.0`, Branch 4 lone-stop returns `1.0` regardless of category mismatch).
- `CRITIQUE_THRESHOLDS['refinement_minimal_edit'] == 1.0` (verified at runtime in inline smoke).
- Function docstring contains `Phase 7` and `primary_type` substrings (verified at runtime).
- 16/16 `TestRefinementMinimalEdit` tests pass.
- 80/80 `tests/unit/test_critique_checks.py` tests pass (no regression in adjacent scorers).
- 2/2 cross-table tests pass (`TestDeterministicChecksRegistration` + `refinement_minimal_edit` regression in `test_eval_agent.py`).
- `make eval-matrix-refinement-structural-check` exits `0` (matrix shape, scorer registration, shared helper all intact).
- `ruff check` + `ruff format --check` both clean on the modified file.
- `git diff app/agent/critique/checks.py` shows all hunks bounded within `refinement_minimal_edit` (lines 355-end); `itinerary_violations` byte-identical.
- Pre-commit hooks (`ruff (legacy alias)`, `ruff format`) both passed during the commit.
- Git log shows commit `ba5afc0` on `worktree-agent-abb98553c6f62c912` branch.

---

*Phase: 07-prompt-rubric-decoupling*
*Plan: 04-scorer-category-extend*
*Completed: 2026-06-04*
