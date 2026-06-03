---
phase: 06-minimal-edit-refinement
plan: 4
subsystem: testing
tags: [refinement, eval-config, pydantic, schema, strictstr, literal]

# Dependency graph
requires:
  - phase: 03-eval-harness-extension
    provides: "EvalQuery.turns + MatrixEntry + EvalMatrixConfig (extended by Phase 6)"
provides:
  - "EvalQuery.threading_mode: Literal['legacy', 'prod'] = 'legacy' (D-06-05)"
  - "ExpectedRefinement nested Pydantic model with target_slot: int = Field(ge=1) (D-06-08)"
  - "EvalQuery.expected_refinement: ExpectedRefinement | None = None (D-06-08)"
  - "MatrixEntry.env: dict[StrictStr, StrictStr] | None = None (D-06-10)"
affects: [06-06-eval-runner-branching, 06-07-yaml-scenario-update, 06-05-refinement-minimal-edit-scorer]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "StrictStr-on-values + mode='before' validator on keys closes YAML coercion-before-validator gap (MEDIUM env-StrictStr fix)"
    - "Opt-in Literal field + nested Pydantic model preserves extra='forbid' invariant while staying backward compat"

key-files:
  created: []
  modified:
    - "app/eval/config.py (+81 / -2; +ExpectedRefinement class, threading_mode, expected_refinement, MatrixEntry.env, env_keys_must_be_strings validator, StrictStr import)"
    - "tests/unit/test_eval_config.py (+177 / -0; +TestPhase6EvalConfigAdditions with 18 test methods)"

key-decisions:
  - "Used dict[StrictStr, StrictStr] for MatrixEntry.env (not dict[str, str] + mode='after' validator) per MEDIUM env-StrictStr fix from REVIEWS.md — closes the coercion-before-validator gap where YAML bool true coerces to str before any mode='after' validator runs"
  - "Added env_keys_must_be_strings field_validator(mode='before') so non-string keys are rejected symmetrically with StrictStr values"
  - "Skipped threading_mode normalization validator: yaml.safe_load already produces lowercase strings and Literal enforces membership; the validator was 'OPTIONAL — planner discretion' per plan and adds no value"
  - "ExpectedRefinement target_slot is 1-indexed (matches user prose, is_refinement_request, refinement_minimal_edit scorer convention)"

patterns-established:
  - "StrictStr-typed dict value + before-mode validator on keys: defends both axes of YAML coercion at type-validation time"
  - "Phase 6 schema additions live in the same commit as their tests (TDD RED→GREEN), satisfying TDD-mode gate"

requirements-completed: [REF-02, REF-03, REF-04]

# Metrics
duration: ~25min
completed: 2026-06-02
---

# Phase 6 Plan 04: Eval Config Schema Additions Summary

**Four backward-compatible Pydantic additions on `app/eval/config.py` (`EvalQuery.threading_mode`, `ExpectedRefinement` nested model with `target_slot`, `EvalQuery.expected_refinement`, `MatrixEntry.env` with StrictStr on both axes) so plans 06-06 / 06-07 can wire the prod threading branch + per-cell env override without YAML 422 errors.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-06-03T01:11:38Z
- **Completed:** 2026-06-03T01:36:57Z
- **Tasks:** 2 (TDD pair: tests then implementation)
- **Files modified:** 2

## Accomplishments

- All four Phase 6 schema additions land with strict Pydantic validation:
  - `threading_mode: Literal["legacy", "prod"] = "legacy"` — Literal-enforced; 30 existing cases keep legacy
  - `ExpectedRefinement.target_slot: int = Field(ge=1)` — 1-indexed, `extra="forbid"` on the nested model
  - `EvalQuery.expected_refinement: ExpectedRefinement | None = None` — opt-in default
  - `MatrixEntry.env: dict[StrictStr, StrictStr] | None = None` + key-symmetry validator
- Zero regression: 851 unit tests pass across the full suite (`tests/unit/`), 55/55 in `test_eval_config.py`
- Backward-compat invariant proven by live YAML round-trips: `load_eval_queries()` returns 33 entries unchanged, `load_eval_matrix()` returns 2 entries with `env is None` unchanged
- YAML-bool regression guard test (`test_matrix_entry_env_yaml_unquoted_boolean_rejected`) proves StrictStr catches the `env: {FLAG: true}` (unquoted) coercion path the prior `mode="after"` design would have silently passed
- ruff check + ruff format + mypy all pass on the two modified files

## Task Commits

Each task was committed atomically:

1. **Task 2: Unit tests (RED)** - `b155fbb` (test) — 18 new test methods; import fails because `ExpectedRefinement` doesn't exist yet; tests confirmed failing before implementation
2. **Task 1: Schema additions (GREEN)** - `151036b` (feat) — `ExpectedRefinement`, `threading_mode`, `expected_refinement`, `MatrixEntry.env` + `StrictStr` import + `env_keys_must_be_strings` validator; all 55 tests pass

_Note: This plan follows the TDD ordering (test commit before feature commit) per the project's TDD-mode gate. Plan tasks were declared `tdd="true"` and executed RED→GREEN. No REFACTOR commit was needed — the GREEN implementation matches the patterns analogs verbatim._

## Files Created/Modified

- `app/eval/config.py` — Added `StrictStr` to pydantic import line, new `ExpectedRefinement` class (between `ExpectedResults` and `EvalQuery`), two new `EvalQuery` fields, one new `MatrixEntry` field, one new `env_keys_must_be_strings` field validator. Updated `EvalQuery` docstring to mention Phase 6 additions. Net: +81/-2 lines.
- `tests/unit/test_eval_config.py` — Added `from app.eval.config import ExpectedRefinement` to the import block; appended `TestPhase6EvalConfigAdditions` test class with 18 methods covering defaults, validation errors, backward-compat YAML loads, and the StrictStr YAML-bool regression guard. Net: +177/-0 lines.

## Decisions Made

- **StrictStr on both keys and values for `MatrixEntry.env`** (not bare `dict[str, str]` + `mode="after"` validator). Rationale: a `mode="after"` validator runs only after Pydantic has already coerced YAML scalars; an unquoted YAML `true` would be coerced to the Python bool `True`, then to the string `"True"`, before a value-type validator could see it. `StrictStr` rejects the coercion at type-validation time. See plan §1.4 + REVIEWS.md MEDIUM env-StrictStr fix.
- **`env_keys_must_be_strings` validator runs `mode="before"`** so non-string keys (e.g. integer `1`) are rejected before Pydantic's standard dict-key coercion can silently turn them into `"1"`.
- **Skipped the optional `threading_mode` normalization validator** — `yaml.safe_load` already emits lowercase strings and the `Literal` type already enforces value membership. Adding a validator would be dead code per the plan's "planner discretion: skip the validator if the YAML loader already produces lowercase strings" allowance.
- **Indexing convention for `target_slot`:** 1-indexed. Matches user-facing prose ("make stop 2 cheaper"), `is_refinement_request` return convention from plan 06-02, and `refinement_minimal_edit` scorer convention from plan 06-03.
- **`ExpectedRefinement` placed directly before `EvalQuery`** (between `ExpectedResults` and `EvalQuery`). Mirrors the existing nested-model placement convention for `ExpectedConstraints` / `ExpectedResults`.

## Deviations from Plan

None — plan executed exactly as written. The TDD ordering (Task 2 tests committed before Task 1 implementation) is the canonical RED→GREEN sequence even though the plan listed Task 1 first numerically; this is the expected reading of "both tasks `tdd="true"`" and produced the required RED-then-GREEN commit pair.

## Issues Encountered

- **Pre-commit hook reformat on first RED commit:** the `ruff format` hook reformatted the test file on the first commit attempt and the commit aborted. Re-staged and re-committed (new commit, not amend per the harness's "create NEW commits rather than amending" rule). Resolved on retry — the reformatted file matched ruff style and the second commit succeeded.

## Verification Performed

All acceptance criteria from the plan verified:

- `grep -n "class ExpectedRefinement" app/eval/config.py` → 1 match (line 115)
- `grep -n "threading_mode: Literal" app/eval/config.py` → 1 match (line 164)
- `grep -n "expected_refinement: ExpectedRefinement | None" app/eval/config.py` → 1 match (line 167)
- `grep -nE "env: dict\[(str|StrictStr), (str|StrictStr)\] \| None" app/eval/config.py` → 1 match (line 272, canonical `StrictStr` form per the MEDIUM fix)
- `python -c "from app.eval.config import load_eval_queries; ..."` → `hand_written count: 33`
- `python -c "from app.eval.config import load_eval_matrix; ..."` → `matrix entries: 2`
- `python -c "EvalQuery(..., threading_mode='prod')..."` → prints `prod`
- `poetry run pytest tests/unit/test_eval_config.py -v` → **55 passed** (37 baseline + 18 new)
- `poetry run pytest tests/unit/ -v -x` → **851 passed, 7 skipped** (no regression)
- `ruff check`, `ruff format --check`, `mypy` → all pass on both modified files

## User Setup Required

None — no external service configuration required. This is a pure-schema plan; no env vars, no secrets, no infra changes.

## Next Phase Readiness

- **Plan 06-05 (refinement_minimal_edit scorer):** can read `state.scratch["refinement_target_slot"]` knowing the YAML side that supplies it (`expected_refinement.target_slot`) now validates
- **Plan 06-06 (eval runner branching + per-cell env consumption):** `case.threading_mode == "prod"` branch + per-cell `env` dict are now type-safe; the runner can consume them without defensive coercion
- **Plan 06-07 (YAML scenario update + baseline):** `refinement_cheaper` can add `threading_mode: prod` and `expected_refinement: { target_slot: 2 }` without 422; the matrix YAML can add per-cell `env: { REFINEMENT_STRUCTURED_PLAN_ENABLED: "true" }`
- No blockers introduced. The four additions are strictly additive and opt-in.

## Self-Check: PASSED

- `app/eval/config.py` exists and contains all four schema additions (verified via grep above)
- `tests/unit/test_eval_config.py` exists and `TestPhase6EvalConfigAdditions` runs 18 methods, all passing
- Commit `b155fbb` (test, RED) exists in git log
- Commit `151036b` (feat, GREEN) exists in git log
- No production YAML changes (eval_queries.yaml + eval_matrix.yaml unchanged on disk)

## TDD Gate Compliance

- RED gate: `b155fbb` (test commit) — failing test before implementation ✓
- GREEN gate: `151036b` (feat commit) — implementation makes RED tests pass ✓
- REFACTOR gate: not used (implementation matched analog patterns directly; no refactor needed)
- MVP+TDD behavior-adding-task gate: not triggered — both tasks are TDD-compliant config-schema additions

---
*Phase: 06-minimal-edit-refinement*
*Completed: 2026-06-02*
