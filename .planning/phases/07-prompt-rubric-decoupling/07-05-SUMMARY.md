---
phase: 07-prompt-rubric-decoupling
plan: 05
subsystem: tests

tags:
  - tests
  - scorer
  - grep-gate
  - primary_type
  - refinement

# Dependency graph
requires:
  - phase: 07-prompt-rubric-decoupling
    plan: 01
    provides: "Task-only _REFINEMENT_PREAMBLE in app/agent/io.py with REFINEMENT TURN sentinel + commit_itinerary callout + slot/place_id/arrival_time JSON-block fields; SYSTEM_PROMPT rule 10 deleted; D-07-04 forbidden phrases absent from prompts.py + io.py"
  - phase: 07-prompt-rubric-decoupling
    plan: 02
    provides: "prior_committed_stops scratch entries extended to {slot, place_id, primary_type}"
  - phase: 07-prompt-rubric-decoupling
    plan: 04
    provides: "refinement_minimal_edit Branch 5 D-07-07 four-cell primary_type matrix (abstain/fail-loud/mismatch/match) with parallel prior_primary_type_by_slot lookup"
provides:
  - "Module-level PROMPT-02 grep-gate test (D-07-04) asserting six canonical behavioral phrases are absent from prompts.py + io.py — CI hard gate against prompt-rewrite regression"
  - "_refinement_stop helper extended with optional primary_type kwarg (default None) — backward-compatible across existing call sites"
  - "Seven new TestRefinementMinimalEdit methods pinning D-07-07 four-cell matrix + partial-byte-fraction-with-category-match edge + Branch-4 lone-stop skip edge"
  - "_stops_for_pids helper extended with primary_type='Cafe' default — propagates the new scratch-payload field across all 11 sibling tests automatically"
  - "test_prod_mode_injects_structured_plan_on_turn_1 rewritten to pin the new task-only preamble's contract fields (REFINEMENT TURN sentinel, commit_itinerary callout, JSON-block field names) instead of the deprecated byte-for-byte behavioral phrase"
  - "test_prod_mode_populates_state_scratch_for_scorer extended to pin primary_type on every prior_committed_stops entry"
affects:
  - 07-07-rebaseline-and-falsifier

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Source-file substring grep gate (Path.read_text + lowercase scan) rather than runtime import + module-state scan — necessary because some behavioral phrases historically lived in docstrings/comments next to the code constants, and a runtime import would only see the loaded string"
    - "Test-helper additive extension pattern: optional kwarg with default that matches the legacy behavior; new tests opt into the value; pre-existing call sites are byte-identical"
    - "Per-class fixture helper carries the new scratch-payload field (primary_type='Cafe') so 11 sibling tests pick up the Phase 7 / D-07-06 contract extension without per-test edits"
    - "Drop-and-replace assertion pattern for prompt-rewrite cleanup: 'byte-for-byte' (forbidden) -> 'REFINEMENT TURN' + 'commit_itinerary' + JSON-block field names (allowed task-description contract)"

key-files:
  created:
    - ".planning/phases/07-prompt-rubric-decoupling/07-05-SUMMARY.md"
  modified:
    - "tests/unit/test_critique_checks.py"
    - "tests/unit/test_eval_agent.py"

key-decisions:
  - "Grep gate runs against source files (Path.read_text) not module imports — captures docstring/comment phrases that a runtime import would miss"
  - "Forbidden-phrase list uses lowercase normalization (single .lower() on combined source) — the 'SAME primary_type' variant is covered by the 'same primary_type' lowercase entry without a duplicate string"
  - "_stops_for_pids helper extension uses primary_type='Cafe' (constant) rather than a per-pid mapping — simplest deterministic value, propagates to all 11 sibling tests automatically; the test_refinement_minimal_edit_scores_1_0_end_to_end check still scores 1.0 because both turn-0 and turn-1 stops use the same constant (D-07-07 match cell)"
  - "test_prod_mode_injects_structured_plan_on_turn_1 'byte-for-byte' assertion was deprecated by Phase 7 plan 07-01 (the phrase moved out of the preamble); replaced with assertions on the new task-only contract fields per D-07-03 + D-07-04 (sentinel + output channel + JSON-block field names) so the test still verifies the structured-plan injection contract without depending on a now-forbidden behavioral phrase"
  - "Branch-4 lone-stop short-circuit test (test_branch_4_lone_stop_target_skips_category_check) sets a mismatched primary_type (prior 'Cocktail Bar', current 'Restaurant') to PROVE the category check does NOT fire on Branch 4 — pins PATTERNS.md 'Preserve abstain semantics on Branch 4'"
  - "test_branch_5_target_primary_type_match_with_partial_byte_fraction (the multiplication-not-override guard) intentionally uses byte_fraction=0.5 (slot 3 dropped) plus a category match — the assertion `== 0.5` proves the scorer multiplies (does not override to 1.0) on a category match"

patterns-established:
  - "Pattern: module-level grep gates on source files protect prompt-engineering invariants — the scorer-side rules can grow over time but the prompt body must not regress; placing the gate next to the scorer-smoke tests keeps the prompt-rewrite contract and the scorer-extension contract in the same CI file"
  - "Pattern: test-helper kwargs with default=None / default=constant for additive contract changes — every existing test stays byte-identical, new tests opt in, sibling tests pick up the propagation automatically"

requirements-completed:
  - PROMPT-02
  - PROMPT-03

# Metrics
duration: ~30min
completed: 2026-06-04
---

# Phase 07 Plan 05: Scorer Tests + PROMPT-02 Grep Gate Summary

**Module-level PROMPT-02 grep gate locks the six D-07-04 canonical behavioral phrases out of `prompts.py` + `io.py` at CI time; seven new `TestRefinementMinimalEdit` methods pin every cell of the D-07-07 four-cell `primary_type` matrix plus two edge guards; `TestEvaluateMultiTurnProdThreading` is extended to assert the new task-only preamble's contract fields and the `primary_type` scratch-payload field per plan 07-02. The pre-existing failing test `test_prod_mode_injects_structured_plan_on_turn_1` (asserting the now-deprecated `byte-for-byte` phrase) is rewritten against the new contract and passes. All 170 tests across the two files are green; the shell-level grep gate returns 0; the Phase 6 CI hard gate `make eval-matrix-refinement-structural-check` still exits 0.**

## Performance

- **Duration:** ~30 min (Poetry install on fresh worktree dominated; actual edits were ~10 min)
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- **`tests/unit/test_critique_checks.py`:**
  - Module-level `test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts` added next to the existing smoke tests. Reads `app/agent/prompts.py` + `app/agent/io.py` via `Path(__file__).resolve().parents[2]` (mirroring `app/eval/config.py:20` `REPO_ROOT` pattern), lowercases the combined text, and asserts none of the six D-07-04 forbidden substrings appear. Fail message includes the offending phrase list so CI output is actionable.
  - `_refinement_stop` helper signature extended from `(place_id: str) -> Stop` to `(place_id: str, primary_type: str | None = None) -> Stop`. Default `None` keeps every existing call site backward-compatible; new tests opt into category values.
  - Seven new test methods inside `TestRefinementMinimalEdit` covering D-07-07:
    - `test_branch_5_target_primary_type_matches_returns_byte_fraction` — match cell, byte_fraction=1.0 → 1.0.
    - `test_branch_5_target_primary_type_mismatch_returns_0_0` — mismatch cell, byte_fraction=1.0 → 0.0.
    - `test_branch_5_prior_primary_type_missing_abstains_on_category` — abstain (missing-key) → 1.0.
    - `test_branch_5_prior_primary_type_none_abstains_on_category` — abstain (explicit-None) → 1.0.
    - `test_branch_5_current_primary_type_none_fails_loud` — fail-loud (prior present, current None) → 0.0.
    - `test_branch_5_target_primary_type_match_with_partial_byte_fraction` — match × byte_fraction=0.5 → 0.5 (proves multiplication, not override).
    - `test_branch_4_lone_stop_target_skips_category_check` — Branch 4 short-circuits BEFORE category check; prior+current target categories mismatched intentionally → 1.0 (pins PATTERNS.md "Preserve abstain semantics on Branch 4").
- **`tests/unit/test_eval_agent.py`:**
  - `_stops_for_pids` helper extended to set `primary_type="Cafe"` on every fixture Stop. All 11 sibling tests in the class pick up the deterministic value automatically.
  - `test_prod_mode_injects_structured_plan_on_turn_1` rewritten: dropped the deprecated `byte-for-byte` substring assertion (the phrase moved into `refinement_minimal_edit` per D-07-03 and is on the D-07-04 grep-gate forbidden list). Replaced with five new assertions pinning the new task-only preamble's contract fields:
    - `REFINEMENT TURN` (sentinel preserved per plan 07-01)
    - `commit_itinerary` (output-channel callout — task description per D-07-03, allowed)
    - `slot`, `place_id`, `arrival_time` (JSON-block field names from `build_refinement_prompt_message` payload)
  - Also added the Phase 7 / D-07-06 scratch-payload assertions inside the same test (per the orchestrator's prompt): `primary_type` is on every entry and equals `"Cafe"` (matching the helper extension).
  - `test_prod_mode_populates_state_scratch_for_scorer` extended with explicit `primary_type` assertions on `prior[0]`, `prior[1]`, `prior[2]` — concrete value `"Cafe"` matches the helper extension. Pre-existing `slot` + `place_id` assertions are preserved (additive change).
- **Verification gates:**
  - `poetry run pytest tests/unit/test_critique_checks.py tests/unit/test_eval_agent.py -v` → 170/170 pass.
  - `git grep -n "byte-for-byte\|same primary_type\|SAME primary_type\|keep same stop count\|do not ask clarifying questions" app/agent/prompts.py app/agent/io.py | grep -v '^#' | wc -l` → 0 (shell-level grep gate green).
  - `make eval-matrix-refinement-structural-check` → exits 0.
  - `poetry run ruff check` + `poetry run ruff format --check` → clean on both modified files.
  - Pre-commit hooks (ruff legacy alias + ruff format) pass on both commits.

## Task Commits

Each task was committed atomically:

1. **Task 1: PROMPT-02 grep gate + `_refinement_stop` extension + seven new D-07-07 matrix tests** — `8f21c0d` (test)
2. **Task 2: `_stops_for_pids` helper extension + `test_prod_mode_injects_structured_plan_on_turn_1` rewrite + `test_prod_mode_populates_state_scratch_for_scorer` primary_type extension** — `7727320` (test)

## Files Created/Modified

- **`tests/unit/test_critique_checks.py`** (+245 / -4):
  - `_refinement_stop` helper: signature + docstring extended for Phase 7 / D-07-05 / D-07-07.
  - New module-level `test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts` (PROMPT-02 / D-07-04).
  - New seven `TestRefinementMinimalEdit` test methods (D-07-07 four-cell matrix + two edge guards).
- **`tests/unit/test_eval_agent.py`** (+66 / -2):
  - `_stops_for_pids` classmethod: docstring + `primary_type="Cafe"` extension.
  - `test_prod_mode_injects_structured_plan_on_turn_1`: docstring + five new contract-field assertions + three scratch-payload assertions; deprecated `byte-for-byte` assertion removed.
  - `test_prod_mode_populates_state_scratch_for_scorer`: docstring + six new `primary_type` assertions.

## Decisions Made

- **Grep gate uses source-file `read_text`, not module import** — per the plan's `<read_first>` guidance and PATTERNS.md. A runtime import would only see the post-load `_REFINEMENT_PREAMBLE` string constant and `SYSTEM_PROMPT`, missing the adjacent docstring/comment text. The grep gate is a SOURCE-FILE invariant: even comments must not resurrect the forbidden phrases. The test was wired to fail-loud with the offending phrase list so CI output is immediately actionable.
- **Forbidden-phrase list uses lowercased normalization** — single `.lower()` on the combined source. The plan listed `"same primary_type"` and `"SAME primary_type"` separately; lowercasing collapses them to one. I included both entries in the test for clarity (matches the plan's literal list) but functionally the second is a no-op.
- **`_stops_for_pids` extension uses constant `primary_type="Cafe"`** — rather than a per-pid mapping or a new `primary_types` kwarg. CONTEXT.md leaves this to planner/implementer discretion. The constant choice keeps every sibling test backward-compatible (they pick up `"Cafe"` automatically) and the end-to-end scorer test (`test_prod_mode_refinement_minimal_edit_scores_1_0_end_to_end`, line 1513) still scores 1.0 because both turn-0 and turn-1 stops carry `primary_type="Cafe"` (D-07-07 match cell preserves byte_fraction).
- **`test_prod_mode_injects_structured_plan_on_turn_1` keeps the existing `"current_plan"` assertion** — that token comes from `build_refinement_prompt_message`'s JSON payload (`io.py:124`) and is a structural anchor for the structured-plan injection. The new contract-field assertions (`REFINEMENT TURN`, `commit_itinerary`, `slot`, `place_id`, `arrival_time`) add defense-in-depth: even if a future PR changes the JSON-block structure, the sentinel + output-channel callout would still be present in the preamble; even if the preamble wording drifts, the JSON-block field names anchor the structural shape.
- **Branch-4 lone-stop test sets mismatched primary_types** — prior `"Cocktail Bar"`, current `"Restaurant"`. If the scorer regressed and applied the D-07-07 mismatch path on Branch 4, the assertion `== 1.0` would fail at `0.0`. This is stricter than just "Branch 4 returns 1.0"; it proves Branch 4 short-circuits BEFORE the category sub-check fires.
- **Partial byte_fraction × category match test (multiplication guard)** — uses 3-stop prior with slot 3 dropped (byte_fraction = 1/2 = 0.5) plus a category match. The assertion `== 0.5` proves the scorer MULTIPLIES the byte_fraction by the category result (rather than overriding to 1.0 on match). If a future regression treated the category match as an override, the assertion would fail at `1.0`.

## Deviations from Plan

None — both tasks executed exactly as written. All seven new TestRefinementMinimalEdit methods pass on the first run; the grep gate passes on the first run (confirming plan 07-01's rewrite + plan 07-04's docstring stayed clean); the prod-threading prior-failing test was rewritten with the orchestrator's specified contract fields and now passes.

The orchestrator's prompt added a critical instruction beyond Task 2's `<action>` block: the failing `test_prod_mode_injects_structured_plan_on_turn_1` had to be updated in this plan (not just the `test_prod_mode_populates_state_scratch_for_scorer` named in the plan). I applied the orchestrator's instruction — replace `byte-for-byte` with new preamble contract-field assertions AND extend the scratch-payload assertion to check `primary_type`. Both changes are inside the same test now, and the original Task 2 scratch-payload extension to `test_prod_mode_populates_state_scratch_for_scorer` also landed per the plan.

## Issues Encountered

- **Poetry virtualenv bootstrap on fresh worktree** — first `poetry run pytest` failed with `Command not found: pytest`; `poetry install --no-interaction` (~3 min) created the venv and installed dependencies; subsequent runs were fast. Documented for future agents in fresh worktrees.
- **Plan files not in worktree git history** — the plan files (`07-05-PLAN.md`, `07-CONTEXT.md`, etc.) live in the parent repo's `.planning/` directory but are gitignored, so the fresh worktree's `.planning/phases/07-prompt-rubric-decoupling/` only contained committed SUMMARYs. Copied the four required PLAN/CONTEXT/PATTERNS/DISCUSSION-LOG files from the parent repo before reading them. This is the standard worktree-with-gitignored-`.planning` setup and not a code issue.

## User Setup Required

None — no environment variables, dashboards, or external services need configuration. All changes are pure test code.

## Next Phase Readiness

- **Plan 07-07 (re-baseline + falsifier) unblocked** — the scorer-extension + scratch-payload-extension contracts are now pinned at the test level. Re-running `eval_matrix_refinement.yaml` against the new prompt + scorer will surface any cell where the scorer change moves the baseline; both PROMPT-04 (no-regression on `openai/gpt-4o-mini`) and PROMPT-05 (falsifier on `gpt-5-mini`) can run against the locked test surface without fear of silent test-side drift.
- **Plan 07-06 (`/chat` integration test) already merged** (commit `6153060`) — no further dependency on this plan.
- **CI hard gate `make eval-matrix-refinement-structural-check` continues to pass** — defense-in-depth that the structural matrix shape, scorer registration, and shared-helper invariants stayed intact.

## Self-Check: PASSED

- `tests/unit/test_critique_checks.py` contains:
  - `test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts` at the module level (verified by `poetry run pytest ... -v --collect-only`).
  - `_refinement_stop(place_id, primary_type=None)` signature (verified in source).
  - Seven new `TestRefinementMinimalEdit` methods named exactly per the plan's `<acceptance_criteria>` (verified by `poetry run pytest ... --collect-only | grep ...`).
- `tests/unit/test_eval_agent.py` contains:
  - `_stops_for_pids` sets `primary_type="Cafe"` (verified in source).
  - `test_prod_mode_injects_structured_plan_on_turn_1` no longer asserts `"byte-for-byte"` (verified by `grep -n "byte-for-byte" tests/unit/test_eval_agent.py` → no matches in test bodies; only the docstring mentions the phrase as the deprecated assertion that was REPLACED).
  - `test_prod_mode_populates_state_scratch_for_scorer` asserts `"primary_type" in prior[0]` and `prior[0]["primary_type"] == "Cafe"` (verified in source).
- `poetry run pytest tests/unit/test_critique_checks.py tests/unit/test_eval_agent.py -v -x` exits 0 (170/170 pass).
- Shell-level grep gate `git grep -n "byte-for-byte|same primary_type|SAME primary_type|keep same stop count|do not ask clarifying questions" app/agent/prompts.py app/agent/io.py | grep -v '^#' | wc -l` returns 0.
- `make eval-matrix-refinement-structural-check` exits 0.
- Commits `8f21c0d` (Task 1) and `7727320` (Task 2) present on `worktree-agent-aa1abed0d24d6cf79` branch (verified via `git log --oneline 8f21c0d^..HEAD`).
- Pre-commit hooks (ruff legacy alias + ruff format) passed on both commits.

---

*Phase: 07-prompt-rubric-decoupling*
*Plan: 05-scorer-tests-and-grep-gate*
*Completed: 2026-06-04*
