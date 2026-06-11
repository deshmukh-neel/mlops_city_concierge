---
phase: 07-prompt-rubric-decoupling
plan: 02
subsystem: eval-harness

tags:
  - eval-harness
  - refinement
  - scoring-contract
  - scratch-payload
  - primary_type

# Dependency graph
requires:
  - phase: 06-minimal-edit-refinement
    provides: "prior_committed_stops scratch contract {slot, place_id}; refinement_minimal_edit scorer Branch-5 byte-equality fraction; ExpectedRefinement.target_slot wiring"
provides:
  - "Turn-0 happy-path scratch entries shape extended from {slot, place_id} to {slot, place_id, primary_type}"
  - "ExpectedRefinement docstring documenting Phase 7 scratch shape extension with cross-reference to canonical refinement_minimal_edit contract"
  - "Data-plane prerequisite for plan 07-04 scorer extension (same-category check on the target slot)"
affects:
  - 07-04-scorer-category-extend
  - 07-05-scorer-tests-and-grep-gate
  - 07-07-rebaseline-and-falsifier

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Additive scratch-payload contract extension (legacy callers continue to read existing keys; new key is opt-in)"
    - "Model-facing vs offline-eval data separation (HIGH-4 mitigation): primary_type lives on scratch payload only, NOT in build_refinement_prompt_message JSON block"

key-files:
  created:
    - ".planning/phases/07-prompt-rubric-decoupling/07-02-SUMMARY.md"
  modified:
    - "scripts/eval_agent.py"
    - "app/eval/config.py"

key-decisions:
  - "D-07-06 honored: primary_type lives on the scratch payload only (offline-eval data plane), NOT in the model-facing build_refinement_prompt_message JSON block — HIGH-4 prompt-injection mitigation preserved byte-identically"
  - "Empty-commit branch unchanged: no per-entry data to extend; refinement_minimal_edit Branch 2 fail-loud semantics intact"
  - "prior_stops_obj key (internal full-Stop carrier) unchanged: scratch-shape extension is purely additive on the externally-readable prior_committed_stops list"

patterns-established:
  - "Pattern: extend scratch-payload contracts additively — preserve legacy reader assertions on existing keys, document the new field on the canonical contract docstring, cross-reference the scorer that consumes it"
  - "Pattern: separate model-facing JSON payload (HIGH-4-whitelisted: slot/place_id/arrival_time) from offline-eval scratch payload (broader: includes primary_type for scorer-only use)"

requirements-completed:
  - PROMPT-03

# Metrics
duration: ~12min
completed: 2026-06-04
---

# Phase 7 Plan 02: Scratch Payload Extend Summary

**Eval-runner turn-0 scratch entries now carry `primary_type` per entry alongside `slot` and `place_id`, enabling the plan 07-04 scorer to enforce the same-category rule that prompt rule 10 used to prescribe — while the model-facing JSON block stays byte-identical (HIGH-4 mitigation preserved).**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-04T03:12:00Z (approximate)
- **Completed:** 2026-06-04T03:24:48Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `scripts/eval_agent.py:_run_prod_threading` turn-0 happy-path now writes `{slot, place_id, primary_type}` per entry on `state.scratch["prior_committed_stops"]`, sourced from `Stop.primary_type` (no new DB call, no new wiring).
- Empty-commit branch (lines 873-878) verified byte-identical to pre-Phase-7 (no per-entry data to extend).
- `prior_stops_obj` internal key (carrier of full `Stop` instances for `build_refinement_prompt_message`) verified unchanged.
- `app/eval/config.py::ExpectedRefinement` docstring extended with a Phase 7 / D-07-06 attribution paragraph documenting the scratch-shape extension and pointing at `app/agent/critique/checks.py::refinement_minimal_edit` as the canonical contract.
- All 16 existing `TestEvaluateMultiTurnProdThreading` + `TestDeterministicChecksRegistration` tests continue to pass under the additive change.
- `make eval-matrix-refinement-structural-check` continues to pass — confirms `build_refinement_prompt_message` JSON shape is unchanged.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend turn-0 scratch happy-path with primary_type per entry** — `e02207e` (feat)
2. **Task 2: Update ExpectedRefinement docstring to document the primary_type scratch extension** — `6956b19` (docs)

## Files Created/Modified

- `scripts/eval_agent.py` — `_run_prod_threading` turn-0 happy-path dict-comprehension extended from `{"slot": i + 1, "place_id": s.place_id}` to `{"slot": i + 1, "place_id": s.place_id, "primary_type": s.primary_type}`. Empty-commit branch and `prior_stops_obj` key untouched.
- `app/eval/config.py` — `ExpectedRefinement` docstring extended with Phase 7 / D-07-06 attribution paragraph explaining the scratch-shape extension and naming `refinement_minimal_edit` as the canonical contract. Pydantic field `target_slot: int = Field(ge=1)` unchanged; `model_config = ConfigDict(extra="forbid")` unchanged.

## Decisions Made

- **Honored D-07-06 strictly:** `primary_type` is added ONLY to the offline-eval scratch payload, NOT to `build_refinement_prompt_message`'s JSON payload (which still surfaces `slot`/`place_id`/`arrival_time` per entry). The HIGH-4 prompt-injection mitigation (`io.py:72-82`) stays byte-identical.
- **Used dict-shaped entries**, not a new `PriorStop(BaseModel)` Pydantic class. CONTEXT.md leaves this to planner discretion; matching the existing 06-06 dict shape keeps the diff minimal and preserves backward-compat with the legacy scratch reader.
- **Empty-commit branch byte-identical:** no per-entry change because the list is empty. PATTERNS.md confirms this expectation.
- **Sourced `primary_type` from `Stop.primary_type` (`app/agent/state.py:202`)** which is already populated by `commit_itinerary` from `places_raw` lookups at turn 0 — no new DB call, no new tool-result wiring, no new dependency on commit-time invariants beyond what plan 04 already established (CAT-01..CAT-04).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Plan defect / overly strict acceptance check] Acceptance check on `build_refinement_prompt_message` source treated literally would always fail**

- **Found during:** Task 1 verification
- **Issue:** The plan's acceptance criterion `assert "primary_type" not in inspect.getsource(build_refinement_prompt_message)` is unsatisfiable in the pre-existing codebase — the function's docstring (`app/agent/io.py:117-120`) legitimately mentions `primary_type` as one of the fields explicitly DROPPED by the HIGH-4 mitigation (`"Client-supplied display strings (name, primary_type, rationale, source, address) are dropped at the helper boundary"`). The plan's intent (verify the JSON payload code body excludes `primary_type`) is satisfied; the literal string check is broken because it does not differentiate docstring from code.
- **Fix:** Instead of asserting the literal substring on the full source, AST-parsed the function, removed the docstring node, and asserted `primary_type not in ast.unparse(remaining_body)`. The code body (the `plan_payload` dict-comprehension at `io.py:133-142`) is verified clean.
- **Files modified:** None (verification-only deviation; source is byte-identical to pre-Phase-7).
- **Verification:** AST-based check returns "OK: build_refinement_prompt_message CODE body has NO primary_type (docstring excluded)" with the unparsed code body printed for inspection. Confirmed the model-facing JSON keys remain `{slot, place_id, arrival_time}` only.
- **Committed in:** No code change required; deviation is a re-interpretation of the acceptance contract to match documented intent (D-07-06 + HIGH-4).

---

**Total deviations:** 1 (verification-method correction; zero source changes)
**Impact on plan:** D-07-06 + HIGH-4 mitigation invariant fully preserved. Plan 07-05 will replace the literal substring check with an explicit code-body check (or the equivalent AST check) when it lands the formal grep gate per D-07-04.

## Issues Encountered

None — both tasks executed cleanly. All 16 existing prod-threading + scorer-registration tests pass; the eval-matrix structural-check gate passes; verification commands all return OK.

## User Setup Required

None — no environment variables, dashboards, or external services need configuration. The change is internal to the offline eval data-plane.

## Next Phase Readiness

- **Plan 07-04 unblocked:** the scorer extension can now read `prior_committed_stops[i]["primary_type"]` to compare against `state.stops[target_slot - 1].primary_type` on Branch 5. D-07-07's missing-prior abstain branch is testable end-to-end (legacy callers that have not been re-run will still surface `{slot, place_id}` only and the scorer abstains on the category check).
- **Plan 07-05 unblocked:** explicit `primary_type` assertions can land alongside the existing `prior[0]["slot"]` + `prior[0]["place_id"]` assertions in `tests/unit/test_eval_agent.py:1442-1448`; the grep gate per D-07-04 can use AST-aware code-body inspection (see Deviation #1).
- **Plan 07-07 unaffected:** re-baseline pipeline reads the same `prior_committed_stops` shape; the extra field is silently ignored by anything that only consumes `slot` and `place_id`.

## Self-Check: PASSED

- `scripts/eval_agent.py` extension present at commit `e02207e` (verified: `{"slot": i + 1, "place_id": s.place_id, "primary_type": s.primary_type}` substring found by AST verifier).
- `app/eval/config.py` ExpectedRefinement docstring extension present at commit `6956b19` (verified: `primary_type` ∈ doc, `D-07-06` ∈ doc, `refinement_minimal_edit` ∈ doc; doc length 1221).
- `build_refinement_prompt_message` CODE body excludes `primary_type` (AST verifier on stripped function body — see Deviation #1).
- 16/16 prod-threading + scorer-registration unit tests pass (`poetry run pytest tests/unit/test_eval_agent.py -k "TestEvaluateMultiTurnProdThreading or TestDeterministicChecksRegistration"`).
- `make eval-matrix-refinement-structural-check` exits 0.
- Git log shows both commits on `worktree-agent-a7ba6dc0d6ef67972` branch (`e02207e` → `6956b19`).

---

*Phase: 07-prompt-rubric-decoupling*
*Plan: 02-scratch-payload-extend*
*Completed: 2026-06-04*
