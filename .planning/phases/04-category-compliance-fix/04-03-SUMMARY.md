---
phase: 04-category-compliance-fix
plan: 03
subsystem: agent
tags: [graph-injection, langchain-tools, json-safety, tdd, d-04-04]

requires:
  - phase: 04-category-compliance-fix
    plan: 01
    provides: optional slot_index kwarg on semantic_search/nearby, family_of() lookup, UserConstraints.requested_primary_types
provides:
  - _inject_primary_type_family helper in app/agent/graph.py mirroring _inject_closure_exclusions
  - Chained injection in act() — closure_exclusions -> primary_type_family -> slot_index strip
  - Functional /chat coverage proving the graph injects primary_type_family for the named slot
affects: [phase-04-category-compliance-fix, graph-injection, eval-gates]

tech-stack:
  added: []
  patterns:
    - Graph-layer tool_call args rewrite mirroring app/agent/swap.py _inject_closure_exclusions
    - JSON-safe filters dict (never a Pydantic SearchFilters instance) per project memory aimessage_tool_call_args_json_safe.md
    - TDD RED/GREEN commits per task

key-files:
  created:
    - tests/unit/test_graph_slot_injection.py
    - .planning/phases/04-category-compliance-fix/04-03-SUMMARY.md
  modified:
    - app/agent/graph.py
    - tests/unit/test_chat_functional.py

key-decisions:
  - "Helper lives in app/agent/graph.py (co-located with act()) for parity with the existing closure-exclusions call site. Imports family_of and SearchFilters from app.tools.filters."
  - "Chain order in act(): closure_exclusions first, then primary_type_family on its dict output, then a final {k: v for k, v in ... if k != 'slot_index'} comprehension to drop the marker arg before tool.invoke."
  - "Defensive int-check on slot_index rejects bool (a subclass of int) so True/False can never sneak in as indices."
  - "When the model emits a non-matching primary_type_family in filters, the graph OVERWRITES it with the slot-derived family — explicit enforcement per D-04-04 / T-04-03-05."

patterns-established:
  - "Multi-helper chain composition on tool_call args stays JSON-safe end-to-end as long as every helper returns a plain dict (verified by test_inject_primary_type_family_result_is_json_dumps_safe across every input shape)."
  - "Functional /chat tests can inject UserConstraints fields via monkeypatch on app.main.UserConstraints, isolating graph behavior from upstream wiring that lands in later plans (04-06 intake pipeline)."

requirements-completed: [CAT-01]

duration: 7 min
completed: 2026-05-22
---

# Phase 04 Plan 03: Graph-Layer Filter Injection Summary

**The agent's `act()` node now writes `filters.primary_type_family` into retrieval tool_call args for each slot the model declares via `slot_index`, sidestepping the critique-loop ↔ commit conflict while preserving the JSON-safety invariant.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-22T22:08:16Z
- **Completed:** 2026-05-22T22:15:13Z
- **Tasks:** 2
- **Files modified:** 3 (1 source, 2 tests)

## Accomplishments

- Added `_inject_primary_type_family(tool_name, args, requested_primary_types)` in `app/agent/graph.py`. Mirrors `_inject_closure_exclusions` (`app/agent/swap.py:456-510`) exactly in shape, defensive branches, and JSON-safety invariants.
- Wired the helper into `act()`'s tool-execution loop as a chain on top of `_inject_closure_exclusions`, plus a final `slot_index` strip via dict comprehension. The AIMessage's `tc["args"]` is NEVER reassigned — the chain only feeds a local `effective_args` dict to `tool.invoke` and the scratch record.
- Added 15 pure-function unit tests covering every helper branch (happy path, all noops, dict vs Pydantic input, model-overwrite, JSON-safety, no-mutation).
- Added 5 functional tests on `act()` driving the helper through a one-shot scripted LLM, including the ADVISORY 4 `kg_traverse` strip-block no-op regression test.
- Added 1 end-to-end `/chat` functional test (`test_chat_graph_injects_primary_type_family_for_slot`) exercising the full FastAPI request stack with `requested_primary_types` pre-populated.

## Task Commits

1. **Task 1 RED: failing helper tests** — `5f62bd3` (test)
2. **Task 1 GREEN: `_inject_primary_type_family` implementation** — `86c8aff` (feat)
3. **Task 2 GREEN: wire chain into act() + functional /chat test** — `9fd29b1` (feat)

Task 2 reused the RED tests committed in 5f62bd3 — the same test file already carries both Task 1's helper tests AND Task 2's functional `act()` tests, mirroring the plan's specification that both layers live in `tests/unit/test_graph_slot_injection.py`.

**Plan metadata:** committed separately with this summary.

## Files Created/Modified

- `app/agent/graph.py` — Added the `_inject_primary_type_family` helper at module level. Updated `act()` to chain `closure_exclusions → primary_type_family → slot_index-strip` into a local `effective_args` dict. Imports `family_of` and `SearchFilters` from `app.tools.filters`. The CRITICAL JSON-safety comment block in `act()` is preserved and extended with one new line describing the Phase 4 chain step.
- `tests/unit/test_graph_slot_injection.py` — New file with 20 tests (15 pure-function + 5 functional `act()` tests, including the ADVISORY 4 kg_traverse no-op regression).
- `tests/unit/test_chat_functional.py` — Added `test_chat_graph_injects_primary_type_family_for_slot`, the end-to-end /chat functional test driving the graph injection through the production FastAPI stack.

## Decisions Made

- **Helper co-location in `app/agent/graph.py`** (not a new module). The plan offered either location; co-locating beside `act()` mirrors `_inject_closure_exclusions` already sitting in `app/agent/swap.py` (its caller is `act()` too) and keeps imports local for the chained call site.
- **Reject `bool` as a `slot_index` value.** `isinstance(True, int)` is `True` in Python, which would silently use `True` (=1) or `False` (=0) as a slot index. The defensive guard `not isinstance(slot_index, int) or isinstance(slot_index, bool)` is one extra line that prevents an obscure bug class.
- **Functional /chat test monkeypatches `app.main.UserConstraints`** instead of waiting for Plan 04-06's intake pipeline. The plan explicitly noted that 04-03 is independent of 04-02 and that the functional test should drive the graph injection, not the upstream wiring. Patching `UserConstraints` at the import site is the smallest seam that isolates the contract this plan owns.
- **Pre-existing functional test in `test_chat_functional.py` already mocked `itinerary_violations`** per project memory `full_suite_db_pool_contamination.md`; I followed the same pattern for the new test to keep it hermetic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Used local `.venv` for verification (no poetry env)**
- **Found during:** Task 1 RED
- **Issue:** `poetry env` for this worktree pointed at an empty cache dir; `poetry run pytest` failed with `ModuleNotFoundError: pytest`. The Wave-1 executors (04-01, 04-02) each carried similar deviations.
- **Fix:** Used the pre-existing parent-repo `.venv` (`/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge/.venv/bin/python`) which already has pytest, mypy, and all project deps installed; `app` resolves correctly because it's editable-installed via `poetry install` from cwd.
- **Files modified:** None tracked
- **Verification:** `.venv/bin/python -m pytest tests/unit -q` → 785 passed, 7 skipped (up from 764 pre-plan, all 21 new tests included).
- **Committed in:** N/A, environment-only.

**2. [Rule 2 — Missing Critical] `bool` admitted as `slot_index` value**
- **Found during:** Task 1 GREEN
- **Issue:** The plan specified `isinstance(slot_index, int)` as the type gate, but `bool` is a subclass of `int` in Python — `True` would silently be treated as slot 1. The plan's behavior contract clearly intended ints only (it says "if None or not an int, early-return"); a literal reading of the plan code would have admitted bools.
- **Fix:** Added `or isinstance(slot_index, bool)` to the rejection guard so True/False return `dict(args)` unchanged.
- **Files modified:** `app/agent/graph.py`
- **Verification:** Implicitly covered by `test_inject_primary_type_family_noop_when_slot_index_not_int` (a non-int string also follows the same rejection path; the bool case is a guard against an even more subtle bug not explicitly tested but locked by the same code path).
- **Committed in:** `86c8aff`.

**3. [Rule 1 — Test Strengthening] Removed unused `_SlotConstraints` subclass in functional test**
- **Found during:** Task 2 test addition
- **Issue:** Initial draft of `test_chat_graph_injects_primary_type_family_for_slot` defined a `_SlotConstraints(_RealUserConstraints)` subclass that was never used — a leftover from an earlier monkeypatch shape exploration. Ruff's pre-commit hook flagged unused class.
- **Fix:** Replaced the dead subclass with a small `_make_constraints` factory function that defaults `requested_primary_types=["Sushi Restaurant"]`.
- **Files modified:** `tests/unit/test_chat_functional.py`
- **Verification:** ruff hook passed on the next commit; test still green.
- **Committed in:** `9fd29b1`.

**4. [Rule 3 — Blocking] Ruff auto-reformat between commit attempts**
- **Found during:** Task 1 RED commit and Task 2 GREEN commit
- **Issue:** The pre-commit ruff hook reformatted code (whitespace, import order) on the first commit attempt, which aborted the commit per the hook's design.
- **Fix:** Re-staged the auto-formatted file and re-ran `git commit`. Per project memory `feedback_precommit_ruff.md`, this is the intended workflow — the hook handles formatting automatically.
- **Files modified:** `tests/unit/test_graph_slot_injection.py`, `app/agent/graph.py`
- **Verification:** Both subsequent commit attempts passed both ruff legacy and ruff format hooks.
- **Committed in:** `5f62bd3` and `9fd29b1`.

**5. [Rule 1 — Test Bug] `N814 camelcase-imported-as-constant` on `HumanMessage as _HM`**
- **Found during:** Task 1 RED commit
- **Issue:** Initial draft of `tests/unit/test_graph_slot_injection.py` aliased `HumanMessage as _HM` inside a function body, hitting ruff's `N814` rule (camelcase-imported-as-constant).
- **Fix:** Removed the aliased import — `HumanMessage` is already imported at module level.
- **Files modified:** `tests/unit/test_graph_slot_injection.py`
- **Verification:** ruff hook passed on the re-commit.
- **Committed in:** `5f62bd3`.

---

**Total deviations:** 5 auto-fixed (Rule 1: 2, Rule 2: 1, Rule 3: 2)
**Impact on plan:** No scope change. Two were Pythonic-correctness reinforcements of the plan's behavior contract (the bool-rejection guard) or pre-commit-hook iteration; the rest were environment / lint feedback during commits.

## Issues Encountered

- The plan files (`04-03-graph-injection-PLAN.md`, `04-CONTEXT.md`, `04-PATTERNS.md`) were NOT yet committed to the worktree base (`811eff3`) — they live in the main repo's working tree only. I read them directly from the parent repo path. This is consistent with the Wave-1 SUMMARY files' note about cross-plan write-boundary discipline.
- The plan's `test_act_strips_slot_index_before_tool_invoke` design asserts that the underlying retrieval mock doesn't receive `slot_index`. Even WITHOUT the strip step, the tool wrapper itself doesn't forward `slot_index` to `_semantic_search` (only `query`, `filters`, `k`). The test is still informative because it documents the contract, but the stronger guard for the strip step is `test_act_injects_primary_type_family_when_model_passes_slot_index` which asserts `slot_index not in scratch[...]['args']` (verifying the strip happened on the recorded effective_args).
- No state/roadmap/requirements files were edited per the parallel-worktree write boundary; the orchestrator will batch-update those after wave 2 settles.

## Verification

- `.venv/bin/python -m pytest tests/unit/test_graph_slot_injection.py -v` → **20 passed** (15 pure-function + 5 functional).
- `.venv/bin/python -m pytest tests/unit/test_chat_functional.py -v` → **5 passed** (4 pre-existing + 1 new).
- `.venv/bin/python -m pytest tests/unit/ -q` → **785 passed, 7 skipped** (full unit suite; +21 net from pre-plan baseline of 764+7).
- `.venv/bin/python -m mypy app/agent/graph.py` → clean.
- `.venv/bin/python -m mypy app/` → clean (34 source files).
- Acceptance probes (per plan):
  - `grep -n "def _inject_primary_type_family" app/agent/graph.py` → 1 line at `48`.
  - `grep -n "from app.tools.filters import" app/agent/graph.py | grep -E "family_of|SearchFilters"` → 1 line at `43` (both symbols).
  - `grep -n "_inject_primary_type_family" app/agent/graph.py | grep -v '^#'` → 2 lines (definition + use in act()).
  - `grep -n "slot_index" app/agent/graph.py | grep -v '^#'` → 10+ lines (helper + chain + strip).
  - `grep -c 'tc\["args"\] =' app/agent/graph.py` → **0** (act() still does NOT mutate tc['args']).
  - Direct Python probes for happy path, json safety, noop on empty requested, noop on out-of-range — all OK.
  - End-to-end chain smoke (closure_exclusions → primary_type_family → strip) JSON-safe → OK.

## Known Stubs

None. The injection helper is fully wired into `act()` and exercised end-to-end via the functional /chat test. The production /chat handler does NOT yet populate `UserConstraints.requested_primary_types` from the user message — that's Plan 04-06's intake-pipeline scope (per CONTEXT.md D-04-01 sequencing). This is NOT a stub for 04-03's deliverable: the eval runner (04-02) already wires constraints from YAML, so the strict scorer will fire as-designed on the Phase 4 eval cases the moment 04-04 (prompts) lands.

## Threat Flags

None. The threat-register entries from the plan (T-04-03-01 through T-04-03-07) are all addressed:

- **T-04-03-01** (tc['args'] mutation): enforced by `grep -c 'tc["args"] =' app/agent/graph.py` returning 0; locked by `test_act_does_not_mutate_tc_args_under_slot_injection`.
- **T-04-03-02** (malicious slot_index): bounds + type + bool checks; locked by 4 noop tests.
- **T-04-03-03** (info disclosure): accepted; no PII in filters.
- **T-04-03-04** (DoS): accepted; chain is exactly 3 O(1) steps.
- **T-04-03-05** (model-emitted family overwrite): mitigation locked by `test_inject_primary_type_family_overwrites_model_emitted_family`.
- **T-04-03-06** (Pydantic in tc): mitigation locked by `test_inject_primary_type_family_result_is_json_dumps_safe` across all input shapes.
- **T-04-03-07** (kg_traverse strip drift): ADVISORY 4 mitigation locked by `test_act_does_not_alter_kg_traverse_args_when_strip_runs`.

No new surfaces (network endpoints, auth paths, file access, trust-boundary schema changes) were introduced beyond what the plan's threat model already covered.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 04-04 (prompts)** can now rely on the graph-layer enforcement: any SYSTEM_PROMPT directive telling the model to emit `slot_index` per slot will produce observable `primary_type_family` injection when it cooperates, AND a measurable non-cooperation signal when it doesn't (D-04-06 trust boundary).
- **Plan 04-06 (intake pipeline)** can wire `UserConstraints.requested_primary_types` from the user message; the functional test in this plan shows exactly how the value flows through `/chat` → `ItineraryState.constraints` → `act()` chain → effective_args.
- **D-04-13 merge gate** is now half-buildable: `category_compliance_strict` (04-01) and `primary_type_family` injection (04-03) together cover the strict scorer's measurement story. Once 04-04's prompt directives land and the eval is re-baselined, the +0.3 delta target on the strict scorer becomes measurable.

## Self-Check: PASSED

- Helper function present at `app/agent/graph.py:48` — FOUND.
- Helper call site in `act()` at `app/agent/graph.py:346` — FOUND.
- New test file `tests/unit/test_graph_slot_injection.py` — FOUND, 20 tests.
- New functional test `test_chat_graph_injects_primary_type_family_for_slot` in `tests/unit/test_chat_functional.py` — FOUND.
- All task commits present: `5f62bd3`, `86c8aff`, `9fd29b1` — FOUND.
- Full unit suite (`tests/unit/` 785 passed, 7 skipped) and mypy (`app/` 34 files clean) green.
- No `tc["args"] =` mutation introduced anywhere in `app/agent/graph.py`.

---
*Phase: 04-category-compliance-fix*
*Completed: 2026-05-22*
