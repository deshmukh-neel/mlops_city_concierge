---
phase: 04-category-compliance-fix
plan: 04
subsystem: agent
tags: [prompts, system-prompt, revision-guidance, text-only]

requires:
  - phase: 04-category-compliance-fix
    plan: 01
    provides: rationale_misaligned RevisionReason literal + slot_index tool kwarg
provides:
  - SYSTEM_PROMPT slot_index directive in step 4
  - SYSTEM_PROMPT step 6 rationale tightening tied to committed primary_type
  - REVISION_GUIDANCE rationale_misaligned bullet under CRITIQUE_ITINERARY
affects: [phase-04-category-compliance-fix, model-contract, graph-injection, revision-loop]

tech-stack:
  added: []
  patterns:
    - Pure-text SYSTEM_PROMPT additions with no new f-string substitutions
    - REVISION_GUIDANCE bullet style mirrored from existing entries
    - Substring-based prompt tests in tests/unit/test_agent_prompts.py

key-files:
  created:
    - .planning/phases/04-category-compliance-fix/04-04-SUMMARY.md
  modified:
    - app/agent/prompts.py
    - tests/unit/test_agent_prompts.py

key-decisions:
  - "Slot-index directive appended to step 4 (PLAN MULTI-STOP itineraries) because multi-stop logic is where slot ordering lives — not step 2's filter discussion."
  - "Step 6 tightening appended to the existing JUSTIFY sentence rather than added as a new step so the rationale rule stays co-located with the rule it refines."
  - "rationale_misaligned bullet placed at the end of the CRITIQUE_ITINERARY list (after hallucinated_place_id) — closest readability fit for a rationale-text-rewrite directive."
  - "Added a coverage test asserting EVERY post-commit RevisionReason has a REVISION_GUIDANCE entry — catches future regressions where someone adds a new RevisionReason without a model-facing bullet."

patterns-established:
  - "REVISION_GUIDANCE bullet coverage test (test_revision_guidance_covers_every_revision_reason_used_by_dispatch) acts as a forward guard for the dispatch ↔ guidance contract."

requirements-completed: [CAT-01, CAT-02, RAT-01]

duration: 4 min
completed: 2026-05-22
---

# Phase 04 Plan 04: Prompts Summary

**SYSTEM_PROMPT now carries the slot_index directive, the step-6 primary_type rationale rule, and a REVISION_GUIDANCE bullet for `rationale_misaligned` — three text-only additions that complete the model-to-graph contract for per-slot category compliance.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-22T22:08:33Z
- **Completed:** 2026-05-22T22:12:07Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Appended a slot_index per-slot directive to SYSTEM_PROMPT step 4 telling the model to pass `slot_index = i` (0-based) on each retrieval tool_call when the user named per-slot categories. This is the model-side counterpart of the `slot_index` kwarg added to `semantic_search` and `nearby` in plan 04-01.
- Appended a new sentence to SYSTEM_PROMPT step 6 ("JUSTIFY every stop") binding the rationale to the committed place's actual `primary_type` from the tool result, NOT the user's requested category — closes the rationale-misalignment hole that D-04-07 / CAT-02 / RAT-01 target.
- Added a `rationale_misaligned` bullet under the CRITIQUE_ITINERARY section of REVISION_GUIDANCE matching the existing bullet style (problem statement + suggested action). The bullet tells the model to REWRITE the rationale prose, not swap the stop — the stop is fine; only the rationale text is misaligned.
- Verified ADVISORY 7 prerequisite: `REVISION_GUIDANCE` IS concatenated into `SYSTEM_PROMPT` (line 170 of `app/agent/prompts.py` uses `+ REVISION_GUIDANCE`), so substring assertions against `SYSTEM_PROMPT` are valid for the new bullet. No deviation needed; the common case held.
- Added four new tests in `tests/unit/test_agent_prompts.py` covering each addition plus a coverage guard for the dispatch ↔ guidance contract.

## Task Commits

1. **Task 1: slot_index directive + step-6 rationale tightening** — `19d7e45` (feat)
2. **Task 2: rationale_misaligned bullet in REVISION_GUIDANCE** — `6aa1748` (feat)

**Plan metadata:** committed separately with this summary.

## Files Created/Modified

- `app/agent/prompts.py` — Appended slot_index directive to step 4, tightened step 6 with the primary_type rationale rule, added rationale_misaligned bullet at the end of the CRITIQUE_ITINERARY section of REVISION_GUIDANCE. Pure text additions; no imports, no new f-string substitutions, no helper functions.
- `tests/unit/test_agent_prompts.py` — Added `test_system_prompt_has_step6_primary_type_directive`, `test_system_prompt_has_slot_index_directive`, `test_revision_guidance_has_rationale_misaligned_bullet`, and `test_revision_guidance_covers_every_revision_reason_used_by_dispatch`. Imported `REVISION_GUIDANCE` alongside the existing `SYSTEM_PROMPT` and `CLARIFYING_STOPS_COUNT_TEMPLATE` imports.

## Decisions Made

- **Placement of slot-index directive (Task 1):** Picked step 4 ("PLAN MULTI-STOP itineraries") over step 2 ("PREFER structured filters") because the slot ordering is a multi-stop concept — stop K maps to slot K. Step 2's filter discussion is about query content vs. filter content, not stop ordering. Step 4 is the natural home.
- **Placement of step 6 sentence (Task 1):** Appended to the existing JUSTIFY sentence rather than added as a new step. The new rule refines the existing rationale instruction; co-locating them prevents the model from treating them as independent rules. The example ("never claim a stop offers omakase if its primary_type is not Sushi Restaurant") makes the rule concrete and binding.
- **Placement of rationale_misaligned bullet (Task 2):** Added at the end of the CRITIQUE_ITINERARY bullet list, after `hallucinated_place_id`. The bullet refers to rationale prose quality — a step-text-level concern — and reads more naturally at the end of the list than wedged between the geographic/temporal/constraint bullets.
- **Added forward-guard coverage test (Task 2):** Beyond the plan's required tests, added `test_revision_guidance_covers_every_revision_reason_used_by_dispatch` that asserts every post-commit `RevisionReason` (from the live `Literal` in `state.py`) has a REVISION_GUIDANCE entry. Rationale: adding a new RevisionReason without a guidance bullet currently silently falls through to the `else` branch in `_hint_for_violation` (per 04-PATTERNS.md "Key invariant" note). The new test catches that class of regression at unit-test time rather than at runtime.
- **`SYSTEM_PROMPT` is the assertion target for `rationale_misaligned` (Task 2):** ADVISORY 7's interpolation check passed (`REVISION_GUIDANCE in SYSTEM_PROMPT` is True because of the `+ REVISION_GUIDANCE` concatenation on line 170), so the model receives the bullet on every plan() step. Asserted both `REVISION_GUIDANCE` and `SYSTEM_PROMPT` for robustness against future refactors that might swap concatenation for a different composition pattern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created Poetry virtualenv on first run**

- **Found during:** Task 1 baseline test run.
- **Issue:** Poetry was installed but no project virtualenv existed; `poetry run pytest` failed with `ModuleNotFoundError: No module named 'pytest'`.
- **Fix:** Ran `poetry install --with dev` to provision the venv with all project + dev dependencies.
- **Files modified:** None tracked (venv lives outside the repo at `~/Library/Caches/pypoetry/virtualenvs/`).
- **Verification:** `poetry run pytest --version` returned `8.4.2`; subsequent test runs all succeeded.
- **Committed in:** N/A, environment-only.

**Total deviations:** 1 auto-fixed (Rule 3: 1)
**Impact on plan:** None. The fix was environmental setup required for the plan's `poetry run pytest` verification commands to work. No source files touched.

## Verification

- `poetry run pytest tests/unit/test_agent_prompts.py -v` — passed, **14 tests** (10 existing + 4 new).
- `poetry run pytest tests/unit/ -v` — passed, **768 passed and 7 skipped** (matches 04-01's 764 + this plan's 4 new tests; no regressions in any existing prompt-dependent test).
- `poetry run python -c "from app.agent.prompts import SYSTEM_PROMPT; assert 'slot_index' in SYSTEM_PROMPT and 'rationale_misaligned' in SYSTEM_PROMPT and ('primary_type' in SYSTEM_PROMPT.lower())"` — exit 0.
- `poetry run python -c "from app.agent.prompts import SYSTEM_PROMPT; SYSTEM_PROMPT.format(current_datetime='x', max_steps=10)"` — exit 0 (existing f-string substitutions still work; no new substitutions introduced).
- `poetry run python -c "from app.agent.prompts import SYSTEM_PROMPT, REVISION_GUIDANCE; assert REVISION_GUIDANCE in SYSTEM_PROMPT"` — exit 0 (ADVISORY 7 prerequisite confirmed before locking in the assertion target).
- `poetry run python -c "from app.agent.state import RevisionReason; from app.agent.prompts import SYSTEM_PROMPT; import typing; reasons = typing.get_args(RevisionReason); covered = [r for r in reasons if r in SYSTEM_PROMPT]; assert 'rationale_misaligned' in covered, f'rationale_misaligned missing; covered: {covered}'"` — exit 0.
- Pre-commit hooks (`ruff legacy alias`, `ruff format`) — passed on both commits.
- Real import statements at the top of `app/agent/prompts.py` unchanged at 3 (`datetime`, `zoneinfo.ZoneInfo`, `app.agent.critique` constants). The plan's "no new imports" invariant is preserved; the apparent `grep -c "import\|from "` count rise from 12 to 13 is from the word "from" appearing in the new prose ("from the tool result" in step 6).

## Known Stubs

None. The prompt additions are concrete model-facing directives; no placeholder text was introduced. The pre-existing test names `test_system_prompt_only_substitutes_known_placeholders` and `test_system_prompt_requires_both_placeholders` describe the f-string substitution mechanism, not stub data.

## Issues Encountered

- `gsd-sdk` was not on PATH; the bundled compatibility CLI was not invoked. Acceptance-criterion shell commands and Python probes were run directly with `poetry run`.
- PLAN files for phase 04 live in the main repo's working tree at `/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge/.planning/phases/04-category-compliance-fix/` but were not committed before the worktree branch was created. Read them from the main-repo path; per `<parallel_execution>` STATE.md and ROADMAP.md are intentionally not modified here.
- A parallel wave-2 executor (04-03 graph injection) is presumably running against `app/agent/graph.py` and friends. This plan touches disjoint files (`app/agent/prompts.py`, `tests/unit/test_agent_prompts.py`), so no merge conflict is expected.

## User Setup Required

None — no external service configuration, secrets, or environment variables are introduced.

## Next Phase Readiness

Plan 04-03 (graph injection) and plan 04-05 (revision-hint dispatch) can now rely on the model emitting `slot_index` on retrieval tool_calls and reacting correctly to `rationale_misaligned` HumanMessages. The prompt is the model's contract surface; the graph (04-03) and revision dispatch (04-05) are the enforcement and feedback layers built on top of these prompt additions.

## Self-Check: PASSED

- Found summary file at `.planning/phases/04-category-compliance-fix/04-04-SUMMARY.md`.
- Found modified files: `app/agent/prompts.py`, `tests/unit/test_agent_prompts.py`.
- Found commits: `19d7e45`, `6aa1748`.

---
*Phase: 04-category-compliance-fix*
*Completed: 2026-05-22*
