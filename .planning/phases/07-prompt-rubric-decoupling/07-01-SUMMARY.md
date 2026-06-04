---
phase: 07-prompt-rubric-decoupling
plan: 01
subsystem: prompts
tags: [prompt-engineering, refinement, llm, system-prompt, structured-plan]

# Dependency graph
requires:
  - phase: 06-minimal-edit-refinement
    provides: "build_refinement_prompt_message shared helper; REFINEMENT_STRUCTURED_PLAN_ENABLED feature flag; CRITIQUE_THRESHOLDS['refinement_minimal_edit']=1.0 binary merge gate; _INJECTION_SENTINEL='REFINEMENT TURN' truth-table contract"
provides:
  - "Task-only _REFINEMENT_PREAMBLE in app/agent/io.py (~362 chars, ~3x shorter than pre-rewrite); REFINEMENT TURN sentinel preserved; commit_itinerary output-channel callout preserved; slot/place_id/arrival_time JSON-block fields named as task description"
  - "SYSTEM_PROMPT rule 10 deleted entirely (rules 1-9 + REVISION_GUIDANCE concat unchanged); SYSTEM_PROMPT shrinks 11194 -> 9856 chars (delta 1338)"
  - "All D-07-04 canonical behavioral phrases absent from prompts.py + io.py combined (grep-gate ready)"
  - "Phase 6 CI hard gate `make eval-matrix-refinement-structural-check` still exits 0"
  - "TestChatRefinementInjection 8-cell truth-table regression suite still passes (9/9, including byte-identity assertion at line 1024)"
affects: [07-02-scratch-payload-extend, 07-04-scorer-category-extend, 07-05-scorer-tests-and-grep-gate, 07-06-chat-refinement-integration-test, 07-07-rebaseline-and-falsifier]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Task description vs behavioral prescription separation: prompt body describes WHAT (user editing one stop, JSON block carries prior plan, commit_itinerary is output channel), scorer enforces HOW (behavioral rules move to refinement_minimal_edit in plan 07-04)"
    - "Output-contract callout retained as task description (not behavioral rule per D-07-03): naming the tool the model must call is task shape, not behavior prescription"
    - "build_refinement_prompt_message kept as the single byte-identity site between /chat (prod) and evaluate_multi_turn_case (eval runner); HIGH-4 mitigation (JSON block stays {slot, place_id, arrival_time}) preserved per D-07-06"

key-files:
  created: []
  modified:
    - "app/agent/io.py (_REFINEMENT_PREAMBLE rewritten 18 lines → 6 lines; build_refinement_prompt_message body byte-identical; surrounding f-string `{_REFINEMENT_PREAMBLE}\\n\\n```json\\n{json_block}\\n```` byte-identical)"
    - "app/agent/prompts.py (rule 10 deleted; rules 6 + 8 + 9 + OUTPUT FORMAT header + REVISION_GUIDANCE concat untouched; no renumbering needed since rule 10 was last)"

key-decisions:
  - "Wording of the new _REFINEMENT_PREAMBLE (per CONTEXT.md Claude's Discretion): used the suggested wording from the plan's <action> block as the baseline, lowered emphasis tone ('one' instead of 'ONE', dropped 'REQUIRED behavior' framing), kept all five required content constraints (a)-(e). Final length 362 chars; well inside the 250-700 bound."
  - "Rule 10 deletion used a single Edit that collapsed the rule-10 block + the blank line that preceded OUTPUT FORMAT into just 'OUTPUT FORMAT (when finalizing):' — preserving exactly ONE blank line between rule 9's closer and OUTPUT FORMAT (PATTERNS.md delete-boundary constraint)."

patterns-established:
  - "Phase 7 'task-only prompt' template: ~3-6 sentence preamble naming (1) the user's intent, (2) the structured-data shape the prompt references, (3) the output channel — with zero 'on this turn you MUST' / 'REQUIRED behavior' framing. The downstream scorer is responsible for behavioral enforcement."
  - "Prompt-site DELETE pattern when the rule is last-numbered: collapse the rule + the blank line before the next section header into a single replacement that ends with the next section header. Renumbering is not required, but the blank-line gap rule (preserve exactly one blank line) is."

requirements-completed: [PROMPT-01, PROMPT-02]

# Metrics
duration: ~18min
completed: 2026-06-04
---

# Phase 07 Plan 01: Prompt Rewrite Summary

**Task-only `_REFINEMENT_PREAMBLE` (~362 chars, 3x shorter) replaces the 18-line behavioral version; SYSTEM_PROMPT rule 10 "STRUCTURED PLAN PRESERVATION" deleted in full; D-07-04 forbidden phrases absent from prompts.py + io.py; Phase 6 CI hard gate + 8-cell injection truth-table both still green.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-04T03:07:50Z (approximate; before `poetry install`)
- **Completed:** 2026-06-04T03:25:49Z
- **Tasks:** 3 (2 code-change tasks + 1 verification-only task)
- **Files modified:** 2 (`app/agent/io.py`, `app/agent/prompts.py`)

## Accomplishments

- **`_REFINEMENT_PREAMBLE` is now task description, not behavioral prescription** (per D-07-03): the model is told what the structured plan is and that `commit_itinerary` is the output channel — but is NOT told to "keep SAME number of stops", "preserve `place_id` byte-for-byte", "use the SAME `primary_type`", or "do not ask clarifying questions". Those rules move into `refinement_minimal_edit` in plan 07-04.
- **`SYSTEM_PROMPT` rule 10 deleted in full** (per D-07-02): the 18-line STRUCTURED PLAN PRESERVATION addendum is gone. Rules 1-9 + OUTPUT FORMAT + REVISION_GUIDANCE concat are byte-identical to pre-Phase-7.
- **D-07-04 forbidden-phrase grep gate is already satisfied** ahead of plan 07-05's test: the combined text of `prompts.py` + `io.py` contains zero occurrences (case-insensitive) of `keep same stop count`, `do not ask clarifying questions`, `preserve \`place_id\` byte-for-byte`, `byte-for-byte`, `same primary_type`. (The smoke verification at the end of Task 2 confirmed this.)
- **Two existing gates stay green** — defense-in-depth that the rewrite did not break structural contracts:
  - `make eval-matrix-refinement-structural-check` exits 0 (Task 3) — the structural assertion is on `"current_plan"` from the JSON block, not on the preamble text, so the rewrite is transparent to this gate per PATTERNS.md.
  - `pytest tests/unit/test_chat_functional.py::TestChatRefinementInjection -x` → 9/9 pass — the truth-table is keyed on the `REFINEMENT TURN` sentinel (preserved) and on byte-identity with `build_refinement_prompt_message` output (preserved since `build_refinement_prompt_message` itself was not modified).

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite `_REFINEMENT_PREAMBLE` in `app/agent/io.py` as task-only** — `f9ccdc7` (refactor)
2. **Task 2: Delete SYSTEM_PROMPT rule 10 in `app/agent/prompts.py`** — `ab6da7f` (refactor)
3. **Task 3: Verify CI structural-check still passes after rewrite** — no commit (verification-only task per the plan; `git status` was clean after Tasks 1+2)

## Files Created/Modified

- **`app/agent/io.py`** — `_REFINEMENT_PREAMBLE` rewritten from 18 lines / ~1100 chars to 6 lines / 362 chars. New text: `"REFINEMENT TURN — the user is editing one stop in the itinerary below. The fenced JSON block carries the prior committed plan: each entry has a 1-indexed \`slot\`, the \`place_id\` of that stop, and its planned \`arrival_time\`. The user's next message names what to change. Produce the updated itinerary by calling the \`commit_itinerary\` tool with the full stop list."`. Surrounding f-string at the (renumbered) line 134 is unchanged. `build_refinement_prompt_message` (now at lines 98-135) is byte-identical to pre-rewrite — the JSON payload still surfaces only `{slot, place_id, arrival_time}` per HIGH-4 strategy (a) / D-07-06.
- **`app/agent/prompts.py`** — Lines 175-193 (the 18-line rule 10 block + its trailing blank line) deleted. Rule 9's closer (line 173: `... semantic_search or nearby.`) is now followed by a single blank line, then `OUTPUT FORMAT (when finalizing):` on what is now line 175. Rules 1-9, the OUTPUT FORMAT section, and the `SYSTEM_PROMPT = ("""...""" + REVISION_GUIDANCE)` concatenation are byte-identical. `SYSTEM_PROMPT` length drops from 11194 to 9856 chars (delta 1338 chars; rule 10 + blank line was ~1338 chars including the leading newlines).

## Decisions Made

- **Preamble wording chosen from the plan's suggested wording** (with minor case-normalization of "ONE" → "one" since the preserved-sentinel test only requires the literal substring "REFINEMENT TURN", not "ONE"). The exact wording is planner discretion per CONTEXT.md; the constraints (a)-(e) from the `<action>` block are all satisfied: sentinel preserved, task described, JSON-block fields named, `commit_itinerary` output contract called out, all six D-07-04 forbidden phrases absent, length within bound.
- **Rule 10 deletion collapsed the rule-block + its trailing blank line into a single Edit** (replacing the whole block + the blank line before `OUTPUT FORMAT` with just `OUTPUT FORMAT (when finalizing):`). This preserves exactly ONE blank line between rule 9 and OUTPUT FORMAT — the PATTERNS.md delete-boundary constraint ("Preserve the blank line; do not collapse to zero blank lines"). Verified by reading the resulting file in the diff.

## Deviations from Plan

None — plan executed exactly as written. All three tasks passed their automated verify commands on the first run; no Rule 1-4 deviations triggered.

## Issues Encountered

- **Worktree environment bootstrap** — the agent worktree was created fresh, so `poetry run python -c ...` initially failed with `ModuleNotFoundError: No module named 'langchain_core'` on the Task 1 verify command. Resolved by running `poetry install --no-interaction` (~few minutes); this is one-time per worktree and not a code issue. After install, Task 1 verify passed on the next attempt.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Plan 07-02 (scratch-payload extend) can start immediately.** It reads the new field `primary_type` on `prior_committed_stops` scratch entries in `scripts/eval_agent.py` — Task 1's rewrite did NOT touch the eval-runner-side scratch wiring; that's plan 07-02's surface.

**Plan 07-04 (scorer extend) can start immediately.** It folds the same-`primary_type` check into `refinement_minimal_edit` Branch 5. The behavioral rules deleted from the prompt now have a home in the scorer.

**Plan 07-05 (grep-gate test) is pre-satisfied.** The D-07-04 forbidden-phrase grep gate is already green against the working tree per Task 2's smoke verification; plan 07-05 just needs to pin it as a unit test that runs in CI.

**Plan 07-06 (chat integration test) can start.** The `REFINEMENT TURN` sentinel + `current_plan` JSON block + `build_refinement_prompt_message` shape are all preserved, so the new PROMPT-01 2-turn test can be authored against the new (task-only) preamble without contradicting the existing 8-cell truth-table tests.

**Plan 07-07 (re-baseline + falsifier) blocks on 07-02/04** as previously planned — the scorer extension must land before re-running `eval_matrix_refinement.yaml` against the new prompt.

## Self-Check: PASSED

- File `app/agent/io.py` exists and contains the new `_REFINEMENT_PREAMBLE` (verified via `poetry run python -c "from app.agent.io import _REFINEMENT_PREAMBLE; ..."` at end of Task 1, OK len=362).
- File `app/agent/prompts.py` exists and `SYSTEM_PROMPT` no longer contains rule 10 (verified at end of Task 2, OK len=9856, delta=1338).
- Commits `f9ccdc7` and `ab6da7f` are present on the worktree branch (verified via `git log --oneline f9ccdc7^..HEAD`).
- `make eval-matrix-refinement-structural-check` exits 0 (verified at start of Task 3).
- `TestChatRefinementInjection` 9-cell suite passes (verified via `pytest tests/unit/test_chat_functional.py::TestChatRefinementInjection -x`).

---
*Phase: 07-prompt-rubric-decoupling*
*Completed: 2026-06-04*
