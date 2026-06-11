---
phase: 07-prompt-rubric-decoupling
plan: 06
subsystem: tests
tags: [integration-test, chat-endpoint, refinement, prompt-decoupling, langgraph]

# Dependency graph
requires:
  - phase: 07-prompt-rubric-decoupling
    plan: 01
    provides: "Task-only _REFINEMENT_PREAMBLE in app/agent/io.py + SYSTEM_PROMPT rule 10 deleted. The new functional test relies on the post-rewrite preamble being the structured-plan body the /chat injection emits, while still observing the (preserved) byte-identity invariant via build_refinement_prompt_message."
provides:
  - "TestChatRefinementInjection.test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied: a single-POST /chat functional test that drives the real LangGraph agent + real FastAPI /chat endpoint with REFINEMENT_STRUCTURED_PLAN_ENABLED=true and asserts the response itinerary has same stop count + byte-equal non-target place_ids + changed target place_id (PROMPT-01 acceptance check per D-07-11)"
  - "Class constant TestChatRefinementInjection._NEW_SLOT2_PLACE_ID='ChIJtest_fixture_NEW2_xxxxxx' (28 chars, matches plan-06-01 Task-3 ^[A-Za-z0-9_-]{20,255}$ validator)"
affects: [07-07-rebaseline-and-falsifier]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-POST refinement-turn test pattern: pre-populate conversation_state.committed_stops to simulate 'turn 0 already happened in a prior conversation', then POST with the refinement message in turn-1 position. Exercises the same code path as a real turn-1 refinement (handler reads committed_stops → build_refinement_prompt_message injects → real LangGraph agent runs → commit_itinerary tool call → response). Matches the 8-cell truth-table pattern already established in TestChatRefinementInjection (lines 800-1026)."
    - "Inline test infrastructure (NOT _post_chat) when the test needs (a) place_ids grounded in scratch via a real prior tool result and (b) max_steps > 2. The class-level _post_chat helper hardcodes both _semantic_search → [] and max_steps=2, which is the right shape for the no-injection truth-table cells (where the LLM ends immediately with no tool calls) but cannot drive a full semantic_search + commit_itinerary trajectory."

key-files:
  created: []
  modified:
    - "tests/unit/test_chat_functional.py — added _NEW_SLOT2_PLACE_ID class constant (3 lines + 4 lines of comment) and one new test method test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied (~210 lines including docstring + comments). All edits are additive inside TestChatRefinementInjection; no existing tests were modified."

key-decisions:
  - "Test built inline (not via TestChatRefinementInjection._post_chat) because PROMPT-01 needs the agent's place_ids GROUNDED in scratch via a prior tool result (per app/agent/commit.py:_grounded_place_ids, lines 23-42, which rejects any place_id not seen via prior tool results) AND enough max_steps for a search + commit + critique trajectory. The _post_chat helper hardcodes _semantic_search → [] and max_steps=2, which is the right shape for the 8 no-injection truth-table cells but cannot drive a full commit_itinerary trajectory. Following the existing test_chat_runs_real_graph_with_tool_call pattern verbatim (lines 37-118) was the cleanest path; the plan's <action> block explicitly authorized this fallback under item (a) of the 'If during implementation' caveat."
  - "Stubbed _semantic_search to return a SINGLE list of 3 PlaceHits covering ALL FOUR place_ids the test references (3 prior committed + 1 new replacement). The agent's _grounded_place_ids set is union-of-all PlaceHits-across-all-scratch-entries (commit.py:23-42 walks every result), so a single semantic_search whose result list contains all four place_ids grounds the entire post-refinement commit list at once. This keeps the trajectory short (1 search + 1 commit) so max_steps=4 is plenty."
  - "Response field is `places` (frontend-contract name in ChatResponse.places, app/main.py:230), NOT `stops` as referenced in the plan's <action> block. The plan's prose says 'data[\"stops\"]' but the actual wire field is `places`. Test assertions use `places` per the existing class pattern (e.g., test_chat_runs_real_graph_with_tool_call line 115). They are the same list — Stop models serialized to PlaceCard dicts via state_to_cards (io.py:43-61)."

patterns-established:
  - "PROMPT-01 functional acceptance template: single-POST refinement turn with pre-populated committed_stops → scripted LLM emitting a search-then-commit trajectory → assert wire-shape invariants on response.places (same length, byte-equal non-target slots, changed target slot, no-op guard). Cleanly portable to future scorer-decoupling integration tests where the prompt no longer prescribes the behavior but the wire shape must still hold."
  - "Place_id fixture convention extended (per PATTERNS.md 'Place_id fixture convention'): _NEW_SLOT2_PLACE_ID='ChIJtest_fixture_NEW2_xxxxxx' joins _CANON_PLACE_ID, _CANON_PLACE_ID_2, and 'ChIJtest_fixture_id_cccccc' as the canonical 4-stop fixture set for refinement-edit-applied tests. All four match ^[A-Za-z0-9_-]{20,255}$ and are namespaced to avoid colliding with the 8-cell truth-table fixtures."

requirements-completed: [PROMPT-01]

# Metrics
duration: ~22min
completed: 2026-06-04
---

# Phase 07 Plan 06: Chat Refinement Integration Test Summary

**One new functional test `test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied` in `TestChatRefinementInjection` drives a single-POST /chat refinement turn through the real LangGraph agent + real FastAPI endpoint with `REFINEMENT_STRUCTURED_PLAN_ENABLED=true` and asserts the response has same stop count + byte-equal non-target `place_id`s + changed target `place_id` — the PROMPT-01 user-observable acceptance check per D-07-11.**

## Performance

- **Duration:** ~22 min (including poetry install on fresh worktree)
- **Started:** 2026-06-04 (immediately after worktree boot + base reset)
- **Completed:** 2026-06-04
- **Tasks:** 1 (single-task plan)
- **Files modified:** 1 (`tests/unit/test_chat_functional.py`)

## Accomplishments

- **PROMPT-01 user-observable acceptance check is now pinned by a deterministic functional test** — Phase 7's prompt rewrite (plan 07-01) deleted the SYSTEM_PROMPT rule that previously told the model to "preserve `place_id` byte-for-byte". This new test verifies that the structured-plan injection + scorer-side enforcement keep the user-observable refinement behavior correct (same stop count, byte-equal non-target slots, changed target slot) even though the prompt no longer prescribes how.
- **Single-POST refinement-turn pattern adopted** per the plan's `<action>` block: pre-populating `conversation_state.committed_stops` is the moral equivalent of "turn 0 already happened in a prior conversation" and exercises the same code path as a true 2-turn flow (handler reads `committed_stops` → `build_refinement_prompt_message` injects → real LangGraph agent runs → `commit_itinerary` tool call → response). This matches the existing class pattern (every test in `TestChatRefinementInjection` is a single POST).
- **All four place_ids grounded via a single `semantic_search` stub** — the agent's `_grounded_place_ids` set (commit.py:23-42) is union-of-all PlaceHits across all scratch entries, so a single search whose result list contains all four place_ids grounds the entire post-refinement commit list at once. Trajectory is therefore short: 1 search + 1 commit + critique + END (finalize-on-commit per `project_finalize_on_commit_fix`). `max_steps=4` is plenty.
- **All 10 `TestChatRefinementInjection` tests pass** (9 existing 8-cell truth-table + Residual-2 + 1 new PROMPT-01 test). All 19 tests in `tests/unit/test_chat_functional.py` pass. No existing tests were modified.
- **Ruff lint + format both pass** post-edit.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add PROMPT-01 2-turn /chat scripted refinement test to TestChatRefinementInjection** — `96603b1` (test)

## Files Created/Modified

- **`tests/unit/test_chat_functional.py`** (+219 lines):
  - Added class constant `_NEW_SLOT2_PLACE_ID = "ChIJtest_fixture_NEW2_xxxxxx"` (28 chars, matches `^[A-Za-z0-9_-]{20,255}$`) with a 4-line PATTERNS.md-reference comment.
  - Added one new test method `test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied` at the end of `TestChatRefinementInjection` (~210 lines including a multi-paragraph block comment that documents the single-POST approach and the inline-vs-_post_chat tradeoff).
  - Test asserts: HTTP 200; `len(places) == 3`; `places[0]["place_id"] == _CANON_PLACE_ID` (slot 1 byte-equal); `places[2]["place_id"] == "ChIJtest_fixture_id_cccccc"` (slot 3 byte-equal); `places[1]["place_id"] == _NEW_SLOT2_PLACE_ID` (slot 2 changed to fixture); `places[1]["place_id"] != _CANON_PLACE_ID_2` (no-op sanity guard).
  - No new imports — `AIMessage`, `ScriptedLLM`, `PlaceHit`, `TestClient`, `build_agent_graph`, `app`, `_stub_loaded_config` were all already in scope.

## Decisions Made

- **Built inline, not via `_post_chat`** — PROMPT-01 needs (a) the agent's `place_id`s GROUNDED in scratch via a prior tool result (commit.py:23-42 rejects any place_id not seen via a prior tool result) and (b) enough `max_steps` for a `semantic_search` + `commit_itinerary` trajectory. `_post_chat` hardcodes `_semantic_search → []` and `max_steps=2`, which is correct for the 8 no-injection truth-table cells (where the LLM ends immediately with no tool calls) but cannot drive `commit_itinerary` through to a non-empty `places` response. The plan's `<action>` block under "If during implementation" explicitly authorized this fallback (item a: follow `test_chat_runs_real_graph_with_tool_call` pattern verbatim).
- **Stubbed `_semantic_search` to return all four place_ids in a single result list** — `_grounded_place_ids` walks every scratch entry's result list and unions all `place_id`s, so one search grounds the entire post-refinement commit list at once. Three PlaceHits (canon-slot-1, NEW-slot-2, canon-slot-3) with realistic Google-Places metadata.
- **Used `places` not `stops` for the response assertion** — the wire field name is `places` (frontend contract, `ChatResponse.places` at `app/main.py:230`). The plan's prose said `data["stops"]` but the actual JSON response uses `places`. They are the same list (Stop pydantic models serialized to PlaceCard dicts via `state_to_cards`).
- **Test name kept verbatim from the plan** — `test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied`. Long but searchable; matches the plan's `<action>` block exactly.

## Deviations from Plan

None — plan executed exactly as written. The plan's `<action>` block anticipated the `_post_chat` infrastructure mismatch and explicitly authorized the inline-fallback path under "If during implementation it becomes clear that the existing `_post_chat` infrastructure cannot drive `commit_itinerary` ... follow `tests/unit/test_chat_functional.py:37-120` `test_chat_runs_real_graph_with_tool_call` pattern verbatim." That is what the test does. The wire-field-name observation (`places` vs the plan's prose `data["stops"]`) is a notation-vs-reality clarification, not a behavioral deviation — the assertion semantics are identical.

## Issues Encountered

- **Worktree environment bootstrap** — fresh worktree, so `poetry run pytest ...` initially failed with `ModuleNotFoundError: No module named 'langchain_core'`. Resolved by running `poetry install --no-interaction` in the worktree (~2 minutes, one-time). After install, the new test passed on the first run.
- **Wire field name mismatch** — the plan's `<action>` block uses `data["stops"]` for the response assertion, but the actual response field is `places` (per `app/main.py:230` `ChatResponse.places: list[dict]`). Resolved by matching the existing class convention (e.g., `test_chat_runs_real_graph_with_tool_call` line 115 uses `body["places"]`). No behavioral impact; the field carries the same data.

## User Setup Required

None — no external service configuration required. Test is hermetic: scripted LLM, stubbed `_semantic_search`, stubbed `itinerary_violations`, `TestClient` for the FastAPI app.

## Next Phase Readiness

- **Plan 07-04 (scorer extend) + Plan 07-05 (scorer tests + grep gate)** — independent of 07-06; can proceed in parallel waves per the original phase plan. Neither depends on this test.
- **Plan 07-07 (re-baseline + falsifier)** — does not depend on this test for execution; the test runs scripted LLMs (no real model calls) so it adds zero baseline cost. However, the test's existence is a defense-in-depth signal that PROMPT-01's user-observable contract holds post-Phase-7 prompt rewrite, which is the protective invariant 07-07's re-baseline assumes.

## Self-Check: PASSED

- File `tests/unit/test_chat_functional.py` contains a method `test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied` (verified via `grep -n` of the file post-format).
- Class constant `_NEW_SLOT2_PLACE_ID = "ChIJtest_fixture_NEW2_xxxxxx"` is defined inside `TestChatRefinementInjection` (verified during write; 28 chars, matches `^[A-Za-z0-9_-]{20,255}$` via `re.fullmatch`).
- Commit `96603b1` is present on the worktree branch (verified via `git log --oneline -3`).
- `poetry run pytest tests/unit/test_chat_functional.py::TestChatRefinementInjection::test_prompt_01_chat_refinement_returns_full_itinerary_with_edit_applied -v -x` exits 0 (1 passed, 2 warnings, ~2s).
- `poetry run pytest tests/unit/test_chat_functional.py::TestChatRefinementInjection -v -x` exits 0 (10/10 passed including the existing 8-cell truth-table + Residual-2 cases).
- `poetry run pytest tests/unit/test_chat_functional.py -v -x` exits 0 (19/19 passed, no broader regressions in the file).
- `poetry run ruff check tests/unit/test_chat_functional.py` exits 0 ("All checks passed!").
- `poetry run ruff format --check tests/unit/test_chat_functional.py` exits 0 (file is formatter-clean post-`ruff format`).

---
*Phase: 07-prompt-rubric-decoupling*
*Completed: 2026-06-04*
