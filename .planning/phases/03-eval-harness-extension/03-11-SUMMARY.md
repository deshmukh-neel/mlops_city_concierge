---
phase: 03-eval-harness-extension
plan: 11
subsystem: tests
tags: [gap-closure, wr-03, wr-04, dry, test-helper, scripted-llm, recording-scripted-llm]
requirements: [EVAL-06, EVAL-08]
dependencies:
  requires:
    - 03-09-scripted-llm-fresh-message  # CR-02 fresh-AIMessage contract the helper respects (loud-fail variant)
  provides:
    - tests._helpers.scripted_llm.ScriptedLLM        # Shared FIFO scripted chat model for graph + /chat tests
    - tests._helpers.scripted_llm.RecordingScriptedLLM  # Records every messages-list it saw; for threading + json-safety assertions
  affects:
    - tests/unit/test_eval_agent.py        # Consumes RecordingScriptedLLM; 5 dead outer-scope `seen` vars removed
    - tests/unit/test_chat_functional.py   # Consumes ScriptedLLM in /chat functional tests
tech-stack:
  added:
    - tests/_helpers/ package marker
  patterns:
    - shared-test-helper-module
    - pydantic-field-default-factory-for-instance-owned-defaults
    - loud-fail-on-test-double-exhaustion
key-files:
  created:
    - tests/_helpers/__init__.py
    - tests/_helpers/scripted_llm.py
    - tests/unit/test_helpers_scripted_llm.py
  modified:
    - tests/unit/test_eval_agent.py
    - tests/unit/test_chat_functional.py
decisions:
  - Helper raises IndexError on script exhaustion (loud-fail) — stricter than production ScriptedChatModel's fresh-AIMessage fallback. Rationale: tests should be exhaustive about their scripts; a mis-count should localize immediately rather than silently flow a marker AIMessage downstream.
  - `RecordingScriptedLLM.seen` uses `Field(default_factory=list)` (not `default=[]` and not a required field). Each instance gets its own clean list; callers do NOT pass `seen=outer_var` (the dead-var bug WR-04 closed).
  - Production `app/llm_factory.py:ScriptedChatModel` intentionally NOT consolidated into the helper. It has scenario-registry semantics (`SCRIPTED_SCENARIOS`, `scenario_id`) and a fresh-AIMessage fallback (CR-02) that don't belong in a test helper. Verifier explicitly flagged this carve-out in 03-VERIFICATION.md.
  - Out-of-scope `_ScriptedLLM` classes in `tests/unit/test_agent_graph.py` and `tests/unit/test_agent_self_correct.py` left untouched. The plan's `files_modified` enumeration scopes this hoist to the two verifier-flagged files (`test_eval_agent.py` + `test_chat_functional.py`). Broader cleanup is deferred (see Deferred Issues below).
metrics:
  duration_minutes: 11
  tasks_completed: 2
  files_created: 3
  files_modified: 2
  diffstat: "5 files changed, 186 insertions(+), 77 deletions(-)"
  unit_tests_passing: 725
  completed: "2026-05-22T16:21:31Z"
---

# Phase 03 Plan 11: Test Helper Hoist Summary

Hoisted the duplicated scripted-LLM-with-recording test classes from `test_eval_agent.py` and `test_chat_functional.py` into a single shared `tests/_helpers/scripted_llm.py` module exporting `ScriptedLLM` and `RecordingScriptedLLM`; removed 5 dead outer-scope `seen: list[list[BaseMessage]] = []` variables in `test_eval_agent.py` (Pydantic deep-copies during validation, so the outer ref was never the same list as `llm.seen`); production `app/llm_factory.py:ScriptedChatModel` intentionally NOT touched per verifier guidance.

## What Shipped

**New files:**

- `tests/_helpers/__init__.py` — package marker with single-line docstring.
- `tests/_helpers/scripted_llm.py` — `ScriptedLLM` (FIFO pop, raises `IndexError` on exhaustion) + `RecordingScriptedLLM(ScriptedLLM)` (snapshots every messages-list via `Field(default_factory=list)`).
- `tests/unit/test_helpers_scripted_llm.py` — 5 unit tests pinning the helper contract:
  - `test_scripted_llm_pops_in_order` (FIFO across consecutive calls)
  - `test_scripted_llm_raises_when_exhausted` (loud-fail contract)
  - `test_recording_scripted_llm_captures_seen_messages` (snapshot before delegating)
  - `test_recording_scripted_llm_default_seen_is_empty_list` (instance-owned default, not shared)
  - `test_bind_tools_returns_self` (no-op binding for both classes)

**Refactored files:**

- `tests/unit/test_chat_functional.py` — local `_ScriptedLLM` definition (lines 25-43) deleted; added `from tests._helpers.scripted_llm import ScriptedLLM`; replaced 4 construction sites; dropped orphaned imports (`BaseChatModel`, `BaseMessage`, `CallbackManagerForLLMRun`, `ChatGeneration`, `ChatResult`, `typing.Any`).
- `tests/unit/test_eval_agent.py` — local `_RecordingScriptedLLM` definition (formerly lines 537-562) deleted; added `from tests._helpers.scripted_llm import RecordingScriptedLLM`; replaced 5 construction sites; dropped orphaned imports (`BaseChatModel`, `BaseMessage`, `CallbackManagerForLLMRun`, `ChatGeneration`, `ChatResult`); removed 5 dead outer-scope `seen` vars (originally lines 584, 606, 648, 672, 708) and their `seen=seen` kwargs; updated two comments to reference the new class name without leading underscore.

## WR-03 + WR-04 Closure

| Gap   | Before                                                                                                                                                                             | After                                                                                                                                          |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| WR-03 | Two near-identical `BaseChatModel` subclasses (`_ScriptedLLM`, `_RecordingScriptedLLM`) across two test files — same `_llm_type`, `_generate` pop pattern, `bind_tools` no-op.     | Single canonical implementation in `tests/_helpers/scripted_llm.py` with a stricter exhaustion contract (raise vs. fallback).                  |
| WR-04 | 5 outer-scope `seen: list[list[BaseMessage]] = []` vars + `seen=seen` kwargs — misleading because Pydantic deep-copies during validation (the outer ref ≠ `llm.seen` instance ref). | Removed. Helper uses `Field(default_factory=list)`, so each instance owns a clean `seen` list with no kwarg needed.                            |

## Commits

- `e501c72` — `test(03-11): add failing tests for tests/_helpers/scripted_llm (WR-03)` (RED gate — module did not exist yet)
- `8f7c4bb` — `feat(03-11): create tests/_helpers/scripted_llm with ScriptedLLM + RecordingScriptedLLM (WR-03)` (GREEN gate — 5 helper tests pass)
- `35d3e2b` — `refactor(03-11): hoist scripted-LLM helpers + drop dead seen vars (WR-03 + WR-04)` (consumer files refactored; full unit suite still green at 725 passing)

## Verification

- `poetry run pytest tests/unit/ -q` → **725 passed** (no regressions; baseline was 49 tests in the two affected files, all still pass)
- `poetry run pytest tests/unit/test_helpers_scripted_llm.py -v` → **5 passed**
- `grep -n "_RecordingScriptedLLM" tests/unit/test_eval_agent.py` → **0 matches** (full removal in scope file)
- `grep -n "class _ScriptedLLM" tests/unit/test_chat_functional.py` → **0 matches** (full removal in scope file)
- `grep -nE "^\s*seen: list\[list\[BaseMessage\]\] = \[\]" tests/unit/test_eval_agent.py` → **0 matches** (all 5 dead vars removed)
- `grep -n "seen=seen" tests/unit/test_eval_agent.py` → **0 matches** (constructor kwargs cleaned up)
- `git diff --name-only $(git merge-base HEAD <base>) HEAD` → only `tests/_helpers/__init__.py`, `tests/_helpers/scripted_llm.py`, `tests/unit/test_chat_functional.py`, `tests/unit/test_eval_agent.py`, `tests/unit/test_helpers_scripted_llm.py` (no production code touched; `app/llm_factory.py` untouched per verifier guidance)

## Decisions Made

1. **Loud-fail exhaustion contract for the helper.** The plan asked the helper to either raise on empty list or mirror the production fresh-AIMessage fallback — we picked raise. Tests should be exhaustive about their scripts; a mis-count should localize immediately rather than silently flow a marker AIMessage to the agent graph. Production retains its fallback because the CI matrix runner needs to never deadlock; tests don't have that constraint.
2. **`Field(default_factory=list)` for `seen`.** Not `default=[]` (which aliases the list across instances) and not a required field (which would force every caller to pass `seen=[]`). This directly enables the WR-04 cleanup — no outer-scope var needed.
3. **`RecordingScriptedLLM` inherits from `ScriptedLLM`** rather than re-implementing `_generate`. Single source of truth for the pop logic; the recording subclass only adds the `seen` snapshot, then delegates via `super()._generate(...)`. If the parent's exhaustion contract changes, the subclass picks it up automatically.
4. **Production `ScriptedChatModel` deliberately NOT folded into the helper.** The verifier flagged this explicitly in 03-VERIFICATION.md WR-03 row: scenario-registry semantics (`SCRIPTED_SCENARIOS`, `scenario_id`) are not test-helper concerns, and the production fresh-AIMessage fallback differs from the helper's loud-fail by design.
5. **Out-of-scope `_ScriptedLLM` classes left alone.** Pre-existing local classes in `tests/unit/test_agent_graph.py` and `tests/unit/test_agent_self_correct.py` were NOT in the plan's `files_modified` enumeration nor in the verifier's WR-03 scope (which named only the three files: `test_eval_agent.py` + `test_chat_functional.py` + `app/llm_factory.py`). Documented in Deferred Issues below.

## Deviations from Plan

None for Rules 1-3 (no auto-fixed bugs, no auto-added critical functionality, no blocking issues encountered). The plan executed exactly as written within its `files_modified` scope.

One discovery noted under Deferred Issues (not a deviation — scope expansion was correctly avoided).

## Deferred Issues

**Out-of-scope `_ScriptedLLM` classes in other test files.** The plan's `<verification>` block specified `grep -rn "_RecordingScriptedLLM\|class _ScriptedLLM" tests/unit/` returns zero matches, but the plan's `files_modified` list scopes the hoist to `test_eval_agent.py` + `test_chat_functional.py` only. Two pre-existing local `_ScriptedLLM` definitions remain:

- `tests/unit/test_agent_graph.py:62` — `_ScriptedLLM(BaseChatModel)` with `RuntimeError("scripted responses exhausted")` on empty list.
- `tests/unit/test_agent_self_correct.py:30` — same shape, same contract.

These are functionally equivalent to `tests._helpers.scripted_llm.ScriptedLLM` except they raise `RuntimeError` instead of `IndexError`. A follow-up gap-closure plan could:

1. Replace both with the shared `ScriptedLLM` import (one-line per file).
2. Decide whether `IndexError` vs `RuntimeError` matters for existing tests (the tests in these files use `pytest.raises` rarely if at all on this path; a quick grep shows zero matches for both).

Not a P1 — DRY hygiene only. The Rule-3 fix-attempt-limit guard kept this out of the current task because it's not blocking, not a bug, and not in the plan's authoritative scope (`files_modified`).

## Production Code Untouched

Confirmed: `git diff --name-only $(merge-base) HEAD` returns only the 5 test-tree files listed above. `app/llm_factory.py:ScriptedChatModel` (the post-03-09 fresh-AIMessage form) was NOT modified by this plan, per the explicit verifier carve-out in 03-VERIFICATION.md WR-03.

## TDD Gate Compliance

The plan declared `tdd="true"` on both tasks. Gate sequence in `git log`:

1. **RED** — `e501c72` `test(03-11): add failing tests for tests/_helpers/scripted_llm (WR-03)` — module did not exist; collection failed with `ModuleNotFoundError: No module named 'tests._helpers'`.
2. **GREEN** — `8f7c4bb` `feat(03-11): create tests/_helpers/scripted_llm with ScriptedLLM + RecordingScriptedLLM (WR-03)` — all 5 helper tests pass.
3. **REFACTOR** — `35d3e2b` `refactor(03-11): hoist scripted-LLM helpers + drop dead seen vars (WR-03 + WR-04)` — consumer test files refactored to use the new helper; pre-existing 49 tests in the affected files still pass; full unit suite still green at 725 passing.

Task 2 (the refactor) reuses the canonical pre-existing test coverage in `test_eval_agent.py` + `test_chat_functional.py` as its behavior gate — no new tests were added for Task 2 because the refactor is behavior-preserving (no new functionality to test) and the existing tests in those files ARE the gate for "the refactor did not regress."

## Self-Check: PASSED

- FOUND: tests/_helpers/__init__.py
- FOUND: tests/_helpers/scripted_llm.py
- FOUND: tests/unit/test_helpers_scripted_llm.py
- FOUND: tests/unit/test_eval_agent.py (modified)
- FOUND: tests/unit/test_chat_functional.py (modified)
- FOUND commit: e501c72 (RED)
- FOUND commit: 8f7c4bb (GREEN)
- FOUND commit: 35d3e2b (REFACTOR)
- VERIFIED: full unit suite 725 passing
- VERIFIED: production app/llm_factory.py untouched
