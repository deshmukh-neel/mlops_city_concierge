---
phase: 03-eval-harness-extension
plan: 09
subsystem: testing
tags: [gap-closure, cr-02, in-05, scripted-chat-model, fresh-aimessage, langgraph-add-messages, llm-factory, regression-test]

# Dependency graph
requires:
  - phase: 03-eval-harness-extension
    provides: ScriptedChatModel (Plan 03-05) — the no-network deterministic BaseChatModel CI uses for the eval matrix; this plan fixes a latent identity-dedup bug in its fallback path.
provides:
  - Fresh-AIMessage-per-call contract for ScriptedChatModel._generate (CR-02 closure)
  - Self-documenting [SCRIPTED CI MODE] fallback content citing scripts/eval_matrix.py (IN-05 closure)
  - Three regression tests pinning the contract (returns_fresh_aimessage_each_call, fallback_content_documents_ci_mode, consumes_scripted_list_when_nonempty)
affects:
  - Phases 04-06 (any future scripted-mode multi-turn scenario that exercises revision/replan loops — without this fix the graph spun until max_steps because add_messages dedupes by identity)
  - CI eval matrix summary.json diffs (reviewers can now distinguish deterministic placeholder output from real LLM failures at a glance)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fresh-construction-per-call for any BaseChatModel emitting messages a LangGraph reducer will see — avoid module-level singleton AIMessages"
    - "Self-documenting CI placeholder strings (marker + originating script reference) so PR diffs are unambiguous"

key-files:
  created: []
  modified:
    - app/llm_factory.py
    - tests/unit/test_llm_factory.py

key-decisions:
  - "Construct AIMessage inline inside _generate (not via a private factory) to keep the fix isomorphic to the original two-line code block — minimal diff for reviewers"
  - "Keep the exact placeholder string verbatim from REVIEW.md IN-05: '[SCRIPTED CI MODE] Deterministic no-network finalize; see scripts/eval_matrix.py.' so the marker is greppable across baselines"
  - "Three tests, not one — the load-bearing CR-02 test (is-not check), the IN-05 content test, and the unchanged-pop-semantics test together pin the full contract and prevent future regressions in any direction"

patterns-established:
  - "Pattern: LangGraph add_messages identity-dedup is a hidden gotcha — any BaseChatModel surface used by an agent MUST emit fresh AIMessage instances per call"
  - "Pattern: CI-placeholder content cites both a marker token (`[SCRIPTED CI MODE]`) AND the originating script path so operators can grep summary.json + jump straight to the source"

requirements-completed: [EVAL-09]

# Metrics
duration: 6min
completed: 2026-05-22
---

# Phase 03 Plan 09: Scripted LLM Fresh Message Summary

**ScriptedChatModel._generate now constructs a fresh AIMessage on every empty-script call with the self-documenting `[SCRIPTED CI MODE]` marker — closes CR-02 (LangGraph add_messages identity-dedup) and IN-05 (ambiguous CI summary.json content) in a single 24-line code edit.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-22T08:11:25Z
- **Completed:** 2026-05-22T08:17:19Z
- **Tasks:** 1 (TDD: RED + GREEN, no REFACTOR needed)
- **Files modified:** 2 (app/llm_factory.py, tests/unit/test_llm_factory.py)

## Accomplishments

- **CR-02 (BLOCKER) closed.** Removed the module-level `_DEFAULT_SCRIPTED_FALLBACK = AIMessage(...)` singleton from `app/llm_factory.py`. `_generate` now constructs a fresh `AIMessage` inline on each empty-script call. Empirical regression check `a is not b` passes against the live module.
- **IN-05 closed.** Fallback content is now `"[SCRIPTED CI MODE] Deterministic no-network finalize; see scripts/eval_matrix.py."` — PR reviewers reading CI `summary.json` files immediately recognize it as deterministic placeholder output instead of a real model failure.
- **Three regression tests added** (all pass): `test_scripted_chat_model_returns_fresh_aimessage_each_call`, `test_scripted_chat_model_fallback_content_documents_ci_mode`, `test_scripted_chat_model_consumes_scripted_list_when_nonempty`.
- **No regressions** in the existing 705-test unit suite (708 total now pass).

## Task Commits

Each gate was committed atomically (TDD):

1. **RED — failing regression tests for fresh-AIMessage + [SCRIPTED CI MODE]** — `157f6b7` (test)
2. **GREEN — scripted-mode finalize returns fresh AIMessage + self-documenting content** — `8c02af9` (feat)

REFACTOR step skipped — the implementation is already minimal (24-line diff, no duplication, no over-engineering).

## Files Created/Modified

- `app/llm_factory.py` — Removed `_DEFAULT_SCRIPTED_FALLBACK` module-level constant. Rewrote `ScriptedChatModel._generate` to construct a fresh `AIMessage` per empty-script call with `[SCRIPTED CI MODE]` content. Updated `ScriptedChatModel` docstring with the CR-02 fix note. Updated the section comment block above the class to document the `[SCRIPTED CI MODE]` + IN-05 rationale.
- `tests/unit/test_llm_factory.py` — Added three regression tests in a new section comment-flagged `# ─── Plan 03-09 Task 1: CR-02 (fresh AIMessage) + IN-05 (self-documenting) ──`. Tests are pure-Python, no DB, no network, no `@pytest.mark.asyncio`.

## Decisions Made

- **Inline `AIMessage(...)` in `_generate`, not a private factory function.** Keeps the diff minimal and isomorphic to the original two-line code block — reviewers see exactly the singleton-vs-fresh swap.
- **Exact placeholder string verbatim from REVIEW.md IN-05:** `"[SCRIPTED CI MODE] Deterministic no-network finalize; see scripts/eval_matrix.py."` — this string is the contract; future changes to it would break IN-05 greppability across baselines.
- **Three tests, not one.** The CR-02 `is not` check is load-bearing; the IN-05 content check pins the placeholder; the unchanged-pop-semantics test guards against accidental refactors of the pop branch.

## Deviations from Plan

None — plan executed exactly as written. All grep-based acceptance criteria pass, all three new tests pass, the empirical `a is not b` regression check passes, the full 705-test baseline unit suite remains green (now 708 with the additions), and `ruff check` + `ruff format --check` both pass.

## Issues Encountered

None during planned work.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- CR-02 latent bug closed before Phase 04-06 begin populating `SCRIPTED_SCENARIOS` with multi-turn revision-loop trajectories that would have exercised the bug.
- IN-05 self-documenting fallback content immediately observable in any future CI matrix run — operators reading PR diffs cite `[SCRIPTED CI MODE]` and recognize it as deterministic.
- 03-08 deliverables unaffected (different file: `scripts/eval_matrix.py`).

## Self-Check: PASSED

- File existence — `app/llm_factory.py`, `tests/unit/test_llm_factory.py`, `.planning/phases/03-eval-harness-extension/03-09-SUMMARY.md` all present.
- Commit existence — RED `157f6b7` (test), GREEN `8c02af9` (feat) both visible in `git log --oneline --all`.
- Empirical regression — `python -c "from app.llm_factory import ScriptedChatModel; m = ScriptedChatModel(); a = m._generate(messages=[]).generations[0].message; b = m._generate(messages=[]).generations[0].message; assert a is not b; assert '[SCRIPTED CI MODE]' in a.content"` exits 0.
- Unit suite — 708 passed, 0 failed, 9 warnings (all pydantic-deprecation from mlflow, pre-existing).
- Lint/format — `ruff check` + `ruff format --check` both green.

---
*Phase: 03-eval-harness-extension*
*Completed: 2026-05-22*
