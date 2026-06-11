---
phase: 10-eval-harness-honesty
plan: "06"
subsystem: eval-harness
tags: [eval, test, llm-factory, async, documentation]
dependency_graph:
  requires: []
  provides: [EVAL-06]
  affects: [tests/unit/test_llm_factory.py, app/agent/critique/vibe.py]
tech_stack:
  added: []
  patterns:
    - monkeypatch + get_settings.cache_clear() for factory dispatch tests
    - async test with asyncio_mode=auto for BaseChatModel executor fallback
    - behavior-preserving doc-comment for verified thread-executor safety
key_files:
  created: []
  modified:
    - tests/unit/test_llm_factory.py
    - app/agent/critique/vibe.py
decisions:
  - "D-10-15: gpt-5-mini routes through OpenAIReasoningChatModel(use_responses_api=True); gpt-4o-mini stays on plain ChatOpenAI — both paths now test-locked"
  - "D-10-16: ScriptedChatModel ainvoke works via BaseChatModel executor fallback (no _agenerate override needed); proven by async test"
  - "D-10-17: vibe_check sync judge_llm.invoke does not block the asyncio event loop — LangGraph 1.2.0 runs sync nodes in a ThreadPoolExecutor thread under ainvoke; doc-comment suffices, no code change"
metrics:
  duration: "5m"
  completed: "2026-06-11"
  tasks: 2
  files: 2
---

# Phase 10 Plan 06: Sync/Async Test Debt (EVAL-06) Summary

Close sync/async test debt: test-lock gpt-5 reasoning dispatch, gpt-4o-mini anchor regression guard, ScriptedChatModel ainvoke executor fallback, and document vibe_check as safe under LangGraph's sync-node thread executor.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Factory-level tests: gpt-5 dispatch + gpt-4o-mini anchor guard + ScriptedChatModel ainvoke | 7d58d1f | tests/unit/test_llm_factory.py |
| 2 | Document vibe_check sync-node-in-executor finding (behavior-preserving) | 9c8ca04 | app/agent/critique/vibe.py |

## What Was Built

### Task 1 — Three new tests in `tests/unit/test_llm_factory.py`

**`test_build_chat_model_gpt5_returns_openai_reasoning_chat_model`** (D-10-15):
Patches `app.llm_factory.OpenAIReasoningChatModel`, calls `build_chat_model("openai", "gpt-5-mini", temperature=1.0)`, asserts the sentinel is returned, and that `kwargs["model"] == "gpt-5-mini"` and `kwargs["use_responses_api"] is True`. Previously zero tests in the suite referenced `use_responses_api` — a silent regression risk.

**`test_build_chat_model_gpt4o_mini_stays_plain_chat_openai`** (D-10-15 regression guard):
Patches both `OpenAIReasoningChatModel` and `ChatOpenAI`, calls `build_chat_model("openai", "gpt-4o-mini", temperature=1.0)`, asserts `out == "plain-llm"` and `reasoning_cls.assert_not_called()`. The v2.0 production anchor is now CI-enforced.

**`test_scripted_chat_model_ainvoke_works`** (D-10-16):
Async test (`asyncio_mode=auto`) constructing `ScriptedChatModel(scripted=[AIMessage(content="hello")])` and awaiting `ainvoke([HumanMessage(content="go")])`. Asserts `result.content == "hello"`. Proves the `BaseChatModel` async executor fallback works for a model that defines only `_generate` — the path the LangGraph agent graph actually uses.

### Task 2 — Doc-comment in `app/agent/critique/vibe.py` line ~78

Added a multi-line comment immediately above the `judge_llm.invoke([HumanMessage(content=prompt)])` call recording the D-10-17 finding: the enclosing `critique` node is a sync LangGraph node; verified against LangGraph 1.2.0 (pinned in this repo) that sync nodes execute in a ThreadPoolExecutor thread under `graph.ainvoke` — the event loop is not blocked. Therefore no async refactor is required. The call stays sync by design.

**Behavior change:** None. `git diff` shows only added comment lines; the `judge_llm.invoke` call is byte-identical.

## Verification

- `poetry run pytest tests/unit/test_llm_factory.py -q` — 42 passed
- `poetry run pytest tests/unit/ -q -k "vibe or critique"` — 102 passed
- `git diff app/agent/critique/vibe.py` — comment-only additions confirmed (already committed clean)
- `poetry run ruff check tests/unit/test_llm_factory.py app/agent/critique/vibe.py` — All checks passed

## Deviations from Plan

None - plan executed exactly as written.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Task 1 is test-only (no production code changed). Task 2 is a comment-only addition to `vibe.py` — behavior-preserving by construction.

## Self-Check: PASSED

- tests/unit/test_llm_factory.py: FOUND (42 tests pass)
- app/agent/critique/vibe.py: FOUND (executor comment at line 78; judge_llm.invoke unchanged)
- Commits: 7d58d1f (test task 1) and 9c8ca04 (docs task 2) both present in git log
