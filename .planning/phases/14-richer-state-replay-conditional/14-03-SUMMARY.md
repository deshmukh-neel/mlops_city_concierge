---
phase: 14
plan: "03"
subsystem: agent-adapters
tags: [replay, adapters, conformance, tests, REPLAY-01]
dependency_graph:
  requires: [14-01]
  provides: [REPLAY-01-conformance-tests]
  affects: [tests/unit/test_adapters.py]
tech_stack:
  added: []
  patterns: [additive-test-blocks, per-message-injection-assertion, flag-off-non-interference]
key_files:
  created: []
  modified:
    - tests/unit/test_adapters.py
decisions:
  - "ABC generic default is sufficient for all four adapters; no per-adapter override needed (D-14-04 confirmed)"
  - "Anthropic multi-replay test asserts content-list injection (not additional_kwargs) per message, honoring the PROV-03 asymmetry"
  - "Anthropic idempotency guard per message verified — no duplicate thinking blocks even in multi-replay path"
  - "Gemini multi-replay test covers the real lcgg 4.x path (__gemini_function_call_thought_signatures__), not just synthetic fixture"
  - "Flag-off non-interference documented per adapter as the 3rd test in each block"
metrics:
  duration: "~10min"
  completed: "2026-06-12"
  tasks_completed: 2
  files_modified: 1
---

# Phase 14 Plan 03: Adapter Conformance and Byte-Identity Summary

Additive multi-message-replay conformance tests (REPLAY-01, D-14-04) for all four provider adapters — per-message injection (flag-on), skip-on-no-state, and flag-off non-interference — while proving the existing 9-test conformance harness passes unchanged.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add multi-replay conformance tests for openai + deepseek adapters | 2f5a8ae | tests/unit/test_adapters.py |
| 2 | Add multi-replay conformance tests for anthropic + gemini, prove full harness unchanged | 2f5a8ae | tests/unit/test_adapters.py (committed together with Task 1 in the single commit) |

Note: Tasks 1 and 2 were committed together in `2f5a8ae` because the ruff pre-commit hook reformatted the file after Task 1 was staged, requiring a re-add. All tests for both tasks are present in that single commit.

## Verification

- `poetry run pytest tests/unit/test_adapters.py -q -k "multi_replay and (openai or deepseek)"`: 6 passed (Task 1 acceptance criteria met)
- `poetry run pytest tests/unit/test_adapters.py -q -k "multi_replay and (anthropic or gemini)"`: 6 passed (Task 2 acceptance criteria met)
- `poetry run pytest tests/unit/test_adapters.py -q`: 49 passed, 4 skipped (all existing + new tests green; 4 skips are fixture-dependent tests that need `make probe-providers`)
- `make test` (full suite): 1406 passed, 53 skipped, 9 deselected — no regressions
- Pre-existing test count confirmed unchanged: only additions in `git diff HEAD~1 HEAD -- tests/unit/test_adapters.py` (all `+` lines, zero `-` lines on existing test code)

## Test Coverage Added (12 new tests)

### OpenAI (3 tests)
- `test_openai_reasoning_adapter_multi_replay_injects_per_message_state`: msg1→r1, msg2→r2 distinct values
- `test_openai_reasoning_adapter_multi_replay_skips_messages_without_state`: no `_reasoning_state` → no inject
- `test_openai_reasoning_adapter_multi_replay_flag_off_path_unchanged`: single-message path still most-recent-only

### DeepSeek (3 tests)
- `test_deepseek_reasoner_adapter_multi_replay_injects_per_message_state`: msg1→ds_r1, msg2→ds_r2
- `test_deepseek_reasoner_adapter_multi_replay_skips_messages_without_state`: no state → untouched
- `test_deepseek_reasoner_adapter_multi_replay_flag_off_path_unchanged`: single-message path unchanged

### Anthropic (3 tests — content-list asymmetry honored)
- `test_anthropic_adapter_multi_replay_injects_per_message_thinking_blocks`: str-content promoted, list-content prepended — per message, not additional_kwargs
- `test_anthropic_adapter_multi_replay_idempotency_guard_per_message`: existing thinking blocks not duplicated (PROV-03 idempotency guard applies through the generic default per message)
- `test_anthropic_adapter_multi_replay_flag_off_path_unchanged`: single-message path targets most-recent only

### Gemini (3 tests — two wire-shape paths)
- `test_gemini_adapter_multi_replay_injects_per_message_real_lcgg_path`: `__gemini_function_call_thought_signatures__` key per message
- `test_gemini_adapter_multi_replay_skips_messages_without_state`: no state → neither lcgg key nor synthetic key injected
- `test_gemini_adapter_multi_replay_flag_off_path_unchanged`: single-message replay writes to most-recent only

## Deviations from Plan

None — plan executed exactly as written. The ABC generic default proved sufficient for all four adapters without any per-adapter override, confirming D-14-04's assertion.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. Tests use synthesized in-memory AIMessages only — no live providers, no DB, no credentials. T-14-06 accepted per plan threat register.

## Self-Check: PASSED

- [x] tests/unit/test_adapters.py modified (additions only — git diff confirms no pre-existing test was touched)
- [x] 12 new tests present (3 per adapter, 4 adapters)
- [x] `poetry run pytest tests/unit/test_adapters.py -q`: 49 passed, 4 skipped
- [x] `make test`: 1406 passed, 53 skipped, 9 deselected
- [x] Commit 2f5a8ae exists in git log
- [x] No app/ files modified (tests-only plan)
