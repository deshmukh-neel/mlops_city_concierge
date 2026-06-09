---
status: complete
phase: 09-per-provider-state-preservation-implementations
source:
  - 09-01-SUMMARY.md
  - 09-02-SUMMARY.md
  - 09-03-SUMMARY.md
  - 09-04-SUMMARY.md
  - 09-05-SUMMARY.md
started: 2026-06-08T20:58:29Z
updated: 2026-06-08T21:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Full unit suite passes
expected: |
  Run `make test-unit`. Suite reports 1060+ passed, 7 skipped, zero failures.
result: pass
observed: 1060 passed, 7 skipped, 9 warnings in 18.42s

### 2. Phase-8 registry invariant catches all 4 PROV adapters
expected: |
  `pytest test_adapters_registry_keys_match_supported_providers -v` passes;
  openai → OpenAIReasoningAdapter, deepseek → DeepSeekReasonerAdapter,
  anthropic → AnthropicAdapter, gemini → GeminiAdapter, others → NoOpAdapter.
result: pass
observed: "1 passed in 0.63s"

### 3. v2.0 anchor (gpt-4o-mini) stays on plain ChatOpenAI
expected: |
  build_chat_model('openai', 'gpt-4o-mini', 1.0) → ChatOpenAI
result: pass
observed: "ChatOpenAI"

### 4. gpt-5 family dispatches to OpenAIReasoningChatModel subclass
expected: |
  build_chat_model('openai', 'gpt-5-mini', 1.0) → OpenAIReasoningChatModel
  with use_responses_api=True
result: pass
observed: "OpenAIReasoningChatModel True"

### 5. AnthropicAdapter wires up with thinking + max_tokens fix
expected: |
  build_chat_model('anthropic', 'claude-sonnet-4-6', 1.0) → ChatAnthropic
  with max_tokens=8192 (> thinking.budget_tokens=4096)
result: pass
observed: "ChatAnthropic 8192 {'type': 'enabled', 'budget_tokens': 4096}"

### 6. DeepSeek-chat regression guard holds (thinking disabled)
expected: |
  `pytest test_deepseek_chat_keeps_thinking_disabled -v` passes.
result: pass
observed: "1 passed in 0.50s"

### 7. CR-01 fix: interactive eval_agent.py no longer KeyErrors on anthropic
expected: |
  resolve_chat_model('anthropic', None) raises ValueError (not KeyError).
result: pass
observed: "ValueError: No chat model for anthropic: pass --chat-model or set ANTHROPIC_MODEL"

### 8. Eval matrix structural check passes on the 6-cell shape
expected: |
  `make eval-matrix-refinement-structural-check` reports 6 cells, exit 0.
result: pass
observed: "structural-check: OK — matrix has 6 cell(s), env-override preserved through _apply_override, scorer registered, shared helper functional"

### 9. Eval baseline JSON includes all 5 reasoning-family cells
expected: |
  baseline JSON providers includes 5 cells (gemini deferred per Option B).
result: pass
observed: ['anthropic/claude-sonnet-4-6', 'deepseek/deepseek-chat', 'deepseek/deepseek-reasoner', 'openai/gpt-4o-mini', 'openai/gpt-5-mini']

### 10. Conformance harness covers all 4 PROV adapters end-to-end
expected: |
  `pytest -m reasoning_conformance -v` passes 9 tests.
result: pass
observed: "9 passed in 0.64s" — PROV-01, PROV-02, PROV-03, PROV-04 real-adapter siblings all green

### 11. Probe artifact contains no leaked secrets
expected: |
  grep for sk-* pattern in probe artifact + script returns no matches.
result: pass
observed: "exit=1" (no matches)

### 12. No new GitHub Actions workflow files added during Phase 9
expected: |
  git log d47c673..HEAD -- .github/workflows/ returns empty.
result: pass
observed: empty output (zero workflow commits in Phase 9 range)

### 13. Revertability claim: cumulative reverse-pop preserves v2.0 anchor
expected: |
  09-05-AUDIT.md confirms v2.0 anchor preserved through revert iterations.
result: pass
observed: "PASS-WITH-FINDINGS"; v2.0 anchor 32 tests pass; ADAPTERS["openai"] preserved; conformance harness intact

## Summary

total: 13
passed: 13
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none — all 13 tests passed cleanly]
