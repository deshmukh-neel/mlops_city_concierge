---
phase: 14-richer-state-replay-conditional
plan: 03
type: execute
wave: 2
depends_on: ["14-01"]
files_modified:
  - tests/unit/test_adapters.py
autonomous: true
requirements: [REPLAY-01]
must_haves:
  truths:
    - "Each of the four adapters (openai, deepseek, anthropic, gemini) has additive multi-replay conformance tests covering flag-on (per-message injection) AND flag-off (single-message path untouched)"
    - "The Anthropic multi-replay test verifies per-message content-list injection (not additional_kwargs), reflecting the asymmetric adapter shape"
    - "The Gemini multi-replay test verifies the correct provider key is injected per message"
    - "The existing 9-test conformance harness passes unchanged (no existing test modified)"
  artifacts:
    - path: "tests/unit/test_adapters.py"
      provides: "additive replay_reasoning_state_multi conformance tests per adapter"
      contains: "replay_reasoning_state_multi"
  key_links:
    - from: "tests/unit/test_adapters.py"
      to: "ProviderAdapter.replay_reasoning_state_multi"
      via: "direct adapter instantiation + synthesized AIMessages"
      pattern: "replay_reasoning_state_multi"
---

<objective>
Add additive multi-message-replay conformance tests for all four provider adapters (REPLAY-01, D-14-04), covering both the flag-on per-message injection path and the flag-off single-message path, while proving the existing 9-test conformance harness passes unchanged. This gives the live A/B runs a green test wall before any API spend and guarantees per-adapter revertability.

Purpose: D-14-04 requires multi-replay coverage on all four adapters even though only three models run, because the generic ABC default must be proven correct per wire format (Anthropic's content-list asymmetry and Gemini's signature-map shape are the risk cells). The flag-off tests document the non-interference contract that keeps the Phase-13 plateau byte-identical.
Output: New per-adapter multi-replay test blocks in tests/unit/test_adapters.py with the existing harness untouched.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/14-richer-state-replay-conditional/14-CONTEXT.md
@.planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add multi-replay conformance tests for openai + deepseek adapters</name>
  <files>tests/unit/test_adapters.py</files>
  <read_first>
    - tests/unit/test_adapters.py (existing per-adapter test structure, naming convention, imports, the existing single-message replay tests for openai + deepseek)
    - app/agent/adapters/__init__.py (the replay_reasoning_state_multi generic ABC default from Plan 14-01 — the method under test)
    - app/agent/adapters/openai_gpt5.py, deepseek.py (the single-message replay implementations the default delegates to — both write additional_kwargs["reasoning_content"])
    - .planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md (section "tests/unit/test_adapters.py — additive multi-path tests": the exact 3-test-per-adapter pattern)
  </read_first>
  <action>
    Add a multi-replay test section (REPLAY-01) for OpenAIReasoningAdapter and DeepSeekReasonerAdapter following the existing test-naming convention (`test_<provider>_reasoning_adapter_multi_replay_...`). For each of the two adapters add three tests: (1) injects per-message state — build a list `[HumanMessage, AIMessage(state r1), AIMessage(state r2)]` where each AIMessage carries its own `additional_kwargs["_reasoning_state"]`, call `adapter.replay_reasoning_state_multi(outbound)`, assert each AIMessage received its OWN `reasoning_content` (msg1 -> r1, msg2 -> r2); (2) skips messages without state — an AIMessage with no `_reasoning_state` is left untouched (its `reasoning_content` stays None); (3) flag-off path unchanged — calling the existing single-message `replay_reasoning_state(outbound, state)` still writes only onto the most-recent AIMessage (documents the non-interference contract). All tests instantiate the adapter directly with synthesized AIMessages — no graph, no LLM, no DB. Do NOT modify any existing test.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_adapters.py -q -k "multi_replay and (openai or deepseek)"</automated>
  </verify>
  <acceptance_criteria>
    - `poetry run pytest tests/unit/test_adapters.py -q -k "multi_replay and (openai or deepseek)"` collects and passes >= 6 tests (3 per adapter)
    - The per-message-injection test asserts msg1 receives r1 AND msg2 receives r2 (distinct values per message — behavior assertion)
    - The skip test asserts a no-state AIMessage's reasoning_content stays None
    - No pre-existing test in test_adapters.py is modified (git diff shows only additions)
  </acceptance_criteria>
  <done>OpenAI and DeepSeek adapters have passing additive multi-replay tests covering per-message injection, skip-on-no-state, and the untouched flag-off single-message path.</done>
</task>

<task type="auto">
  <name>Task 2: Add multi-replay conformance tests for anthropic + gemini, then prove the full 9-test harness is unchanged</name>
  <files>tests/unit/test_adapters.py</files>
  <read_first>
    - tests/unit/test_adapters.py (existing anthropic + gemini single-message replay tests and their fixture payload shapes)
    - app/agent/adapters/anthropic.py (replay writes onto message.content as a block list, with the idempotency guard; the generic default calls replay_reasoning_state(outbound[:i+1], state) so message i is the reverse-walk target)
    - app/agent/adapters/gemini.py (two capture paths: bytes thought_signature and the function_call_thought_signatures dict map; confirm which additional_kwargs key the single-message replay writes)
    - .planning/phases/14-richer-state-replay-conditional/14-PATTERNS.md (Anthropic asymmetry callout + Gemini callout + the "existing 9-test harness MUST pass unchanged" rule)
  </read_first>
  <action>
    Add the same three-test multi-replay block (REPLAY-01, D-14-04) for AnthropicAdapter and GeminiAdapter, adapting the assertions to each adapter's wire format. Anthropic: build AIMessages whose `additional_kwargs["_reasoning_state"]` carries a `{"provider":"anthropic","thinking_blocks":[...]}` payload and whose `.content` is a str (the pruned shape); assert `replay_reasoning_state_multi` injects the captured thinking blocks onto EACH targeted message's `.content` (block-list promotion per message), respecting the existing idempotency guard (a message already carrying thinking blocks is not duplicated). Gemini: build AIMessages with per-message `_reasoning_state` in the canonical gemini payload shape; assert each message receives the correct gemini key (`thought_signature` / `__gemini_function_call_thought_signatures__`) per the single-message implementation. Add the flag-off non-interference test for each. Then add an explicit assertion (test or a documented run) that the existing 9-test conformance harness is unchanged: run the full test_adapters.py file and confirm the pre-Phase-14 tests still pass with zero modifications. Do NOT modify any existing test.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_adapters.py -q</automated>
  </verify>
  <acceptance_criteria>
    - `poetry run pytest tests/unit/test_adapters.py -q -k "multi_replay and (anthropic or gemini)"` passes >= 6 tests (3 per adapter)
    - The Anthropic multi-replay test asserts content-list injection per message (NOT additional_kwargs) and respects the idempotency guard (no duplicate thinking blocks)
    - The Gemini multi-replay test asserts the correct provider-specific additional_kwargs key is set per message
    - The full `poetry run pytest tests/unit/test_adapters.py -q` passes; the count of pre-existing tests is unchanged and none were edited (git diff shows additions only)
  </acceptance_criteria>
  <done>Anthropic and Gemini adapters have passing multi-replay tests honoring their wire-format asymmetry, the idempotency guard holds, and the full conformance harness (existing 9 tests + new additive tests) is green with no existing test changed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| synthesized AIMessages → adapter under test | Test-only in-memory objects; no external input, no network |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-14-06 | Tampering | adapter test fixtures | accept | Tests instantiate adapters directly with synthesized messages; no live providers, no credentials, no DB — zero attack surface |
| T-14-07 | Repudiation | conformance regression masking | mitigate | The flag-off non-interference tests + the "existing 9-test harness unchanged" assertion ensure a future change to the multi path cannot silently alter the single path (per-adapter revertability, Phase-9 precedent) |
| T-14-SC | Tampering | npm/pip/cargo installs | mitigate | No new package installs (existing pytest + langchain_core only); slopcheck N/A |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_adapters.py -q` passes (existing 9 tests + new additive multi-replay tests)
- `make test` passes (full suite — mandatory when touching the agent test surface per CLAUDE.md; guards against DB-pool contamination from real-graph tests)
- `make lint` passes (ruff) on the new test code
- Per-adapter revertability: the multi path is exercised only via the ABC default; no adapter file was modified by this plan (tests-only change)
</verification>

<success_criteria>
- All four adapters have additive multi-replay conformance tests (flag-on per-message injection + flag-off non-interference)
- Anthropic's content-list asymmetry and Gemini's signature-key shape are explicitly covered
- The existing 9-test conformance harness passes unchanged
</success_criteria>

<output>
Create `.planning/phases/14-richer-state-replay-conditional/14-03-SUMMARY.md` when done
</output>
