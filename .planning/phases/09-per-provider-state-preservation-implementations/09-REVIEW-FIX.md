---
phase: 09-per-provider-state-preservation-implementations
fixed_at: 2026-06-05
fix_scope: critical_warning
findings_in_scope: 6
fixed: 6
skipped: 0
iteration: 1
status: all_fixed
source_review: .planning/phases/09-per-provider-state-preservation-implementations/09-REVIEW.md
---

# Phase 09 Code Review Fix Report

**Branch:** `gsd/phase-09-per-provider-state-preservation-implementations`
**Source review:** `09-REVIEW.md` (commit `837cb7a`)
**Fix scope:** `critical_warning` (1 BLOCKER + 5 WARNINGs; INFOs excluded per default scope)
**Iteration:** 1 (single pass; `--auto` not specified)

## Summary

- Findings in scope: **6** (CR-01 BLOCKER + WR-01..WR-05 WARNINGs)
- Fixed: **6**
- Bundled bonus fix: **IN-01** (tightly coupled to CR-01 — same `LlmProvider`/`SUPPORTED_PROVIDERS` drift defect)
- Skipped: **0**
- Tests touched: 124 passing (33 + 11 + 39 + 41)
- Status: **all_fixed**

## Fix Commits (in order)

| Commit | Subject | Findings |
|--------|---------|----------|
| `c6ead45` | `fix(09): CR-01 + IN-01 — resolve_chat_model handles all SUPPORTED_PROVIDERS` | CR-01, IN-01 |
| `23f561f` | `fix(09): WR-01 + WR-05 — _lift_reasoning_blocks per-block copy + unit tests` | WR-01, WR-05 |
| `45d80fe` | `fix(09): WR-02 — remove dead Path 3 (asymmetric tool_calls capture-vs-replay)` | WR-02 |
| `a9ea411` | `refactor(09): WR-03 — drive registry invariant test off single-source mapping` | WR-03 |
| `f348679` | `fix(09): WR-04 — add observability log to AnthropicAdapter signature-mismatch skip` | WR-04 |

## Fixed Issues

### CR-01 + IN-01: `resolve_chat_model` KeyError on `--llm-provider anthropic` + stale `LlmProvider` Literal

**Severity:** BLOCKER + INFO
**Files:** `scripts/eval_agent.py`
**Commit:** `c6ead45`

Replaced hardcoded `env_var = {"deepseek": ..., "kimi": ...}[provider]` (which raised `KeyError` on `anthropic`/`openai`/`gemini`/`scripted`) with `env_var_map.get(provider)` covering all `SUPPORTED_PROVIDERS`, followed by a fall-through to a user-friendly `ValueError("No chat model for {provider}: pass --chat-model or set {env_var}")`. Updated `LlmProvider = Literal[...]` to add `"anthropic"`. Bundled IN-01 (stale Literal) into the same commit because it is the same drift defect.

Verified independently: `resolve_chat_model("anthropic", None)` now raises the structured `ValueError`, not `KeyError`. Matrix runner still works because cells pass `--chat-model` explicitly (the bug only surfaces in the interactive `python scripts/eval_agent.py --llm-provider anthropic` path).

### WR-01 + WR-05: `OpenAIReasoningChatModel._lift_reasoning_blocks` per-block copy + unit-test coverage

**Severity:** WARNING + WARNING
**Files:** `app/llm_factory.py`, `tests/unit/test_llm_factory.py`
**Commit:** `23f561f`

- **WR-05 code fix:** changed `list(reasoning_blocks)` to `[dict(block) for block in ...]`. The outer-list-only copy was leaking inner-dict aliases between `msg.content` and `msg.additional_kwargs["reasoning_content"]`, violating the docstring's shallow-copy promise. Mutating one would silently mutate the other.
- **WR-01 tests:** added 7 unit tests covering list-content lift, str-content no-op, inner-dict isolation regression guard (closes WR-05), no-reasoning-block no-op, non-AIMessage generation guard, and **sync+async parity** (the asymmetric-coverage failure mode that bit Anthropic in Wave 3). Future `langchain-openai` version bumps now surface lift breakage at unit-test time.

The two findings were bundled because the new isolation test IS the regression guard for the code change.

### WR-02: GeminiAdapter Path 3 (per-tool-call surfacing) removed as dead code

**Severity:** WARNING
**Files:** `app/agent/adapters/gemini.py`, `tests/unit/test_adapters.py`
**Commit:** `45d80fe`

The Wave 4 live probe confirmed lcgg 4.x surfaces per-call signatures EXCLUSIVELY at `additional_kwargs["__gemini_function_call_thought_signatures__"]` (Path 1). Path 3 (scanning `message.tool_calls` for `thought_signature` bytes) was dead code AND had an asymmetric round-trip: capture read from `tool_calls[i]` but replay wrote to `additional_kwargs`, silently dropping the signature on the wire.

Chose "remove dead code" over "track source in payload and write to tool_calls on replay" because (a) the live probe confirms it never fires under the pinned `langchain-google-genai>=4.0.0,<5.0.0`, and (b) dead code in a security-adjacent context (byte-identical signature roundtrip) is a regression hazard. Repurposed the existing Path 3 capture test as a regression guard asserting per-tool_call signatures are ignored. Module + class docstrings updated to document the removal date and rationale.

### WR-03: Registry-invariant test refactored to single-source mapping

**Severity:** WARNING
**Files:** `tests/unit/agent/test_adapters.py`
**Commit:** `a9ea411`

Replaced the hardcoded `("openai", "deepseek", "anthropic", "gemini")` skip-tuple with a `_EXPECTED_REGISTRY_MAPPING: dict[str, type]` containing the expected adapter class for every provider. The test now asserts:
1. `set(ADAPTERS.keys()) == set(SUPPORTED_PROVIDERS)`
2. `_EXPECTED_REGISTRY_MAPPING.keys() == SUPPORTED_PROVIDERS` (with diff message naming missing/extra keys)
3. Each provider's `ADAPTERS[name]` matches its expected class

Catches two failure modes the previous shape missed: (a) new reasoning provider added to `SUPPORTED_PROVIDERS` without updating the skip-list (would silently assert NoOp on a real-adapter wiring), and (b) typo'd `ADAPTERS` entries matching `SUPPORTED_PROVIDERS` via set-equality but silently bypassing per-provider class checks.

### WR-04: AnthropicAdapter signature-mismatch observability

**Severity:** WARNING
**Files:** `app/agent/adapters/anthropic.py`, `tests/unit/test_adapters.py`
**Commit:** `f348679`

Behavior unchanged (target wins when its content already has thinking blocks — required for the Wave 3 live-probe `400` idempotency fix). Added `logging.getLogger(__name__)` and a `_log.debug(...)` call that fires ONLY when existing and captured signature sets differ (`sorted(existing_sigs) != sorted(captured_sigs)`). The log names both signature sets so future bug-hunts have telemetry to spot silent payload-discard caused by revision steps or graph reducer mis-order.

Added 2 unit tests: (1) positive — log fires and names both sets when signatures differ; (2) negative — log does NOT fire when signatures match (guards against noisy prod logs every agent turn).

## Skipped Issues

None. All BLOCKER + WARNING findings fixed.

The 4 INFOs (IN-02 punctuation in JSON, IN-03 hardcoded probe paths, IN-04 narrow secret regex) were out of `critical_warning` scope. IN-01 was the exception — fixed alongside CR-01 because both originate from the same drift defect.

## Test Results

```
tests/unit/test_adapters.py — 33 passed
tests/unit/agent/test_adapters.py — 11 passed
tests/unit/test_llm_factory.py — 39 passed
tests/unit/test_eval_matrix.py — 41 passed
TOTAL: 124 passed in 1.53s (zero failures, zero skipped)
```

Independent verification ran by orchestrator: all 5 fixes are real and substantive, code grep-confirms claims, full focused-suite runs green.

## Pre-commit hooks

All 5 commits passed pre-commit (`ruff` + `ruff format`) without `--no-verify`. No manual `ruff format` invocations (per project convention `feedback_precommit_ruff`).

---

_Fixed: 2026-06-05_
_Fixer: gsd-code-fixer (single iteration; no `--auto`)_
_Verified: orchestrator (grep + pytest 124/124 pass)_
