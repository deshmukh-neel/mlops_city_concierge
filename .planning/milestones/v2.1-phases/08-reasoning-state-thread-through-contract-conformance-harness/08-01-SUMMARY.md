---
phase: 08-reasoning-state-thread-through-contract-conformance-harness
plan: 01
subsystem: agent
tags: [reasoning-state, provider-adapter, contract, abc, registry, phase-8, v2.1, tdd]
dependency_graph:
  requires:
    - app/llm_factory.py:SUPPORTED_PROVIDERS  # registry keys mirror this tuple
  provides:
    - app/agent/adapters/__init__.py:ProviderAdapter  # ABC for Phase 9 sub-phases
    - app/agent/adapters/__init__.py:StatePayload     # opaque dict alias (D-08-03)
    - app/agent/adapters/__init__.py:NoOpAdapter      # default registered for all providers
    - app/agent/adapters/__init__.py:MockReasoningAdapter  # test-only, NOT in ADAPTERS
    - app/agent/adapters/__init__.py:ADAPTERS         # registry plan 03 will look up
  affects:
    - app/agent/graph.py  # plan 03 closes adapter over build_agent_graph + plan()
tech_stack:
  added: []  # no new deps — abc + typing + langchain_core (already in)
  patterns:
    - "Register-then-dispatch idiom mirrored from app/llm_factory.py:59-186 (D-08-04)"
    - "Opaque payload pattern mirrored from ConversationState round-trip via /chat"
    - "Two-pure-method ABC (capture + replay) — testable in isolation per shape"
    - "Dict-comprehension registry from SUPPORTED_PROVIDERS — no parallel-edit drift"
key_files:
  created:
    - app/agent/adapters/__init__.py
    - tests/unit/agent/test_adapters.py
  modified: []
decisions:
  - "D-08-01: ProviderAdapter lives in app/agent/adapters/ subpackage; __init__.py exports the public surface — Phase 9 adds one file per provider in the same subpackage"
  - "D-08-02: ABC has EXACTLY two abstract methods — capture_reasoning_state + replay_reasoning_state (not one combined hook, not three)"
  - "D-08-03: StatePayload = dict[str, Any] — opaque, no Union, no discriminator; a fifth shape is registry addition only"
  - "D-08-04: Registry-keyed dispatch at graph-build time (mirrors llm_factory.py); per-message provider sniffing rejected"
  - "D-08-08: openai registered as NoOpAdapter in Phase 8 — gpt-5-mini matrix behavior unchanged from Phase 7 (the falsifier signal)"
  - "D-08-09: MockReasoningAdapter exported for tests but NOT in ADAPTERS — test-only segregation enforced by unit test"
metrics:
  duration_minutes: 12
  completed: "2026-06-04T21:25:00Z"
---

# Phase 8 Plan 1: Adapters Subpackage Summary

**One-liner:** Ships the `app/agent/adapters/` typed contract surface (`ProviderAdapter` ABC + `StatePayload` opaque dict + `NoOpAdapter` default + `MockReasoningAdapter` test-only + `ADAPTERS` dict-comprehension registry over `SUPPORTED_PROVIDERS`) — zero behavior change, RED→GREEN TDD with 11/11 unit tests passing.

## Outputs

**Created file:** `app/agent/adapters/__init__.py`

**Five exported public symbols** (`__all__`):

| Symbol | Kind | Purpose |
|---|---|---|
| `ProviderAdapter` | ABC | Two abstract methods: `capture_reasoning_state(message) -> StatePayload \| None` and `replay_reasoning_state(outbound, state) -> list[BaseMessage]` (D-08-02) |
| `StatePayload` | type alias | `dict[str, Any]` — opaque per-provider payload (D-08-03) |
| `NoOpAdapter` | concrete | `capture → None`; `replay → outbound` unchanged (the Phase 7 byte-identity guarantee) |
| `MockReasoningAdapter` | concrete | Test-only: `capture` returns stored payload; `replay` walks `outbound` in reverse and tags the most-recent `AIMessage` with `additional_kwargs["_reasoning_state"] = state` (D-08-06, D-08-09) |
| `ADAPTERS` | `dict[str, ProviderAdapter]` | `{p: NoOpAdapter() for p in SUPPORTED_PROVIDERS}` — dict comprehension keeps the registry in sync with `app/llm_factory.py:62` automatically; D-08-08 swap point for Phase 9 |

## Confirmations

- **ADAPTERS is dict-comprehension-driven from SUPPORTED_PROVIDERS** — verified by `grep`:
  ```
  ADAPTERS: dict[str, ProviderAdapter] = {p: NoOpAdapter() for p in SUPPORTED_PROVIDERS}
  ```
  Hard-coding the five keys (`openai`, `gemini`, `deepseek`, `kimi`, `scripted`) was rejected per the plan note — the comprehension means Phase 9 adding a sixth provider in `llm_factory.py` automatically extends the registry without a parallel edit here.
- **MockReasoningAdapter exported but NOT registered** — `__all__` includes it; `ADAPTERS` does not. Asserted by `test_mock_reasoning_adapter_not_registered_in_prod_registry`.

## TDD Gate Compliance

- **RED commit** `871c32d` — 11 unit tests in `tests/unit/agent/test_adapters.py` fail at collection with `ModuleNotFoundError: No module named 'app.agent.adapters'` (verified before commit).
- **GREEN commit** `00d214a` — `app/agent/adapters/__init__.py` created; 11/11 tests pass; 58/58 unit tests across `tests/unit/agent/` + `tests/unit/test_agent_graph.py` pass (no regression).
- **REFACTOR** — Not needed; implementation is minimal (130 LOC, single source file).

Gate sequence in git log: `test(08-01): RED ...` → `feat(08-01): GREEN ...` ✓

## Verification

Inline verify (from `<verify><automated>` block in the plan):
```
poetry run python -c "from app.agent.adapters import ProviderAdapter, StatePayload, NoOpAdapter, MockReasoningAdapter, ADAPTERS; ..."
→ VERIFY OK
poetry run ruff check app/agent/adapters/__init__.py
→ All checks passed!
```

All 9 acceptance criteria from `<acceptance_criteria>` met:
1. File `app/agent/adapters/__init__.py` exists ✓
2. `grep -c "class ProviderAdapter(ABC):"` → 1 ✓
3. `grep -c "@abstractmethod"` → 2 (exactly two abstract methods) ✓
4. `grep -c "class NoOpAdapter(ProviderAdapter):"` → 1 ✓
5. `grep -c "class MockReasoningAdapter(ProviderAdapter):"` → 1 ✓
6. `grep -E "ADAPTERS\s*[:=]"` matches the dict-comprehension line ✓
7. `grep -c "StatePayload\s*=\s*dict\[str, Any\]"` → 1 (docstring rewritten to avoid double match while preserving informative content) ✓
8. Inline verify command exits 0 (asserts contract surface) ✓
9. Mock-not-in-prod-registry assertion exits 0 ✓
10. `ruff check` exits 0 ✓

## Test Coverage

`tests/unit/agent/test_adapters.py` (NEW, 151 LOC, 11 tests):

| Test | Decision Covered |
|---|---|
| `test_provider_adapter_defines_two_abstract_methods` | D-08-02 |
| `test_state_payload_is_dict_of_str_to_any` | D-08-03 |
| `test_noop_adapter_capture_returns_none` | D-08-08 (NoOp shape) |
| `test_noop_adapter_replay_returns_outbound_unchanged` | D-08-08 (identity preservation) |
| `test_noop_adapter_replay_with_none_state_returns_outbound_unchanged` | D-08-08 (None-safe) |
| `test_adapters_registry_keys_match_supported_providers` | D-08-08 (registry == SUPPORTED_PROVIDERS) |
| `test_mock_reasoning_adapter_captures_stored_payload` | D-08-09 (capture shape) |
| `test_mock_reasoning_adapter_replay_tags_most_recent_ai_message` | D-08-09 + D-08-06 (kwarg tagging) |
| `test_mock_reasoning_adapter_replay_with_none_state_is_noop` | D-08-09 (None-safe) |
| `test_mock_reasoning_adapter_replay_tags_last_ai_message_when_multiple` | D-08-09 (reverse-walk) |
| `test_mock_reasoning_adapter_not_registered_in_prod_registry` | D-08-09 (test-only segregation) |

The plan's behavior block listed 6 tests; this expansion to 11 strengthens REASON-02 acceptance by adding multi-AIMessage reverse-walk + None-safe variants for both adapter classes (extra Rule-2 hardening — failure modes Phase 9 sub-phases will trip if they regress the contract).

## Deviations from Plan

**None for the contract; one minor docstring change to satisfy grep acceptance criterion exactly:**

The first GREEN draft included the line `StatePayload = dict[str, Any]` verbatim in the module docstring, which caused `grep -c` to return 2 instead of the criterion-required 1. Rewrote the docstring line to describe the alias prose-style ("`StatePayload` opaque dict alias — no discriminator, no Union (D-08-03)") so only the actual assignment matches. Pure cosmetic; no behavior change.

**No Rule 1/2/3 deviations triggered.** No auth gates. No architectural questions.

## Self-Check: PASSED

Verified after writing this Summary:

- `app/agent/adapters/__init__.py`: FOUND
- `tests/unit/agent/test_adapters.py`: FOUND
- Commit `871c32d` (RED): FOUND in `git log`
- Commit `00d214a` (GREEN): FOUND in `git log`
- `make test` equivalent (`pytest tests/unit/agent/ tests/unit/test_agent_graph.py`): 58 passed, 0 failed
- Ruff: clean
- Phase 8 directory created in `.planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/`

## Next Plan

Plan 08-02 (`08-02-prune-kwargs-preservation-PLAN.md`) extends `_prune_for_llm` (`app/agent/graph.py:222-224`) with the single-kwarg patch `additional_kwargs=m.additional_kwargs` on the pre-cutoff stub constructor (D-08-07). Plan 08-03 (`08-03-plan-capture-replay-wiring-PLAN.md`) then imports `ADAPTERS` + `NoOpAdapter` from this subpackage and closes the resolved adapter over `build_agent_graph` + `plan()`.
