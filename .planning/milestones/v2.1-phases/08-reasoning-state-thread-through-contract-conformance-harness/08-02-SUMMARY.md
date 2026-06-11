---
phase: 08-reasoning-state-thread-through-contract-conformance-harness
plan: 02
subsystem: agent/graph
tags: [reasoning-state, prune, additional_kwargs, REASON-04, D-08-07]
dependency_graph:
  requires: []
  provides:
    - "additional_kwargs preservation across _prune_for_llm pre-cutoff stub"
  affects:
    - "Plan 03 capture/replay round-trip across _RECENT_TOOL_EXCHANGES_KEPT cutoff"
tech_stack:
  added: []
  patterns:
    - "LangChain additional_kwargs as cross-turn extension surface"
key_files:
  created: []
  modified:
    - app/agent/graph.py
    - tests/unit/test_agent_graph.py
decisions:
  - "D-08-07 implemented: one-kwarg patch to the pre-cutoff stub constructor"
metrics:
  duration: ~10 minutes
  tasks_completed: 1
  files_modified: 2
  completed_date: 2026-06-04
requirements: [REASON-04]
---

# Phase 8 Plan 2: _prune_for_llm additional_kwargs preservation — Summary

One-line: Patched `_prune_for_llm`'s pre-cutoff AIMessage stub constructor to forward `additional_kwargs` (D-08-07), unblocking Plan 03's adapter capture/replay across the `_RECENT_TOOL_EXCHANGES_KEPT=2` cutoff window without changing any observable behavior on the gpt-4o-mini path.

## What Shipped

**Single-line patch (`app/agent/graph.py` lines 219-229):**

The content-only `AIMessage` constructor inside the `if isinstance(m, AIMessage) and m.tool_calls:` branch of `_prune_for_llm` now passes `additional_kwargs=m.additional_kwargs` so the kwarg dict survives the stub replacement. An inline `# D-08-07:` rationale comment documents the link to the canonical decision. No other line of `_prune_for_llm` changed — same content coercion, same tool-call stripping, same `ToolMessage` drop, same post-cutoff pass-through.

**New unit test (`tests/unit/test_agent_graph.py:566-598`):**

`test_prune_for_llm_preserves_additional_kwargs_on_stub` — builds a message list with 3 tool-issuing AIMessages so the oldest gets stubbed; the oldest AIMessage carries `additional_kwargs={"reasoning_content": "carried-over"}`. The test asserts the stub at the same index has matching `additional_kwargs`, empty `tool_calls`, and unchanged `content`. Structurally identical to `test_prune_for_llm_drops_oldest_tool_exchanges` with the kwargs added — same helper-list style, same import surface, no new imports needed.

## Tasks Completed

| Task | Name                                                                                         | Commit  | Files                                              |
| ---- | -------------------------------------------------------------------------------------------- | ------- | -------------------------------------------------- |
| 1.RED   | test(08-02): add failing test for additional_kwargs preservation on _prune stub          | 8a2273f | tests/unit/test_agent_graph.py                     |
| 1.GREEN | feat(08-02): preserve additional_kwargs on _prune_for_llm pre-cutoff stub                | 0b923f0 | app/agent/graph.py                                 |

TDD gate sequence: RED (test fails — `additional_kwargs == {}`) → GREEN (test passes — kwarg dict carried through). REFACTOR not needed: the patch is one extra kwarg on a constructor call; nothing to clean up.

## Verification

**Plan-mandated automated check:**

```
poetry run pytest tests/unit/test_agent_graph.py::test_prune_for_llm_preserves_additional_kwargs_on_stub \
                  tests/unit/test_agent_graph.py::test_prune_for_llm_keeps_short_history_intact \
                  tests/unit/test_agent_graph.py::test_prune_for_llm_drops_oldest_tool_exchanges -v
```

Result (post-GREEN):

```
tests/unit/test_agent_graph.py::test_prune_for_llm_preserves_additional_kwargs_on_stub PASSED
tests/unit/test_agent_graph.py::test_prune_for_llm_keeps_short_history_intact         PASSED
tests/unit/test_agent_graph.py::test_prune_for_llm_drops_oldest_tool_exchanges        PASSED
============================== 3 passed in 0.55s ==============================
```

**Acceptance criteria — every line of the plan's `<acceptance_criteria>` block:**

| Criterion                                                                                                            | Result   |
| -------------------------------------------------------------------------------------------------------------------- | -------- |
| `grep -c "additional_kwargs=m.additional_kwargs" app/agent/graph.py` returns 1                                       | 1 PASS   |
| `grep -c "D-08-07" app/agent/graph.py` returns 1                                                                     | 1 PASS   |
| `grep -c "def test_prune_for_llm_preserves_additional_kwargs_on_stub" tests/unit/test_agent_graph.py` returns 1      | 1 PASS   |
| All three pytest names pass                                                                                          | PASS     |
| `pytest tests/unit/test_agent_graph.py -v -k "prune_for_llm"` exits 0                                                | PASS (3 selected, 3 passed) |
| `ruff check app/agent/graph.py tests/unit/test_agent_graph.py` exits 0                                               | PASS     |
| Exactly one `AIMessage(` constructor inside the `if isinstance(m, AIMessage) and m.tool_calls:` branch               | 1 PASS   |

**Wider regression check:** full `tests/unit/test_agent_graph.py` (44 tests) passes post-patch.

**REASON-04 spirit (byte-identity on gpt-4o-mini path):** The two pre-existing prune tests are byte-identical to their pre-patch shape and still pass. For any input where every `AIMessage.additional_kwargs == {}` (the gpt-4o-mini case in practice), the new code is observationally indistinguishable from the old code — the stub still has `additional_kwargs={}`. The formal byte-identity regression test (D-08-15) is owned by Plan 04/05 per the Phase 8 plan-split.

## Deviations from Plan

None — plan executed exactly as written. The patch wording, the test fixture shape (3 tool-issuing AIMessages with the kwarg on the oldest), the test name, the inline comment, and the placement (immediately after `test_prune_for_llm_drops_oldest_tool_exchanges`, before `_state_with_grounded`) all match the `<action>` block verbatim.

## Decisions Made

- **D-08-07 (planner decision)** implemented as written: one extra kwarg on one constructor call inside `_prune_for_llm`. No structural change to the function.
- **TDD ordering:** RED commit first (`8a2273f` — failing test only), then GREEN commit (`0b923f0` — one-line patch). Each commit is independently revertable.
- **REFACTOR skipped:** the patch is structurally trivial; nothing to extract or rename.

## What's NOT in This Plan

- Plan 02 does NOT wire the `ProviderAdapter` contract into `plan()` — that is Plan 03. The kwargs-preservation guarantee shipped here is the *precondition* for Plan 03's capture/replay to round-trip across the cutoff window; without it, Plan 03's `replay_reasoning_state` could never see state stashed > 2 turns ago because the pruner would have already dropped the carrier.
- Plan 02 does NOT add the `app/agent/adapters/` subpackage or the `NoOpAdapter` — that is Plan 01.
- Plan 02 does NOT touch `_RECENT_TOOL_EXCHANGES_KEPT` (line 122 of `app/agent/graph.py`) — explicitly out per CONTEXT §canonical_refs.
- Plan 02 does NOT add the formal byte-identity regression test (D-08-15) — that is Plan 04/05.

## Self-Check: PASSED

- File `app/agent/graph.py` modified — verified by `grep -c "additional_kwargs=m.additional_kwargs"` returning 1.
- File `tests/unit/test_agent_graph.py` modified — verified by `grep -c "def test_prune_for_llm_preserves_additional_kwargs_on_stub"` returning 1.
- Commit `8a2273f` (RED) exists — verified by `git log --oneline --all | grep 8a2273f`.
- Commit `0b923f0` (GREEN) exists — verified by `git log --oneline --all | grep 0b923f0`.
- Three plan-mandated pytest names all PASS.

## TDD Gate Compliance

This plan is `type=auto` with `tdd="true"` on its single task. Two commits in the expected order:

- `test(08-02): ...` (8a2273f) — RED gate (failing test alone)
- `feat(08-02): ...` (0b923f0) — GREEN gate (one-line patch makes it pass)

No `refactor(...)` commit because no structural change was warranted. Plan-checker / verifier should accept this as a complete TDD cycle.
