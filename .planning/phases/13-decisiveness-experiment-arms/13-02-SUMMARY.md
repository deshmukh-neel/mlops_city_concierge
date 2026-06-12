---
phase: 13-decisiveness-experiment-arms
plan: "02"
subsystem: agent-prompts
tags: [decisiveness, prompt-engineering, tdd, flag-gated]
dependency_graph:
  requires: []
  provides: [rule8_viability_addendum, DEC-01-prompt-builder]
  affects: [app/agent/prompts.py, tests/unit/test_agent_prompts.py]
tech_stack:
  added: []
  patterns: [flag-gated additive prompt extension, LOW_SIMILARITY_THRESHOLD import convention]
key_files:
  created: []
  modified:
    - app/agent/prompts.py
    - tests/unit/test_agent_prompts.py
decisions:
  - "rule8_viability_addendum returns '' for flag-off so flag-off is byte-identical to baseline"
  - "Threshold parameter defaults to imported LOW_SIMILARITY_THRESHOLD, never hardcoded 0.55"
  - "Addendum concatenated after SYSTEM_PROMPT.format(...) by caller (plan 13-04) — prompts.py stays pure"
  - "TDD RED/GREEN pattern: failing import test committed first, then implementation"
metrics:
  duration: "~2 minutes"
  completed: "2026-06-12"
  tasks: 2
  files_changed: 2
requirements: [DEC-01]
---

# Phase 13 Plan 02: Viability Contract Prompt Summary

**One-liner:** Flag-gated rule8_viability_addendum function adds cosine-threshold sentence to rule 8 only when VIABILITY_CONTRACT_ENABLED=1, keeping flag-off byte-identical to baseline.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Failing tests for rule8_viability_addendum | c5fe419 | tests/unit/test_agent_prompts.py |
| 1 (GREEN) | Add flag-gated viability addendum builder | 3d9bb1c | app/agent/prompts.py |
| 2 | Both-flag-state lock tests (included in RED/GREEN) | c5fe419, 3d9bb1c | tests/unit/test_agent_prompts.py |

## What Was Built

### `app/agent/prompts.py`

Added `rule8_viability_addendum(enabled: bool, threshold: float | None = None) -> str`:

- **Flag-off** (`enabled=False`): returns `""` — byte-identical baseline, cosine text absent
- **Flag-on** (`enabled=True`): returns `"   A result with cosine similarity >= {threshold} and matching primary_type IS viable — do not keep searching past it.\n"`
- Imports `LOW_SIMILARITY_THRESHOLD` from `app.agent.revision` — no hardcoded 0.55
- Contains neither "byte-for-byte" nor "SAME primary_type" (Phase-7 forbidden-phrase gate)
- Does not modify SYSTEM_PROMPT — appended by caller (plan 13-04)

### `tests/unit/test_agent_prompts.py`

Added 8 new tests covering:
1. `test_rule8_viability_addendum_flag_off_returns_empty` — flag-off returns `""`
2. `test_rule8_viability_addendum_flag_on_contains_cosine` — "cosine" present
3. `test_rule8_viability_addendum_flag_on_contains_is_viable` — "is viable" present
4. `test_rule8_viability_addendum_flag_on_contains_threshold_value` — threshold value in output
5. `test_rule8_viability_addendum_no_forbidden_phrases` — no "byte-for-byte" / "SAME primary_type"
6. `test_rule8_viability_addendum_custom_threshold` — custom threshold parameter works
7. `test_viability_contract_addendum_is_additive` — flag-ON: all existing rule-8 locks still hold + new phrases present
8. `test_viability_contract_flag_off_unchanged` — flag-OFF: byte-identical, no "cosine similarity"

`VIABILITY_CONTRACT_ENABLED` appears 5 times in the test file (greppable).

## Verification

- `poetry run pytest tests/unit/test_agent_prompts.py tests/unit/test_agent_io.py -v` — **35/35 passed**
- `make lint` — **All checks passed**
- `grep -c "VIABILITY_CONTRACT_ENABLED" tests/unit/test_agent_prompts.py` — **5** (>= 1 required)
- `grep -n "0\.55" app/agent/prompts.py` — one docstring-comment only, no functional hardcode
- Phase-7 CI grep gate (`test_agent_io.py`) — **green in both flag states**

## Deviations from Plan

None — plan executed exactly as written.

## TDD Gate Compliance

- RED gate: `c5fe419` — `test(13-02): add failing tests for rule8_viability_addendum (RED)`
- GREEN gate: `3d9bb1c` — `feat(13-02): add flag-gated rule8_viability_addendum to prompts.py (GREEN)`
- REFACTOR: not needed (implementation was clean on first pass)

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. The addendum is
a pure string function gated by an in-process boolean — no trust boundary crossing.

## Self-Check: PASSED

- `app/agent/prompts.py` — FOUND: contains `rule8_viability_addendum`
- `tests/unit/test_agent_prompts.py` — FOUND: contains 8 new DEC-01 tests
- Commit `c5fe419` — FOUND (RED gate)
- Commit `3d9bb1c` — FOUND (GREEN gate)
- All 35 tests pass, lint clean
