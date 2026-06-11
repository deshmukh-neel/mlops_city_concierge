---
phase: 08-reasoning-state-thread-through-contract-conformance-harness
plan: 05
subsystem: tests/unit
tags: [reasoning-state, byte-identity, regression, fixture, NoOpAdapter, REASON-06, D-08-15, phase-8, v2.1, tdd]
dependency_graph:
  requires:
    - tests/unit/test_agent_graph.py:test_prune_for_llm_preserves_additional_kwargs_on_stub  # Plan 08-02
    - app/agent/graph.py:_prune_for_llm                                                       # Plan 08-02 (kwargs preservation)
    - app/agent/adapters/__init__.py:NoOpAdapter                                              # Plan 08-01
  provides:
    - REASON-06 byte-identity hard gate enforced at the unit-test level
    - tests/unit/fixtures/reason_04_prune_baseline.json committed baseline
    - tests/unit/test_agent_graph.py:test_reason_04_noop_adapter_byte_identical_to_pre_phase8
  affects:
    - Phase 9 sub-phases (each must keep the NoOpAdapter ADAPTERS entry byte-identical for unchanged providers; tampering with NoOpAdapter or _prune_for_llm flips the regression gate)
    - Phase 10 cross-model baseline regen (operates on a different set of files; this fixture is the unit-level no-regression guarantee, eval-matrix staleness is the empirical one)
tech_stack:
  added: []
  patterns:
    - "Committed JSON fixture as deterministic regression baseline"
    - "Regen CLI flag gated by __main__ guard mirrors the configs/eval_baselines/ pre-regen-snapshot pattern"
    - "Test-side helper used on BOTH sides of the equality check (fixture generation AND runtime comparison)"
key_files:
  created:
    - tests/unit/fixtures/reason_04_prune_baseline.json
  modified:
    - tests/unit/test_agent_graph.py
    - .gitignore
decisions:
  - "D-08-15 implemented: byte-identity regression for the gpt-4o-mini path via a committed JSON fixture (strongest possible REASON-06 guarantee)"
  - "Helper _serialize_messages_for_fixture is the single source of truth for the comparison shape: same function generates the fixture AND serializes the runtime output (no comparison-side bias possible)"
  - "Regen path gated behind an explicit --regen-reason-04-fixture CLI flag (T-08-13 mitigation: no accidental drift) with a __main__ block in test_agent_graph.py rather than a separate scripts/ entrypoint"
  - "Rule 3 deviation: .gitignore re-include for tests/unit/fixtures/*.json — required because the repo's blanket *.json rule would silently hide the committed baseline; mirrors the existing exception pattern at .gitignore:25-28 for configs/eval_baselines/"
metrics:
  duration_minutes: ~15
  tasks_completed: 1
  files_modified: 3
  completed_date: 2026-06-04
requirements: [REASON-06]
---

# Phase 8 Plan 5: Byte-Identity Regression Summary

**One-liner:** Ships the strongest-possible REASON-06 guarantee per D-08-15 — a committed JSON fixture (`tests/unit/fixtures/reason_04_prune_baseline.json`, 8 entries) plus a unit test (`test_reason_04_noop_adapter_byte_identical_to_pre_phase8`) that asserts the post-Phase-8 `_prune_for_llm` + `NoOpAdapter().replay_reasoning_state(..., None)` pipeline produces byte-identical output for a realistic refinement-turn message list (the gpt-4o-mini case: empty `additional_kwargs`, no `_reasoning_state`). The fixture is committed alongside the test so it cannot drift silently; a `--regen-reason-04-fixture` `__main__` block regenerates it on intentional changes.

## What Shipped

### (a) `tests/unit/fixtures/reason_04_prune_baseline.json` — NEW committed fixture

- **76 lines, 8 entries** (`system + human + ai-stub + ai-with-tools + tool + ai-with-tools + tool + ai-final`).
- Deterministic — two runs of `--regen-reason-04-fixture` produce byte-identical SHA-1 (`eec5e3a7ce796014cbc553bdff883a60c618559b`).
- Contains the load-bearing edge cases for the prune cutoff:
  - 2 entries of type `"ai"` with truthy `tool_calls` (kept post-cutoff: `nearby`, `semantic_search`)
  - 2 entries of type `"ai"` with `tool_calls: null` (the stubbed pre-cutoff AI + the final no-tool-call AI)
- Sorted-key, indent=2 JSON for human-readable diffs on intentional regenerations.

### (b) `tests/unit/test_agent_graph.py` — EXTENDED

| Addition | Purpose |
|----------|---------|
| New top-of-file imports: `json`, `Path` (pathlib), `NoOpAdapter` from `app.agent.adapters` | Wire the regression test to the adapter contract from Plan 01 and the fixture path |
| Module docstring expansion | Documents the regen command, T-08-13 mitigation, and the D-08-15 link |
| Helper `_serialize_messages_for_fixture(msgs)` | Single source of truth for the comparison shape (used on BOTH sides of the equality check) |
| Helper `_reason_04_input_messages()` | Constructs the realistic 9-message input list deterministically (system + human + 3 tool-issuing AIMessages + 3 ToolMessages + 1 final AIMessage) — exercises the pre-cutoff stub branch (`_RECENT_TOOL_EXCHANGES_KEPT=2` forces the oldest AI to be stubbed) |
| Helper `_reason_04_pipeline_output(msgs)` | Runs the input through `_prune_for_llm` then `NoOpAdapter().replay_reasoning_state(..., None)` — the exact pipeline REASON-06 guards |
| Test `test_reason_04_noop_adapter_byte_identical_to_pre_phase8` | Loads the fixture, runs the pipeline, asserts byte-identity with a regen-instructions failure message |
| `if __name__ == "__main__":` block with `--regen-reason-04-fixture` CLI flag | One-shot fixture regeneration; unrecognised flags print an error + exit 2 (T-08-13 mitigation: no accidental drift) |

### (c) `.gitignore` — EXTENDED (Rule 3 deviation, see Deviations section)

Re-include `tests/unit/fixtures/` and `tests/unit/fixtures/*.json` so the committed fixture is not silently swallowed by the blanket `*.json` rule at line 11. Mirrors the existing exception block at lines 25-28 for `configs/eval_baselines/`.

## Tasks Completed

| Task | Name                                                                                          | Commit  | Files                                                                                          |
| ---- | --------------------------------------------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------- |
| 1.RED   | test(08-05): RED — byte-identity regression for NoOpAdapter on gpt-4o-mini path (D-08-15) | 04c6f4c | tests/unit/test_agent_graph.py                                                                |
| 1.GREEN | feat(08-05): GREEN — commit reason_04_prune_baseline.json fixture + un-ignore tests/unit/fixtures/ (D-08-15) | 3389428 | .gitignore, tests/unit/fixtures/reason_04_prune_baseline.json                                  |

TDD gate sequence in `git log`: `test(08-05): RED …` (04c6f4c) → `feat(08-05): GREEN …` (3389428).
- RED fails with `FileNotFoundError` because the fixture does not yet exist — the desired test-first failure mode.
- GREEN generates the fixture via the `__main__` regen helper and re-includes the fixtures directory in `.gitignore`, transitioning the test to PASS.
- No REFACTOR commit because no structural change to the helper or pipeline was warranted; the regen-helper / equality-check split is already minimal.

## Verification

### Plan-mandated automated check

```bash
poetry run pytest \
  tests/unit/test_agent_graph.py::test_reason_04_noop_adapter_byte_identical_to_pre_phase8 \
  tests/unit/test_agent_graph.py::test_prune_for_llm_preserves_additional_kwargs_on_stub \
  tests/unit/test_agent_graph.py::test_prune_for_llm_keeps_short_history_intact \
  tests/unit/test_agent_graph.py::test_prune_for_llm_drops_oldest_tool_exchanges -v \
  && test -s tests/unit/fixtures/reason_04_prune_baseline.json \
  && python -c "import json; data = json.load(open('tests/unit/fixtures/reason_04_prune_baseline.json')); \
       assert isinstance(data, list) and len(data) >= 5, f'fixture has {len(data)} entries, expected >= 5'; \
       print('OK')"
```

Result (post-GREEN):

```
tests/unit/test_agent_graph.py::test_reason_04_noop_adapter_byte_identical_to_pre_phase8 PASSED
tests/unit/test_agent_graph.py::test_prune_for_llm_preserves_additional_kwargs_on_stub  PASSED
tests/unit/test_agent_graph.py::test_prune_for_llm_keeps_short_history_intact           PASSED
tests/unit/test_agent_graph.py::test_prune_for_llm_drops_oldest_tool_exchanges          PASSED
============================== 4 passed in 0.58s ==============================
OK — fixture has 8 entries
```

### Every acceptance criterion from `<acceptance_criteria>`

| AC | Expected                                                                                                                | Actual | Status |
|----|-------------------------------------------------------------------------------------------------------------------------|--------|--------|
| AC1 | `tests/unit/fixtures/reason_04_prune_baseline.json` exists, valid JSON, list, >= 5 entries                              | 8 entries | PASS |
| AC2 | `grep -c "def test_reason_04_noop_adapter_byte_identical_to_pre_phase8" tests/unit/test_agent_graph.py` returns 1       | 1      | PASS |
| AC3 | `grep -c "_serialize_messages_for_fixture" tests/unit/test_agent_graph.py` returns at least 2                           | 5      | PASS |
| AC4 | `grep -c "reason_04_prune_baseline.json" tests/unit/test_agent_graph.py` returns at least 1                             | 6      | PASS |
| AC5 | `grep -c "NoOpAdapter" tests/unit/test_agent_graph.py` returns at least 1                                                | 11     | PASS |
| AC6 | The test passes against the post-Phase-8 code                                                                            | PASS   | PASS |
| AC7 | All four prune-related tests pass (new + 3 existing)                                                                     | 4/4    | PASS |
| AC8 | Fixture has type=ai with truthy tool_calls AND type=ai with empty/null tool_calls                                        | 2 + 2  | PASS |
| AC9 | `poetry run ruff check tests/unit/test_agent_graph.py` exits 0                                                            | clean  | PASS |
| AC10 | Full `tests/unit/test_agent_graph.py` sweep exits 0                                                                       | 48/48  | PASS |

### Wider regression sweep

```
poetry run pytest tests/unit/
=========== 1012 passed, 7 skipped, 9 warnings in 36.50s ===========
```

One more pass than the post-Plan-03 count of 1011 — the +1 is the new byte-identity regression test. No regressions across the full unit suite.

### Regen-fixture CLI invocation (for future intentional baseline moves)

```bash
poetry run python tests/unit/test_agent_graph.py --regen-reason-04-fixture
```

The regen path is gated behind the explicit `--regen-reason-04-fixture` flag (T-08-13). Any intentional baseline move MUST document the rationale in the commit message AND in the next Phase SUMMARY. Phase 9 sub-phases should NOT need to regenerate this fixture (they only swap individual `ADAPTERS` entries; `NoOpAdapter` itself is left unchanged).

### Determinism check

```bash
$ shasum tests/unit/fixtures/reason_04_prune_baseline.json
eec5e3a7ce796014cbc553bdff883a60c618559b  …
$ poetry run python tests/unit/test_agent_graph.py --regen-reason-04-fixture
regen-reason-04-fixture: wrote 8 entries to …/reason_04_prune_baseline.json
$ shasum tests/unit/fixtures/reason_04_prune_baseline.json
eec5e3a7ce796014cbc553bdff883a60c618559b  …   # IDENTICAL
```

The fixture is fully deterministic; running the regen twice produces byte-identical output.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking issue] `.gitignore` `*.json` rule would silently hide the committed fixture**

- **Found during:** Task 1 GREEN step, immediately after `git add tests/unit/fixtures/reason_04_prune_baseline.json` returned no staged change.
- **Issue:** The repo's `.gitignore` line 11 (`*.json`) inherits from the Python template and catches every JSON file outside of the explicit re-include block at lines 25-28 for `configs/eval_baselines/`. `git check-ignore -v tests/unit/fixtures/reason_04_prune_baseline.json` confirmed the fixture was ignored.
- **Why this is a blocking issue (Rule 3, not Rule 1 or 4):** The plan EXPLICITLY requires "The fixture lives at tests/unit/fixtures/reason_04_prune_baseline.json — committed alongside the test so it cannot drift silently" (truths block, line 19). Without the .gitignore exception, the fixture cannot be committed and D-08-15's "byte-identity regression … committed-fixture" promise breaks.
- **Fix:** Added a new exception block immediately after the existing `configs/eval_baselines/` block (`.gitignore` lines 30-37) re-including `tests/unit/fixtures/` and `tests/unit/fixtures/*.json`. Pattern is structurally identical to the existing eval-baselines exception so future test fixtures don't need a parallel .gitignore edit.
- **Files modified:** `.gitignore`
- **Commit:** 3389428 (folded into the GREEN commit since both fixture-creation and .gitignore-unblock are the same logical step)

### No other deviations

No Rule 1 (bug fixes), no Rule 2 (missing critical functionality), no Rule 4 (architectural questions), no auth gates triggered.

## Decisions Made

- **D-08-15 implemented as written:** committed-fixture byte-identity regression for the gpt-4o-mini path; realistic refinement-turn message list (system + human + 3 tool-issuing AIMessages + 3 ToolMessages + final AIMessage) so the pre-cutoff stub branch fires; serialization helper used on both sides of the equality check.
- **Regen entrypoint placement:** chose the `if __name__ == "__main__":` block in `test_agent_graph.py` rather than a separate `scripts/regen_reason_04_fixture.py`. Rationale per the plan's `<action>` block: keeps the input-list definition co-located with the test that asserts on it (single source of truth).
- **`_RECENT_TOOL_EXCHANGES_KEPT` value:** unchanged at 2; the input list has 3 tool-issuing AIMessages so the oldest gets stubbed — exactly the branch Plan 02 patched and that Plan 05 needs to exercise.
- **Tool-calls normalization in the helper:** empty list `[]` is normalized to `None` in the fixture. Rationale: the fixture's JSON intent is "no tool_calls here" and JSON `null` is more readable than `[]` for diff purposes. The helper is the only place this normalization happens, so both fixture and runtime output go through it.
- **`sort_keys=True` in the regen output:** chosen so future intentional regenerations produce diffs that are insertion-order-independent — only the actual content moves are visible in `git diff`.
- **TDD gate sequence honored:** RED commit first (fixture missing, test fails); GREEN commit second (fixture committed, test passes). Each commit is independently revertable.

## What's NOT in This Plan

- Plan 05 does **NOT** add the conformance harness or the REASON-05 gate — that is Plan 08-04 (`08-04-conformance-harness-and-reason-05-PLAN.md`).
- Plan 05 does **NOT** add fixtures for the reasoning-content case (Mock/real adapter with non-None state). The parametrize harness for that is REASON-02's territory in Plan 04, not Plan 05.
- Plan 05 does **NOT** modify `_prune_for_llm`, `build_agent_graph`, the existing prune tests, the Plan 02 test, the Plan 03 tests, or `app/agent/adapters/__init__.py` — these are all explicitly out-of-scope per the plan's `<action>` block ("Do NOT modify any of: …").
- Plan 05 does **NOT** touch the empirical eval-matrix baselines under `configs/eval_baselines/` — REASON-06's eval-matrix half is covered by the existing CI hard gate from v2.0 Phase 3 EVAL-10 (`scripts/check_baselines_fresh.py`), which is unaffected by Phase 8.

## REASON-06 Verdict

**PASSED on post-Phase-8 code.** Byte-identical output for the gpt-4o-mini-shaped input (empty `additional_kwargs`, no `_reasoning_state` marker) is now guarded by:

1. **Unit-level byte-identity test** (this plan) — fails on ANY change to `_prune_for_llm` or `NoOpAdapter` that alters the gpt-4o-mini path's serialized output, with a clear regen-instructions failure message.
2. **Plan 02 unit-level kwargs-preservation test** — fails on regression of the pre-cutoff stub's `additional_kwargs` propagation.
3. **Plan 03 full test_agent_graph.py sweep** — the 47-test pre-existing suite is byte-identical to its pre-Plan-3 shape; the new 48th test is the Plan 05 hard gate.
4. **Eval-matrix staleness gate** (v2.0 Phase 3 EVAL-10) — unchanged by Phase 8; continues to enforce empirical no-regression on the gpt-4o-mini baselines.

Combined, REASON-06 is enforced at the hardest possible level for the locked v2.0 prod anchor.

## Self-Check: PASSED

Verified after writing this Summary:

- `tests/unit/test_agent_graph.py`: FOUND (modified)
- `tests/unit/fixtures/reason_04_prune_baseline.json`: FOUND (created, 8 entries)
- `.gitignore`: FOUND (modified — re-include for `tests/unit/fixtures/*.json`)
- Commit `04c6f4c` (RED): FOUND in `git log`
- Commit `3389428` (GREEN): FOUND in `git log`
- `tests/unit/test_agent_graph.py`: 48/48 PASS
- Full `tests/unit/` sweep: 1012 passed, 7 skipped
- Ruff: clean across the modified test file
- `git check-ignore -v tests/unit/fixtures/reason_04_prune_baseline.json` → exit 0 with the `!tests/unit/fixtures/*.json` re-include rule firing (file NOT ignored)
- Fixture determinism: two consecutive regen runs produce byte-identical SHA-1

## TDD Gate Compliance

This plan is `type=auto` with `tdd="true"` on its single task. Two commits in the expected order:

- `test(08-05): RED …` (04c6f4c) — RED gate (test fails because fixture missing — `FileNotFoundError`)
- `feat(08-05): GREEN …` (3389428) — GREEN gate (fixture committed + .gitignore unblock; test passes)

No `refactor(...)` commit because no structural change was warranted — the helper / regen-block / equality-check split is already minimal. Plan-checker / verifier should accept this as a complete TDD cycle.

## Next Plan

Phase 8 wave 3 complete with this plan; Phase 8 itself wraps with this plan (final plan in the phase). The conformance harness and REASON-05 gate from Plan 04 are sequenced alongside this work in wave 3; the Phase 8 verifier ties together: (1) Plan 01's adapter contract surface, (2) Plan 02's kwargs preservation, (3) Plan 03's capture/replay wiring, (4) Plan 04's conformance harness + REASON-05 gate, (5) Plan 05's byte-identity regression.

Phase 9 sub-phases (per-provider adapter impls in order: gpt-5 family → DeepSeek reasoner → Claude Sonnet 4.6 → Gemini 3) can now ship independently. Each sub-phase swaps one `ADAPTERS` entry; the NoOpAdapter byte-identity guarantee shipped here ensures the un-swapped providers stay byte-identical to the pre-Phase-9 baseline as each new adapter lands.
