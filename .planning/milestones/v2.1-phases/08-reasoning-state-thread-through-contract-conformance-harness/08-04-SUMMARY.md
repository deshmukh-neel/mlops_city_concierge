---
phase: 08-reasoning-state-thread-through-contract-conformance-harness
plan: 04
subsystem: agent/tests
tags: [reasoning-state, conformance-harness, REASON-02, REASON-03, REASON-05, gate-decision, phase-8, v2.1, quarantine]
dependency_graph:
  requires:
    - app/agent/adapters/__init__.py:ProviderAdapter   # Plan 08-01
    - app/agent/adapters/__init__.py:MockReasoningAdapter   # Plan 08-01
    - app/agent/adapters/__init__.py:ADAPTERS   # Plan 08-01
    - app/agent/graph.py:build_agent_graph(*, provider="openai")   # Plan 08-03
    - app/agent/graph.py:plan() capture/replay around ainvoke   # Plan 08-03
    - app/agent/graph.py:_prune_for_llm additional_kwargs preservation   # Plan 08-02
  provides:
    - tests/integration/test_reasoning_state_roundtrip.py:test_reason_02_four_shape_roundtrip   # 4-shape parametrize harness (REASON-02)
    - tests/integration/test_reasoning_state_roundtrip.py:test_reason_05_graph_invoke_preserves_reasoning_state   # REASON-05 gate (PASSED)
    - tests/integration/test_reasoning_state_roundtrip.py:RecordingLLM   # BaseChatModel test double
    - pyproject.toml:reasoning_conformance marker registration + addopts exclusion
    - Makefile:test-reasoning-conformance target
  affects:
    - Phase 9 sub-phases (PROV-01..04) — REASON-05 PASSED, so Phase 9 proceeds on LangGraph as written; no v2.1.1 imperative-loop replan needed
    - Phase 10 (BASE-03) — will promote `reasoning_conformance` marker from quarantined to required CI gate
tech_stack:
  added: []
  patterns:
    - "Module-level pytestmark for marker-based quarantine (mirrors test_swap_real_db.py:21 idiom, swapped APP_ENV skipif for reasoning_conformance marker per D-08-14)"
    - "BaseChatModel + Field(default_factory=list) for scripted/recorded_inputs mirrors ScriptedChatModel (app/llm_factory.py:107-160)"
    - "list(messages) snapshot copy in _generate prevents downstream mutation tainting assertion history"
    - "pytest.mark.parametrize over EXACT D-08-13 dict literals — Phase 9 swaps Mock for real adapters one shape at a time; harness body unchanged"
    - "addopts addition `-m 'not reasoning_conformance'` is additive — existing `-v --tb=short` flags preserved"
key_files:
  created:
    - tests/integration/test_reasoning_state_roundtrip.py
    - .planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/08-04-SUMMARY.md
  modified:
    - pyproject.toml
    - Makefile
  conditional_not_created:
    - .planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/08-REASON-05-BLOCKER.md   # D-08-11 branch A — gate PASSED, blocker doc NOT materialized
decisions:
  - "D-08-09 + D-08-10 + D-08-12 implemented: REASON-05 gate test uses REAL build_agent_graph + REAL graph.ainvoke + RecordingLLM with the D-08-12 verbatim turn-1 AIMessage. Assertion is at the additional_kwargs level — NOT a system-message marker, NOT a content-string marker."
  - "D-08-13 implemented: 4-shape parametrize cases match the EXACT dict literals from CONTEXT.md specifics (OpenAI str, Anthropic signed thinking_blocks, DeepSeek str, Gemini bytes thought_signature)."
  - "D-08-14 implemented: pytestmark = pytest.mark.reasoning_conformance + addopts `-m 'not reasoning_conformance'` + Makefile test-reasoning-conformance target. Default `make test` excludes the harness; explicit Make target runs it."
  - "D-08-11 RESOLVED — branch A (REASON-05 PASSED): no 08-REASON-05-BLOCKER.md created, no xfail marker added. Phase 9 proceeds on LangGraph as written; v2.1.1 imperative-loop replan stays in deferred (ARCH-FUT-01)."
  - "Claude's Discretion: RecordingLLM is named RecordingLLM (matches PATTERNS.md template), lives in the test file (NOT in tests/integration/conftest.py), and snapshots via list(messages) for mutation-safety."
  - "Grep-fit fixup: rewrote one docstring sentence referencing `pytestmark = pytest.mark.reasoning_conformance` so grep -c returns 1 (acceptance criterion). Mirrors the docstring grep-fit pattern Plans 08-01 + 08-03 SUMMARYs document."
metrics:
  duration_minutes: ~30
  tasks_completed: 3   # Task 1 (harness), Task 2 (quarantine plumbing), Task 3 (gate-decision checkpoint resolved as PASSED)
  files_created: 2   # test file + this Summary
  files_modified: 2   # pyproject.toml + Makefile
  completed_date: 2026-06-04
requirements: [REASON-02, REASON-03, REASON-05]
---

# Phase 8 Plan 4: Conformance Harness + REASON-05 Gate Summary

**One-liner:** Ships the parametrized 4-shape `ProviderAdapter` conformance harness + the REASON-05 `graph.ainvoke` architectural decision gate + the `reasoning_conformance` pytest marker / addopts quarantine / `make test-reasoning-conformance` target. **REASON-05 gate outcome: PASSED — Phase 9 proceeds on LangGraph as written; no v2.1.1 imperative-loop replan triggered.** 5/5 tests pass (4 REASON-02 parametrize cases + 1 REASON-05 gate); `08-REASON-05-BLOCKER.md` is NOT materialized (D-08-11 branch A).

## What Shipped

### (a) Conformance harness — `tests/integration/test_reasoning_state_roundtrip.py` (NEW, 281 LOC)

| Test item | Purpose | Outcome |
|---|---|---|
| `test_reason_02_four_shape_roundtrip[payload0]` (OpenAI str) | REASON-02: contract supports `{"provider": "openai", "reasoning_content": "foo"}` | PASS |
| `test_reason_02_four_shape_roundtrip[payload1]` (Anthropic signed dict) | REASON-02: contract supports `{"provider": "anthropic", "thinking_blocks": [{"type": "thinking", "signature": "abc", "thinking": "..."}]}` | PASS |
| `test_reason_02_four_shape_roundtrip[payload2]` (DeepSeek str) | REASON-02: contract supports `{"provider": "deepseek", "reasoning_content": "bar"}` | PASS |
| `test_reason_02_four_shape_roundtrip[payload3]` (Gemini bytes) | REASON-02: contract supports `{"provider": "gemini", "thought_signature": b"\x00\x01\x02"}` | PASS |
| `test_reason_05_graph_invoke_preserves_reasoning_state` | **REASON-05 architectural decision gate** — `graph.ainvoke` preserves `additional_kwargs["_reasoning_state"]` across the LangGraph `add_messages` reducer | **PASS** |

5 items collected under the `reasoning_conformance` marker; 5/5 pass.

### (b) `RecordingLLM` helper

`BaseChatModel` subclass mirroring `ScriptedChatModel` (`app/llm_factory.py:107-160`):

- `scripted: list[AIMessage] = Field(default_factory=list)` — scripted responses popped in order.
- `recorded_inputs: list[list[BaseMessage]] = Field(default_factory=list)` — snapshot of every `_generate` call's input message list, so the assertion can read the EXACT outbound payload the graph handed to the LLM at each turn.
- `_generate` snapshots via `list(messages)` (immutability against downstream adapter / graph mutation), pops the next scripted response, falls back to `AIMessage(content="done")` if exhausted (graph terminates cleanly).
- `bind_tools(...) -> self` no-op binding mirrors the ScriptedChatModel idiom.

### (c) Quarantine plumbing

**`pyproject.toml` `[tool.pytest.ini_options]`:**

```toml
addopts = "-v --tb=short -m 'not reasoning_conformance'"
markers = [
    "reasoning_conformance: reasoning-state conformance harness (quarantined; run via make test-reasoning-conformance)",
]
```

- Additive `-m 'not reasoning_conformance'` extension preserves existing `-v --tb=short` flags.
- `markers` registration suppresses `PytestUnknownMarkWarning`.

**`Makefile`** — new target inserted after `test-integration:` and before the CLOUD_SQL block:

```makefile
.PHONY: test-reasoning-conformance
test-reasoning-conformance: ## Run reasoning-state conformance harness (quarantined; not in make test)
	$(POETRY_RUN) pytest -m reasoning_conformance -v
```

Tab-indented recipe line; help-string format matches sibling targets so `make help` picks it up.

### (d) Confirmation that default `make test` excludes the harness

| Command | Result |
|---|---|
| `pytest tests/integration/test_reasoning_state_roundtrip.py --collect-only -q` (default addopts) | `5 deselected / 0 selected` — silent quarantine via marker exclusion |
| `pytest tests/integration/test_reasoning_state_roundtrip.py --collect-only -q -m reasoning_conformance` | `5 tests collected` |
| `pytest -m reasoning_conformance -v` (across entire suite) | `5 passed, 1060 deselected` — the rest of the suite is correctly excluded |
| `pytest tests/unit/test_agent_graph.py --collect-only -q` | `47 tests collected` — unaffected by the addopts extension |

## REASON-05 Gate Outcome: PASSED

**The gate fired clean.** `additional_kwargs["_reasoning_state"]` survives `graph.ainvoke`'s `add_messages` reducer end-to-end with the canonical OpenAI-shape marker. This validates the core architectural assumption behind Phases 8-9: LangGraph's reducer DOES preserve `AIMessage` `additional_kwargs` across turn boundaries, so the Phase 8 contract (capture state on the AIMessage's `additional_kwargs["_reasoning_state"]` per D-08-06) is sound.

**What this means downstream:**

- **Phase 9 proceeds on LangGraph as written.** No replan needed. Sub-phases PROV-01 (gpt-5) → PROV-02 (DeepSeek) → PROV-03 (Claude) → PROV-04 (Gemini 3) each swap one `ADAPTERS` entry from `NoOpAdapter` to a real implementation; the wiring around them stays put.
- **No `08-REASON-05-BLOCKER.md` materialized.** Per D-08-11 the blocker doc only exists when the gate fires. This is the gate-passes acceptance branch.
- **No `xfail` marker on the REASON-05 test.** It passes clean and stays a live gate against future regressions (Phase 10 / BASE-03 promotes it to a required CI step).
- **ARCH-FUT-01 (replace LangGraph with custom imperative loop)** stays in deferred. The gate proved LangGraph is not lossy for this codebase's reasoning-state shape.

**Phase 8 success criterion #5 from ROADMAP.md is PASSED** ("If the conformance harness shows that `graph.invoke` itself drops the reasoning-state additional_kwargs at the reducer boundary, surface as a Phase 8 blocker (`08-REASON-05-BLOCKER.md`) AND replan v2.1 around a custom imperative loop — Phase 8 still ships with the contract + harness + blocker per Phase 6 D-06-09 part 2 precedent.").

## REASON-02 Contract Coverage

All four parametrize cases pass — the contract is shape-agnostic across:

| Shape family | Payload Python type | Provider literal |
|---|---|---|
| string | `str` | `"openai"` (`reasoning_content`), `"deepseek"` (`reasoning_content`) |
| signed dict | `list[dict[str, str]]` | `"anthropic"` (`thinking_blocks`) |
| bytes | `bytes` | `"gemini"` (`thought_signature`) |

`StatePayload = dict[str, Any]` (the opaque-dict choice from D-08-03) handles all three Python types without any Union/discriminator surgery. Adding a fifth shape in a future phase is an additive change to `ADAPTERS` only — exactly as REASON-02 requires.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | test(08-04): conformance harness — 4-shape REASON-02 + REASON-05 gate | `9e4599a` | tests/integration/test_reasoning_state_roundtrip.py (+281 LOC) |
| 2 | chore(08-04): register reasoning_conformance marker + quarantine + Makefile target | `d1163d1` | pyproject.toml, Makefile (+12 / -1) |
| 3 | checkpoint:human-verify — REASON-05 architectural decision gate | (no commit — gate-passes branch is a no-op per D-08-11) | n/a |

## Verification

### Acceptance criteria from PLAN.md

**Task 1 (10 criteria, all PASS):**

| AC | Expected | Actual |
|---|---|---|
| File exists | non-zero size | 281 LOC, exists |
| `grep -c "pytestmark = pytest.mark.reasoning_conformance"` | 1 | 1 (post grep-fit) |
| `grep -c "class RecordingLLM"` | 1 | 1 |
| `grep -c "test_reason_02_four_shape_roundtrip"` | ≥ 1 | 1 |
| `grep -c "test_reason_05_graph_invoke_preserves_reasoning_state"` | ≥ 1 | 1 |
| `"provider": "openai"` / anthropic / deepseek / gemini all present | yes each | yes (2 / 1 / 1 / 1 respectively) |
| `grep -c "thought_signature"` | ≥ 1 | 2 |
| `grep -c "thinking_blocks"` | ≥ 1 | 2 |
| `grep -c "_reasoning_state"` | ≥ 2 | 11 |
| `grep -c "D-08-11"` | ≥ 1 | 4 |
| `grep -c "graph.ainvoke"` | ≥ 1 | 8 |
| Collection `pytest --collect-only -q -m reasoning_conformance` | ≥ 5 items | 5 items |
| `ruff check` | exit 0 | exit 0 |

**Task 2 (10 criteria, all PASS):**

| AC | Expected | Actual |
|---|---|---|
| `grep -c "reasoning_conformance" pyproject.toml` | ≥ 2 | 2 (addopts + markers) |
| `grep -c "not reasoning_conformance" pyproject.toml` | 1 | 1 |
| `grep -c "markers" pyproject.toml` | ≥ 1 | 1 |
| `grep -c "test-reasoning-conformance:" Makefile` | 1 | 1 |
| `grep -c "pytest -m reasoning_conformance" Makefile` | 1 | 1 |
| `grep -c "^.PHONY: test-reasoning-conformance" Makefile` | 1 | 1 |
| Default collection skips harness | "0 selected" / "no tests ran" | "5 deselected / 0 selected" |
| Explicit `-m reasoning_conformance` collection includes harness | ≥ 5 items | 5 |
| `tests/unit/` count unchanged by addopts | identical | 47 in test_agent_graph.py (unchanged) |
| `make test-reasoning-conformance` is a valid target | resolves cleanly | `make -n` shows `poetry run pytest -m reasoning_conformance -v` |
| No `PytestUnknownMarkWarning` when running harness file directly | warning absent | absent |

### Plan-mandated automated checks

```bash
$ pytest tests/integration/test_reasoning_state_roundtrip.py \
    -m reasoning_conformance -v --tb=long
============================ 5 passed in 0.65s =============================
```

```bash
$ pytest tests/integration/test_reasoning_state_roundtrip.py --collect-only -q
collected 5 items / 5 deselected / 0 selected
============== no tests collected (5 deselected) in 0.51s ==================
```

```bash
$ pytest tests/integration/test_reasoning_state_roundtrip.py \
    --collect-only -q -m reasoning_conformance
collected 5 items
========================== 5 tests collected in 0.45s =====================
```

```bash
$ grep -c "reasoning_conformance" pyproject.toml
2
$ grep -c "test-reasoning-conformance:" Makefile
1
```

```bash
$ ruff check tests/integration/test_reasoning_state_roundtrip.py
All checks passed!
```

### Wider regression sweep

```bash
$ pytest tests/unit/test_agent_graph.py
=================== 47 passed in 2.16s ===================
```

No regression on the Plan 03 unit-level capture/replay tests (`test_build_agent_graph_provider_default_is_noop_adapter`, `test_plan_captures_reasoning_state_via_adapter`, `test_plan_replays_reasoning_state_into_outbound`).

## Deviations from Plan

### Cosmetic — docstring rewrite to satisfy strict `grep -c` count

**Found during:** post-implementation acceptance-criteria check.

**Issue:** The initial draft module docstring included the literal phrase `pytestmark = pytest.mark.reasoning_conformance` to describe the quarantine mechanism. Acceptance criterion specified `grep -c "pytestmark = pytest.mark.reasoning_conformance"` returns 1 — the duplication made it return 2.

**Fix:** Rewrote the docstring sentence in prose form: "Module-level `pytestmark` assignment to the `reasoning_conformance` marker (defined below) — every test in this file inherits the marker." Semantically equivalent; mirrors the same docstring-grep-fit pattern Plans 08-01 + 08-03 SUMMARYs document. Pure cosmetic; no behavior change.

**Files modified:** `tests/integration/test_reasoning_state_roundtrip.py`
**Commit:** `9e4599a` (same as Task 1 — fix applied before commit landed)

### Rule 3 — Auto-fixed blocking issue: REASON-05 initial-state seeding

**Found during:** Task 1, first run of `test_reason_05_graph_invoke_preserves_reasoning_state`.

**Issue:** Initial draft of the REASON-05 test seeded the verbatim D-08-12 turn-1 `AIMessage` INTO `ItineraryState.messages` and scripted `RecordingLLM` with just `[AIMessage(content="done")]`. Because `"done"` has no `tool_calls`, the graph terminated after a single `plan()` call. Only ONE entry in `recorded_inputs`; that input's most-recent `AIMessage` had `additional_kwargs={"reasoning_content": "..."}` (the seeded RAW provider field) but NOT `_reasoning_state` (because `capture` had not yet run — it only runs on the AIMessage RETURNED from `ainvoke`, not on history). The assertion fired, but it was a TEST design bug — not a REASON-05 gate failure.

**Fix:** Re-read CONTEXT.md D-08-12 — the turn-1 `AIMessage` is what `RecordingLLM` EMITS as its turn-1 response, NOT what the initial state contains. Updated the test to:
- Initial state: just `HumanMessage("find a bar in mission")`.
- `RecordingLLM` scripted: `[turn1_ai, AIMessage(content="done", tool_calls=[])]` — turn-1 has the tool_call so the graph loops through `act()` → `critique()` → `plan()` again, giving `recorded_inputs` two entries (the second of which carries the marker via adapter replay).

**Files modified:** `tests/integration/test_reasoning_state_roundtrip.py`
**Commit:** `9e4599a` (fix applied before the commit landed)

This is a fix-before-commit cycle — not a separate RED commit — because the production code under test (Plans 08-01/02/03) is already correct; only the test scaffolding had a misreading of D-08-12. After the fix, all 5 tests pass on first run.

### Rule 1 — Auto-fixed bug: SIM108 ruff lint

**Found during:** post-implementation ruff check.

**Issue:** `RecordingLLM._generate` initial draft used `if self.scripted: msg = self.scripted.pop(0); else: msg = AIMessage(content="done")` — ruff SIM108 fired (suggested ternary).

**Fix:** Collapsed to `msg = self.scripted.pop(0) if self.scripted else AIMessage(content="done")`. Pure refactor; no behavior change. Commit `9e4599a` post-cleanup.

### No other deviations

- No Rule 2 (missing critical functionality) — the Phase 8 contract is intentionally narrow (the four shapes, the kwarg key, the marker quarantine).
- No Rule 4 (architectural questions) — D-08-11 already documented both gate outcomes; gate passed, branch A applies cleanly.
- No auth gates — the harness uses no real network / DB / API keys.
- No `08-REASON-05-BLOCKER.md` materialized — D-08-11 branch A explicitly forbids it on a passing gate.

## TDD Gate Compliance

Plan 04 Task 1 is `tdd="true"` for a NEW integration test file. The production code under test (Plan 08-01 adapters subpackage, Plan 08-02 `_prune_for_llm` kwarg preservation, Plan 08-03 capture/replay wiring + `provider=` parameter) is already in place from Waves 1 + 2. The natural TDD cadence here is one commit per task:

- **Task 1** commit `9e4599a` (`test(08-04): ...`) — adds the conformance harness. The transient RED moment was the REASON-05 mis-seeding caught during the first `pytest` run; the test was corrected before the commit landed (per the established pattern from Plan 08-03 where the in-flight Rule 3 fix was rolled into the same commit). All 5 tests pass at the commit.
- **Task 2** commit `d1163d1` (`chore(08-04): ...`) — adds the quarantine plumbing. No new tests; verification is collection-shape assertions plus a regression sweep.
- **Task 3** no commit — REASON-05 gate-passes branch is a no-op per D-08-11.

Gate sequence in git log: `test(08-04): conformance harness ...` → `chore(08-04): register reasoning_conformance marker ...` ✓

## What's NOT in This Plan

- Plan 04 does **NOT** ship real provider adapter implementations — every `ADAPTERS` entry is still `NoOpAdapter` from Plan 08-01. Phase 9 sub-phases (PROV-01..04) swap entries one shape at a time; the harness body is invariant.
- Plan 04 does **NOT** ship the byte-identity regression fixture for NoOp path. That is Plan 08-05 (`08-05-byte-identity-regression-PLAN.md`).
- Plan 04 does **NOT** promote the conformance harness to a required CI gate. That is Phase 10 (BASE-03).
- Plan 04 does **NOT** create `08-REASON-05-BLOCKER.md` — D-08-11 branch A (gate PASSED) explicitly forbids it.
- Plan 04 does **NOT** add `@pytest.mark.xfail` to the REASON-05 test — same reason.
- Plan 04 does **NOT** touch `tests/integration/conftest.py` or `app/agent/adapters/__init__.py` — `MockReasoningAdapter` already exported by Plan 01; the harness imports it directly.

## Self-Check: PASSED

Verified after writing this Summary:

- `tests/integration/test_reasoning_state_roundtrip.py`: FOUND (created)
- `pyproject.toml`: FOUND (modified — addopts + markers)
- `Makefile`: FOUND (modified — test-reasoning-conformance target)
- `.planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/08-04-SUMMARY.md`: FOUND (this file)
- `.planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/08-REASON-05-BLOCKER.md`: ABSENT (correct — gate PASSED, D-08-11 branch A)
- Commit `9e4599a` (test harness): FOUND in `git log`
- Commit `d1163d1` (marker + Makefile): FOUND in `git log`
- `pytest -m reasoning_conformance`: 5 passed, 1060 deselected
- `pytest tests/unit/test_agent_graph.py`: 47/47 PASS
- `ruff check tests/integration/test_reasoning_state_roundtrip.py`: clean
- Default `pytest` collection of harness: 5 deselected / 0 selected (silent quarantine)

## Next Plan

Plan 08-05 (`08-05-byte-identity-regression-PLAN.md`) elevates the byte-identity guarantee from the unit-level sweep proof (Plan 03's 47/47 unaffected) to a hard fixture-based regression test in `tests/unit/test_agent_graph.py`. With REASON-02, REASON-03, and REASON-05 all met by this plan, the only outstanding Phase 8 requirements after Plan 05 ships are REASON-01 (`ProviderAdapter` ABC + `StatePayload` + registry + `NoOpAdapter` default — already met by Plan 01), REASON-04 (delegation under-test on gpt-4o-mini happy path — met by Plan 03's unit-level + Plan 05's byte-identity), and REASON-06 (no regression on the locked anchor — gated by the v2.0 baselines, picked up by Plan 05's regression fixture).

After Plan 05, Phase 8 is complete and Phase 9 can start with PROV-01 (gpt-5 family) — the milestone anchor gate.
