---
phase: 08-reasoning-state-thread-through-contract-conformance-harness
plan: 03
subsystem: agent/graph
tags: [reasoning-state, provider-adapter, wiring, capture-replay, REASON-01, REASON-04, D-08-04, D-08-05, D-08-06, D-08-16, phase-8, v2.1, tdd]
dependency_graph:
  requires:
    - app/agent/adapters/__init__.py:ADAPTERS  # Plan 08-01
    - app/agent/adapters/__init__.py:NoOpAdapter  # Plan 08-01
    - app/agent/adapters/__init__.py:ProviderAdapter  # Plan 08-01
    - app/agent/adapters/__init__.py:MockReasoningAdapter  # Plan 08-01
    - app/agent/graph.py:_prune_for_llm  # Plan 08-02 (additional_kwargs preservation)
  provides:
    - app/agent/graph.py:build_agent_graph(*, provider="openai")  # adapter dispatched at build time
    - app/agent/graph.py:plan()  # POST-PRUNE capture+replay around ainvoke
    - additional_kwargs["_reasoning_state"] storage contract on AIMessage
  affects:
    - Plan 08-04 conformance harness (uses build_agent_graph(provider=...) + MockReasoningAdapter)
    - Plan 08-05 byte-identity regression (asserts NoOpAdapter path is byte-identical)
    - Phase 9 sub-phases (each swaps one ADAPTERS entry; wiring is now load-bearing)
tech_stack:
  added: []
  patterns:
    - "Keyword-only param with default mirrors judge_llm placement convention"
    - "Resolve-once-at-build-time + close-over-plan() mirrors tool_by_name lookup pattern"
    - "Reverse-walk to most-recent AIMessage mirrors MockReasoningAdapter.replay shape (D-08-06)"
    - "Attribute-access on ActiveModelConfig via getattr(..., default) — defensive for missing-field cases"
key_files:
  created: []
  modified:
    - app/agent/graph.py
    - app/main.py
    - scripts/eval_agent.py
    - tests/unit/test_agent_graph.py
    - tests/unit/test_lifespan.py
decisions:
  - "D-08-04 implemented: build_agent_graph gains keyword-only `provider: str = \"openai\"`; ADAPTERS.get(provider, NoOpAdapter()) resolves the adapter ONCE at graph-build time and is closed over plan()"
  - "D-08-05 implemented: capture+replay are POST-PRUNE in plan() — _prune_for_llm body unchanged; one replay call before ainvoke, one capture call after"
  - "D-08-06 implemented: captured state lives on AIMessage.additional_kwargs[\"_reasoning_state\"] — NOT on ItineraryState, NOT in a module-level dict (load-bearing for the REASON-05 gate in Plan 04)"
  - "D-08-16 implemented: app/main.py:409 threads provider from MLflow params (attribute-access via getattr); scripts/eval_agent.py:907 + 1047 thread provider from args.llm_provider"
  - "PATTERNS.md correction honored: scripts/eval_matrix.py is NOT touched (it subprocesses out to eval_agent.py with --llm-provider already on the CLI)"
  - "Rule 1 deviation: planner assumed loaded.params was a dict; it is an ActiveModelConfig Pydantic model. Auto-fixed by switching to getattr(loaded.params, \"llm_provider\", \"openai\")"
metrics:
  duration_minutes: ~25
  tasks_completed: 1
  files_modified: 5
  completed_date: 2026-06-04
requirements: [REASON-01, REASON-04]
---

# Phase 8 Plan 3: Capture/Replay Wiring Summary

**One-liner:** Wires the typed `ProviderAdapter` contract from Plan 08-01 into the agent graph — `build_agent_graph` gains a keyword-only `provider: str = "openai"` parameter that resolves the adapter once via `ADAPTERS.get(provider, NoOpAdapter())` at build time; `plan()` wraps `ainvoke` with POST-PRUNE replay (reads `additional_kwargs["_reasoning_state"]` from the most-recent AIMessage) and post-ainvoke capture (writes the payload back onto the just-returned AIMessage). Phase 8 ships NoOpAdapter for every `SUPPORTED_PROVIDERS` entry so the gpt-4o-mini path is byte-identical to pre-Phase-8 behavior (REASON-04). The two real call sites — `app/main.py:409` and `scripts/eval_agent.py:907 + 1047` — thread `provider=` end-to-end.

## What Shipped

### (a) Keyword-only signature change — `app/agent/graph.py`

```python
def build_agent_graph(
    llm: BaseChatModel,
    max_steps: int = 8,
    judge_llm: BaseChatModel | None = None,
    *,
    provider: str = "openai",
):
```

The bare `*` enforces keyword-only on `provider`. Default `"openai"` keeps all pre-existing positional callers in tests byte-identical (`build_agent_graph(fake_llm)` and `build_agent_graph(fake, max_steps=4)` still work and route to `NoOpAdapter`). The adapter is resolved ONCE inside the function body before `plan()` is defined:

```python
adapter: ProviderAdapter = ADAPTERS.get(provider, NoOpAdapter())
```

— so `plan()` closes over a single adapter instance for the lifetime of the compiled graph.

### (b) Replay/capture insertion points in `plan()` — `app/agent/graph.py`

POST-PRUNE wrap around `ai = await llm_with_tools.ainvoke(messages_for_llm)`:

```python
messages_for_llm = _prune_for_llm(messages_in)  # unchanged

# D-08-05 / D-08-06: POST-PRUNE replay
captured_state = None
for m in reversed(messages_for_llm):
    if isinstance(m, AIMessage):
        captured_state = m.additional_kwargs.get("_reasoning_state")
        break
messages_for_llm = adapter.replay_reasoning_state(messages_for_llm, captured_state)

ai = await llm_with_tools.ainvoke(messages_for_llm)

# D-08-05 / D-08-06: post-ainvoke capture
state_payload = adapter.capture_reasoning_state(ai)
if state_payload is not None:
    ai.additional_kwargs["_reasoning_state"] = state_payload
```

`_prune_for_llm` body is **unchanged**. The SystemMessage prepend at `step_count == 0` is unchanged. The `_RECENT_TOOL_EXCHANGES_KEPT` constant is unchanged. Reverse-walk to most-recent AIMessage mirrors `MockReasoningAdapter.replay_reasoning_state`'s own shape (single source of truth for "which AIMessage carries the marker").

### (c) Call-site updates with line numbers

| File | Line | Before | After |
|------|------|--------|-------|
| `app/main.py` | 409-414 | `app.state.agent_graph = build_agent_graph(loaded.llm)` | `app.state.agent_graph = build_agent_graph(loaded.llm, provider=getattr(loaded.params, "llm_provider", "openai"))` |
| `scripts/eval_agent.py` | 902 | `evaluate_cases(cases, llm, max_steps)` | `evaluate_cases(cases, llm, max_steps, provider)` (new required param) |
| `scripts/eval_agent.py` | 909 | `graph = build_agent_graph(llm, max_steps=max_steps)` | `graph = build_agent_graph(llm, max_steps=max_steps, provider=provider)` |
| `scripts/eval_agent.py` | 1047 | `await evaluate_cases(cases, llm, max_steps=args.max_steps)` | `await evaluate_cases(cases, llm, max_steps=args.max_steps, provider=provider)` (provider local was already defined at line 1035 from `args.llm_provider`) |
| `tests/unit/test_lifespan.py` | 55-57 | `build_agent_graph.assert_called_once_with(fake_llm)` | `build_agent_graph.assert_called_once_with(fake_llm, provider="openai")` |

### (d) PATTERNS.md correction honored — `scripts/eval_matrix.py` NOT modified

```
$ grep -c "build_agent_graph" scripts/eval_matrix.py
0
```

`scripts/eval_matrix.py` subprocesses out to `scripts/eval_agent.py` with `--llm-provider` already on the CLI; the `provider=` kwarg lands in `evaluate_cases` via `args.llm_provider` inside `eval_agent.py`. The original PLAN's `canonical_refs` referenced `eval_matrix.py`; the PATTERNS.md correction propagated through to this plan's `<action>` block and was honored.

### (e) Three new unit tests in `tests/unit/test_agent_graph.py`

| Test | Status | What it proves |
|------|--------|----------------|
| `test_build_agent_graph_provider_default_is_noop_adapter` | PASS | D-08-04 + D-08-08: omitting `provider=` routes through NoOpAdapter; no `_reasoning_state` written on the default path (byte-identity sentinel for gpt-4o-mini). |
| `test_plan_captures_reasoning_state_via_adapter` | PASS | D-08-05 + D-08-06: with `MockReasoningAdapter` monkey-patched into `ADAPTERS["scripted"]`, plan() writes the captured payload onto `ai.additional_kwargs["_reasoning_state"]`. |
| `test_plan_replays_reasoning_state_into_outbound` | PASS | D-08-05 (REASON-05 precursor at unit level): across two plan() turns, the marker stashed in turn 1 lands on the most-recent AIMessage of turn 2's outbound payload BEFORE `ainvoke`. Uses a new `_RecordingLLM` helper. |

### (f) Full `tests/unit/test_agent_graph.py` sweep — 47/47 PASS

```
============================== 47 passed in 1.57s ==============================
```

Including the 3 new tests + the 44 pre-existing tests (graph terminate / retry / max-steps / parallel-tool / commit-finalize / prune / commit-stops / enrich / retime / closure-swap). Default `provider="openai"` + NoOpAdapter routing keeps every positional call site byte-identical.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1.RED | test(08-03): RED — failing tests for ProviderAdapter wiring (D-08-04..06) | f72d802 | tests/unit/test_agent_graph.py |
| 1.GREEN(a) | feat(08-03): GREEN — wire ProviderAdapter into build_agent_graph + plan() (D-08-04..06, D-08-16) | 46ab721 | app/agent/graph.py |
| 1.GREEN(b) | feat(08-03): thread provider through real call sites + docstring grep-fit (D-08-16) | ab90265 | app/agent/graph.py, app/main.py, scripts/eval_agent.py |
| 1.test-fixup | test(08-03): update lifespan mock to expect provider= keyword (D-08-16) | b5d5a5c | tests/unit/test_lifespan.py |

TDD gate sequence in git log: `test(08-03): RED ...` (f72d802) → `feat(08-03): GREEN ...` (46ab721). Two further feat/test commits split the call-site threading and the lifespan-mock fixup from the core GREEN commit so each piece is independently revertable.

## Verification

### Plan-mandated automated check

```bash
poetry run pytest tests/unit/test_agent_graph.py -v -k "build_agent_graph or prune_for_llm or plan_captures_reasoning or plan_replays_reasoning"
```

Result: **6 passed, 41 deselected**.

```bash
poetry run python -c "from app.agent.graph import build_agent_graph; ... assert sig.parameters['provider'].kind == inspect.Parameter.KEYWORD_ONLY; assert sig.parameters['provider'].default == 'openai'; print('OK')"
```

Result: **OK**.

```bash
poetry run ruff check app/agent/graph.py app/main.py scripts/eval_agent.py tests/unit/test_agent_graph.py
```

Result: **All checks passed!**

### All 13 acceptance criteria from `<acceptance_criteria>`

| AC | Expected | Actual | Status |
|----|----------|--------|--------|
| AC1 | `grep -c "provider: str = \"openai\"" app/agent/graph.py` returns 1 | 1 | PASS |
| AC2 | `grep -c "ADAPTERS.get(provider" app/agent/graph.py` returns 1 (resolver) | 1 | PASS (after docstring rewrite to drop canonical phrase) |
| AC3 | `grep -c "adapter.replay_reasoning_state" app/agent/graph.py` returns 1 | 1 | PASS |
| AC4 | `grep -c "adapter.capture_reasoning_state" app/agent/graph.py` returns 1 | 1 | PASS |
| AC5 | `grep -c "_reasoning_state" app/agent/graph.py` returns at least 2 | 6 | PASS |
| AC6 | `grep -c "from app.agent.adapters import" app/agent/graph.py` returns 1 | 1 | PASS |
| AC7 | `grep -c "provider=" app/main.py` returns at least 1 | 3 | PASS |
| AC8 | `grep -c "provider=" scripts/eval_agent.py` returns at least 2 + signature | 3 | PASS |
| AC9 | `grep -c "D-08-16" app/main.py` returns 1 | 1 | PASS |
| AC10 | `grep -c "D-08-16" scripts/eval_agent.py` returns 1 | 1 | PASS |
| AC11 | `grep -c "build_agent_graph" scripts/eval_matrix.py` returns 0 | 0 | PASS |
| AC12 | `inspect.Parameter.KEYWORD_ONLY` and default `"openai"` | both match | PASS |
| AC13 | 3 new tests pass + 3 existing prune tests pass + full test_agent_graph.py sweep + ruff clean + test_main equivalent | all pass | PASS |

### Wider regression sweep

```
poetry run pytest tests/unit/
====== 1011 passed, 7 skipped, 9 warnings in 16.75s ======
```

The 7 skipped are integration tests (require APP_ENV=integration). No failures.

## Deviations from Plan

### Rule 1 — Auto-fixed bug: `loaded.params` is a Pydantic model, not a dict

**Found during:** Task 1, after the initial `app/main.py` edit caused 19 `test_chat_functional.py` tests to return 503.

**Issue:** PLAN.md instructed `provider=loaded.params.get("llm_provider", "openai")`. But `loaded.params` is an `ActiveModelConfig` Pydantic `BaseModel` (defined at `app/main.py:182`), not a dict. Calling `.get()` raised `AttributeError: 'ActiveModelConfig' object has no attribute 'get'` inside the `try:` block, which propagated to the `except Exception:` handler and silently set `app.state.agent_graph = None` — every `/chat` call then returned 503 from the agent-unavailable guard.

**Fix:** Switched to `getattr(loaded.params, "llm_provider", "openai")` — attribute access on the Pydantic model with a defensive default for the missing-field branch (mirrors the original PLAN intent of fallback to `"openai"` for degraded configs).

**Files modified:** `app/main.py`
**Commit:** ab90265
**Verification:** All 19 `tests/unit/test_chat_functional.py` tests went from FAIL→PASS; full `tests/unit/` sweep is 1011 passed.

### Rule 3 — Auto-fixed blocking issue: `test_lifespan.py` mock assertion

**Found during:** Task 1, after the `app/main.py` `provider=` threading landed.

**Issue:** `tests/unit/test_lifespan.py:55` asserted `build_agent_graph.assert_called_once_with(fake_llm)` — strict positional/kwargs match. Adding `provider="openai"` to the lifespan call site broke this assertion (`Kwargs: assert {'provider': 'openai'} == {}`). This is a test-expectation update, not a behavior regression.

**Fix:** Updated the assertion to `build_agent_graph.assert_called_once_with(fake_llm, provider="openai")` with an inline D-08-16 comment explaining why.

**Files modified:** `tests/unit/test_lifespan.py`
**Commit:** b5d5a5c

### Cosmetic — docstring rewrite to satisfy strict `grep -c` count

**Found during:** post-implementation acceptance-criteria check.

**Issue:** The original GREEN draft included the canonical phrase `ADAPTERS.get(provider, NoOpAdapter())` in the `build_agent_graph` docstring AND in the resolver line. Acceptance criterion AC2 specified `grep -c "ADAPTERS.get(provider" app/agent/graph.py` returns 1 (i.e., the resolver only). The duplication made it return 2.

**Fix:** Rewrote the docstring sentence in prose form: "Resolved ONCE via the ADAPTERS registry (with NoOpAdapter fallback) ..." — semantically equivalent, mirrors the same docstring-grep-fit pattern Plan 08-01 SUMMARY documents for `StatePayload`. Pure cosmetic; no behavior change.

**Files modified:** `app/agent/graph.py`
**Commit:** ab90265

### No other deviations

No Rule 2 (missing critical functionality), no Rule 4 (architectural questions), no auth gates triggered.

## TDD Gate Compliance

- **RED commit** `f72d802` — two of the three new tests fail with `TypeError: build_agent_graph() got an unexpected keyword argument 'provider'` (Tests B and C). Test A (the byte-identity sentinel) passes pre-wiring because the absent kwarg means no `_reasoning_state` is ever written — the invariant the sentinel guards is trivially true. Documented in the RED commit message.
- **GREEN commit** `46ab721` — all 3 new tests pass; all 44 pre-existing tests in `tests/unit/test_agent_graph.py` continue to pass (47/47).
- **REFACTOR commits** `ab90265` + `b5d5a5c` — call-site threading + lifespan-mock fixup. These are not refactors in the structural sense; they're integration-completion commits that bring the broader code surface in line with the new contract.

Gate sequence in git log: `test(08-03): RED ...` → `feat(08-03): GREEN ...` → `feat(08-03): thread provider ...` → `test(08-03): update lifespan mock ...` ✓

## What's NOT in This Plan

- Plan 03 does **NOT** ship a conformance harness. That is Plan 08-04 (`08-04-conformance-harness-and-reason-05-PLAN.md`). The unit-level precursor (`test_plan_replays_reasoning_state_into_outbound`) shipped here is a smoke check, not the REASON-05 gate.
- Plan 03 does **NOT** ship a byte-identity regression fixture. That is Plan 08-05. The full `tests/unit/test_agent_graph.py` sweep passing is the *unit-level* byte-identity proof for the NoOpAdapter path; Plan 05 elevates this to a hard fixture-based regression guarantee.
- Plan 03 does **NOT** touch `scripts/eval_matrix.py`. PATTERNS.md's correction is honored: `eval_matrix.py` fans out to `eval_agent.py` as subprocesses with `--llm-provider` already on the CLI.
- Plan 03 does **NOT** touch `app/llm_factory.py` — Phase 9 territory (per-provider real adapters land there).
- Plan 03 does **NOT** add a real provider adapter. Every entry in `ADAPTERS` is still `NoOpAdapter` (D-08-08); Phase 9 sub-phases swap entries one provider at a time.

## Self-Check: PASSED

Verified after writing this Summary:

- `app/agent/graph.py`: FOUND (modified)
- `app/main.py`: FOUND (modified)
- `scripts/eval_agent.py`: FOUND (modified)
- `tests/unit/test_agent_graph.py`: FOUND (modified)
- `tests/unit/test_lifespan.py`: FOUND (modified)
- Commit `f72d802` (RED): FOUND in `git log`
- Commit `46ab721` (GREEN graph.py): FOUND in `git log`
- Commit `ab90265` (call-sites + grep-fit): FOUND in `git log`
- Commit `b5d5a5c` (lifespan test): FOUND in `git log`
- `tests/unit/test_agent_graph.py`: 47/47 PASS
- Full `tests/unit/` sweep: 1011 passed, 7 skipped
- Ruff: clean across all five modified files
- `inspect` signature check: provider is KEYWORD_ONLY with default `"openai"`
- `grep -c "build_agent_graph" scripts/eval_matrix.py`: 0 (PATTERNS.md correction honored)

## Next Plan

Plan 08-04 (`08-04-conformance-harness-and-reason-05-PLAN.md`) builds the conformance harness on top of this wiring: a per-provider 2-turn `graph.invoke` loop that asserts the captured `_reasoning_state` payload survives the LangGraph `add_messages` reducer end-to-end. If the harness passes in isolation but fails when run through `graph.invoke`, Plan 08-04 flips the harness-swap decision gate and Phase 9 re-plans around a custom imperative loop (the harness-swap branch documented in `08-CONTEXT.md`). The unit-level precursor `test_plan_replays_reasoning_state_into_outbound` shipped here is a smoke check, not the gate itself.

Plan 08-05 (`08-05-byte-identity-regression-PLAN.md`) elevates the byte-identity guarantee from the unit-level sweep proof (this plan) to a hard fixture-based regression test.
