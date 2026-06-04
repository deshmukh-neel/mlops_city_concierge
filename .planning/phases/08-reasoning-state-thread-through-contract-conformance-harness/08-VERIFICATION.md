---
phase: 08-reasoning-state-thread-through-contract-conformance-harness
verified: 2026-06-04T22:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 8: Reasoning-State Thread-Through Contract + Conformance Harness — Verification Report

**Phase Goal:** A typed `ProviderAdapter` contract exists and is proven to round-trip reasoning state through `graph.invoke`, or the harness-swap decision gate fires and v2.1 replans around a custom imperative loop.

**Verified:** 2026-06-04T22:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

The phase goal is a disjunction: either (a) the contract exists and `graph.invoke` round-trips state, OR (b) the harness-swap gate fires and a blocker doc + xfail land. Branch (a) is the observed outcome on this branch: REASON-05 conformance gate test (`test_reason_05_graph_invoke_preserves_reasoning_state`) PASSES against real `build_agent_graph` + real `graph.ainvoke` + real `add_messages` reducer, and `08-REASON-05-BLOCKER.md` is correctly ABSENT. Per D-08-11 branch A this is the success branch.

### Observable Truths

| # | Truth (from ROADMAP success criteria + PLAN must-haves) | Status | Evidence |
|---|---------------------------------------------------------|--------|----------|
| 1 | A typed `ProviderAdapter` ABC exists with stable `capture_reasoning_state` + `replay_reasoning_state` methods; opaque `StatePayload`; fifth shape is interface extension | VERIFIED | `app/agent/adapters/__init__.py` — ABC at line 35, two `@abstractmethod` decorators at lines 46+54; `StatePayload = dict[str, Any]` at line 32; `ADAPTERS` is dict-comprehension-driven from `SUPPORTED_PROVIDERS` (line 121); 11/11 unit tests pass in `tests/unit/agent/test_adapters.py` |
| 2 | Contract type-stubs cover all four state shapes (OpenAI str / Anthropic signed dict / DeepSeek str / Gemini bytes); each has a dedicated unit test | VERIFIED | `tests/integration/test_reasoning_state_roundtrip.py:105-115` — exact dict literals for 4 shapes; 4 parametrize cases (`payload0..payload3`) all PASS under `make test-reasoning-conformance` |
| 3 | `_prune_for_llm` delegates state preservation to `ProviderAdapter.replay_reasoning_state`; gpt-4o-mini path unchanged; both regression-unit-tested | VERIFIED | `app/agent/graph.py:299-312` — POST-PRUNE replay + capture wired in `plan()`; `additional_kwargs` preservation patch at `_prune_for_llm:223-229`; `test_reason_04_noop_adapter_byte_identical_to_pre_phase8` PASSES (fixture-based byte-identity gate); `test_prune_for_llm_keeps_short_history_intact` + `test_prune_for_llm_drops_oldest_tool_exchanges` PASS unchanged |
| 4 | `tests/integration/test_reasoning_state_roundtrip.py` exists, runs 2-turn loop with mocked provider, asserts state survives; quarantined | VERIFIED | File exists (281 LOC); module-level `pytestmark = pytest.mark.reasoning_conformance` at line 56; default `pytest --collect-only -q` reports `0 selected / 5 deselected`; explicit `-m reasoning_conformance` collects 5 items |
| 5 | REASON-05 conformance test passes end-to-end through `graph.invoke` — harness-swap decision gate | VERIFIED | `test_reason_05_graph_invoke_preserves_reasoning_state` PASSED under live `make test-reasoning-conformance` run; uses REAL `build_agent_graph` + REAL `graph.ainvoke` + real `add_messages` reducer; `additional_kwargs["_reasoning_state"]` marker survives end-to-end; `08-REASON-05-BLOCKER.md` correctly ABSENT (branch A acceptance per D-08-11) |
| 6 | After refactor, v2.0 baselines (gpt-4o-mini × refinement_cheaper + all committed baselines) do not regress; `scripts/check_baselines_fresh.py` passes | VERIFIED | `poetry run python scripts/check_baselines_fresh.py` exits 0 with "OK — app/agent/ changed and 2 baseline file(s) refreshed"; byte-identity fixture asserts identical pruner output for gpt-4o-mini-shaped input |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app/agent/adapters/__init__.py` | ABC + StatePayload + NoOpAdapter + MockReasoningAdapter + ADAPTERS registry | VERIFIED | 131 LOC; exports 5 symbols via `__all__`; ADAPTERS dict-comprehension from SUPPORTED_PROVIDERS (5 keys: openai, gemini, deepseek, kimi, scripted); MockReasoningAdapter NOT in registry |
| `app/agent/graph.py` | `build_agent_graph(*, provider="openai")` keyword-only + POST-PRUNE replay/capture in `plan()` + `_prune_for_llm` additional_kwargs preservation | VERIFIED | Signature confirmed via inspect.signature: KEYWORD_ONLY, default `"openai"`; replay at line 312; capture at line 321-323; `_prune_for_llm` patch at line 223-229 with D-08-07 comment |
| `app/main.py` | Threads provider from MLflow loaded.params | VERIFIED | Line 411-413: `build_agent_graph(loaded.llm, provider=getattr(loaded.params, "llm_provider", "openai"))`; D-08-16 comment present |
| `scripts/eval_agent.py` | Threads provider through evaluate_cases | VERIFIED | Line 901-909: `evaluate_cases(... provider)`, `build_agent_graph(llm, max_steps=max_steps, provider=provider)`; line 1047 caller threads `provider=provider` |
| `scripts/eval_matrix.py` | UNTOUCHED (per PATTERNS.md correction) | VERIFIED | `grep -c "build_agent_graph" scripts/eval_matrix.py` returns 0 |
| `tests/integration/test_reasoning_state_roundtrip.py` | Parametrized 4-shape + REASON-05 gate + RecordingLLM | VERIFIED | 5 test items collected under marker; module-level pytestmark; 4 parametrize literals match CONTEXT.md specifics verbatim; RecordingLLM class present (lines 59-99) |
| `tests/unit/agent/test_adapters.py` | REASON-01 contract tests | VERIFIED | 11 tests, all PASS; covers ABC shape, NoOp identity, ADAPTERS == SUPPORTED_PROVIDERS, Mock not in prod registry |
| `tests/unit/test_agent_graph.py` | REASON-04 byte-identity + capture/replay + kwargs-preservation tests | VERIFIED | `test_reason_04_noop_adapter_byte_identical_to_pre_phase8` + `test_prune_for_llm_preserves_additional_kwargs_on_stub` + `test_build_agent_graph_provider_default_is_noop_adapter` + `test_plan_captures_reasoning_state_via_adapter` + `test_plan_replays_reasoning_state_into_outbound` — all PASS; 48/48 full sweep PASS |
| `tests/unit/fixtures/reason_04_prune_baseline.json` | Byte-identity baseline | VERIFIED | Valid JSON, 8 entries; contains both AI-with-tool_calls and stubbed-AI (tool_calls null) shapes; deterministic SHA-1 |
| `Makefile` | `test-reasoning-conformance` target | VERIFIED | Lines 164-166: `.PHONY: test-reasoning-conformance` + `$(POETRY_RUN) pytest -m reasoning_conformance -v` |
| `pyproject.toml` | reasoning_conformance marker registered + addopts exclusion | VERIFIED | Line 69: `addopts = "-v --tb=short -m 'not reasoning_conformance'"`; line 71: marker registered in `markers` array |
| `08-REASON-05-BLOCKER.md` | ABSENT on REASON-05 PASS (D-08-11 branch A) | VERIFIED | File does not exist — correct acceptance for branch A |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `app/agent/adapters/__init__.py:ADAPTERS` | `app/llm_factory.py:SUPPORTED_PROVIDERS` | dict-comprehension import | WIRED | Line 121: `{p: NoOpAdapter() for p in SUPPORTED_PROVIDERS}`; runtime assertion confirms `set(ADAPTERS) == set(SUPPORTED_PROVIDERS) == {"openai", "gemini", "deepseek", "kimi", "scripted"}` |
| `build_agent_graph` | `ADAPTERS` registry | `ADAPTERS.get(provider, NoOpAdapter())` | WIRED | `app/agent/graph.py:279`; resolves once at build time, closed over `plan()` |
| `plan()` | `adapter.replay_reasoning_state` + `adapter.capture_reasoning_state` | POST-PRUNE wrap around ainvoke | WIRED | `app/agent/graph.py:312` (replay) + `:321` (capture); reverse-walk for captured_state at line 308-311 |
| `app/main.py` | `build_agent_graph(provider=...)` | `getattr(loaded.params, "llm_provider", "openai")` | WIRED | Line 411-413 |
| `scripts/eval_agent.py:907` | `build_agent_graph(provider=...)` | threaded via `evaluate_cases(provider)` from `args.llm_provider` | WIRED | Lines 901, 909, 1047 |
| `tests/integration/test_reasoning_state_roundtrip.py:pytestmark` | `pyproject.toml:markers` | `pytest.mark.reasoning_conformance` | WIRED | Marker registered; no PytestUnknownMarkWarning |
| `pyproject.toml:addopts` | default `make test` exclusion | `-m 'not reasoning_conformance'` | WIRED | Default collection reports `0 selected / 5 deselected` |
| `Makefile:test-reasoning-conformance` | `pytest -m reasoning_conformance` | POETRY_RUN | WIRED | `make -n test-reasoning-conformance` resolves to `poetry run pytest -m reasoning_conformance -v` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `plan()` `captured_state` | reverse-walk `additional_kwargs["_reasoning_state"]` | most-recent AIMessage from prior turn | YES (in conformance harness via MockReasoningAdapter); None in NoOp/prod gpt-4o-mini path (by design) | FLOWING |
| `plan()` `state_payload` (capture) | `adapter.capture_reasoning_state(ai)` | adapter implementation | YES (Mock returns stored payload); None for NoOp (by design) | FLOWING |
| `RecordingLLM.recorded_inputs[-1]` | list of inbound messages on turn 2 | snapshot of `messages_for_llm` post-replay | YES — actual marker payload survives the reducer (REASON-05 gate evidence) | FLOWING |
| `ADAPTERS["openai"]` (production resolution) | NoOpAdapter instance | dict-comprehension over SUPPORTED_PROVIDERS | NoOp by design (Phase 9 swaps real adapter); gpt-4o-mini path is byte-identical | FLOWING (intentional NoOp in Phase 8) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Contract surface importable | `poetry run python -c "from app.agent.adapters import ProviderAdapter, StatePayload, NoOpAdapter, MockReasoningAdapter, ADAPTERS"` | OK | PASS |
| `build_agent_graph(provider=)` is keyword-only with default `"openai"` | `inspect.signature(build_agent_graph)` | KEYWORD_ONLY, default `'openai'` | PASS |
| ADAPTERS keys == SUPPORTED_PROVIDERS | runtime assertion | `{'openai','gemini','deepseek','kimi','scripted'}` matches | PASS |
| MockReasoningAdapter NOT in ADAPTERS | runtime assertion | True (no instance is MockReasoningAdapter) | PASS |
| Default `pytest` collection excludes harness | `pytest tests/integration/test_reasoning_state_roundtrip.py --collect-only -q` | `0 selected / 5 deselected` | PASS |
| Explicit marker collection includes harness | `pytest --collect-only -q -m reasoning_conformance` | 5 items | PASS |
| Makefile target resolves | `make -n test-reasoning-conformance` | `poetry run pytest -m reasoning_conformance -v` | PASS |
| Conformance harness (5 tests, including REASON-05 gate) | `make test-reasoning-conformance` | 5 passed in 0.54s | PASS |
| Adapter contract unit tests | `pytest tests/unit/agent/test_adapters.py` | 11 passed | PASS |
| Phase 8 unit tests in test_agent_graph.py | `pytest tests/unit/test_agent_graph.py -k "reason_04 or prune_for_llm or build_agent_graph or plan_captures or plan_replays"` | 7 passed | PASS |
| Full unit sweep — no regression | `pytest tests/unit/` | 1012 passed, 7 skipped | PASS |
| REASON-06 empirical staleness gate | `python scripts/check_baselines_fresh.py` | OK — baselines refreshed | PASS |

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` declared for this phase; the Makefile target `test-reasoning-conformance` serves the same role and is executed in Behavioral Spot-Checks. PLAN.md/SUMMARY.md do not declare additional probes. Skipped: not applicable.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| REASON-01 | 08-01, 08-03 | Typed ProviderAdapter contract with stable capture/replay interface | SATISFIED | ABC in `app/agent/adapters/__init__.py` with exactly two abstract methods; wired into `build_agent_graph` (Plan 03); 11+7 unit tests assert contract |
| REASON-02 | 08-01, 08-04 | Contract supports OpenAI str / Anthropic signed dict / DeepSeek str / Gemini bytes shapes | SATISFIED | 4 parametrize cases in `test_reason_02_four_shape_roundtrip` — all PASS; opaque `StatePayload = dict[str, Any]` is shape-agnostic |
| REASON-03 | 08-04 | Per-provider conformance test harness; runs in CI as quarantined integration test | SATISFIED | `test_reasoning_state_roundtrip.py` exists with module-level `reasoning_conformance` marker; pyproject.toml addopts exclusion; Makefile target |
| REASON-04 | 08-02, 08-03 | `_prune_for_llm` delegates state preservation; gpt-4o-mini path unchanged; unit-tested | SATISFIED | POST-PRUNE wiring in `plan()` (Plan 03); additional_kwargs preservation patch (Plan 02); `test_prune_for_llm_preserves_additional_kwargs_on_stub` + existing prune tests PASS unchanged |
| REASON-05 | 08-04 | Conformance test passes end-to-end through `graph.invoke` — architectural decision gate | SATISFIED | `test_reason_05_graph_invoke_preserves_reasoning_state` PASSES under live `make test-reasoning-conformance` run; `08-REASON-05-BLOCKER.md` correctly ABSENT (branch A per D-08-11) — Phase 9 proceeds on LangGraph as written |
| REASON-06 | 08-05 | After refactor, v2.0 baselines do not regress; staleness CI hard gate continues to pass | SATISFIED | Byte-identity fixture-based regression test PASSES (`test_reason_04_noop_adapter_byte_identical_to_pre_phase8`); `scripts/check_baselines_fresh.py` exits 0; full 1012-test unit sweep clean |

All 6 Phase 8 requirements (REASON-01..06) are SATISFIED. No orphaned requirements found — REQUIREMENTS.md maps no additional IDs to Phase 8 beyond what plans claim.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | No `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, or `PLACEHOLDER` markers in any Phase 8 file | — | — |

`grep -E "TBD|FIXME|XXX"` across all Phase 8 modified/created files returned zero matches. The 4 warnings + 6 info items from `08-REVIEW.md` are forward-compat hazards for Phase 9 (per the review's own classification: "0 critical, 4 warning, 6 info" and "No BLOCKERS found").

### Code Review Findings (from 08-REVIEW.md)

The standalone code review already ran and classified 0 critical / 4 warning / 6 info. All 4 warnings are forward-compatibility concerns that Phase 9 sub-phases will encounter, not Phase 8 blockers:

- **WR-01**: replay step reads from kept-as-is AIMessage, not stub — D-08-07 preservation is dead code in Phase 8 (becomes load-bearing in Phase 9)
- **WR-02**: in-place mutation contract for `replay_reasoning_state` could leak adapter state into `state.messages` (mitigation: NoOpAdapter doesn't mutate; Phase 9 adapters must be careful)
- **WR-03**: `test_lifespan` only verifies `provider="openai"` path (Phase 9 should parametrize across SUPPORTED_PROVIDERS)
- **WR-04**: Gemini `bytes` payload not JSON-serializable — Phase 9 transport (LangGraph checkpoints, MLflow callbacks) will crash; confirmed empirically (`json.dumps(payload)` raises `TypeError`)

These are pre-existing known issues documented in the review, not new findings from this verification. They are appropriately deferred to Phase 9.

### Human Verification Required

None. The Phase 8 deliverable surface is entirely programmatic:
- Contract surface is asserted by 11 unit tests
- Capture/replay wiring is asserted by 7 unit tests
- 4-shape contract coverage is asserted by 4 parametrize cases
- REASON-05 graph.invoke gate is asserted by 1 integration test through the real reducer
- REASON-06 byte-identity is asserted by a committed fixture + 1 unit test plus the existing baseline-staleness CI gate
- Quarantine plumbing is verified by `pytest --collect-only` shape comparison
- Phase boundary (no Phase 9 real adapter behavior) is asserted by `MockReasoningAdapter NOT in ADAPTERS`

No visual/UX/external-service behavior is in scope for this phase. The REASON-05 decision-gate human-checkpoint (Plan 04 Task 3) already resolved to "approved" (branch A — PASSED) per `08-04-SUMMARY.md`, with the absence of `08-REASON-05-BLOCKER.md` as the load-bearing evidence.

### Gaps Summary

No gaps. The phase goal is achieved by branch A: typed `ProviderAdapter` contract exists, all 4 state shapes have dedicated parametrize coverage, REASON-05 `graph.invoke` round-trip PASSES end-to-end, REASON-06 byte-identity is enforced by a committed fixture + CI staleness gate, quarantine plumbing keeps the harness out of default `make test`, and the v2.0 gpt-4o-mini production anchor is byte-identical (every `ADAPTERS` entry is `NoOpAdapter` per D-08-08).

Phase 9 (per-provider adapter implementations) can proceed on LangGraph as written; no v2.1.1 imperative-loop replan is triggered.

---

_Verified: 2026-06-04T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
