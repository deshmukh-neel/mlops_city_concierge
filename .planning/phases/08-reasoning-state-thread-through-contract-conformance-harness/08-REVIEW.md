---
phase: 08-reasoning-state-thread-through-contract-conformance-harness
reviewed: 2026-06-04T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - .gitignore
  - Makefile
  - app/agent/adapters/__init__.py
  - app/agent/graph.py
  - app/main.py
  - pyproject.toml
  - scripts/eval_agent.py
  - tests/integration/test_reasoning_state_roundtrip.py
  - tests/unit/agent/test_adapters.py
  - tests/unit/fixtures/reason_04_prune_baseline.json
  - tests/unit/test_agent_graph.py
  - tests/unit/test_lifespan.py
findings:
  critical: 0
  warning: 4
  info: 6
  total: 10
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-06-04
**Depth:** standard
**Files Reviewed:** 9 source files (plus fixture + integration test)
**Status:** issues_found

## Summary

Phase 8 ships the typed `ProviderAdapter` contract for reasoning-state thread-through. The implementation correctly threads `provider` through `build_agent_graph` (default `"openai"`), wires post-prune replay + capture inside `plan()`, preserves `additional_kwargs` across the pruner cutoff (D-08-07), and registers `NoOpAdapter` for every `SUPPORTED_PROVIDERS` value (D-08-08).

The core contract is sound. The four-shape conformance harness, the REASON-05 architectural gate, and the REASON-06 byte-identity fixture are genuine end-to-end exercises, not tautologies. Pydantic copy-semantics on `AIMessage.additional_kwargs` were verified to prevent aliasing between the stub and the original message (which protects future Phase 9 in-place mutations).

No BLOCKERS found. Four WARNINGs flag forward-compatibility hazards and test-coverage gaps that will bite Phase 9 if not addressed. Six INFO items capture style / robustness / documentation polish.

Key invariants verified positively:
- `_prune_for_llm` stub correctly preserves `additional_kwargs` for empty AND populated kwargs (Pydantic deep-copies, so the stub's kwargs are independent of the source).
- `ADAPTERS` registry exactly matches `SUPPORTED_PROVIDERS` (5 entries; all `NoOpAdapter`).
- Provider param is keyword-only (`*, provider: str = "openai"`).
- Pytest marker `reasoning_conformance` is registered in `pyproject.toml` and excluded from default `make test` via `addopts`.
- Legacy `build_agent_graph(...)` call sites without `provider=` get NoOpAdapter via the openai default — byte-identical to pre-Phase-8.
- Fixture round-trips cleanly through the regen pipeline (verified by replaying `_prune_for_llm + NoOpAdapter().replay_reasoning_state(None)` on the input).

## Warnings

### WR-01: replay step reads `_reasoning_state` from the kept-as-is AIMessage, not the stub — D-08-07 preservation is mostly dead code in Phase 8

**File:** `app/agent/graph.py:299-312`

**Issue:** The replay loop walks `reversed(messages_for_llm)` and binds `captured_state` to the **first** `AIMessage` found — which is the most-recent one. By construction of `_prune_for_llm`, the most-recent AIMessage in the pruned list is **always a kept-as-is message from after the cutoff** (because `_RECENT_TOOL_EXCHANGES_KEPT >= 2` AIMessages survive verbatim). The pre-cutoff stub (where D-08-07's `additional_kwargs` preservation actually fires) is only reachable as `captured_state` source when **every** post-cutoff message is non-AIMessage — i.e., effectively never in normal flow.

This means D-08-07's load-bearing kwargs-preservation patch is preventing **future** loss when a Phase 9 real adapter eventually reads from a stub — but the current replay-side code path almost never exercises it. The unit test `test_prune_for_llm_preserves_additional_kwargs_on_stub` validates the pruner in isolation but does NOT prove the replay step actually consumes from a stub.

This is a forward-compat concern: when Phase 9 wires a real OpenAI / Anthropic adapter that depends on multi-turn reasoning state, the first long-horizon test with >2 tool exchanges may surface that the kept-as-is path dominates (which is fine) and the stub path was never validated end-to-end.

**Fix:** Add a focused integration test that proves the replay step reads from a stub when ALL kept-as-is messages happen to be non-AIMessage (e.g., a synthesized state where the only AIMessage left in the pruned list IS the stub). At minimum, document this constraint inline at `graph.py:299`:

```python
# NOTE: the most-recent AIMessage in messages_for_llm is almost always a
# kept-as-is (post-cutoff) message — the pre-cutoff stub branch (D-08-07)
# only matters when the LATEST tool exchange has no AIMessage successor,
# which is rare in normal multi-turn flow.
captured_state = None
for m in reversed(messages_for_llm):
    if isinstance(m, AIMessage):
        captured_state = m.additional_kwargs.get("_reasoning_state")
        break
```

---

### WR-02: in-place mutation contract for `replay_reasoning_state` will leak adapter state back into `state.messages`

**File:** `app/agent/adapters/__init__.py:60-63`, `app/agent/graph.py:312`

**Issue:** The `ProviderAdapter.replay_reasoning_state` docstring states: *"Implementations must not mutate the list spine; in-place edits on individual messages' additional_kwargs are acceptable per D-08-06."*

`MockReasoningAdapter.replay_reasoning_state` mutates `msg.additional_kwargs["_reasoning_state"] = state` in-place (line 112). When `messages_for_llm` shares message-object identity with `state.messages` (which it does for kept-as-is messages from the pruner), this mutation propagates back into the graph's persistent state. For NoOpAdapter (no mutation) this is benign. For Phase 9 real adapters, mutating the kept-as-is AIMessage's kwargs will:

1. Cause repeat-mutation on each plan() turn (the SAME `_reasoning_state` key gets overwritten — fine if idempotent, broken if cumulative).
2. Leak across `state.model_copy(deep=True)` boundaries only if the adapter mutated BEFORE the copy (eval_agent.py line 682 does this).
3. Surface in MLflow traces / observability callbacks that snapshot `state.messages` mid-turn.

The contract permits this, but no test currently asserts that NoOpAdapter (the only registered Phase 8 adapter) **does not** mutate. A regression where someone adds mutation to NoOpAdapter would not be caught.

**Fix:** Either tighten the contract docstring to forbid in-place mutation (forcing all adapters to use `AIMessage.model_copy(update=...)` which is safer but allocates), or add an explicit unit test that NoOpAdapter is byte-identical:

```python
def test_noop_adapter_does_not_mutate_message_kwargs() -> None:
    """NoOpAdapter must not mutate any inbound AIMessage's additional_kwargs."""
    adapter = NoOpAdapter()
    msg = AIMessage(content="x", additional_kwargs={"existing": "value"})
    before = dict(msg.additional_kwargs)
    adapter.replay_reasoning_state([msg], {"provider": "test", "rc": "y"})
    assert msg.additional_kwargs == before  # no in-place edit
```

---

### WR-03: `test_lifespan` only verifies `provider="openai"` — `gemini`/`deepseek`/`kimi` paths through `getattr(loaded.params, "llm_provider", "openai")` are untested

**File:** `tests/unit/test_lifespan.py:27-58`

**Issue:** The lifespan test fixes `ActiveModelConfig.llm_provider = "openai"` and asserts `build_agent_graph.assert_called_once_with(fake_llm, provider="openai")`. This passes for the trivial case but does not exercise:

1. **Non-openai providers**: gemini, deepseek, kimi, scripted. If a future refactor breaks `getattr(loaded.params, "llm_provider", ...)` (e.g., someone renames the field to `provider`), the test still passes because the openai default would mask the bug.
2. **The fallback default**: the `"openai"` literal in `getattr(..., "openai")`. If `loaded.params` is unexpectedly missing `llm_provider`, the fallback triggers — but no test pins this behavior.

This matters because Phase 9 will swap adapters per provider; if lifespan silently passes `"openai"` when the loaded config says `"gemini"`, the wrong adapter gets bound at startup with no test surfacing it.

**Fix:** Parametrize the test across `SUPPORTED_PROVIDERS` so each provider's threading is verified:

```python
@pytest.mark.parametrize("provider", ["openai", "gemini", "deepseek", "kimi", "scripted"])
async def test_lifespan_threads_provider_into_build_agent_graph(mocker, provider) -> None:
    # ... build ActiveModelConfig(llm_provider=provider, ...)
    # ... assert build_agent_graph.assert_called_once_with(fake_llm, provider=provider)
```

---

### WR-04: REASON-02 conformance harness uses `bytes` payload for Gemini, but bytes are not JSON-serializable — any state-checkpoint or observability layer will crash

**File:** `tests/integration/test_reasoning_state_roundtrip.py:114`

**Issue:** The Gemini parametrize case uses `{"provider": "gemini", "thought_signature": b"\x00\x01\x02"}` — raw `bytes`. The MockReasoningAdapter happily stuffs this into `AIMessage.additional_kwargs["_reasoning_state"]`. The test passes because the assertion is dict equality on a Python object — never crossing a JSON boundary.

In production, LangGraph's persistent checkpoint backends (e.g., SQLite, Postgres) and MLflow's callback handlers (`langgraph_callbacks()` in `app/main.py:794`) typically serialize `additional_kwargs` to JSON for tracing. `bytes` will crash these layers with `TypeError: Object of type bytes is not JSON serializable`. The conformance harness greenlights a shape that the production observability stack cannot transport.

When Phase 9 wires a real Gemini adapter, the FIRST production trace with a non-empty `thought_signature` will fail in MLflow's callback even though the in-memory graph round-trip succeeds. The harness should validate JSON-serializability of every shape OR the contract docstring should explicitly require base64-encoding of binary state.

**Fix:** Either change the Gemini payload to a base64-encoded string (matching how the real Gemini adapter should encode the signature for transport), or add an explicit `json.dumps` round-trip in the test:

```python
import json
# Validate the shape survives JSON transport (langgraph checkpoints,
# MLflow callbacks, etc.) — bytes will crash here.
try:
    json.dumps(payload, default=str)
except (TypeError, ValueError) as exc:
    pytest.fail(
        f"REASON-02 payload {payload['provider']!r} is not JSON-serializable: "
        f"{exc}. Production transport (LangGraph checkpoints, MLflow) will crash."
    )
```

And document on `StatePayload` (adapters/__init__.py:32) that payloads MUST be JSON-serializable for compatibility with persistent state and observability.

## Info

### IN-01: `NoOpAdapter()` instantiated N times via dict-comprehension despite being stateless

**File:** `app/agent/adapters/__init__.py:121`

**Issue:** `ADAPTERS: dict[str, ProviderAdapter] = {p: NoOpAdapter() for p in SUPPORTED_PROVIDERS}` creates 5 distinct `NoOpAdapter` instances even though the class is stateless. Marginal memory waste; harmless.

**Fix:** If preferred, share a single instance:
```python
_NOOP = NoOpAdapter()
ADAPTERS: dict[str, ProviderAdapter] = {p: _NOOP for p in SUPPORTED_PROVIDERS}
```

Trivial; flagged only because the comment at line 117-120 emphasizes "single source of truth" — a single shared instance would slightly better embody that.

---

### IN-02: `NoOpAdapter` fallback in `build_agent_graph` silently swallows typos in `provider=` kwarg

**File:** `app/agent/graph.py:279`

**Issue:** `adapter: ProviderAdapter = ADAPTERS.get(provider, NoOpAdapter())` — an unknown provider string (e.g., `provider="openaii"` typo) silently falls back to `NoOpAdapter()`. The docstring at line 269-270 acknowledges this is intentional Phase-8 defensive default but warns "Phase 9 sub-phases may add stricter validation when real adapters land."

For Phase 8 this is correct behavior; flagging for Phase 9 follow-up. Consider adding a debug log:
```python
adapter: ProviderAdapter = ADAPTERS.get(provider)
if adapter is None:
    logger.debug("build_agent_graph: unknown provider %r — falling back to NoOpAdapter", provider)
    adapter = NoOpAdapter()
```

---

### IN-03: `_RecordingLLM` in `test_agent_graph.py` has a mutable default arg `recorded_inputs: list[list[BaseMessage]] = []` — Pydantic noqa is misleading

**File:** `tests/unit/test_agent_graph.py:753`

**Issue:** The `# noqa: RUF012 — pydantic mutable default` comment is technically valid (Pydantic Field would handle this), but the bare `= []` does NOT use `Field(default_factory=list)`. Pydantic v2 may or may not deep-copy the default per-instance depending on the field config; the safer idiom is what the conformance harness uses:

```python
recorded_inputs: list[list[BaseMessage]] = Field(default_factory=list)
```

(See `tests/integration/test_reasoning_state_roundtrip.py:75` for the correct pattern.) The current code WORKS but breaks the contract that mutable defaults are shared. Test `test_plan_replays_reasoning_state_into_outbound` explicitly initializes `recorded_inputs=[]` at construction (line 838) to dodge the issue — but that's a defensive workaround for a code smell.

**Fix:** Migrate to `Field(default_factory=list)`:
```python
from pydantic import Field
recorded_inputs: list[list[BaseMessage]] = Field(default_factory=list)
```

Then drop the noqa and the explicit `recorded_inputs=[]` at construction sites.

---

### IN-04: REASON-06 fixture path uses `Path(__file__).parent / "fixtures"` — fragile if test layout changes

**File:** `tests/unit/test_agent_graph.py:726, 1623`

**Issue:** Two separate sites compute the fixture path. If `tests/unit/test_agent_graph.py` ever moves (e.g., into a subdirectory), both call sites need updating in lockstep. The `--regen-reason-04-fixture` entrypoint silently writes to the wrong path if the move is incomplete.

**Fix:** Extract a module-level constant:
```python
_REASON_04_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reason_04_prune_baseline.json"
```

Use everywhere. Minor maintenance hygiene.

---

### IN-05: `.gitignore` re-include patterns rely on negation order — surrounding rules could silently re-block the fixture directory

**File:** `.gitignore:36-37`

**Issue:** The re-include pattern:
```
!tests/unit/fixtures/
!tests/unit/fixtures/*.json
```

works because the only ignore rule that catches them is `*.json` at line 11. However, git's pattern matching is order-sensitive AND a parent-directory ignore (e.g., a future rule `tests/` to exclude test reports) would re-block these files because git does not let you re-include files inside an ignored directory unless the directory itself is also re-included.

Currently safe. For robustness, document the dependency in a comment:

```
# IMPORTANT: re-include order matters. `*.json` (line 11) is the only blanket
# ignore that catches these. If a future rule excludes `tests/` itself, BOTH
# negate patterns must be moved above it AND the parent `tests/` re-included
# via `!tests/` first.
```

---

### IN-06: `pyproject.toml` `addopts` quoting works in pytest but is brittle to env-passing

**File:** `pyproject.toml:69`

**Issue:** `addopts = "-v --tb=short -m 'not reasoning_conformance'"` uses single quotes inside a double-quoted TOML string. Pytest's shlex parsing handles this correctly. However, when a CI script tries to override via `PYTEST_ADDOPTS` or shell-splices the value, the quoting can break (the shell strips the single quotes).

For example, a developer running `pytest $(grep addopts pyproject.toml | cut -d= -f2)` would see `-m not reasoning_conformance` get split into `-m`, `not`, `reasoning_conformance` — passing `not` as the marker expression.

Currently no CI script does this; flagging for awareness. The standard `make test-reasoning-conformance` and `make test` targets work correctly because they don't re-shell the addopts.

---

_Reviewed: 2026-06-04_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
