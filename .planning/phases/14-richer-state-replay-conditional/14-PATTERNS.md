# Phase 14: Richer State Replay (CONDITIONAL) - Pattern Map

**Mapped:** 2026-06-12
**Files analyzed:** 8 (5 modified, 1 extended, 1 new, 1 documented-only)
**Analogs found:** 8 / 8

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `app/agent/graph.py` | agent-loop, middleware | event-driven (replay site REPLAY-01 + preservation REPLAY-02) | `app/agent/graph.py` itself (Phase-13 block ~285-310) | self-reference — exact |
| `app/agent/adapters/__init__.py` | ABC provider contract | request-response | `app/agent/adapters/__init__.py` (Phase-9 ProviderAdapter ABC) | self-reference — exact |
| `app/agent/adapters/openai_gpt5.py` | service adapter | request-response | `app/agent/adapters/deepseek.py` (symmetric replay shape) | exact |
| `app/agent/adapters/deepseek.py` | service adapter | request-response | `app/agent/adapters/openai_gpt5.py` (symmetric replay shape) | exact |
| `app/agent/adapters/anthropic.py` | service adapter | request-response | `app/agent/adapters/openai_gpt5.py` (asymmetric — content list, not additional_kwargs) | role-match |
| `app/agent/adapters/gemini.py` | service adapter | request-response | `app/agent/adapters/openai_gpt5.py` | role-match |
| `tests/unit/test_adapters.py` | test | CRUD (adapter contract) | `tests/unit/test_adapters.py` (existing 9-test harness) + `tests/unit/test_graph_forced_commit.py` (flag-gated test pattern) | exact |
| `scripts/eval_agent.py` | script, data pipeline | batch | `scripts/eval_agent.py` (existing `arm_flags` dict ~line 928) | self-reference — exact |
| `docs/replay_arm_verdicts.md` (NEW) | documentation | N/A | `docs/decisiveness_arm_verdicts.md` | exact structural mirror |

---

## Pattern Assignments

### `app/agent/graph.py` — REPLAY-01 replay site + REPLAY-02 prune preservation

**Analog:** `app/agent/graph.py` Phase-13 arm-flag block (lines 285-310) + existing replay site (lines 336-341) + `_prune_for_llm` replacement block (lines 228-235)

---

#### Pattern 1: Env-flag reads at graph-build time (Phase-13 precedent, lines 285-310)

```python
# Phase 13 / DEC arm-flag reads — resolved ONCE at graph-build time and
# closed over the inner functions. With all three flags unset/0, behavior
# is byte-identical to the baseline path (flag-off is the default state).
_forced_commit_step: int = int(os.environ.get("FORCED_COMMIT_STEP", "0") or "0")
_viability_contract_enabled: bool = env_flag("VIABILITY_CONTRACT_ENABLED")
_parallel_tool_execution_enabled: bool = env_flag("PARALLEL_TOOL_EXECUTION_ENABLED")
# Pre-compute the prompt addendum at build time (pure string; empty when flag off).
_viability_prompt_addendum: str = rule8_viability_addendum(_viability_contract_enabled)
```

**Copy pattern for Phase 14:** Add two new bool reads immediately after the three DEC reads, before the inner-function definitions. Use `env_flag()` (the `app.config` helper — case-insensitive, truthy set "1"/"true"/"yes"/"on"):

```python
# Phase 14 / REPLAY arm-flag reads — same build-time resolve + closure pattern.
# Flag-off path must be byte-identical to the Phase-13 plateau (re-verify before merge).
_replay_multi_message_enabled: bool = env_flag("REPLAY_MULTI_MESSAGE_ENABLED")
_replay_content_blocks_enabled: bool = env_flag("REPLAY_CONTENT_BLOCKS_ENABLED")
```

---

#### Pattern 2: REPLAY-01 site — current most-recent-only injection (lines 333-341)

```python
# D-08-05 / D-08-06: POST-PRUNE reasoning-state replay. Read the most
# recent AIMessage's _reasoning_state kwarg (stashed by the previous
# turn's capture, preserved across the _RECENT_TOOL_EXCHANGES_KEPT
# cutoff by D-08-07's additional_kwargs forwarding in _prune_for_llm).
# The adapter decides how to inject it; NoOpAdapter returns the list
# unchanged so this is byte-identical for non-reasoning providers.
captured_state = None
for m in reversed(messages_for_llm):
    if isinstance(m, AIMessage):
        captured_state = m.additional_kwargs.get("_reasoning_state")
        break
messages_for_llm = adapter.replay_reasoning_state(messages_for_llm, captured_state)
```

**Copy pattern for Phase 14 (REPLAY-01 branch):** When `_replay_multi_message_enabled` is True, replace the single-message extraction above with a call to the new multi-message method. Flag-off path must fall through to the existing block unchanged:

```python
if _replay_multi_message_enabled:
    messages_for_llm = adapter.replay_reasoning_state_multi(messages_for_llm)
else:
    # Flag-off: existing single-message path (byte-identical to Phase-13 plateau)
    captured_state = None
    for m in reversed(messages_for_llm):
        if isinstance(m, AIMessage):
            captured_state = m.additional_kwargs.get("_reasoning_state")
            break
    messages_for_llm = adapter.replay_reasoning_state(messages_for_llm, captured_state)
```

---

#### Pattern 3: REPLAY-02 site — `_prune_for_llm` str() collapse (lines 228-235)

```python
if isinstance(m, AIMessage) and m.tool_calls:
    # Replace with a content-only AIMessage so we don't strand the
    # LLM thinking it issued tool_calls that were never answered.
    # D-08-07: preserve additional_kwargs (e.g. _reasoning_state)
    # across the cutoff window so adapter capture/replay can survive.
    pruned.append(
        AIMessage(
            content=m.content if isinstance(m.content, str) else str(m.content),
            additional_kwargs=m.additional_kwargs,
        )
    )
    continue
```

**Copy pattern for Phase 14 (REPLAY-02 branch):** When `_replay_content_blocks_enabled` is True, preserve the original `m.content` shape (list or str) verbatim. Tool_calls must still be stripped (AIMessage constructor without `tool_calls` handles this). Flag-off path must be byte-identical:

```python
if isinstance(m, AIMessage) and m.tool_calls:
    if _replay_content_blocks_enabled:
        # D-14-06: preserve original content shape (list or str) verbatim
        # instead of collapsing to str. tool_calls excluded by constructor default.
        pruned.append(
            AIMessage(
                content=m.content,
                additional_kwargs=m.additional_kwargs,
            )
        )
    else:
        # Flag-off: existing str() collapse (byte-identical to Phase-13 plateau)
        pruned.append(
            AIMessage(
                content=m.content if isinstance(m.content, str) else str(m.content),
                additional_kwargs=m.additional_kwargs,
            )
        )
    continue
```

**CRITICAL:** `_replay_content_blocks_enabled` must be passed to `_prune_for_llm` as a parameter (not read as a module-level var), matching the "resolve once at build time, pass as parameter" precedent. The `_prune_for_llm` signature becomes:

```python
def _prune_for_llm(
    messages: list[BaseMessage],
    *,
    preserve_content_blocks: bool = False,
) -> list[BaseMessage]:
```

And the call site in `plan()` becomes:
```python
messages_for_llm = _prune_for_llm(messages_in, preserve_content_blocks=_replay_content_blocks_enabled)
```

---

### `app/agent/adapters/__init__.py` — new `replay_reasoning_state_multi` ABC method

**Analog:** Existing `ProviderAdapter` ABC (lines 33-61) + `NoOpAdapter` (lines 64-78)

**Existing ABC contract (lines 44-61):**

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def capture_reasoning_state(self, message: AIMessage) -> StatePayload | None:
        ...

    @abstractmethod
    def replay_reasoning_state(
        self, outbound: list[BaseMessage], state: StatePayload | None
    ) -> list[BaseMessage]:
        ...
```

**New method to add — generic default in ABC (not abstract):**

```python
def replay_reasoning_state_multi(
    self, outbound: list[BaseMessage]
) -> list[BaseMessage]:
    """Replay per-message _reasoning_state for every in-window AIMessage.

    REPLAY-01 (D-14-03): iterates outbound, and for each AIMessage that
    carries a ``_reasoning_state`` in its ``additional_kwargs``, calls
    the existing single-message ``replay_reasoning_state`` on the
    sub-list up to and including that message.

    Generic default: iterate all AIMessages in outbound and apply
    per-message injection via the existing ``replay_reasoning_state``
    contract. Per-adapter overrides only where wire format demands.
    Flag-off path (the existing ``replay_reasoning_state``) is UNTOUCHED.
    """
    for i, m in enumerate(outbound):
        if isinstance(m, AIMessage):
            per_msg_state = m.additional_kwargs.get("_reasoning_state")
            if per_msg_state is not None:
                self.replay_reasoning_state(outbound[:i + 1], per_msg_state)
    return outbound
```

**NoOpAdapter must also gain the method (explicit no-op override for clarity):**

```python
class NoOpAdapter(ProviderAdapter):
    def replay_reasoning_state_multi(
        self, outbound: list[BaseMessage]
    ) -> list[BaseMessage]:
        return outbound
```

**ADAPTERS registry (lines 168-175):** No changes needed — the registry entries point to instances of the existing adapter classes; the new method lands on the ABC default or per-adapter override.

**`__all__` list (lines 186-196):** No new exports needed (method lands on existing exported classes).

---

### `app/agent/adapters/openai_gpt5.py`, `deepseek.py`, `anthropic.py`, `gemini.py` — per-provider multi-replay

**Analog:** Each provider's existing `replay_reasoning_state` implementation.

**OpenAI/DeepSeek pattern (symmetric — both read/write `additional_kwargs["reasoning_content"]`):**

From `openai_gpt5.py` lines 66-81 / `deepseek.py` lines 76-93:

```python
def replay_reasoning_state(
    self, outbound: list[BaseMessage], state: StatePayload | None
) -> list[BaseMessage]:
    if state is None:
        return outbound
    reasoning = state.get("reasoning_content")
    if reasoning is None:
        return outbound
    # Walk in reverse to find the most-recent AIMessage.
    for msg in reversed(outbound):
        if isinstance(msg, AIMessage):
            msg.additional_kwargs["reasoning_content"] = reasoning
            break
    return outbound
```

**Phase-14 multi-replay for OpenAI/DeepSeek:** The ABC generic default is sufficient for these two — `additional_kwargs["reasoning_content"]` is per-message storage already. No per-adapter override needed unless wire format requires it.

**Anthropic asymmetry callout:** `AnthropicAdapter` reads/writes `message.content` (block list), not `additional_kwargs`. The ABC generic default iterates `outbound` and calls `self.replay_reasoning_state(outbound[:i+1], per_msg_state)` per message — the single-message method handles the content-list injection correctly for each target message. Likely no per-adapter override needed.

**Gemini callout:** `GeminiAdapter` has two capture paths (bytes `thought_signature` and dict `function_call_thought_signatures` map). The ABC generic default applies per-message. Likely no per-adapter override needed.

**Revertability rule (Phase 9 precedent):** Each adapter's multi-path changes must be independently removable without touching the single path. The ABC default achieves this — removing the default method from the ABC restores flag-off byte-identity.

---

### `tests/unit/test_adapters.py` — additive multi-path tests

**Analog:** Existing per-adapter test structure (PROV-01..04, lines 53-913). Tests are numbered by provider and behavior.

**Existing test naming convention:**

```python
def test_openai_reasoning_adapter_capture_returns_payload_when_kwarg_present() -> None:
def test_openai_reasoning_adapter_replay_writes_kwarg_on_most_recent_ai_message() -> None:
def test_openai_reasoning_adapter_replay_returns_outbound_unchanged_when_state_none() -> None:
def test_openai_reasoning_adapter_capture_does_not_mutate_input_message() -> None:
```

**Additive test pattern for multi-replay (flag-on AND flag-off):**

Each new test section should follow the same format: adapter instantiated directly, synthesized AIMessages (no graph, no LLM), behavior asserted on the returned list. Mirror the existing 5-test-per-provider structure:

```python
# ─── Multi-replay (REPLAY-01) — per-adapter conformance ──────────────────────


def test_openai_reasoning_adapter_multi_replay_injects_per_message_state() -> None:
    """REPLAY-01: replay_reasoning_state_multi writes per-message _reasoning_state
    onto each in-window AIMessage's additional_kwargs["reasoning_content"]."""
    adapter = OpenAIReasoningAdapter()
    msg1 = AIMessage(
        content="first turn",
        additional_kwargs={
            "_reasoning_state": {"provider": "openai", "reasoning_content": "r1"}
        },
    )
    msg2 = AIMessage(
        content="second turn",
        additional_kwargs={
            "_reasoning_state": {"provider": "openai", "reasoning_content": "r2"}
        },
    )
    outbound = [HumanMessage(content="h"), msg1, msg2]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    # Both messages received their own reasoning_content.
    assert msg1.additional_kwargs.get("reasoning_content") == "r1"
    assert msg2.additional_kwargs.get("reasoning_content") == "r2"


def test_openai_reasoning_adapter_multi_replay_skips_messages_without_state() -> None:
    """REPLAY-01: messages with no _reasoning_state are left untouched."""
    adapter = OpenAIReasoningAdapter()
    msg_no_state = AIMessage(content="no state")
    outbound = [HumanMessage(content="h"), msg_no_state]

    result = adapter.replay_reasoning_state_multi(outbound)

    assert result is outbound
    assert msg_no_state.additional_kwargs.get("reasoning_content") is None


def test_openai_reasoning_adapter_multi_replay_flag_off_path_unchanged() -> None:
    """REPLAY-01 flag-off: existing single-message replay_reasoning_state signature
    is UNTOUCHED. Existing Test 3 (single-message replay) continues to pass
    unchanged — this test documents the non-interference contract."""
    adapter = OpenAIReasoningAdapter()
    outbound = [HumanMessage(content="h"), AIMessage(content="a"), AIMessage(content="b")]
    state = {"provider": "openai", "reasoning_content": "r"}

    result = adapter.replay_reasoning_state(outbound, state)

    # Same list spine; most-recent AIMessage got reasoning_content.
    assert result is outbound
    assert outbound[-1].additional_kwargs.get("reasoning_content") == "r"
    assert outbound[1].additional_kwargs.get("reasoning_content") is None
```

**Repeat the same 3-test block for each adapter:** DeepSeek, Anthropic, Gemini. Anthropic test must verify content-list injection per message (not additional_kwargs). Gemini test must verify the correct key (`reasoning_content` vs `thought_signature` vs `__gemini_function_call_thought_signatures__`).

**Existing 9-test harness MUST pass unchanged.** The flag-off path test above (single-message replay) is a copy of the existing Test 3 — run it to confirm no signature change. The CI step `tests/unit/test_adapters.py` must remain green with zero modifications to existing tests.

---

### `scripts/eval_agent.py` — `arm_flags` extension

**Analog:** Existing `arm_flags` dict assembly (lines 928-934):

```python
arm_flags={
    "viability_contract": env_flag("VIABILITY_CONTRACT_ENABLED"),
    "forced_commit_step": int(os.environ.get("FORCED_COMMIT_STEP", "0") or "0"),
    "parallel_tool": env_flag("PARALLEL_TOOL_EXECUTION_ENABLED"),
    "viability_threshold_override": os.environ.get("LOW_SIMILARITY_THRESHOLD_OVERRIDE")
    or None,
},
```

**Phase-14 extension — extend, do NOT replace the Phase-13 keys:**

```python
arm_flags={
    # Phase-13 DEC arm keys (unchanged)
    "viability_contract": env_flag("VIABILITY_CONTRACT_ENABLED"),
    "forced_commit_step": int(os.environ.get("FORCED_COMMIT_STEP", "0") or "0"),
    "parallel_tool": env_flag("PARALLEL_TOOL_EXECUTION_ENABLED"),
    "viability_threshold_override": os.environ.get("LOW_SIMILARITY_THRESHOLD_OVERRIDE")
    or None,
    # Phase-14 REPLAY arm keys (new)
    "replay_multi_message": env_flag("REPLAY_MULTI_MESSAGE_ENABLED"),
    "replay_content_blocks": env_flag("REPLAY_CONTENT_BLOCKS_ENABLED"),
},
```

**Smoke verification contract (D-14-02):** Before every full n=5 spend, run n=1 and assert the printed `arm_flags` dict matches the intended flag config. For R1: `{'..., 'replay_multi_message': True, 'replay_content_blocks': False}`. For R2: `{'..., 'replay_multi_message': False, 'replay_content_blocks': True}`. This is the same verification used for Phase-13 arms (see `docs/decisiveness_arm_verdicts.md` per-arm "Smoke arm_flags verification" lines).

---

### `docs/replay_arm_verdicts.md` (NEW) — verdict document structure

**Analog:** `docs/decisiveness_arm_verdicts.md` (full document)

**Mirror the DEC-05 document structure exactly:**

```
# Replay Experiment Arm Verdicts

**Role:** REPLAY-05 record — the canonical per-arm verdict document for Phase 14.

**Cross-link:** Closes out `docs/decisiveness_arm_verdicts.md` (Phase-13 record, immutable).

---

## INST-05 Falsifier Definition
[copy falsifier definition from decisiveness_arm_verdicts.md, updated with Phase-14 comparison points]

---

## Run Budget Contract
[≤4 full live matrix runs; R1 + R2 + conditional R3 + discretionary valve]

---

## R1: Multi-Message Reasoning-State Replay

**Flag config:** `REPLAY_MULTI_MESSAGE_ENABLED=1`
**DEC arm flags:** ALL UNSET (pure replay effect)

### Run Dirs
| Run | Dir |
|-----|-----|
| Smoke (n=1) | [fill] |
| Full (n=5) | [fill] |

**Smoke arm_flags verification:** [paste from run JSON]

### Per-model results

| Model | Pooled commit rate | Delta vs flag-off floor | Delta vs A2 (0.500) | omakase | refinement_cheaper | Falsifier verdict |
|---|---|---|---|---|---|---|
| openai/gpt-5-mini | | | | | | |
| openai/gpt-4o-mini (anchor) | | | | | | |
| deepseek/deepseek-reasoner | | | | | | |

**Falsifier exit code:** [fill]

**Falsifier per-scenario breakdown (pasted verbatim):**
[fill]

### Closing verdict
[fill]

---

## R2: Content-Block Preservation Through _prune_for_llm

### R2 Evidence Audit (D-14-05)
[Fill before R1/R2 live spend: audit of Phase-12/13 run-dir JSONs for list-content AIMessages — this subsection lives HERE under R2, before R2's run dirs]

[remaining structure same as R1]

---

## R3: Conditional Combo (R1 + R2)
[same structure as A4 in DEC-05]

---

## Closing Verdict

### Per-Arm Summary Table

| Arm | Flag Config | gpt-5-mini pooled | Delta vs floor | Delta vs A2 | gpt-4o-mini anchor | Falsifier exit |
|---|---|---|---|---|---|---|

### ARCH-FUT-01 Evaluation (on plateau)
[a] cumulative evidence chain  [b] ARCH-FUT-01 contingency  [c] recommendation — USER CHECKPOINT

### Explicit Closing Line
[one sentence verdict]

### Phase-15 Consequence
[winning flag config + run-dir path, OR documented plateau + user checkpoint before Phase-15 scope]
```

**Key structural differences from DEC-05:**
- Per-arm tables have THREE delta columns (pooled rate, Δ vs flag-off floor, Δ vs A2 0.500) per D-14-07
- Cross-links DEC-05 document rather than appending to it
- Has ARCH-FUT-01 section on plateau
- R2 section must include the evidence-audit result before any live spend (D-14-05)

---

## Shared Patterns

### Env-flag truthy parsing
**Source:** `app/config.py` lines 14-23
**Apply to:** All new flag reads in `graph.py` and `eval_agent.py`

```python
def env_flag(name: str) -> bool:
    """Return True if the named environment variable is set to a truthy value.
    Accepted truthy values: "1", "true", "yes", "on" (case-insensitive, whitespace-stripped).
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
```

### Flag-off byte-identity verification
**Source:** Phase-13 practice (`[skip-baseline]` commit prefix, Phase-13 commit `99d056c`)
**Apply to:** All graph changes gated by Phase-14 flags

Rule: after implementing both flags, run the full eval matrix with both flags unset. The run JSON `arm_flags` must show `replay_multi_message: false, replay_content_blocks: false` and the output must be byte-identical to the Phase-13 plateau baseline. Only then add `[skip-baseline]` to the commit message.

### Flag-gated test pattern (monkeypatch.setenv)
**Source:** `tests/unit/test_graph_forced_commit.py` lines 219-234, 366-369

```python
def test_my_flag_on_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag name appears in graph.py source (greppable check pattern)."""
    monkeypatch.setenv("REPLAY_MULTI_MESSAGE_ENABLED", "1")
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "0")
    # ... build_agent_graph after monkeypatch so flag reads the new value

def test_my_flag_off_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPLAY_MULTI_MESSAGE_ENABLED", raising=False)
    monkeypatch.setenv("FORCED_COMMIT_STEP", "0")
    monkeypatch.setenv("VIABILITY_CONTRACT_ENABLED", "0")
    monkeypatch.setenv("PARALLEL_TOOL_EXECUTION_ENABLED", "0")
```

**Greppable flag check test (mirror `test_forced_commit_step_flag_reads_at_build_time`):**

```python
def test_replay_flags_read_at_build_time() -> None:
    """REPLAY_MULTI_MESSAGE_ENABLED and REPLAY_CONTENT_BLOCKS_ENABLED must appear in graph.py."""
    import inspect
    import app.agent.graph as graph_module

    src = inspect.getsource(graph_module)
    assert "REPLAY_MULTI_MESSAGE_ENABLED" in src
    assert "REPLAY_CONTENT_BLOCKS_ENABLED" in src
```

### Adapter mutation safety invariant
**Source:** `app/agent/adapters/openai_gpt5.py` lines 61-64, `deepseek.py` lines 70-74
**Apply to:** Any new per-message injection in `replay_reasoning_state_multi`

```python
# Return a fresh dict — never alias the message's own kwargs container
# (T-09-0x-T3 mutation safety: callers may mutate the returned payload
# without affecting the originating message).
return {"provider": self.PROVIDER_KEY, "reasoning_content": reasoning}
```

### PROV-05 isolation rule
**Source:** `app/agent/adapters/openai_gpt5.py` docstring lines 27-30
**Apply to:** All per-provider adapter files

Each `<provider>.py` imports ONLY from `app.agent.adapters` base + `langchain_core` + stdlib. Never from a sibling adapter file. This ensures a per-adapter revert is a single commit touching only `__init__.py` + the one adapter file.

---

## No Analog Found

All files have close analogs. No file requires falling back to RESEARCH.md patterns.

---

## Metadata

**Analog search scope:** `app/agent/`, `tests/unit/`, `scripts/`, `docs/`, `app/config.py`
**Files scanned:** 10 (graph.py, adapters/__init__.py, openai_gpt5.py, deepseek.py, anthropic.py, gemini.py, test_adapters.py, test_graph_forced_commit.py, test_graph_parallel_tools.py, eval_agent.py, config.py, docs/decisiveness_arm_verdicts.md)
**Pattern extraction date:** 2026-06-12
