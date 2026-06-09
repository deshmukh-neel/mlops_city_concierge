---
phase: 09-per-provider-state-preservation-implementations
reviewed: 2026-06-05T00:00:00Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - app/agent/adapters/__init__.py
  - app/agent/adapters/anthropic.py
  - app/agent/adapters/deepseek.py
  - app/agent/adapters/gemini.py
  - app/agent/adapters/openai_gpt5.py
  - app/llm_factory.py
  - configs/eval_baselines/refinement_cheaper.json
  - configs/eval_matrix_refinement.yaml
  - scripts/eval_agent.py
  - scripts/probe_gpt5_capture.py
  - tests/integration/test_reasoning_state_roundtrip.py
  - tests/unit/agent/test_adapters.py
  - tests/unit/test_adapters.py
  - tests/unit/test_eval_matrix.py
  - tests/unit/test_llm_factory.py
findings:
  critical: 1
  blocker: 1
  warning: 5
  info: 4
  total: 10
status: issues_found
---

# Phase 9: Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Phase 9 wires four real `ProviderAdapter` implementations (OpenAI gpt-5, DeepSeek reasoner, Anthropic Claude, Gemini 3) against the Phase 8 contract + `ADAPTERS` registry. The four wire shapes are intentionally asymmetric and the implementations are largely correct: the `AnthropicAdapter` carries an explicit live-probe idempotency fix; `GeminiAdapter` supports both the synthetic fixture path and the live lcgg-4.x function-call map path; mutation-safety contracts (T-09-*-T3) are well-tested.

The adversarial review surfaces one BLOCKER (live-integration coverage gap that mirrors the four Anthropic bugs caught in the Wave-3 probe), several WARNINGs covering stale type contracts, a third-path round-trip lossiness in `GeminiAdapter`, and a registry-invariant test that allows partial drift. The code shipped through Wave 1-4 already accounted for most of the "obvious" defects via the live probes, so the remaining surface area is in:

1. `scripts/eval_agent.py` — `LlmProvider` type Literal and `resolve_chat_model` hardcoded dict are out of sync with `SUPPORTED_PROVIDERS` since Phase 9 added `"anthropic"`.
2. `app/llm_factory.py:OpenAIReasoningChatModel` — the gpt-5 reasoning-lift subclass has zero unit-test coverage (mirrors the live-bug pattern in Anthropic where unit tests passed but live calls 400'd).
3. `app/agent/adapters/gemini.py` — Path 3 (per-tool-call surfacing) capture + replay is asymmetric: capture pulls bytes from `tool_calls[i]` but replay writes back to `additional_kwargs`, dropping the tool-call binding entirely.

## Critical Issues

### CR-01: `scripts/eval_agent.py:resolve_chat_model` crashes with KeyError on `--llm-provider anthropic`

**File:** `scripts/eval_agent.py:241`
**Severity:** BLOCKER
**Issue:**
Phase 9 PROV-03 added `"anthropic"` to `SUPPORTED_PROVIDERS`. The argparse `--llm-provider` choices were correctly fixed to derive from `SUPPORTED_PROVIDERS` (lines 168-180), but `resolve_chat_model()` was NOT updated. When called with `provider="anthropic"` and `chat_model=None`, control flows to line 241:

```python
env_var = {"deepseek": "DEEPSEEK_MODEL", "kimi": "MOONSHOT_MODEL"}[provider]
```

This dict-lookup raises `KeyError: 'anthropic'` — an unhandled exception that crashes with a non-actionable traceback, NOT the user-friendly `ValueError(f"No chat model for {provider}: ...")` on line 244.

The matrix runner (`configs/eval_matrix_refinement.yaml`) always passes an explicit `--chat-model claude-sonnet-4-6` so this defect is masked in CI. But a direct invocation (`python scripts/eval_agent.py --llm-provider anthropic`) — exactly the pattern used in the 09-03 live probes that found four bugs — crashes immediately. Same root-cause pattern as the original argparse `choices` drift (memory `project_eval_multi_turn_threading_bug` plus the 09-03 SUMMARY).

Additionally, the `LlmProvider` Literal type alias at line 49 is now stale:
```python
LlmProvider = Literal["openai", "gemini", "deepseek", "kimi", "scripted"]
```
This omits `"anthropic"`. Type-checkers (mypy) running on `resolve_chat_model(provider="anthropic", ...)` will report a Literal-mismatch error or — worse — silently accept it because `provider` is annotated `LlmProvider` and runtime never enforces the type. The argparse `choices=list(SUPPORTED_PROVIDERS)` now feeds string values that violate the function signature.

**Fix:**
```python
# Line 49 — Literal type follows SUPPORTED_PROVIDERS
LlmProvider = Literal["openai", "gemini", "deepseek", "kimi", "anthropic", "scripted"]

# Lines 236-245 — handle anthropic + future-proof via settings lookup
def resolve_chat_model(provider: LlmProvider, chat_model: str | None) -> str:
    if chat_model and chat_model.strip():
        return chat_model.strip()
    if provider == "scripted":
        return "scripted-default"
    settings = get_settings()
    if provider == "openai":
        return settings.openai_chat_model
    if provider == "gemini":
        return settings.gemini_chat_model
    env_var_map = {
        "deepseek": "DEEPSEEK_MODEL",
        "kimi": "MOONSHOT_MODEL",
        "anthropic": "ANTHROPIC_MODEL",
    }
    if provider not in env_var_map:
        raise ValueError(f"No chat model resolver for {provider}: pass --chat-model")
    env_var = env_var_map[provider]
    model = os.getenv(env_var)
    if not model:
        raise ValueError(f"No chat model for {provider}: pass --chat-model or set {env_var}")
    return model
```

Same fix prevents this pattern from recurring when PROV-FUT providers join `SUPPORTED_PROVIDERS`.

---

## Warnings

### WR-01: `OpenAIReasoningChatModel` reasoning-lift subclass has zero unit-test coverage

**File:** `app/llm_factory.py:59-138`
**Severity:** WARNING
**Issue:**
`OpenAIReasoningChatModel._lift_reasoning_blocks` (lines 92-118) is the sole bridge between the OpenAI Responses-API content-block format and the documented `additional_kwargs["reasoning_content"]` contract that `OpenAIReasoningAdapter` reads. If the lift function fails or silently no-ops (e.g., because the content shape changes in a future `langchain-openai` version, or because tool-call responses come back with `content` as `str` and bypass the `isinstance(content, list)` guard on line 109), the adapter sees an empty `additional_kwargs` and silently returns `None` on every capture — the entire PROV-01 wire breaks with no test failure.

There is NO unit test that verifies:
- `_lift_reasoning_blocks` copies blocks into `additional_kwargs["reasoning_content"]` when content is a list with reasoning blocks
- The lift is a SHALLOW COPY (mutation of `additional_kwargs["reasoning_content"]` does not reach back into `msg.content`)
- The lift is a no-op when content is a `str` (Chat Completions shape) — does not crash
- `_generate` and `_agenerate` both wire through `_lift_reasoning_blocks` (sync + async parity)

This is the EXACT pattern that bit Anthropic: unit tests passed against synthetic AIMessages (the conformance `OpenAIReasoningAdapter` sibling test plants `additional_kwargs["reasoning_content"]` directly), but the LIVE wire shape was never exercised. PROV-01 SHIPPED-WITH-GAP at 0.4 vs ≥0.6 — the unit-tested adapter path itself works, but the gpt-5-mini wire path is not validated.

The probe script `scripts/probe_gpt5_capture.py` was the live-shape check but it has no assertions and the artifact is a one-shot informational dump.

**Fix:**
Add a unit-test file (or extend `tests/unit/test_llm_factory.py`) with at least these tests:

```python
def test_openai_reasoning_chat_model_lifts_reasoning_blocks_into_kwargs():
    from app.llm_factory import OpenAIReasoningChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    msg = AIMessage(content=[
        {"type": "reasoning", "summary": "thought process"},
        {"type": "text", "text": "the answer"},
    ])
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    lifted = OpenAIReasoningChatModel._lift_reasoning_blocks(result)
    out_msg = lifted.generations[0].message
    assert out_msg.additional_kwargs["reasoning_content"] == [
        {"type": "reasoning", "summary": "thought process"},
    ]

def test_openai_reasoning_chat_model_lift_is_noop_on_str_content():
    """Tool-call responses or refusals may carry str content; lift must not crash."""
    msg = AIMessage(content="plain string response")
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    lifted = OpenAIReasoningChatModel._lift_reasoning_blocks(result)
    assert "reasoning_content" not in lifted.generations[0].message.additional_kwargs

def test_openai_reasoning_chat_model_lift_shallow_copies_list():
    """T-09-01-T3 mutation safety at the lift boundary."""
    block = {"type": "reasoning", "summary": "x"}
    msg = AIMessage(content=[block])
    result = ChatResult(generations=[ChatGeneration(message=msg)])
    lifted = OpenAIReasoningChatModel._lift_reasoning_blocks(result)
    out_msg = lifted.generations[0].message
    # Mutate the lifted list; the original content list must be unchanged.
    out_msg.additional_kwargs["reasoning_content"].append({"type": "reasoning", "summary": "TAMPERED"})
    assert len(msg.content) == 1
```

---

### WR-02: `GeminiAdapter` Path 3 (tool_calls surfacing) capture+replay round-trip is LOSSY

**File:** `app/agent/adapters/gemini.py:184-200`
**Severity:** WARNING
**Issue:**
The Path 3 capture path scans `message.tool_calls` for a `thought_signature` key and wraps the bytes value in `{"provider": "gemini", "thought_signature": <bytes>}` — the same shape Path 2 produces. But `replay_reasoning_state` writes this bytes value back to `additional_kwargs["thought_signature"]` (lines 230-234), NOT back to any tool_call.

If lcgg 4.x's outbound serializer reads `tool_calls[i]["thought_signature"]` (Path 3's source-of-truth) but NOT `additional_kwargs["thought_signature"]` (Path 3's replay target), the round-trip silently drops the signature on the wire. The docstring acknowledges "PROV-04 ships single-signature; per-call alignment is deferred to a future v2.2 / Phase 10 follow-up" — but that's a design defer for the multi-signature case, NOT for the single-signature case which currently round-trips to the wrong key.

This is masked by:
1. The conformance harness test `test_reason_02_gemini_real_adapter` only exercises Path 2 (kwargs primary).
2. The unit test `test_gemini_adapter_capture_returns_payload_when_tool_calls_carry_signature` asserts capture only — never that the captured payload round-trips to a wire-correct shape for Gemini.
3. The live probe data shows real Gemini traffic uses Path 1 (`__gemini_function_call_thought_signatures__`), so Path 3 is currently dormant in production.

Path 3 is documented as a "lcgg-version surfacing variant" — so it might silently NEVER fire in production with the pinned `langchain-google-genai>=4.0.0,<5.0.0`. If that's true, Path 3 is dead code. If it does fire, the replay is wrong.

**Fix:**
Either:
1. Add a unit test that exercises Path 3 round-trip end-to-end and validates the wire-correct replay target. If lcgg only reads the tool_call key, replay must write back to the tool_call:
   ```python
   if isinstance(signature, bytes):
       # Path 3 replay: write back to tool_calls if that's where it was captured.
       # Track the source key in the payload so replay knows where to put it.
       msg.additional_kwargs[_SYNTHETIC_FIXTURE_KEY] = signature
       # And/or rewrite tool_calls[0]:
       if msg.tool_calls:
           msg.tool_calls[0]["thought_signature"] = signature
   ```
2. Or REMOVE Path 3 entirely if it's a speculative path that never fires in practice (the live probe in Wave 4 should have confirmed this). Dead code in a security-adjacent context (state round-trip drives provider 400s) is a regression hazard.

The current shape — capture path 3 exists, replay path doesn't match — is the worst of both worlds.

---

### WR-03: `test_adapters_registry_keys_match_supported_providers` provider-classification is fragile to spelling

**File:** `tests/unit/test_adapters.py:126-134`
**Severity:** WARNING
**Issue:**
The loop "providers PROV-01..04 did NOT swap stay on NoOpAdapter" iterates `SUPPORTED_PROVIDERS` and `continue`s for any provider in the hardcoded tuple `("openai", "deepseek", "anthropic", "gemini")`. If a Phase 9.5/10 PR adds a NEW reasoning provider (e.g., `"claude_v6"`) WITHOUT updating this tuple, the test would falsely assert `isinstance(ADAPTERS["claude_v6"], NoOpAdapter)` even when the provider is correctly wired to a real adapter — causing a confusing test failure.

Conversely, if a typo'd entry is added to `ADAPTERS` (e.g., `"openai2": NoOpAdapter()`), the test passes the registry equality but the typo provider never gets exercised.

The registry-equality assertion (`set(ADAPTERS.keys()) == set(SUPPORTED_PROVIDERS)`) is the actual invariant — the per-provider isinstance checks are a "configuration snapshot" that risks drift.

**Fix:**
Refactor to a single source of truth for the (provider, adapter-class) mapping, then derive both the test and ADAPTERS from it:
```python
_PROVIDER_TO_ADAPTER_CLASS = {
    "openai": OpenAIReasoningAdapter,
    "deepseek": DeepSeekReasonerAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "kimi": NoOpAdapter,
    "scripted": NoOpAdapter,
}

def test_adapters_registry_matches_expected_mapping():
    assert set(ADAPTERS.keys()) == set(SUPPORTED_PROVIDERS)
    for provider, expected_cls in _PROVIDER_TO_ADAPTER_CLASS.items():
        assert isinstance(ADAPTERS[provider], expected_cls), (
            f"ADAPTERS[{provider!r}] expected {expected_cls.__name__}, "
            f"got {type(ADAPTERS[provider]).__name__}"
        )
```

This catches BOTH "new provider added but adapter not wired" AND "typo'd ADAPTERS entry" failure modes.

---

### WR-04: `AnthropicAdapter.replay_reasoning_state` silently discards captured payload when target has different thinking blocks

**File:** `app/agent/adapters/anthropic.py:143-151`
**Severity:** WARNING
**Issue:**
The idempotency fix (correctly added per the live-probe 400 in Wave 3) detects "any thinking block already present" and skips replay. But the check is `any(isinstance(b, dict) and b.get("type") == "thinking" for b in existing)` — it does NOT compare the existing block's signature against the captured payload's signature.

Scenario where this matters:
1. Turn 0: Anthropic returns AIMessage A with `thinking_blocks=[{signature: "sig_A", ...}]`.
2. Plan loop captures A's state into `additional_kwargs["_reasoning_state"]`.
3. The graph reducer somehow injects a DIFFERENT AIMessage B (carrying `thinking_blocks=[{signature: "sig_B", ...}]`) as the most-recent AIMessage — e.g., a revision step rewrote content.
4. Turn 1 begins; replay walks reverse, finds B has thinking blocks, skips replay.
5. The captured `sig_A` payload is silently discarded.

Whether this can actually happen depends on the agent loop's invariants (which the review can't fully verify from the adapter alone). The conformance test `test_anthropic_adapter_replay_is_idempotent_when_thinking_blocks_already_present` only exercises the case where the existing block IS the original. A defensive improvement: log a warning when replay skips because target already has thinking blocks BUT the signatures don't match.

**Fix:**
At minimum, add a debug log so the silent-discard path is observable:
```python
if already_has_thinking:
    existing_sigs = [
        b.get("signature") for b in existing
        if isinstance(b, dict) and b.get("type") == "thinking"
    ]
    captured_sigs = [b.get("signature") for b in blocks]
    if set(existing_sigs) != set(captured_sigs):
        import logging
        logging.getLogger(__name__).debug(
            "AnthropicAdapter.replay: target AIMessage already has thinking "
            "blocks with signatures %r but captured payload carried %r — "
            "skipping replay (target wins; captured payload discarded).",
            existing_sigs, captured_sigs,
        )
    break
```

This trades zero behavior change for observability so a future bug-hunt has telemetry.

---

### WR-05: `OpenAIReasoningChatModel` lift does NOT shallow-copy individual reasoning block dicts

**File:** `app/llm_factory.py:111-117`
**Severity:** WARNING
**Issue:**
The list-comprehension `reasoning_blocks = [block for block in content if ...]` followed by `msg.additional_kwargs["reasoning_content"] = list(reasoning_blocks)` shallow-copies the LIST but the individual block DICTS are shared references with `msg.content`. A downstream consumer that mutates `msg.additional_kwargs["reasoning_content"][0]["summary"]` would mutate the same dict still living in `msg.content`.

The docstring says "The list is shallow-copied so downstream mutation of the additional_kwargs entry cannot reach back into `content`" — but that's wrong by exactly one level. List shallow-copy preserves the inner-dict aliasing.

Compare to `AnthropicAdapter.capture_reasoning_state` (line 95), which correctly does `[dict(b) for b in thinking_blocks]` — per-block dict copy.

The actual exploit is narrow: `OpenAIReasoningAdapter.capture_reasoning_state` then wraps `reasoning` again into a new dict, and adapters downstream don't mutate the inner blocks. But this is a latent footgun that the docstring promises has been mitigated. Defect-in-comment + defect-in-code is hazardous: the next maintainer reads the docstring and assumes safety.

**Fix:**
Either fix the code to match the documented behavior:
```python
reasoning_blocks = [
    dict(block)  # per-block shallow copy
    for block in content
    if isinstance(block, dict) and block.get("type") == "reasoning"
]
if reasoning_blocks:
    msg.additional_kwargs["reasoning_content"] = reasoning_blocks
```
Or fix the docstring to match the code (and accept the alias). The first is preferred — it costs O(blocks) per response and removes a documented contract violation.

---

## Info

### IN-01: `LlmProvider` Literal type alias unused throughout `eval_agent.py`

**File:** `scripts/eval_agent.py:49`
**Issue:**
`LlmProvider = Literal["openai", "gemini", "deepseek", "kimi", "scripted"]` is defined and used as a type annotation on `resolve_chat_model` and `build_eval_llm`. But argparse populates `args.llm_provider` as a plain `str` driven from `SUPPORTED_PROVIDERS`, and the runtime values are never type-checked. The alias's only effect is to mislead mypy and casual readers about which strings are valid. CR-01 above proposes the fix.

**Fix:** Drop `LlmProvider` entirely and use `str` (or import it from a shared location), OR keep it but make it the single source of truth that `SUPPORTED_PROVIDERS` derives from. Two sources of truth (Literal + tuple) is exactly the drift this whole phase was supposed to fix at the argparse layer.

---

### IN-02: `configs/eval_baselines/refinement_cheaper.json` carries em-dash and ellipsis characters in `_observations`

**File:** `configs/eval_baselines/refinement_cheaper.json:4,7,75,143,211,279`
**Issue:**
The `_observations` strings contain Unicode em-dashes (U+2014) and an ellipsis or two. Per CLAUDE.md, no emoji rule applies; these aren't emojis but the JSON data file is committed and may be parsed by tools that expect ASCII. Not a correctness issue (Python JSON parsing handles UTF-8 fine), but a stylistic note: future reviewers grep'ing for verdicts may be tripped up by non-ASCII punctuation. Many of these strings are also very long single-line entries that hurt git-diff readability. This is not a blocker but worth flagging.

**Fix:** Optional — normalize punctuation to ASCII (e.g., `--` instead of em-dash) or accept Unicode in observation strings. Either is fine; the file currently round-trips through `json.dumps`/`json.loads` cleanly.

---

### IN-03: `scripts/probe_gpt5_capture.py` writes outside the source tree to `.planning/` via hardcoded relative pathing

**File:** `scripts/probe_gpt5_capture.py:39-47`
**Issue:**
The probe computes `REPO_ROOT = Path(__file__).resolve().parents[1]` and writes the artifact to `.planning/phases/09-.../09-PROV-01-PROBE.md`. This is fine for a one-shot script that lives in the repo. But the script is committed and `things_to_watch_for` flagged it for lighter review. The pathing assumes the script is exactly two directory levels below the repo root — accurate today, would silently break if the script moves. The artifact path is also hardcoded to phase 09, so if this script is ever re-used in Phase 10 the artifact would land in the wrong directory.

**Fix:** This is acceptable for a one-shot. If anyone is tempted to re-run the script for a future phase, they should fork or parameterize the artifact path via `--output`. Not blocking.

---

### IN-04: `probe_gpt5_capture.py` defensive secret-redaction pattern misses non-`sk-` API key formats

**File:** `scripts/probe_gpt5_capture.py:63`
**Issue:**
`_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")` only matches the legacy OpenAI `sk-` format. OpenAI now also issues `sk-proj-...`, `sk-svcacct-...` (covered by the existing pattern since it allows any chars after `sk-`), and Anthropic uses `sk-ant-api03-...` (also covered). But other providers' keys do NOT start with `sk-`:
- Google API keys typically `AIza...` (39 chars, base64-ish)
- DeepSeek/Moonshot use opaque tokens
- Anthropic admin tokens may differ

The probe only invokes OpenAI so the narrow regex is OK in practice. But the comment and final defensive scan suggest a generalized redaction; readers may misread it as a comprehensive guard. Worth a comment clarifying scope.

**Fix:** Either expand the pattern OR add a comment "OpenAI-key-only redaction; the probe never touches other providers' keys". Optional.

---

## Notes on items checked but found clean

- **D-09-07 import isolation**: each `app/agent/adapters/<provider>.py` imports only from `app.agent.adapters` base + `langchain_core` + stdlib. No sibling-adapter imports. Verified.
- **Mutation safety for capture (T-09-*-T3)**: each adapter returns a fresh dict and (for Anthropic + Gemini Path 1) shallow-copies nested mutable containers. Unit tests exercise the tamper-then-compare contract.
- **`temp=1.0` invariant per CLAUDE.md**: every place that hardcodes temperature does so with documented rationale (Anthropic clamp, Kimi clamp, DeepSeek reasoner test). No drift.
- **`app` poetry editable-install**: no `sys.path.insert` / `REPO_ROOT` bootstrap in any of the reviewed files. `scripts/probe_gpt5_capture.py` uses `REPO_ROOT` only for `.env` loading and artifact pathing, NOT for sys.path.
- **gpt-4o-mini stays on plain `ChatOpenAI`**: `_is_openai_reasoning_model` dispatch is correctly scoped to `chat_model.startswith("gpt-5")`. The v2.0 anchor path is preserved.
- **No emoji in source files**: grepped, none found.
- **Phase 8 invariant `test_adapters_registry_keys_match_supported_providers`**: asserts all four PROV adapters are swapped and the remaining (`kimi`, `scripted`) stay on NoOp. WR-03 flags fragility but the current intent is correct.
- **Live-integration coverage for Gemini real-wire path**: tests 8-12 in `tests/unit/test_adapters.py` exercise the lcgg-4.x function-call map shape end-to-end. This is the gold-standard pattern WR-01 calls for on the OpenAI side.
- **`OpenAIReasoningAdapter`, `DeepSeekReasonerAdapter`, `AnthropicAdapter` reverse-walk replay**: all correctly break on the first AIMessage encountered, leaving earlier AIMessages untouched. Unit-tested.
- **`ScriptedChatModel` fresh-AIMessage fix (CR-02)**: still in place; not regressed by Phase 9.

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
