---
phase: 14-richer-state-replay-conditional
reviewed: 2026-06-12T21:52:52Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - app/agent/adapters/__init__.py
  - app/agent/graph.py
  - scripts/audit_list_content_aimessages.py
  - scripts/eval_agent.py
  - tests/unit/test_adapters.py
  - tests/unit/test_agent_graph.py
  - tests/unit/test_audit_list_content.py
  - tests/unit/test_eval_agent.py
  - docs/replay_arm_verdicts.md
findings:
  critical: 0
  warning: 4
  info: 5
  total: 9
status: issues_found
---

# Phase 14: Code Review Report

**Reviewed:** 2026-06-12T21:52:52Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Narrative Findings (AI reviewer)

## Summary

Reviewed the Phase 14 REPLAY experiment-arm changes (diff base `758bbae^`): two flags
(`REPLAY_MULTI_MESSAGE_ENABLED`, `REPLAY_CONTENT_BLOCKS_ENABLED`), the
`replay_reasoning_state_multi` ABC default, the `preserve_content_blocks` threading through
`_prune_for_llm`, the zero-spend audit script, the verdicts document, and ~900 lines of new
tests. All four per-adapter `replay_reasoning_state` implementations (openai_gpt5, deepseek,
anthropic, gemini) were also read to verify the ABC default's interaction with each.

**Flag-off byte-identity (the contractual Critical check): VERIFIED at the code level — no
Critical findings.** The diff shows the flag-off branches in `graph.py` (`_prune_for_llm`
str-collapse at lines 249-256; single-message replay at lines 371-377) are character-for-character
the pre-phase code. `env_flag` returns `False` on unset/empty. The ABC's new
`replay_reasoning_state_multi` is non-abstract and unreached when the flag is off; existing
subclasses implementing only the two abstract methods are unaffected. The REASON-06 fixture test
(`test_reason_04_noop_adapter_byte_identical_to_pre_phase8`) independently pins the flag-off
prune+replay pipeline output. The only flag-off-visible delta is two additive `arm_flags` keys in
the eval report JSON (`scripts/eval_agent.py:933-935`), which follows the Phase-13 precedent and
is verified in the verdicts doc.

The `replay_reasoning_state_multi` ABC default is mechanically correct against all four adapters:
the slice `outbound[: i + 1]` ends at the target AIMessage, every adapter's single-message replay
walks `reversed(...)` and finds exactly that message, all four mutate messages in place (the
discarded return value is contract-consistent with D-08-06), and the Anthropic per-message
idempotency guard prevents duplicate thinking blocks across repeated plan() invocations.

The substantive issues found are evidence-integrity issues, not logic bugs: in-code comments
asserting a wire-safety property the live R2 run refuted, an audit script whose hardcoded
verdict is now known-false (with unit tests pinning the falsified claim), a test whose name
contradicts its assertion, and an experiment-attribution inconsistency where R1's "+0.500
positive signal" is claimed despite R1 being structurally payload-identical to flag-off for the
run models.

## Warnings

### WR-01: `_prune_for_llm` flag-on comments assert a safety property the R2 run empirically refuted

**File:** `app/agent/graph.py:207-212, 240-247`
**Issue:** The `preserve_content_blocks` docstring says "tool_calls are still stripped
regardless" and the flag-on branch comment says "tool_calls excluded by constructor default,"
implying the preserved-content path still honors the unanswered-tool_call wire contract. The R2
live run proved this false for Responses-API models: gpt-5-mini's `AIMessage.content` block list
embeds `function_call` items, so preserving it verbatim while `_prune_for_llm` still drops the
paired pre-cutoff ToolMessages produced deterministic `400 "No tool output found for function
call"` on 10/10 episodes (documented in `docs/replay_arm_verdicts.md:300-316`). The code carries
no guard, no warning log, and no pointer to the negative verdict — a future maintainer reading
only `graph.py` would conclude the flag is safe to enable.
**Fix:** Either (a) make the flag-on branch actually safe by filtering function-call-type blocks
out of preserved list content:
```python
if preserve_content_blocks:
    content = m.content
    if isinstance(content, list):
        content = [
            b for b in content
            if not (isinstance(b, dict) and b.get("type") in ("function_call", "tool_use"))
        ]
    pruned.append(AIMessage(content=content, additional_kwargs=m.additional_kwargs))
```
or (b) at minimum replace the refuted comments with the measured finding and a cross-reference:
"KNOWN-BROKEN for Responses-API models (deterministic 400s, see docs/replay_arm_verdicts.md R2
verdict) — function_call state also lives inside the content block list, not only in
`.tool_calls`."

### WR-02: Audit script hardcodes the refuted EXPECTED-NULL verdict with no correction marker; unit tests pin the falsified claim

**File:** `scripts/audit_list_content_aimessages.py:166-209, 235-244` and `tests/unit/test_audit_list_content.py:249-272, 313-323`
**Issue:** `_ADAPTER_CLASSIFICATIONS` classifies the openai row as `content_shape: "str"` /
`str_collapse_effect: "NO-OP"`, and `_structural_analysis()` emits "VERDICT: R2 EXPECTED-NULL on
all tested cells." The verdicts doc's own POST-RUN ANNOTATION (`docs/replay_arm_verdicts.md:214-225`)
declares this row "INCORRECT for gpt-5-mini" — the gpt-5 family routes through
`OpenAIReasoningChatModel` with `use_responses_api=True`, whose `AIMessage.content` IS a block
list, and R2 measured catastrophically negative, not null. The root cause of the audit's error
(`_smoke_check_adapters` verifies adapter capture behavior, never the chat-model content shape —
the exact blind spot) is also unannotated. Re-running the script today still prints the falsified
verdict with full confidence, and `test_expected_null_verdict_for_run_models` plus
`test_all_run_models_have_str_content` actively enforce the false claim, so correcting the script
requires editing tests that look like regression protection.
**Fix:** Add a prominent post-run correction to the script's module docstring and printed verdict
(mirroring the doc's POST-RUN ANNOTATION), update the openai row to
`content_shape: "list (Responses API, gpt-5 family) / str (Chat Completions, gpt-4o-mini)"`, and
re-point the unit tests at the corrected classification. The script is the artifact a future
phase will re-run; it must not contradict the measured record.

### WR-03: Test name and docstring say "three" but the assertion is `== 2`

**File:** `tests/unit/test_audit_list_content.py:300-311`
**Issue:** `test_run_model_count_is_three` has docstring "exactly three RUN models are
classified" but asserts `result["run_model_count"] == 2`, with an inline comment explaining the
classification table has only two RUN-model rows. The test passes, but its name/docstring state
the opposite of what it verifies — anyone grepping or reading the test report will be misled, and
a future fix that makes the count genuinely 3 would "break" a test whose name says 3 is expected.
**Fix:** Rename to `test_run_model_classification_rows_is_two`, fix the docstring to match the
assertion, and move the explanatory comment into the docstring. Alternatively, split the openai
classification row into gpt-5-family and gpt-4o-mini rows (which WR-02's fix requires anyway)
and assert 3.

### WR-04: R1's "+0.500 positive signal vs flag-off floor" is structurally unattributable to the flag — internal inconsistency feeding the valve precondition

**File:** `docs/replay_arm_verdicts.md:100-144, 398-411` (mechanism: `app/agent/adapters/__init__.py:76-81`, `app/agent/graph.py:240-256`)
**Issue:** For the two adapters actually in the run matrix (OpenAI, DeepSeek), multi-replay is
provably payload-identical to flag-off: `_prune_for_llm` forwards `additional_kwargs` wholesale
across the cutoff (D-08-07), capture never strips the native `reasoning_content` key, so every
in-window AIMessage already carries its own `reasoning_content` with the flag OFF; the multi path
merely rewrites the identical value back onto each message (`openai_gpt5.py:79`,
`deepseek.py:91`). Yet the doc reports R1 as "+0.500 delta vs flag-off floor (0.000)" and
classifies it as "positive signal," which is then load-bearing in the valve precondition check
(line 398: "Best replay arm (R1) ... showed positive signal — SATISFIED"). A wire-no-op flag
cannot cause a +0.500 delta; the inconsistency traces to the floor proxy — the "flag-off floor"
is A1, a Phase-13 arm that ran with `VIABILITY_CONTRACT_ENABLED=1` (a prompt-modifying flag), not
a true flag-off control (the doc acknowledges no fresh control was run, line 49). The doc's own
conclusion ("R1 adds nothing over A2, delta ±0.000") is consistent with the no-op reading, but
the "+0.500 positive signal" framing contradicts it within the same document.
**Fix:** Add a note to the R1 mechanism/closing-verdict sections stating that R1 is structurally
payload-identical to flag-off for the OpenAI/DeepSeek wire shapes (reasoning_content already
survives pruning via D-08-07), so the +0.500 vs the A1-proxy floor reflects floor-proxy error or
run drift, not a flag effect — and that the valve precondition's "R1 positive signal" rests on
that proxy. This strengthens (not weakens) the doc's plateau conclusion and matters for the
Phase-15 scoping checkpoint that consumes this record.

## Info

### IN-01: `RUN_MODELS` constant is dead code

**File:** `scripts/audit_list_content_aimessages.py:63-67`
**Issue:** `RUN_MODELS` is defined and never referenced anywhere in the script or its tests
(verified by grep); the analysis uses `_ADAPTER_CLASSIFICATIONS` instead.
**Fix:** Delete it, or use it to drive/validate the `run_model` fields in the classification
table so the two can't drift.

### IN-02: Hardcoded line references invalidated by this same phase's edit

**File:** `scripts/audit_list_content_aimessages.py:6, 32, 467` and `docs/replay_arm_verdicts.md:152, 202`
**Issue:** References to "graph.py:232" and "graph.py:228-235" point at the pre-Phase-14 location
of the str() collapse; the Phase-14 diff moved it to ~lines 249-256.
**Fix:** Replace line numbers with a stable anchor ("the pre-cutoff str() collapse branch in
`_prune_for_llm`") in the script; the doc references can stand as historical (they described the
code as it was at audit time) but a parenthetical "(pre-Phase-14 line numbers)" would prevent
confusion.

### IN-03: "Byte-Identity Verification: PASS" claim is stronger than the cited evidence

**File:** `docs/replay_arm_verdicts.md:31-42`
**Issue:** The section titled "Flag-Off Floor and Byte-Identity Verification" declares "Flag-off
path is byte-identical to Phase-13 plateau" but the only evidence shown is the `arm_flags` dict
from a scripted n=1 smoke — which verifies flag VALUES, not payload bytes. The actual
byte-identity guarantee comes from the code diff (flag-off branches unchanged) and the REASON-06
fixture test in `tests/unit/test_agent_graph.py`.
**Fix:** Cite the REASON-06 fixture test and the flag-off-branch diff inspection alongside the
arm_flags smoke, so the PASS claim is backed by evidence matching its strength.

### IN-04: `_smoke_check_adapters` docstring says "four adapters," imports three

**File:** `scripts/audit_list_content_aimessages.py:255-267`
**Issue:** Docstring: "Import the four adapters and verify the content-shape assumption." Only
`AnthropicAdapter`, `DeepSeekReasonerAdapter`, and `OpenAIReasoningAdapter` are imported;
`GeminiAdapter` is never checked, so a Gemini content-shape regression would go undetected by the
smoke check while the classification table still asserts Gemini is str-content.
**Fix:** Import and check `GeminiAdapter` (capture returns None on a plain str-content message),
or correct the docstring to "three."

### IN-05: No graph-level test that the flag-OFF path avoids `replay_reasoning_state_multi`

**File:** `tests/unit/test_agent_graph.py:1937-2003`
**Issue:** `test_replay_multi_message_flag_on_routes_through_multi_replay` asserts flag-on calls
multi and not single, but there is no inverse spy test asserting flag-off calls single and never
multi. The flag-off contract is currently protected only indirectly (REASON-06 fixture pins
prune+replay output for the no-state path).
**Fix:** Add the paired test: unset `REPLAY_MULTI_MESSAGE_ENABLED`, build the graph with the same
spy adapter, assert `single_calls >= 1` and `multi_calls == 0`.

---

_Reviewed: 2026-06-12T21:52:52Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
