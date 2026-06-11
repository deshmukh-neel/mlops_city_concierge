---
phase: 09-per-provider-state-preservation-implementations
plan: 04
subsystem: agent
tags: [adapter, gemini, reasoning-state, experimental, prov-04]
verdict: SHIPPED-STRUCTURAL
dependency_graph:
  requires: [09-03 AnthropicAdapter, Phase 8 ProviderAdapter contract, langchain-google-genai>=4.2.0]
  provides: [GeminiAdapter, ADAPTERS["gemini"] real-adapter wiring, real-lcgg-4.x wire-shape coverage]
  affects: [ADAPTERS registry consolidation to explicit literal — all 4 PROV adapters wired]
tech_stack:
  added: []
  patterns: [explicit-dict-literal registry consolidation (Option B); dual-path adapter for synthetic-fixture + live-wire wire shapes]
key_files:
  created:
    - app/agent/adapters/gemini.py
    - .planning/phases/09-per-provider-state-preservation-implementations/09-04-SUMMARY.md
  modified:
    - app/agent/adapters/__init__.py
    - tests/unit/test_adapters.py
    - tests/unit/agent/test_adapters.py
    - tests/integration/test_reasoning_state_roundtrip.py
    - configs/eval_matrix_refinement.yaml
    - tests/unit/test_eval_matrix.py
decisions:
  - D-09-08 PROV-04 EXPERIMENTAL — no merge gate; empirical median logged-not-gated.
  - D-09-09 Phase 8 fixture targets `additional_kwargs["thought_signature"]: bytes`; real lcgg 4.x surfaces signatures on `additional_kwargs["__gemini_function_call_thought_signatures__"]: dict[tool_call_id, base64_str]` (live-probe-confirmed 2026-06-05).
  - D-09-07 Option B registry consolidation — dict-comp + 4 per-cell mutations replaced by explicit dict literal at one site.
  - User-approved Option B (2026-06-05) — defer empirical n=5 measurement to Phase 10 BASE-01 due to exhausted OpenAI embeddings quota.
metrics:
  duration_minutes: ~45
  completed_date: 2026-06-05
  task_count: 1 task (Task 1 implementation; Task 2 empirical-matrix checkpoint deferred per user-approved Option B)
  files_created: 2
  files_modified: 6
  commits: 3
  unit_tests_added: 13 (GeminiAdapter coverage — synthetic + real lcgg 4.x wire shapes)
  integration_tests_added: 1 (test_reason_02_gemini_real_adapter conformance sibling)
  conformance_tests_total: 9 (all pass)
---

# Phase 9 Plan 4: Gemini 3 Experimental Adapter Summary

GeminiAdapter ships structurally for PROV-04 with dual-path coverage for the synthetic Phase 8 fixture AND the real lcgg 4.x wire shape; ADAPTERS registry consolidated to explicit literal at one site with all 4 reasoning providers swapped off NoOp.

## Outcome (verdict: SHIPPED-STRUCTURAL)

D-09-08's PROV-04 charter — "no merge gate; bytes payload survives `graph.ainvoke` end-to-end via the conformance harness" — is delivered. The state-preservation half of the PROV-04 contract is complete; the empirical/decisiveness half (refinement_minimal_edit median) is informational-only per D-09-08 and is deferred to Phase 10 BASE-01 because the OpenAI embeddings quota is exhausted (semantic_search 429s on every matrix cell regardless of LLM provider, per the same blocker that capped PROV-03 at n=1). Carry-forward is documented in REQUIREMENTS.md.

Three atomic commits per `feedback_small_focused_commits`:

| Order | Hash      | Type     | Subject                                                                |
| ----- | --------- | -------- | ---------------------------------------------------------------------- |
| 1     | `10e88b9` | `feat`   | GeminiAdapter (initial bytes-shape coverage) + ADAPTERS registry consolidation (PROV-04) |
| 2     | `bf6ff83` | `feat`   | GeminiAdapter handles real lcgg 4.x wire shape (PROV-04)               |
| 3     | `17e9187` | `chore`  | add gemini-3.1-pro-preview cell to refinement matrix (PROV-04)         |

## Implementation summary

### `app/agent/adapters/gemini.py` — `GeminiAdapter`

- **Asymmetry vs PROV-01 / PROV-02 / PROV-03 adapters:** Gemini's reasoning payload is a `dict[tool_call_id, base64_str]` on a NAMESPACED kwarg key (`__gemini_function_call_thought_signatures__`), NOT a single str (`reasoning_content` — PROV-01 / PROV-02) and NOT a heterogeneous block list (`thinking_blocks` — PROV-03). The Phase 8 fixture (`thought_signature: bytes`) was IDEALIZED — it never appears in real lcgg 4.x output.
- **Three capture paths in priority order:**
  1. **Real lcgg 4.x wire shape** (primary, live-probe-confirmed): `additional_kwargs["__gemini_function_call_thought_signatures__"]` → captured as `{"provider": "gemini", "function_call_thought_signatures": <dict>}`.
  2. **Phase 8 fixture / synthetic shape** (REASON-02 conformance): `additional_kwargs["thought_signature"]` bytes → captured as `{"provider": "gemini", "thought_signature": <bytes>}` matching `FOUR_SHAPE_PAYLOADS[3]`.
  3. **Per-tool-call fallback** (CONTEXT.md `<specifics>` PROV-04 variant): scan `message.tool_calls[i]["thought_signature"]` for bytes; captured as the synthetic-shape payload.
- **Replay** writes back to whichever wire key the captured payload identifies, so live traffic round-trips through Path 1 (lcgg outbound serializer reads the same key) AND the conformance harness still round-trips through Path 2 byte-for-byte.
- **No base64 encoding/decoding inside the adapter** — lcgg owns that boundary at `_convert_message_to_dict` time. Avoids the historical foot-gun documented in memory `project_gemini3_thought_signatures` (W10-era static-bypass hack encoded bytes as base64 strings and lost the round-trip).
- **D-09-07 import isolation:** `from app.agent.adapters import ProviderAdapter, StatePayload` + `langchain_core.messages` only. No sibling-adapter imports. The lcgg internal key (`__gemini_function_call_thought_signatures__`) is pinned as a string literal with a comment cross-referencing the library source — avoids a circular-import / coupling to lcgg's internal symbol while still tracking its stable public-ish wire format.

### `app/agent/adapters/__init__.py` — Option B consolidation

Pre-Plan-09-04 state (Option A — dict-comp + 4 per-cell mutations after Plans 09-01..09-03):

```python
ADAPTERS: dict[str, ProviderAdapter] = {p: NoOpAdapter() for p in SUPPORTED_PROVIDERS}
from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter
ADAPTERS["openai"] = OpenAIReasoningAdapter()
from app.agent.adapters.deepseek import DeepSeekReasonerAdapter
ADAPTERS["deepseek"] = DeepSeekReasonerAdapter()
from app.agent.adapters.anthropic import AnthropicAdapter
ADAPTERS["anthropic"] = AnthropicAdapter()
```

Post-Plan-09-04 state (Option B — explicit dict literal, per PATTERNS.md §`app/agent/adapters/__init__.py:121` recommendation once 4+ entries are non-NoOp):

```python
from app.agent.adapters.anthropic import AnthropicAdapter
from app.agent.adapters.deepseek import DeepSeekReasonerAdapter
from app.agent.adapters.gemini import GeminiAdapter
from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter

ADAPTERS: dict[str, ProviderAdapter] = {
    "openai": OpenAIReasoningAdapter(),       # PROV-01
    "gemini": GeminiAdapter(),                 # PROV-04 (EXPERIMENTAL)
    "deepseek": DeepSeekReasonerAdapter(),     # PROV-02
    "kimi": NoOpAdapter(),                     # PROV-FUT-02 (library-blocked)
    "anthropic": AnthropicAdapter(),           # PROV-03
    "scripted": NoOpAdapter(),                 # CI/test only
}
```

D-09-07 isolation rule is preserved — `__init__.py` IS the only file allowed to import across sibling adapter files; the per-provider `<provider>.py` modules still import only from `app.agent.adapters` base + `langchain_core` + stdlib.

### Phase-8 invariant test tightened (`tests/unit/agent/test_adapters.py`)

`test_adapters_registry_keys_match_supported_providers` now asserts all four reasoning providers wired off NoOp:

```python
assert isinstance(ADAPTERS["openai"], OpenAIReasoningAdapter)
assert isinstance(ADAPTERS["deepseek"], DeepSeekReasonerAdapter)
assert isinstance(ADAPTERS["anthropic"], AnthropicAdapter)
assert isinstance(ADAPTERS["gemini"], GeminiAdapter)
for provider in SUPPORTED_PROVIDERS:
    if provider in ("openai", "deepseek", "anthropic", "gemini"):
        continue
    assert isinstance(ADAPTERS[provider], NoOpAdapter), ...
```

Remaining NoOp providers post-PROV-04: `kimi` (PROV-FUT-02 library-blocked) and `scripted` (CI-only, never has reasoning state).

### Matrix YAML cell + cell-count test

`configs/eval_matrix_refinement.yaml` now has 6 cells (5th → 6th):

```yaml
# Phase 9 / D-09-12 / PROV-04 EXPERIMENTAL (no merge gate per D-09-08).
# Empirical refinement_minimal_edit median is LOGGED-NOT-GATED — the
# critique-loop fix is DEFERRED per project_w10_migration_necessary_not_sufficient.
- provider: gemini
  model: gemini-3.1-pro-preview
  env:
    REFINEMENT_STRUCTURED_PLAN_ENABLED: "true"
```

`tests/unit/test_eval_matrix.py::test_repo_eval_matrix_refinement_yaml_loads_via_load_eval_matrix` updated cell count 5 → 6, adds `("gemini", "gemini-3.1-pro-preview")` to the providers assertion.

`make eval-matrix-refinement-structural-check` exits 0:

```
structural-check: OK — matrix has 6 cell(s), env-override preserved through _apply_override, scorer registered, shared helper functional
```

`scripts/check_baselines_fresh.py origin/main` exits 0 — the existing `refinement_cheaper.json` was already refreshed by PROV-03's commits (`generated_at: 2026-06-05T20-29-56Z`).

## Live 1-call Gemini probe outcome (2026-06-05)

The probe surfaced critical live-integration findings the synthetic Phase 8 fixture missed. Two probes ran:

**Probe 1 — simple no-tool query** (`"Reply with just the number 42."`):
- `content` shape: `list` of block dicts (Anthropic-style block list, not str)
- Sample block: `{'type': 'text', 'text': '42', 'extras': {'signature': '<base64-str>'}}`
- The `extras.signature` is the thought_signature for reasoning content blocks
- `additional_kwargs` is **empty** (no thought_signature key at all)
- `response_metadata` does NOT carry the signature either
- `usage_metadata.output_token_details.reasoning: 70` — Gemini IS doing reasoning, but signatures live INSIDE the content block list under `extras.signature`

**Probe 2 — tool-calling query** (search-for-coffee-shop with a single `search` tool):
- `additional_kwargs` keys: `['function_call', '__gemini_function_call_thought_signatures__']`
- `__gemini_function_call_thought_signatures__` is `dict[tool_call_id, base64_signature_str]`
- This is the real wire shape lcgg 4.x reads from on outbound (verified by reading lcgg source: `langchain_google_genai/chat_models.py:127, 717-735`)
- `tool_calls[0]` carries `{name, args, id, type}` — no thought_signature at the per-tool-call level

**Implication for the adapter:** the original implementation (which only read `additional_kwargs["thought_signature"]: bytes`) would have returned None on EVERY real Gemini response, silently failing PROV-04's actual charter. Commit 2 (`bf6ff83`) extended the adapter to handle the real lcgg 4.x wire shape AS PRIMARY (Path 1) while keeping the Phase 8 fixture path (Path 2) so REASON-02 conformance continues to assert byte-for-byte survival through the LangGraph reducer.

This is exactly the lesson from Plan 09-03 (PROV-03 had 4 live-integration bug-fixes the synthetic tests missed — `max_tokens=8192`, eval_agent SUPPORTED_PROVIDERS, replay idempotency, temperature clamp). The Wave 4 probe averted the same class of regression for PROV-04 before any matrix run.

## Asymmetry visualization

| Provider | PROV# | Where reasoning state lives                                                | Payload type                | Wire key                                              |
| -------- | ----- | -------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------- |
| openai   | 01    | `AIMessage.additional_kwargs["reasoning_content"]` (via Responses-API lift) | `str` (content blocks list) | Same                                                  |
| deepseek | 02    | `AIMessage.additional_kwargs["reasoning_content"]` (lcdeepseek native)      | `str`                       | Same                                                  |
| anthropic| 03    | `AIMessage.content[i]` (heterogeneous block list)                          | signed `dict` blocks        | `content` (with `signature` field)                    |
| gemini   | 04    | `AIMessage.additional_kwargs["__gemini_function_call_thought_signatures__"]` (real lcgg 4.x) | `dict[tc_id, base64_str]`   | Same (lcgg outbound serializer reads from this key)   |

Phase 8's contract is shape-agnostic enough that all four asymmetric shapes round-trip through the same ProviderAdapter ABC — REASON-02 acceptance is met across all four shapes (8 conformance tests pass, plus the new `test_reason_02_gemini_real_adapter` sibling).

## Test results

```
$ poetry run pytest tests/unit/test_adapters.py tests/unit/agent/test_adapters.py \
       tests/integration/test_reasoning_state_roundtrip.py \
       -m "reasoning_conformance or not reasoning_conformance" -v
============================== 51 passed in 0.79s ==============================

$ poetry run pytest tests/unit/ -q
================= 1051 passed, 7 skipped, 9 warnings in 17.89s =================

$ poetry run python -c 'from app.agent.adapters import ADAPTERS; ...'
OK - all 4 reasoning adapters wired; kimi+scripted on NoOp

$ make eval-matrix-refinement-structural-check
structural-check: OK — matrix has 6 cell(s), env-override preserved through _apply_override, scorer registered, shared helper functional

$ poetry run python scripts/check_baselines_fresh.py origin/main
check_baselines_fresh: OK — app/agent/ changed and 1 baseline file(s) refreshed:
  - configs/eval_baselines/refinement_cheaper.json
```

## Deviations from Plan

### Adapter shape expansion (live-probe-driven; Rule 3)

**Found during:** Live 1-call Gemini probe (Step 4 of execution prompt).

**Issue:** The Plan + PATTERNS.md + Phase 8 fixture all designed the adapter around `additional_kwargs["thought_signature"]: bytes`. The live probe revealed that real `langchain-google-genai==4.x` traffic surfaces function-call signatures at `additional_kwargs["__gemini_function_call_thought_signatures__"]: dict[tool_call_id, base64_str]` instead. The original adapter would have returned `None` on every real Gemini response.

**Fix:** Extended `GeminiAdapter` to a three-path priority capture (real lcgg 4.x dict primary; synthetic-fixture bytes fallback; per-tool-call bytes fallback) and symmetric three-key replay. The synthetic-fixture path is preserved so REASON-02 conformance harness still passes byte-for-byte.

**Files modified:** `app/agent/adapters/gemini.py`, `tests/unit/test_adapters.py` (5 new tests for the real-wire path).

**Commit:** `bf6ff83`.

**Rationale:** Plan explicitly anticipated this — "If the executor finds lcgg 4.x surfaces the field on `tool_calls[i]` instead of `additional_kwargs`, they adapt inline — PROV-04 has no merge gate so iteration cost is low." The actual surfacing wasn't on `tool_calls[i]` either; it was on a NAMESPACED `additional_kwargs` key the synthetic fixture didn't predict. Same class of fix, same low iteration cost. Lesson from Plan 09-03's 4 live-integration bug-fixes (per `09-03-SUMMARY.md`) was the precedent: probe LIVE before signing off.

### Empirical matrix run skipped (Rule 4 → user-approved Option B)

**Found during:** Pre-execution planning (user-approved Option B per execution prompt's `<deferred_empirical_context>`).

**Issue:** The Plan's Task 2 is a `checkpoint:human-verify` requiring a local 6-cell × 5-runs matrix to capture the gemini cell's median + refresh existing cells' n=5 stats. That requires the OpenAI embeddings quota to be intact (`semantic_search` 429s otherwise, regardless of LLM provider). Quota exhausted during PROV-03's matrix retry (per `09-03-SUMMARY.md`).

**Fix:** User explicitly pre-approved Option B at execution-prompt time: ship PROV-04 STRUCTURALLY (adapter + registry swap + tests + matrix YAML cell), defer empirical n=5 measurement to Phase 10 BASE-01. PROV-04 has no merge gate per D-09-08 so this is a clean pass for the actual charter.

**Files modified / not modified:** Matrix YAML cell IS added (`configs/eval_matrix_refinement.yaml`); baseline JSON NOT updated (will be populated when Phase 10 BASE-01 runs the matrix under restored OpenAI quota).

**Commit:** Not applicable — the deferred work is carry-forward only.

### Note: Baseline JSON cell NOT added (intentional, per user-approved Option B)

The Plan's PATTERNS.md PROV-04 row anticipates a `gemini/gemini-3.1-pro-preview` cell in `configs/eval_baselines/refinement_cheaper.json` carrying the `_observations` string `"Phase 9 PROV-04 EXPERIMENTAL (no merge gate per D-09-08). n=5 runs at temp=1.0, GeminiAdapter active. Critique-loop fix (LOW_SIMILARITY_THRESHOLD=0.55) deferred per project_w10_migration_necessary_not_sufficient — expect median=0.0 until v2.2."`

That cell is intentionally NOT added in this commit. Adding it with `n=0` / `n=1` synthetic placeholder values would dilute the baseline JSON's role as the empirical-medians source of truth (per Plan 09-03's pattern of carrying n=1 single-probe values explicitly tagged "SHIPPED-WITH-GAP"). Phase 10 BASE-01 will populate the cell with real n=5 medians once the OpenAI quota is restored — that's the natural place per the milestone charter.

`scripts/check_baselines_fresh.py origin/main` still exits 0 because `refinement_cheaper.json` was already refreshed by PROV-03's commits and `app/agent/` mtime ordering is preserved.

## Carry-forward to Phase 10 BASE-01

Phase 10 BASE-01 must populate the following deferred measurements into `configs/eval_baselines/refinement_cheaper.json`:

1. **Re-measure `anthropic/claude-sonnet-4-6` at n=5** — deferred from Plan 09-03 (currently n=1 single-probe observation; refinement_minimal_edit=0.0 / committed_itinerary_rate=1.0). The PROV-03 strict-gate (median ≥ 1.0 over n=5) is a SHIPPED-WITH-GAP that can be confirmed or contradicted only by a fresh n=5 run under restored OpenAI quota.
2. **First-time measure `gemini/gemini-3.1-pro-preview` at n=5** — deferred from Plan 09-04 per user-approved Option B. GeminiAdapter is wired structurally + conformance-tested; the empirical median is informational-only per D-09-08 (logged-not-gated). Expected median: 0.0 until the deferred critique-loop fix (`LOW_SIMILARITY_THRESHOLD=0.55`) lands in v2.2 / a Phase 10.5 follow-up.

Both measurements require:
- OpenAI embeddings quota top-up (current state: exhausted; `semantic_search` 429s on every tool call).
- All 4 provider API keys: OPENAI, DEEPSEEK, ANTHROPIC, GEMINI.
- `cloud-sql-proxy mlops--city-concierge:us-west2:postgres-18 --port 5433` running.
- `APP_ENV=eval make eval-matrix-refinement RUNS=5`.

## Deferred critique-loop fix (D-09-08 SUMMARY.md requirement)

Per D-09-08, PROV-04 SUMMARY.md MUST link memory `project_w10_migration_necessary_not_sufficient` and carry forward the deferred critique-loop fix. Doing both now:

- Memory: **`project_w10_migration_necessary_not_sufficient`** — W10 migration to langchain-google-genai 4.x was necessary but NOT sufficient for Gemini 3 to converge on the agent's tool-loop tasks. The deeper blocker is the critique-loop in `app/agent/revision.py:21` (`LOW_SIMILARITY_THRESHOLD=0.55`) which causes Gemini 3 to re-search forever instead of committing. This phase ships the state-preservation half of PROV-04; the decisiveness half is unowned.
- Deferred fix: **`LOW_SIMILARITY_THRESHOLD=0.55`** in `app/agent/revision.py:21` — the critique-loop sends "low_similarity" critiques back to the planner when the search results' embedding similarity to the query falls below this threshold. Gemini 3 then "rephrases more broadly" and re-searches, perpetually. Lowering the threshold (or rewriting the critique prompt to be commit-pressure-positive) would let Gemini converge, but the fix cross-cuts ALL providers including the v2.0 gpt-4o-mini anchor — needs its own falsifier-shaped phase (v2.2 reasoning-decisiveness follow-up, or Phase 10.5).
- Cross-link: also see project memory **`project_reasoning_models_break_agent_loop`** for the broader architectural framing — every reasoning model on this codebase has the same convergence gap; the locked gpt-4o-mini anchor is a feature given the pre-Phase-9 architecture.

PROV-04 acceptance under D-09-08: state-preservation contract delivered, decisiveness contract carry-forward. PROV-FUT-01 (promote Gemini 3 from experimental to gated) requires BOTH (a) PROV-04 conformance pass (done) AND (b) critique-loop fix lands AND (c) Phase 10 cross-model baseline regen captures a `gemini-3.1-pro-preview` median.

## Wave 5 readiness

All 4 reasoning providers (PROV-01..04) are now wired off NoOpAdapter. The registry consolidation to explicit literal at one site simplifies the import-isolation audit Plan 09-05 will run (`scripts/audit_adapter_isolation.py` or equivalent — D-09-07 PROV-05 revert-atomicity simulation). The per-adapter imports in `__init__.py` are the only cross-sibling imports in the subpackage; each individual `<provider>.py` follows the isolation rule.

PROV-05 (Plan 09-05) is unblocked. The 4 reasoning adapters are auditably revert-atomic via `git revert <hash>` on the four PROV commits across Plans 09-01..09-04. Plan 09-05 ships the audit + revert simulation that proves it.

## Self-Check: PASSED

Files created/modified verified to exist:
- `app/agent/adapters/gemini.py` — FOUND
- `.planning/phases/09-per-provider-state-preservation-implementations/09-04-SUMMARY.md` — FOUND (this file)
- `app/agent/adapters/__init__.py` — modified, commit `10e88b9`
- `tests/unit/test_adapters.py` — modified, commits `10e88b9` + `bf6ff83`
- `tests/unit/agent/test_adapters.py` — modified, commit `10e88b9`
- `tests/integration/test_reasoning_state_roundtrip.py` — modified, commit `10e88b9`
- `configs/eval_matrix_refinement.yaml` — modified, commit `17e9187`
- `tests/unit/test_eval_matrix.py` — modified, commit `17e9187`

Commits exist in `gsd/phase-09-per-provider-state-preservation-implementations`:
- `10e88b9` — FOUND
- `bf6ff83` — FOUND
- `17e9187` — FOUND

---

*Phase 9 / Plan 09-04 complete (SHIPPED-STRUCTURAL). Next: Plan 09-05 (revertability-audit), Wave 5.*
