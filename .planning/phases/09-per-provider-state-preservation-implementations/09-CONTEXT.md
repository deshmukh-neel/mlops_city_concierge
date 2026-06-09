# Phase 9: Per-Provider State Preservation Implementations - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 9 ships four `ProviderAdapter` implementations — one per reasoning provider family — into the `app/agent/adapters/` subpackage created in Phase 8, and swaps each provider's `ADAPTERS` registry entry from `NoOpAdapter` to the real adapter. Each sub-phase lands as an independently revertable commit per PROV-05, and the milestone anchor gate (`gpt-5-mini × refinement_cheaper × prod × flag-on` commits 3 stops in median 5/5 runs at temp=1.0) is met by the OpenAI sub-phase. Per-cell baseline JSON entries are written alongside each sub-phase's adapter; the wholesale honest baseline regen under DB-up conditions is deferred to Phase 10 (BASE-01).

**In scope:**
- `app/agent/adapters/openai_gpt5.py` (or executor-chosen name) + `OpenAIReasoningAdapter` + registry swap for `"openai"` (PROV-01).
- `app/agent/adapters/deepseek.py` + `DeepSeekReasonerAdapter` + registry swap for `"deepseek"` + model-level conditional in `build_chat_model` for `deepseek-reasoner` thinking-mode (PROV-02).
- `app/agent/adapters/anthropic.py` + `AnthropicAdapter` + new `"anthropic"` entry in `SUPPORTED_PROVIDERS` + new `build_chat_model` branch + `langchain-anthropic` in `pyproject.toml` + ADAPTERS registration (PROV-03).
- `app/agent/adapters/gemini.py` + `GeminiAdapter` + registry swap for `"gemini"` (PROV-04, experimental).
- One `scripts/probe_*.py` for PROV-01 only (D-09-03) to dump real gpt-5-mini AIMessage shape post langchain-openai 1.2; output informs the adapter design before it's written.
- Per-sub-phase: matching update to `configs/eval_matrix_refinement.yaml` (add/promote the cell + YAML gate-status comment) + per-cell update to `configs/eval_baselines/refinement_cheaper.json` (median from the local n=5 run).
- PROV-01..03 phase-completion gates verified by local `make eval-matrix-refinement RUNS=5` with all keys present + cloud-sql-proxy; medians recorded in each sub-phase SUMMARY.md.
- PROV-04 conformance test pass (real `GeminiAdapter` substituted into the Phase 8 parametrized harness; the bytes `thought_signature` payload survives `graph.ainvoke`).

**Out of scope:**
- Touching `SYSTEM_PROMPT` / `_REFINEMENT_PREAMBLE` — Phase 7 territory, already shipped.
- Changing `ProviderAdapter` ABC, `StatePayload`, the `_prune_for_llm` post-prune hook, the `provider=` kwarg on `build_agent_graph`, or the conformance-harness body — Phase 8 territory, already shipped.
- The Gemini 3 `low_similarity` critique-loop fix (`LOW_SIMILARITY_THRESHOLD=0.55` in `app/agent/revision.py`) — deferred. Memory `project_w10_migration_necessary_not_sufficient` is the source. PROV-04 is "experimental, no merge gate" precisely because this loop is unfixed.
- Wholesale honest regen of all `configs/eval_baselines/*.json` files — Phase 10 BASE-01.
- Adding `gpt-5-mini`, `claude-sonnet-4-6`, `deepseek-reasoner` to the broader `configs/eval_matrix.yaml` cross-model matrix — Phase 10 BASE-02. (Phase 9 only updates `configs/eval_matrix_refinement.yaml`, the per-phase Phase 6 matrix.)
- Promoting the `reasoning_conformance` pytest marker from quarantined to required CI gate — Phase 10 BASE-03.
- New CI live-provider secrets / Cloud SQL proxy in CI / 5-runs-per-PR matrix execution in CI — Phase 10 BASE-03.
- Promoting Gemini 3 from experimental to gated — PROV-FUT-01 (v2.2+).
- O-series (o3, o4-mini) adapter — PROV-FUT-03.
- Kimi K2 / Moonshot adapter — PROV-FUT-02 (library-blocked per memory).

</domain>

<decisions>
## Implementation Decisions

### Sub-phase Sequencing & Atomicity (Area 1)

- **D-09-01:** Single `gsd/phase-09-per-provider-state-preservation-implementations` branch, 4 sequential commits PROV-01 → PROV-02 → PROV-03 → PROV-04, one PR. Each sub-phase is independently revertable via `git revert <sha>` per PROV-05. Matches Phase 8 shipping pattern (one branch, multi-plan ship via PR #102). Parallel-branch alternative explicitly rejected: registry-edit conflicts in `app/agent/adapters/__init__.py` would force rebases for every sibling.
- **D-09-02 (RE-SCOPED 2026-06-05 per user-approved Option A; original strict scorer was a Phase-6 baseline-saturation artifact — see memories `project_phase6_gate_failed`, `project_phase6_d_06_09_root_cause`):** PROV-01 PR-blocking gate on `openai/gpt-5-mini × refinement_cheaper × prod × flag-on × temp=1.0` is now a 2-part gate — **Part A (hard):** `committed_itinerary_rate ≥ 0.6`; **Part B (advisory):** `refinement_minimal_edit median ≥ 0.5`. Part A is the PR-blocking measurement; Part B is logged but does not block. Rationale: the v2.0 anchor `gpt-4o-mini` itself sits at `refinement_minimal_edit` median 0.0 / max 0.5 under the current scorer (post-D-07-05 + D-07-07 tightening), so holding gpt-5-mini to the original strict median 1.0 was asymmetric with the anchor. The Phase 9 charter is "preserve provider reasoning state cross-turn" — committed-itinerary-rate measures that signal directly; edit-distance measures refinement-quality which is prompt/critique-loop territory (deferred to v2.1 phase 2). Original strict wording is preserved in `09-PROV-01-BLOCKER.md` for the historical record. PROV-02..04 cannot rescue the PR if Part A still fails.

  Original wording (preserved for archaeology): "PROV-01 gate failure (gpt-5-mini × refinement_cheaper × prod × flag-on median < 5/5 commits at temp=1.0) BLOCKS the entire PR".
- **D-09-07:** PROV-05 revert atomicity enforced by CONTEXT.md convention, not test: `app/agent/adapters/<provider>.py` imports ONLY from `app.agent.adapters` base + `langchain_core` + stdlib; never from a sibling adapter file. Audited at code review. 4 files in scope is small enough that mechanical enforcement (a unit test scanning imports) is overkill; the convention is recorded here so reviewers know what to look for.

### PROV-01 — OpenAI gpt-5 Family Adapter (Area 1)

- **D-09-03:** PROV-01 plan MUST begin with a probe step. A short `scripts/probe_gpt5_capture.py` (or executor-chosen name) hits `gpt-5-mini` with a representative tool_call shaped like the agent's own (e.g. `semantic_search`), dumps the returned `AIMessage` (`.additional_kwargs` keys, `.content` shape, `.response_metadata`), and commits the probe output as a markdown artifact in the phase dir. The adapter's `capture_reasoning_state` implementation is finalized only after the probe lands.

  **Why probe-first:** Memory `project_agent_loses_reasoning_state_all_providers` (2026-05-17) confirmed that `langchain-openai` 1.x at that snapshot did NOT capture vendor `reasoning_content` onto `AIMessage.additional_kwargs` at all — bypassing `_prune_for_llm` did not help because the data was already gone at the library boundary. The current pin is `langchain-openai>=1.2.0,<2.0.0`; the 1.2 line added passthrough for OpenAI Responses-API `reasoning` field. Whether that actually surfaces on `AIMessage.additional_kwargs['reasoning_content']` for `gpt-5-mini` via the Chat Completions wrapper or via the Responses API path is what we don't know. The probe answers it in 30 lines.

  Two plausible adapter shapes downstream of the probe:
  - **Path A (read-the-kwarg):** `capture` reads `msg.additional_kwargs.get("reasoning_content")` or `msg.response_metadata.get("reasoning")`; `replay` injects it back as `additional_kwargs["reasoning_content"]` on the most-recent AIMessage so the next outbound request includes it.
  - **Path B (custom subclass):** PROV-01 ships a `OpenAIReasoningChatModel(ChatOpenAI)` in `app/llm_factory.py` that overrides `_generate` to lift the field from the raw response BEFORE LangChain's normalizer drops it; adapter is symmetric to A but reads from the subclass's enriched message.

  Probe output decides A vs B. Plan doc must reflect this branch point explicitly.

### PROV-02 — DeepSeek Reasoner Adapter (Area 1)

- **D-09-04:** PROV-02 uses **provider=`deepseek` (unchanged), model=`deepseek-reasoner`**. `app/llm_factory.py:218` `build_chat_model` gets a model-level conditional: when `chat_model.startswith("deepseek-reasoner")`, drop the hardcoded `extra_body={"thinking": {"type": "disabled"}}` (or flip it to enabled). `DeepSeekAdapter.capture` reads `additional_kwargs.get("reasoning_content")` (langchain-deepseek populates it for the reasoner model per documented contract); `replay` re-attaches it to the most-recent AIMessage so the next request body carries it on the assistant tool_call message. Mirrors the per-model policy precedent at `app/llm_factory.py:65-72` (`_KIMI_FORCED_TEMPERATURE`, `_GEMINI_THINKING_ONLY`).

  **Rejected alternatives:**
  - Separate `"deepseek_reasoner"` provider key — conflates vendor with model family; doubles registry surface.
  - Custom `ChatDeepSeek` subclass — bigger surface than the conditional; library handles the reasoning_content field already per documented contract for the reasoner model.

### PROV-03 — Anthropic Claude Wiring (Area 2)

- **D-09-05:** PROV-03 uses **provider key `"anthropic"`** (vendor naming, not the model family `"claude"`). Rationale: `app/config.py:159` `resolve_llm_api_key` already dispatches on `"anthropic"` and `Settings.anthropic_api_key` (`app/config.py:93`) already exists since pre-W10. Roadmap text says "add claude to SUPPORTED_PROVIDERS" — that maps to `provider="anthropic", model="claude-sonnet-4-6"`, matching the openai/gpt-5-mini and gemini/gemini-3.1-pro-preview vendor-vs-model patterns. Other Anthropic models (Haiku, Opus, future) share the same provider key.

  Concrete edits in PROV-03:
  - `app/llm_factory.py:62` — `SUPPORTED_PROVIDERS += ("anthropic",)`.
  - `app/llm_factory.py:186` — new `if provider == "anthropic":` branch in `build_chat_model` returning `ChatAnthropic(...)`.
  - `pyproject.toml` — `langchain-anthropic = ">=N.N,<M.M"` dep add (executor picks compatible major). `poetry lock` regenerated.
  - `app/agent/adapters/anthropic.py` — new `AnthropicAdapter` reading the `thinking_blocks` content-block array and replaying it on the most-recent AIMessage.
  - `app/agent/adapters/__init__.py` — `ADAPTERS["anthropic"] = AnthropicAdapter()` (registry rewrite from `NoOpAdapter()`).
  - `configs/eval_matrix_refinement.yaml` — new cell `{provider: anthropic, model: claude-sonnet-4-6, env: {REFINEMENT_STRUCTURED_PLAN_ENABLED: "true"}}` + YAML comment marking `gated: median ≥ 1.0 (PROV-03)`.
  - `configs/eval_baselines/refinement_cheaper.json` — new cell entry with the measured n=5 median.

- **D-09-06:** PROV-03 Claude default policy: **`thinking=enabled, temp=1.0`** as a deliberate carve-out from memory `feedback_temp1_reasoning_off_all_models` ("disable thinking for ALL providers"). Rationale: Claude Sonnet 4.6 with `thinking` disabled is just regular Sonnet 4.6 — non-reasoning, no `thinking_blocks` to round-trip, no PROV-03 signal at all. Mirrors the `_GEMINI_THINKING_ONLY` hard-floor pattern (Gemini 3 runs at `thinking_level="low"` because `thinking_budget=0` 400s). CONTEXT.md documents the rule's exception for future grep'ers. `budget_tokens` value is Claude's-Discretion (planner picks ~4096 as a reasonable default; Phase 10 baseline regen may tune).

### PROV-04 — Gemini 3 Experimental Adapter (Area 3)

- **D-09-08:** PROV-04 ships when the Gemini parametrize case in `tests/integration/test_reasoning_state_roundtrip.py::test_reason_02_four_shape_roundtrip` passes with the real `GeminiAdapter` swapped into `ADAPTERS["gemini"]` (the bytes `thought_signature` payload survives `graph.ainvoke`). No empirical refinement_cheaper gate — `gemini-3.1-pro-preview × refinement_cheaper` may still score 0.0 and that does NOT block PROV-04 commit; the failure mode is documented as belonging to a separate critique-loop fix (deferred, see `<deferred>` below).

  PROV-04 SUMMARY.md MUST link memory `project_w10_migration_necessary_not_sufficient` and explicitly carry forward the deferred critique-loop fix so the milestone-archive audit catches it.

- **D-09-09:** `GeminiAdapter` is designed against the Phase 8 fixture payload `{"provider": "gemini", "thought_signature": b"\x00\x01\x02"}` (D-08-13). `capture` reads `additional_kwargs.get("thought_signature")` (bytes); `replay` re-attaches it to the most-recent AIMessage. No probe step — `langchain-google-genai>=4.2.0` is pinned and W10 confirmed it has thought_signature plumbing (17 refs vs 0 in lcgg 2.1). If the executor finds the field name differs (e.g. `thought_signatures` plural, or under `.tool_calls[i].thought_signature`), they adapt inline; PROV-04 has no merge gate so iteration cost is low.

### Per-Sub-Phase Gates & Baselines (Area 4)

- **D-09-10:** PROV-01..03 empirical gates execute LOCALLY via `make eval-matrix-refinement RUNS=5` with all 4 provider API keys + `cloud-sql-proxy :5433`. Matches the Phase 7 plan 07-07 procedure verbatim. Each sub-phase SUMMARY.md records the n=5 median; the median IS the gate measurement. CI continues to run `make eval-matrix-refinement-structural-check` (Phase 6 D-06-10 structural-only CI gate) and `scripts/check_baselines_fresh.py` — no new live-provider CI surface in Phase 9. Phase 10 BASE-03 is responsible for promoting any of these to live-CI.

- **D-09-11:** Each sub-phase commits its `configs/eval_baselines/refinement_cheaper.json` cell update alongside its adapter, so `scripts/check_baselines_fresh.py origin/main` stays exit-0 throughout Phase 9. Phase 7 wrote n=5 medians for 3 cells; PROV-01 promotes the gpt-5-mini cell's "logged-not-gated" measurement to "gated"; PROV-02 adds a new `deepseek-reasoner` cell; PROV-03 adds `anthropic/claude-sonnet-4-6`; PROV-04 adds `gemini/gemini-3.1-pro-preview` as logged-only. The wholesale honest regen under DB-up conditions remains Phase 10 BASE-01's scope.

- **D-09-12:** Each sub-phase edits `configs/eval_matrix_refinement.yaml` to add or promote its cell + a YAML comment annotating gate status. PROV-01 changes the gpt-5-mini cell comment from "logged-not-gated" to "gated: 2-part — Part A (hard) `committed_itinerary_rate ≥ 0.6`, Part B (advisory) `refinement_minimal_edit median ≥ 0.5` per D-09-02 re-scope 2026-06-05". PROV-02..04 add new cells. The broader `configs/eval_matrix.yaml` (cross-model matrix) is NOT touched in Phase 9 — that belongs to Phase 10 BASE-02.

### Claude's Discretion

- Exact module file names inside `app/agent/adapters/` (e.g., `openai_gpt5.py` vs `openai_reasoning.py`; `anthropic.py` vs `claude.py`; `deepseek.py` vs `deepseek_reasoner.py`; `gemini.py` vs `gemini3.py`). Planner picks names that read clearly. Convention: vendor-named where possible (`anthropic.py`), model-family-disambiguated where the same vendor has reasoning vs non-reasoning models that share a provider key (`deepseek_reasoner.py` vs `deepseek_chat.py` if that disambiguation matters).
- The exact PROV-01 probe script name and location (`scripts/probe_gpt5_capture.py` vs `scripts/phase09_probe_gpt5.py` vs in-test). Planner picks.
- The `budget_tokens` value for Claude `thinking={"type": "enabled", "budget_tokens": N}` (D-09-06). Planner picks ~4096 as a reasonable starting point.
- The langchain-anthropic minor-version pin range. Planner picks a compatible major (`langchain-anthropic = ">=N.N,<M.0.0"`) and updates the lockfile.
- Whether `app/agent/adapters/__init__.py`'s `ADAPTERS` dict is mutated cell-by-cell per sub-phase (PROV-01 swaps `"openai"`, PROV-02 swaps `"deepseek"`, etc.) OR whether the dict-comprehension at line 121 is replaced with an explicit dict literal in PROV-01 and then per-cell-edited in PROV-02..04. Planner picks; the explicit literal probably reads better once 4 entries are non-NoOp.
- Whether `langchain-anthropic` import in `build_chat_model` is top-level or lazy (`if provider == "anthropic": from langchain_anthropic import ChatAnthropic`). Planner picks; top-level matches the existing factory style.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 8 Contract Surface (consumed by Phase 9)
- `app/agent/adapters/__init__.py` — `ProviderAdapter` ABC, `StatePayload`, `NoOpAdapter`, `MockReasoningAdapter`, `ADAPTERS` registry (currently all-NoOp). Phase 9 mutates `ADAPTERS` values in place per sub-phase.
- `app/agent/graph.py` §`plan()` lines 262-330 — Post-prune `replay_reasoning_state` and post-`ainvoke` `capture_reasoning_state` calls. Phase 9 adapters fit here; do NOT modify this wiring (Phase 8 territory).
- `app/agent/graph.py` §`build_agent_graph` line 244 + `provider: str = "openai"` kwarg — Phase 9 verifies main.py and scripts/eval_matrix.py keep passing this kwarg correctly.
- `tests/integration/test_reasoning_state_roundtrip.py` — Phase 8 4-shape conformance harness. Phase 9 sub-phases adapt the parametrize cases one-by-one as real adapters land. Test BODY does not change.
- `tests/unit/fixtures/reason_04_prune_baseline.json` — Byte-identity fixture for the gpt-4o-mini anchor path. Phase 9 must not regress this.

### Phase 9 Edit Targets
- `app/llm_factory.py:62` `SUPPORTED_PROVIDERS` — Append `"anthropic"` in PROV-03.
- `app/llm_factory.py:186` `build_chat_model` — Add `anthropic` branch (PROV-03); add `deepseek-reasoner` model-conditional inside existing deepseek branch (PROV-02).
- `app/llm_factory.py:65-78` per-model policy block — Add `_DEEPSEEK_REASONER_THINKING_ENABLED` or equivalent constant if D-09-04 wants to be symmetric with `_KIMI_FORCED_TEMPERATURE` / `_GEMINI_THINKING_ONLY`.
- `app/config.py:159` `resolve_llm_api_key` — Already has `"anthropic"` branch; PROV-03 does NOT need to edit this (verify by re-grep).
- `app/config.py:93` `anthropic_api_key` Setting — Already exists; PROV-03 does NOT need to add it (verify by re-grep).
- `pyproject.toml` `[tool.poetry.dependencies]` — Add `langchain-anthropic` in PROV-03; `poetry lock` updates `poetry.lock`.
- `configs/eval_matrix_refinement.yaml` — Per-sub-phase edits per D-09-12. The existing entries for openai/gpt-4o-mini, deepseek/deepseek-chat, openai/gpt-5-mini stay (D-07-08 logged-not-gated rules unchanged for gpt-4o-mini and deepseek-chat).
- `configs/eval_baselines/refinement_cheaper.json` — Per-sub-phase cell updates per D-09-11. `_snapshots/` pre-phase snapshot NOT required for Phase 9 (Phase 7 already snapshot'd; git history is enough).

### Phase 9 New Files (one per sub-phase)
- `app/agent/adapters/openai_gpt5.py` (or planner-chosen name) — `OpenAIReasoningAdapter` (PROV-01).
- `app/agent/adapters/deepseek.py` — `DeepSeekReasonerAdapter` (PROV-02).
- `app/agent/adapters/anthropic.py` — `AnthropicAdapter` (PROV-03).
- `app/agent/adapters/gemini.py` — `GeminiAdapter` (PROV-04).
- `scripts/probe_gpt5_capture.py` (or planner-chosen name) — One-shot script that hits gpt-5-mini and dumps AIMessage shape; output committed as `.planning/phases/09-.../09-PROV-01-PROBE.md` (PROV-01 only, per D-09-03).

### Roadmap / Requirements / State
- `.planning/ROADMAP.md` §Phase 9 — Goal, depends-on (Phase 8), requirements (PROV-01..05), success criteria. SC #1 is the milestone anchor gate (gpt-5-mini × refinement_cheaper × prod × flag-on × 5/5).
- `.planning/REQUIREMENTS.md` §Per-Provider State Preservation Implementations (Phase 9) — PROV-01..05 verbatim. Each PROV's gate threshold lives here.
- `.planning/STATE.md` — Current focus is Phase 9 (per status field).
- `.planning/PROJECT.md` §Current Milestone v2.1 + Key Decisions — Phase 9 sequencing rationale; "stay on LangGraph for v2.1" decision resolved by Phase 8 REASON-05 PASSED.
- `.planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/08-CONTEXT.md` — Phase 8 implementation decisions D-08-01 through D-08-16, especially D-08-01 (sub-phase file convention), D-08-13 (4-shape fixture payloads — Phase 9 GeminiAdapter consumes the gemini case verbatim), D-08-14 (conformance harness marker quarantine — Phase 10 promotes, Phase 9 does not).
- `.planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/08-04-SUMMARY.md` — D-08-11 Branch A acceptance: REASON-05 PASSED; Phase 9 proceeds on LangGraph as written; no v2.1.1 imperative-loop replan.
- `.planning/phases/07-prompt-rubric-decoupling/07-07-SUMMARY.md` — gpt-5-mini median = 0.0 falsifier outcome; state-loss dominates; Phase 9 stays at full scope (no narrowing). `configs/eval_baselines/refinement_cheaper.json` post-Phase-7 cell values are the inheritance baseline for D-09-11.

### Existing Configuration the Adapters Lean On
- `app/main.py:342` — Threads `provider=` resolved from MLflow Model Registry into `build_agent_graph`. Phase 9 doesn't touch main.py wiring; the provider string already arrives at `build_agent_graph` correctly.
- `scripts/eval_matrix.py` — Threads `provider=` per matrix cell. Phase 9 doesn't touch eval_matrix.py wiring.
- `Makefile` `eval-matrix-refinement` target — Phase 9 uses verbatim per D-09-10. The `make eval-matrix-refinement-structural-check` CI gate remains the only CI enforcement.
- `scripts/check_baselines_fresh.py` — CI hard gate on baseline staleness; Phase 9 keeps it green by committing baseline updates alongside agent changes per D-09-11.

### Project Memories (v2.1 Architecture Context)
- `project_agent_loses_reasoning_state_all_providers` — The architectural bug Phase 8 contract + Phase 9 impls fix. Key data: `langchain-openai` 1.x at the W10 snapshot did NOT capture vendor `reasoning_content` onto AIMessage. PROV-01 probe (D-09-03) confirms whether langchain-openai 1.2 changed this.
- `project_reasoning_models_break_agent_loop` — Locked gpt-4o-mini anchor is a feature given pre-Phase-9 architecture; Phase 9 changes that.
- `project_w10_migration_necessary_not_sufficient` — Gemini 3 critique-loop is the SECOND blocker beyond signatures plumbing. Linked from PROV-04 SUMMARY per D-09-08.
- `project_deepseek_decisiveness_gap` — DeepSeek decisiveness gap on the v2.0 anchor was on `deepseek-chat` (non-reasoning); PROV-02 is `deepseek-reasoner` (reasoning). Different signal.
- `project_gemini3_thought_signatures` — Historical context for thought-signature handling pre-W10; lcgg 4.x supersedes the static-bypass hack.
- `project_mlflow_prod_alias_gemini` — Shared MLflow `production` alias is currently on broken Gemini v2. Phase 9 does NOT touch the alias (Phase 2 `RAG_MODEL_OVERRIDE` env-var is the safe path).
- `feedback_temp1_reasoning_off_all_models` — General rule; D-09-06 documents the Claude carve-out.
- `feedback_test_layering` — Each PROV adapter ships unit + smoke + integration coverage.
- `feedback_small_focused_commits` — 4 commits in PROV-01..04 order (D-09-01).

### Project Instructions
- `CLAUDE.md` — Project-wide guidance; v2.0 Phase 6 `REFINEMENT_STRUCTURED_PLAN_ENABLED` flag + `openai/gpt-4o-mini` locked prod anchor + `make eval-matrix-refinement-structural-check` CI hard gate. PROV-01..03 must not regress the gpt-4o-mini anchor cell.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`ADAPTERS` registry** (`app/agent/adapters/__init__.py:121`) — Dict-comprehension over `SUPPORTED_PROVIDERS` keyed by provider string. Phase 9 sub-phases mutate entries in place: `ADAPTERS["openai"] = OpenAIReasoningAdapter()`, etc. The dict-comprehension may be rewritten to an explicit literal once 4+ entries are non-NoOp (Claude's Discretion).
- **`MockReasoningAdapter`** (`app/agent/adapters/__init__.py:83`) — Phase 8 test-only adapter that the conformance harness already drives end-to-end. Phase 9 real adapters follow the same shape: capture reads `additional_kwargs`, replay walks `outbound` in reverse and tags the most-recent `AIMessage`.
- **`_KIMI_FORCED_TEMPERATURE` / `_GEMINI_THINKING_ONLY`** (`app/llm_factory.py:65-78`) — Existing per-model policy constants. D-09-04 (deepseek-reasoner thinking-enabled) and D-09-06 (Claude thinking-enabled budget_tokens) follow the same shape — module-level constant + conditional inside `build_chat_model`.
- **`ChatDeepSeek` + `ChatOpenAI` + `ChatGoogleGenerativeAI` + (new) `ChatAnthropic`** — All inherit `BaseChatModel`; all support `bind_tools`; all integrate with the LangGraph reducer. Phase 9 uses each verbatim — no agent-graph-level rewires.
- **`additional_kwargs` on `AIMessage`** — D-08-06 / Phase 8 contract: state lives on `ai.additional_kwargs["_reasoning_state"]` so it survives `add_messages` reducer between turns. Phase 9 adapters write to this exact key; Phase 8 conformance harness verifies kwarg survival.
- **Phase 7 baseline** (`configs/eval_baselines/refinement_cheaper.json`) — 3-cell post-Phase-7 baseline (openai/gpt-4o-mini, deepseek/deepseek-chat, openai/gpt-5-mini). PROV-01 promotes the gpt-5-mini cell's measurement to "gated"; the other two cells stay as-is.

### Established Patterns
- **Per-model policy via module-level constants** (`_KIMI_FORCED_TEMPERATURE`, `_GEMINI_THINKING_ONLY`) — D-09-04 deepseek-reasoner extension follows this.
- **Vendor-named provider keys** (openai, gemini, deepseek, kimi) — D-09-05 uses `"anthropic"` to match this; `"claude"` would be the only model-family-named exception.
- **Probe-then-build** — New pattern introduced by D-09-03 for PROV-01 only. Justified by the high-staleness of memory `project_agent_loses_reasoning_state_all_providers` relative to the current `langchain-openai>=1.2.0` pin. Not generalized to PROV-02..04 because their library contracts are documented (DeepSeek reasoner reasoning_content) or already covered by Phase 8 conformance (Gemini thought_signature fixture).
- **Per-sub-phase atomic commits** (D-09-01 / Phase 8 plan-by-plan ship) — Mirrors Phase 8's 5-plan single-PR ship pattern.
- **Local empirical gate, CI structural gate** (D-09-10 / Phase 6 D-06-10 + Phase 7 plan 07-07) — Live providers + Cloud SQL proxy + local Make target; CI runs only the structural variant.
- **Baseline JSON freshness via `check_baselines_fresh.py`** — Phase 9 keeps green by per-sub-phase cell updates per D-09-11.
- **Test layering** (memory `feedback_test_layering`) — Each PROV adapter ships unit (capture/replay isolated), integration (conformance harness real-adapter swap), and the local empirical gate is the "functional" layer.

### Integration Points
- **`plan() → adapter.replay_reasoning_state` / `adapter.capture_reasoning_state`** (`app/agent/graph.py:312-321`) — The single insertion point for adapter behavior. Phase 9 does not modify these call sites; they were locked in Phase 8 plan 08-03.
- **`ADAPTERS[provider]` lookup at graph-build time** (`app/agent/graph.py:279`) — Adapter resolution happens once; `plan()` closes over the chosen adapter. Phase 9 just changes which adapter instance the lookup returns.
- **`build_chat_model` per-provider branches** (`app/llm_factory.py:206-232`) — PROV-02 edits the existing deepseek branch (model conditional); PROV-03 adds a new anthropic branch; PROV-01 and PROV-04 do NOT edit `build_chat_model` (their adapter changes are in adapters/ only — PROV-01 may add a `ChatOpenAI` subclass per probe outcome, in which case build_chat_model.openai branch changes).
- **`configs/eval_matrix_refinement.yaml` cell injection** (`scripts/eval_matrix.py`) — Each new cell runs through the same scripted-LLM-or-live-provider dispatch; nothing new required at the runner level.
- **`scripts/check_baselines_fresh.py`** — CI gate that compares agent-touching files against baseline-touching files. Phase 9 baseline-JSON updates per sub-phase keep this green; if a sub-phase commits an adapter without its baseline cell update, CI fails.

</code_context>

<specifics>
## Specific Ideas

- **PROV-01 probe output format** (D-09-03): a markdown file at `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` containing (a) `langchain-openai.__version__` at probe time, (b) the gpt-5-mini chat model used, (c) the AIMessage `.additional_kwargs.keys()`, `.response_metadata` if non-empty, `.content` shape (str vs list), and `.usage_metadata`, (d) raw `dict(message)` dump for fidelity, (e) one-sentence verdict: "kwarg path works" / "subclass required" / "neither — escalate". Planner stubs this artifact in the PROV-01 plan; executor fills it.

- **PROV-02 reasoning_content shape on `deepseek-reasoner`**: per the DeepSeek docs, the model emits `reasoning_content` as a top-level string on the assistant message, and the API requires it replayed on the assistant tool_call message of the NEXT request. `langchain-deepseek>=1.0.0,<2.0.0` (current pin) populates `AIMessage.additional_kwargs["reasoning_content"]` accordingly. If the executor finds langchain-deepseek puts it elsewhere (e.g. `response_metadata`), they adapt; if it's missing entirely, escalate (analogous to PROV-01 subclass path).

- **PROV-03 Anthropic `thinking_blocks` shape**: per Anthropic docs, `messages.create(thinking={"type": "enabled", "budget_tokens": N})` returns content as a heterogeneous list including `{"type": "thinking", "thinking": "...", "signature": "..."}` blocks alongside `{"type": "text", ...}` blocks. `langchain-anthropic` surfaces these as content-blocks on the `AIMessage.content` list (not on `additional_kwargs`). `AnthropicAdapter.capture` should extract the thinking blocks from `message.content` (when content is a list); `replay` re-attaches them to the most-recent AIMessage's content list. The signature MUST round-trip byte-identical or the next request 400s. This is different from the other 3 adapters which all use `additional_kwargs` — `AnthropicAdapter` reads/writes `content` instead. Plan must call out this asymmetry.

- **PROV-04 Gemini thought_signature shape**: per lcgg 4.x docs, `thought_signature` lives on individual `tool_call` items in the AIMessage (`AIMessage.tool_calls[i]` may have a `thought_signature` bytes field, OR it lives on `additional_kwargs["thought_signature"]` as a top-level bytes). Phase 8 fixture assumes the latter (`additional_kwargs["thought_signature"]: bytes`); the conformance test passes against `MockReasoningAdapter`. If lcgg 4.x actually surfaces it on `tool_calls[i]`, PROV-04 GeminiAdapter must iterate tool_calls and stash a list of signatures (one per call), then replay them by aligning indices on the next outbound. Executor adapts; PROV-04 has no merge gate so iteration cost is low.

- **Anthropic API key env var**: `app/config.py:93` `anthropic_api_key` reads from `ANTHROPIC_API_KEY` via pydantic-settings's default env-var mapping. PROV-03 plan must add a Makefile / docs callout that the developer needs `ANTHROPIC_API_KEY` set for the local gate run (D-09-10).

- **DeepSeek reasoner model name**: planner should verify the exact model name with `langchain-deepseek` docs. Most likely `deepseek-reasoner`; could be `deepseek-r1` or `deepseek-reasoner-v2` depending on the library's enum. The matrix YAML cell uses whatever string the library accepts.

- **Gemini 3 model name in PROV-04**: matrix already uses `gemini-3.1-pro-preview` per Phase 7 / `_GEMINI_THINKING_ONLY`. PROV-04 cell uses the same string for consistency.

</specifics>

<deferred>
## Deferred Ideas

- **Gemini 3 critique-loop fix** (memory `project_w10_migration_necessary_not_sufficient`): `LOW_SIMILARITY_THRESHOLD=0.55` in `app/agent/revision.py:21` + the low_similarity step-critique → "rephrase more broadly" prompt cause Gemini 3 to re-search forever and never commit. PROV-04 ships experimental WITHOUT this fix. Belongs in v2.2 or a Phase 10.5 follow-up; the fix touches prompt/critique contract and could cross-cut all providers including the gpt-4o-mini anchor — needs its own falsifier-shaped phase.
- **Promote Gemini 3 from experimental to gated** — PROV-FUT-01 in REQUIREMENTS.md. Triggered when both (a) PROV-04 conformance test passes AND (b) the critique-loop fix lands AND (c) Phase 10 cross-model baseline regen has a `gemini-3.1-pro-preview` measurement.
- **Kimi K2 / Moonshot adapter** — PROV-FUT-02. Memory `project_agent_loses_reasoning_state_all_providers` confirms Kimi 400s on the assistant tool_call payload because `langchain-moonshot` (or its predecessor) doesn't expose `reasoning_content` at the library boundary. Adapter is library-blocked. Phase 9 leaves `ADAPTERS["kimi"]` as `NoOpAdapter()`.
- **OpenAI o-series adapter** (o3, o4-mini) — PROV-FUT-03. Encrypted reasoning blocks are a different shape than gpt-5 family `reasoning_content` (the latter is what PROV-01 targets). PROV-01 ships gpt-5 specifically; o-series would be a separate sub-phase in a future milestone.
- **Custom imperative loop replacing LangGraph** — ARCH-FUT-01. Was conditional on Phase 8 REASON-05 firing; gate PASSED, so this stays in Future Requirements. Phase 9 proceeds on LangGraph.
- **Per-provider live-CI gate** (run all 4 cells in CI with secrets + cloud-sql-proxy) — Phase 10 BASE-03 explicitly. Phase 9 stays local-empirical + CI-structural per D-09-10.
- **Wholesale honest regen of all `configs/eval_baselines/*.json`** — Phase 10 BASE-01. Phase 9 does per-cell incremental updates only.
- **Cross-model `configs/eval_matrix.yaml` expansion** (gpt-5-mini, claude-sonnet-4-6, deepseek-reasoner as cross-model anchors) — Phase 10 BASE-02. Phase 9 only edits `configs/eval_matrix_refinement.yaml`.
- **`configs/eval_baselines/_snapshots/refinement_cheaper.pre-phase9.json`** — Considered (mirrors Phase 7 pattern) and rejected for Phase 9 since git history already preserves the pre-phase state and the per-sub-phase atomic commits make each cell change locatable. Phase 10 BASE-01 may re-snapshot before wholesale regen.
- **Import-isolation unit test for adapters** — Considered (a test that scans `app/agent/adapters/*.py` for cross-sibling imports to mechanically enforce PROV-05). Rejected in favor of CONTEXT.md convention (D-09-07) given the small surface; could be added in Phase 10 BASE-03 alongside CI promotion.
- **`budget_tokens` empirical tuning for Claude** — D-09-06 leaves the planner picking a default (~4096); Phase 10 baseline regen with claude-sonnet-4-6 as a cross-model anchor is the natural place to revisit if scores plateau.

</deferred>

---

*Phase: 09-per-provider-state-preservation-implementations*
*Context gathered: 2026-06-04*
