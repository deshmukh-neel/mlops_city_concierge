---
phase: 09-per-provider-state-preservation-implementations
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/probe_gpt5_capture.py
  - .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md
  - app/agent/adapters/openai_gpt5.py
  - app/agent/adapters/__init__.py
  - app/llm_factory.py
  - tests/unit/test_adapters.py
  - tests/integration/test_reasoning_state_roundtrip.py
  - configs/eval_matrix_refinement.yaml
  - configs/eval_baselines/refinement_cheaper.json
autonomous: false
requirements: [PROV-01]
requirements_addressed: [PROV-01]
user_setup:
  - service: openai
    why: "Local empirical gate (D-09-10) — gpt-5-mini live calls"
    env_vars:
      - name: OPENAI_API_KEY
        source: ".env or shell — already used by current openai/gpt-4o-mini path"
  - service: cloud-sql-proxy
    why: "make eval-matrix-refinement runs the real /chat path against pgvector; needs cloud-sql-proxy :5433 (memory project_local_backend_prod_db)"
    dashboard_config:
      - task: "Run `cloud-sql-proxy mlops--city-concierge:us-west2:postgres-18 --port 5433` in another terminal"
        location: "local terminal; DB instance is the double-dash mlops--city-concierge, DB inside is single-dash"

must_haves:
  truths:
    - "Running `python scripts/probe_gpt5_capture.py` with OPENAI_API_KEY set produces `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` with a verdict line (kwarg path works / subclass required / neither)"
    - "After PROV-01, `ADAPTERS['openai']` resolves to a real `OpenAIReasoningAdapter` instance (NOT `NoOpAdapter`)"
    - "`make eval-matrix-refinement RUNS=5` with `openai/gpt-5-mini × refinement_cheaper × prod × flag-on × temp=1.0` satisfies the D-09-02 2-part PR-blocking gate (re-scoped 2026-06-05 per user-approved Option A): Part A (hard) `committed_itinerary_rate ≥ 0.6`; Part B (advisory) `refinement_minimal_edit median ≥ 0.5`"
    - "`openai/gpt-4o-mini × refinement_cheaper` cell median unchanged vs the post-Phase-7 baseline (no regression on v2.0 prod anchor)"
    - "The OpenAI parametrize case of `test_reason_02_four_shape_roundtrip` passes with `OpenAIReasoningAdapter` swapped into `ADAPTERS['scripted']` (or the openai key) and a RecordingLLM scripted to emit an AIMessage carrying the openai-shape state"
  artifacts:
    - path: "scripts/probe_gpt5_capture.py"
      provides: "One-shot probe that hits gpt-5-mini and dumps AIMessage shape to a markdown artifact"
      min_lines: 30
    - path: ".planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md"
      provides: "Committed probe artifact recording langchain-openai version + additional_kwargs keys + verdict (Path A vs Path B)"
      contains: "## Verdict"
    - path: "app/agent/adapters/openai_gpt5.py"
      provides: "OpenAIReasoningAdapter(ProviderAdapter) — capture/replay for gpt-5 family reasoning_content"
      exports: ["OpenAIReasoningAdapter"]
    - path: "tests/unit/test_adapters.py"
      provides: "Unit tests for the new OpenAIReasoningAdapter — capture/replay isolated"
      contains: "test_openai_reasoning_adapter"
  key_links:
    - from: "app/agent/adapters/__init__.py"
      to: "app/agent/adapters/openai_gpt5.py"
      via: "ADAPTERS['openai'] = OpenAIReasoningAdapter()"
      pattern: "ADAPTERS\\[\"openai\"\\] *= *OpenAIReasoningAdapter"
    - from: "app/agent/graph.py:plan()"
      to: "OpenAIReasoningAdapter.capture_reasoning_state"
      via: "adapter resolved at build_agent_graph time + closed over plan()"
      pattern: "adapter.capture_reasoning_state"
    - from: "configs/eval_baselines/refinement_cheaper.json"
      to: "openai/gpt-5-mini cell"
      via: "median measurement from local n=5 run"
      pattern: "openai/gpt-5-mini"
---

<objective>
Land the first `ProviderAdapter` implementation — `OpenAIReasoningAdapter` for the gpt-5 family — and meet the v2.1 milestone anchor gate (D-09-02 re-scoped 2026-06-05 per user-approved Option A to a 2-part gate: Part A (hard) `committed_itinerary_rate ≥ 0.6`; Part B (advisory) `refinement_minimal_edit median ≥ 0.5`) per PROV-01 and D-09-02. This plan begins with a mandatory probe step (D-09-03) because the existing snapshot memory `project_agent_loses_reasoning_state_all_providers` was taken before the current `langchain-openai>=1.2.0` pin landed Responses-API passthrough; we don't yet know whether `gpt-5-mini`'s `reasoning_content` surfaces on `AIMessage.additional_kwargs` (Path A) or requires a `ChatOpenAI` subclass to lift it before the LangChain normalizer drops it (Path B). The probe produces a committed markdown artifact; the adapter implementation is finalized only after the probe lands.

Purpose: PROV-01 gate failure BLOCKS the entire Phase 9 PR (D-09-02). Without this adapter the milestone anchor gate cannot be met and PROV-02/03/04 cannot rescue the phase. Probe-then-build avoids guessing at the library boundary and keeps the iteration cost low.
Output: Probe artifact + `OpenAIReasoningAdapter` + ADAPTERS swap + conformance parametrize case + unit test + matrix YAML promotion + baseline JSON cell update + one revertable commit per task per D-09-01.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/09-per-provider-state-preservation-implementations/09-CONTEXT.md
@.planning/phases/09-per-provider-state-preservation-implementations/09-PATTERNS.md
@.planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/08-04-SUMMARY.md
@.planning/phases/07-prompt-rubric-decoupling/07-07-SUMMARY.md
@CLAUDE.md
@app/agent/adapters/__init__.py
@scripts/w10_convergence_check.py
@tests/integration/test_reasoning_state_roundtrip.py
@configs/eval_matrix_refinement.yaml
</context>

<tasks>

<task type="checkpoint:human-action" gate="blocking-human">
  <name>Task 1: PROV-01 probe — capture real gpt-5-mini AIMessage shape (D-09-03)</name>
  <files>scripts/probe_gpt5_capture.py, .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md</files>
  <read_first>
    - scripts/w10_convergence_check.py — closest analog for the dotenv-load + build_chat_model + call-provider pattern (see PATTERNS.md §scripts/probe_gpt5_capture.py)
    - .planning/phases/09-per-provider-state-preservation-implementations/09-CONTEXT.md §specifics first bullet — exact markdown sections required in the probe artifact
    - .planning/phases/09-per-provider-state-preservation-implementations/09-PATTERNS.md §scripts/probe_gpt5_capture.py (PROV-01 only) — D-09-03 spec for probe output sections + the "No analog found" note for the markdown-emit step (planner stubs from scratch)
    - CLAUDE.md — `make ingest` / dotenv conventions; `app` is poetry editable-installed so the script imports `from app.llm_factory import build_chat_model` with no sys.path hacks
  </read_first>
  <action>
    Create `scripts/probe_gpt5_capture.py` (~30-60 LOC). The script (a) loads `.env` via `dotenv` (mirrors `scripts/w10_convergence_check.py:198`), (b) calls `from app.llm_factory import build_chat_model` and constructs `gpt-5-mini` with `temperature=1.0` (no thinking-disable — we WANT to see whether reasoning_content surfaces), (c) sends a single `HumanMessage` that mimics the agent's own first-turn shape (e.g. "search for a bar in mission" — short, tool-call-prone) via `.invoke` or `.ainvoke`, (d) inspects the returned `AIMessage` for `additional_kwargs.keys()`, `response_metadata`, `content` shape (str vs list), `usage_metadata`, and `tool_calls`, (e) writes a markdown report to `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` with these REQUIRED sections (header names verbatim so the conformance grep gate below can verify): `## langchain-openai version` (via `importlib.metadata.version("langchain-openai")`), `## gpt-5-mini chat_model used`, `## AIMessage additional_kwargs keys`, `## AIMessage response_metadata`, `## AIMessage content shape`, `## AIMessage usage_metadata`, `## Raw dict(message) dump`, `## Verdict` (one of three exact strings: `kwarg path works`, `subclass required`, or `neither — escalate`). Per CONTEXT.md threat T1: the probe MUST NOT log raw `OPENAI_API_KEY` value or any `Authorization:` header; redact / never write the SecretStr's contents. Per memory `project_app_editable_install`: do NOT add `sys.path.insert` hacks — `from app.llm_factory import build_chat_model` works as-is. Per memory `feedback_temp1_reasoning_off_all_models`: temp=1.0 always; reasoning is left ON in this probe because we are explicitly checking whether reasoning_content surfaces.

    Then RUN the probe LOCALLY (this is the checkpoint:human-action half — needs a real OPENAI_API_KEY). Commit BOTH the script AND the resulting `09-PROV-01-PROBE.md` artifact in a single commit (`feat(09-01): probe gpt-5-mini AIMessage shape (PROV-01)`). The verdict line in the artifact dictates whether downstream Task 2 takes Path A (read additional_kwargs) or Path B (ChatOpenAI subclass). DO NOT proceed to Task 2 until the probe artifact is committed.
  </action>
  <what-built>A ~30-60 LOC probe script + a committed markdown artifact recording the live langchain-openai 1.2.x behavior for gpt-5-mini's reasoning_content field.</what-built>
  <how-to-verify>
    1. `ls scripts/probe_gpt5_capture.py` — exists.
    2. Run `OPENAI_API_KEY=... python scripts/probe_gpt5_capture.py` in a terminal where `.env` is loaded.
    3. After the script finishes, `cat .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` — confirm all required `##` sections are present and a verdict line exists.
    4. `grep -E 'sk-[A-Za-z0-9]{8}' .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` returns no matches (T1 secret-redaction check).
    5. Approve when the artifact is committed AND the verdict is one of the three exact strings.
  </how-to-verify>
  <resume-signal>Type "approved: Path A" or "approved: Path B" (or "approved: escalate" — in which case PR is blocked per D-09-02 and we open a Phase 9 blocker note).</resume-signal>
  <acceptance_criteria>
    - Source: `scripts/probe_gpt5_capture.py` exists; `grep -c "build_chat_model" scripts/probe_gpt5_capture.py` returns ≥ 1
    - Source: `scripts/probe_gpt5_capture.py` does NOT contain `sys.path.insert` (memory `project_app_editable_install` enforcement); `grep -c "sys.path.insert" scripts/probe_gpt5_capture.py` returns 0
    - Source: `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` exists and contains all of: `## langchain-openai version`, `## AIMessage additional_kwargs keys`, `## Verdict` (grep -c each returns ≥ 1)
    - Behavior / security: `grep -E 'sk-[A-Za-z0-9]{20,}' .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` returns no matches (T1 mitigation)
    - Behavior: `grep -E '^## Verdict$' -A 5 .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` shows one of: `kwarg path works`, `subclass required`, `neither — escalate`
    - Git: one commit named `feat(09-01): probe gpt-5-mini AIMessage shape (PROV-01)` (or equivalent single-line message per `feedback_small_focused_commits`) containing both the script and the markdown artifact
  </acceptance_criteria>
  <verify>
    <automated>test -f scripts/probe_gpt5_capture.py && test -f .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md && grep -q '^## Verdict' .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md</automated>
    <human-check>Verdict line is one of the three exact strings.</human-check>
  </verify>
  <done>Probe script committed, probe artifact committed, verdict determined (A or B), no leaked secrets.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement OpenAIReasoningAdapter + swap ADAPTERS['openai'] + conformance case + unit test</name>
  <files>app/agent/adapters/openai_gpt5.py, app/agent/adapters/__init__.py, app/llm_factory.py, tests/unit/test_adapters.py, tests/integration/test_reasoning_state_roundtrip.py</files>
  <read_first>
    - .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md — the verdict line dictates Path A vs Path B (Task 1 output)
    - app/agent/adapters/__init__.py:35-114 — `ProviderAdapter` ABC + `StatePayload` + `MockReasoningAdapter` (the exact analog to copy per PATTERNS.md §`app/agent/adapters/openai_gpt5.py`)
    - .planning/phases/09-per-provider-state-preservation-implementations/09-PATTERNS.md §`app/agent/adapters/openai_gpt5.py` and §"ABC contract (verbatim — adapters MUST match these signatures)" — interface signatures + Mock reference impl
    - .planning/phases/09-per-provider-state-preservation-implementations/09-PATTERNS.md §"Imports pattern" — exact import order; PROV-05 isolation rule (D-09-07)
    - .planning/phases/09-per-provider-state-preservation-implementations/09-PATTERNS.md §`app/agent/adapters/__init__.py:121 ADAPTERS registry mutation` — Option A (cell-by-cell) vs Option B (explicit literal). Recommendation: Option A here in Plan 09-01 since "anthropic" is not yet in SUPPORTED_PROVIDERS until Plan 09-03 — a hard-coded literal with "anthropic" would KeyError in unit tests; cell-by-cell mutation is safe.
    - tests/integration/test_reasoning_state_roundtrip.py:100-174 — `FOUR_SHAPE_PAYLOADS` literal + the parametrize body. The body does NOT change (D-08-13); the OpenAI case currently uses `MockReasoningAdapter` swapped into `ADAPTERS["scripted"]`. Plan 09-01 adds a NEW sibling test (or a new parametrize id) that swaps the REAL `OpenAIReasoningAdapter` into the same key and scripts an AIMessage carrying the openai-shape state.
    - app/llm_factory.py:205-206 — current openai branch (Path B may modify; Path A leaves it untouched)
    - .planning/phases/09-per-provider-state-preservation-implementations/09-CONTEXT.md §D-09-07 — PROV-05 isolation rule: imports ONLY from `app.agent.adapters` base + `langchain_core` + stdlib
  </read_first>
  <behavior>
    - Test 1 (Path A): `OpenAIReasoningAdapter().capture_reasoning_state(AIMessage(content="x", additional_kwargs={"reasoning_content": "thinking..."}))` returns `{"provider": "openai", "reasoning_content": "thinking..."}` (matches `FOUR_SHAPE_PAYLOADS[0]` shape)
    - Test 2: `capture_reasoning_state(AIMessage(content="x"))` (no reasoning_content key) returns `None` (the contract default per D-08-02; non-reasoning messages return None)
    - Test 3: `replay_reasoning_state(outbound=[HumanMessage("h"), AIMessage("a"), AIMessage("b")], state={"provider":"openai","reasoning_content":"r"})` returns the SAME list spine with the most-recent AIMessage's `additional_kwargs["reasoning_content"]` set to `"r"` (walks in reverse; matches MockReasoningAdapter shape but writes the provider-native field, NOT the `_reasoning_state` key — graph.py owns that key per PATTERNS.md §Shared Patterns)
    - Test 4: `replay_reasoning_state(outbound, state=None)` returns `outbound` byte-identical (no mutation)
    - Test 5 (mutation safety): after calling `capture_reasoning_state(msg)` on a message, the original `msg.additional_kwargs` is unchanged (T3 mitigation — capture returns a NEW dict; replay may mutate the outbound message but capture must not mutate its input)
    - Integration test: `test_reason_02_openai_real_adapter` (new sibling test in the same file) — swaps `OpenAIReasoningAdapter()` into `ADAPTERS["scripted"]` via `monkeypatch.setitem`, scripts a `RecordingLLM` with a turn-1 AIMessage carrying `additional_kwargs={"reasoning_content": "thinking..."}`, runs `graph.ainvoke`, asserts the turn-2 input's most-recent AIMessage carries the same `reasoning_content` value (via `additional_kwargs["reasoning_content"]`, NOT via `_reasoning_state` — graph.py stashes capture output at `_reasoning_state` but replay writes the provider-native key)
  </behavior>
  <action>
    PROV-01 implementation has two paths gated by the probe verdict from Task 1:

    **Path A (verdict: "kwarg path works"):** Create `app/agent/adapters/openai_gpt5.py` containing `OpenAIReasoningAdapter(ProviderAdapter)`. Mirror `MockReasoningAdapter` shape (PATTERNS.md). `capture_reasoning_state` reads `message.additional_kwargs.get("reasoning_content")`; returns `{"provider": "openai", "reasoning_content": <str>}` if present, else `None`. `replay_reasoning_state` walks `outbound` in reverse, finds the most-recent `AIMessage`, sets `msg.additional_kwargs["reasoning_content"] = state["reasoning_content"]`, returns `outbound` unchanged on `state is None`. Imports: ONLY `from __future__ import annotations`, stdlib (`typing`), `langchain_core.messages` (`AIMessage`, `BaseMessage`), `from app.agent.adapters import ProviderAdapter, StatePayload` per D-09-07. Do NOT import from sibling adapter files; do NOT modify `app/llm_factory.py` openai branch.

    **Path B (verdict: "subclass required"):** Same `app/agent/adapters/openai_gpt5.py` shape, PLUS edit `app/llm_factory.py:205-206` to introduce an `OpenAIReasoningChatModel(ChatOpenAI)` subclass at module scope that overrides `_generate` (or async `_agenerate`) to lift the raw response's `reasoning_content` field into `AIMessage.additional_kwargs["reasoning_content"]` BEFORE LangChain's normalizer drops it. The openai branch then returns `OpenAIReasoningChatModel(model=chat_model, ...)` when `chat_model.startswith("gpt-5")`, falling back to `ChatOpenAI(...)` for `gpt-4o-mini` and other non-reasoning models. Adapter is symmetric to Path A — reads from the subclass-enriched message. Document the subclass with a comment citing the probe verdict + langchain-openai version + memory `project_agent_loses_reasoning_state_all_providers` as the diagnosis history.

    Both paths then: swap `ADAPTERS["openai"]` from `NoOpAdapter()` to `OpenAIReasoningAdapter()` in `app/agent/adapters/__init__.py`. Use cell-by-cell mutation (PATTERNS.md Option A) — append `ADAPTERS["openai"] = OpenAIReasoningAdapter()` after the dict-comprehension at line 121 with an import added at the top of the file. DO NOT switch to the explicit-literal Option B in this plan because plan 09-03 will add `"anthropic"` to `SUPPORTED_PROVIDERS` — a hard-coded literal with `"anthropic"` would KeyError before 09-03 lands. Plan 09-04 is the right moment to consolidate to Option B (only after `SUPPORTED_PROVIDERS` is final).

    Create `tests/unit/test_adapters.py` (new file). Add `test_openai_reasoning_adapter` covering the 5 behaviors above. Use the same pytest style as existing `tests/unit/test_agent_graph.py` (sync functions; instantiate adapter directly; no graph wiring). Import the adapter as `from app.agent.adapters.openai_gpt5 import OpenAIReasoningAdapter` to verify the module path is correct.

    Extend `tests/integration/test_reasoning_state_roundtrip.py` with `test_reason_02_openai_real_adapter` (new sibling, marked with the same `reasoning_conformance` marker — module-level `pytestmark` already applies). Pattern matches `test_reason_02_four_shape_roundtrip` but swaps `OpenAIReasoningAdapter()` (not `MockReasoningAdapter`) into `ADAPTERS["scripted"]`, scripts a turn-1 AIMessage with `additional_kwargs={"reasoning_content": "thinking..."}` instead of the empty additional_kwargs the Mock case used, and asserts the turn-2 input's most-recent AIMessage carries `reasoning_content` (the provider-native field). DO NOT modify the existing `test_reason_02_four_shape_roundtrip[payload0]` (D-08-13 + canonical_refs lock it).

    Single commit when all tests are green: `feat(09-01): OpenAIReasoningAdapter + swap ADAPTERS['openai'] (PROV-01)`. Per `feedback_precommit_ruff`: do NOT run ruff manually; the pre-commit hook handles it.
  </action>
  <verify>
    <automated>pytest tests/unit/test_adapters.py::test_openai_reasoning_adapter -v && pytest tests/integration/test_reasoning_state_roundtrip.py -m reasoning_conformance -v && pytest tests/unit/test_agent_graph.py -v</automated>
  </verify>
  <acceptance_criteria>
    - Source: `grep -c "class OpenAIReasoningAdapter(ProviderAdapter)" app/agent/adapters/openai_gpt5.py` returns 1
    - Source: `grep -E "^from (sibling|app\\.agent\\.adapters\\.(anthropic|deepseek|gemini))" app/agent/adapters/openai_gpt5.py` returns no matches (D-09-07 import isolation)
    - Source: `grep -c "ADAPTERS\\[\"openai\"\\]" app/agent/adapters/__init__.py` returns ≥ 1 (swap landed)
    - Source: `grep -c "OpenAIReasoningAdapter" app/agent/adapters/__init__.py` returns ≥ 1 (import + swap)
    - Behavior / unit: `pytest tests/unit/test_adapters.py::test_openai_reasoning_adapter -v` exits 0
    - Behavior / integration: `pytest tests/integration/test_reasoning_state_roundtrip.py::test_reason_02_openai_real_adapter -m reasoning_conformance -v` exits 0
    - Behavior / regression: `pytest tests/integration/test_reasoning_state_roundtrip.py -m reasoning_conformance -v` reports 6 passed (5 from Phase 8 + 1 new openai real-adapter case)
    - Behavior / regression: `pytest tests/unit/test_agent_graph.py -v` reports 47 passed (no Phase 8 regression)
    - Behavior / mutation safety: the unit test asserts `capture_reasoning_state` does NOT mutate the input message's `additional_kwargs` (T3 mitigation; the test compares `dict(msg.additional_kwargs)` before/after the call)
    - Path B only: `grep -c "class OpenAIReasoningChatModel" app/llm_factory.py` returns 1; the openai branch dispatches by `chat_model.startswith("gpt-5")`
    - Path B only: existing `pytest tests/unit/test_llm_factory.py -v` exits 0 (no regression on openai/gpt-4o-mini construction)
  </acceptance_criteria>
  <done>OpenAIReasoningAdapter implemented (Path A or B per probe verdict), `ADAPTERS["openai"]` swapped, unit + integration tests pass, no regression on existing 47 test_agent_graph tests, no cross-adapter imports.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking-human">
  <name>Task 3: Local empirical gate — gpt-5-mini × refinement_cheaper × D-09-02 2-part gate (D-09-02 re-scoped, D-09-10) + matrix YAML + baseline JSON</name>
  <files>configs/eval_matrix_refinement.yaml, configs/eval_baselines/refinement_cheaper.json</files>
  <read_first>
    - configs/eval_matrix_refinement.yaml — current 3-cell shape (openai/gpt-4o-mini, deepseek/deepseek-chat, openai/gpt-5-mini logged-not-gated)
    - configs/eval_baselines/refinement_cheaper.json — current cell shape; openai/gpt-5-mini cell at scorers > refinement_minimal_edit currently has median = 0.0 from Phase 7 falsifier
    - .planning/phases/09-per-provider-state-preservation-implementations/09-PATTERNS.md §`configs/eval_matrix_refinement.yaml — per-cell pattern` and §`configs/eval_baselines/refinement_cheaper.json — per-cell pattern` (PROV-01 row of the per-sub-phase table)
    - .planning/phases/09-per-provider-state-preservation-implementations/09-CONTEXT.md §D-09-10, §D-09-11, §D-09-12 — local empirical gate procedure + per-cell baseline updates + matrix YAML comment changes
    - Makefile:132-145 — `eval-matrix-refinement` target shape (LIVE, APP_ENV=eval)
    - .planning/phases/07-prompt-rubric-decoupling/07-07-SUMMARY.md — Phase 7 procedure for `make eval-matrix-refinement RUNS=5` invocation
    - CLAUDE.md — `make eval-matrix-refinement-structural-check` is the CI hard gate; this plan's gate is LOCAL only (D-09-10)
  </read_first>
  <action>
    First update `configs/eval_matrix_refinement.yaml`: change the gpt-5-mini cell comment block from the Phase 7 "logged-not-gated" wording to PROV-01 GATED per PATTERNS.md §`configs/eval_matrix_refinement.yaml — per-cell pattern` PROV-01 bullet:

    ```
    # Phase 9 / D-09-12 / PROV-01: GATED. Milestone anchor gate (D-09-02 re-scoped
    # 2026-06-05 per user-approved Option A — see 09-PROV-01-BLOCKER.md Resolution):
    # 2-part — Part A (hard): committed_itinerary_rate ≥ 0.6 on temp=1.0 with
    # OpenAIReasoningAdapter; Part B (advisory): refinement_minimal_edit median ≥ 0.5.
    # Gate enforced by make eval-matrix-refinement (local) and
    # scripts/check_baselines_fresh.py (CI).
    ```

    The entry block (provider/model/env) stays byte-identical.

    Commit this YAML change as `chore(09-01): promote gpt-5-mini cell to gated (PROV-01)` — separate from the empirical gate run so the commit history clearly distinguishes "config promotion" from "data refresh".

    Then run the LOCAL empirical gate per D-09-10:
      1. Start `cloud-sql-proxy mlops--city-concierge:us-west2:postgres-18 --port 5433` in a second terminal (memory `project_local_backend_prod_db`).
      2. Confirm `OPENAI_API_KEY` (and `DEEPSEEK_API_KEY` for the deepseek cell — even though we're not gating it here, the matrix runs all 3 cells).
      3. Run `APP_ENV=eval make eval-matrix-refinement RUNS=5` (mirrors Phase 7 plan 07-07 procedure verbatim).
      4. Wait for completion (~5-10 minutes per cell × 3 cells × 5 runs).
      5. Inspect the per-cell aggregate JSON under `eval_reports/<timestamp>/` for the `openai/gpt-5-mini` cell's `refinement_minimal_edit` and `committed_itinerary_rate` medians.

    Apply the result to `configs/eval_baselines/refinement_cheaper.json`:
      - Update top-level `closure_check_confirmed` and `generated_at` to current dates.
      - Replace the `generated_by` string to reference Phase 9 PROV-01.
      - Update the `openai/gpt-5-mini` cell's `_observations` to: `"Phase 9 PROV-01 GATED. n=5 runs at temp=1.0, REFINEMENT_STRUCTURED_PLAN_ENABLED=true, OpenAIReasoningAdapter active. Milestone anchor gate met: refinement_minimal_edit median=<X>, committed_itinerary_rate=<Y>/5."` (per PATTERNS.md PROV-01 row).
      - Update each scorer in the gpt-5-mini cell with the fresh n=5 stats (max / median / min / n / stdev).
      - Update the `openai/gpt-4o-mini` and `deepseek/deepseek-chat` cells' stats with their fresh n=5 numbers (per D-09-11 — these cells run in the same matrix and their data refreshes; their `_observations` keep their Phase 7 / D-04-11 status text). NO REGRESSION on `openai/gpt-4o-mini × refinement_cheaper × refinement_minimal_edit` median vs the post-Phase-7 baseline (PROV-05 + D-09-02 spirit: don't break the v2.0 anchor).

    Commit the JSON refresh as `data(09-01): refresh refinement_cheaper baselines with PROV-01 n=5 medians (PROV-01)` — keeps `scripts/check_baselines_fresh.py origin/main` exit-0 per D-09-11.

    **D-09-02 PR-blocking gate (re-scoped 2026-06-05 per user-approved Option A — see 09-PROV-01-BLOCKER.md Resolution):** the gate is 2-part. **Part A (hard):** `openai/gpt-5-mini × committed_itinerary_rate` (mean across n=5) ≥ 0.6. **Part B (advisory):** `openai/gpt-5-mini × refinement_minimal_edit` median ≥ 0.5. If Part A fails, STOP — the Phase 9 PR cannot ship until Part A clears. If only Part B fails, log it in SUMMARY.md as accept-with-notes and continue. Either failure path opens / updates `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-BLOCKER.md` capturing per-run rationale_stop_alignment / refinement_minimal_edit / committed_itinerary_rate / revision_reasons distribution + the quantified gap (e.g. "Part A: 0.4 vs ≥0.6 — 1 more commit out of 5 would clear it"). Original strict wording ("median ≠ 1.0 / not 5/5 commits scoring 1.0") preserved in BLOCKER.md for the historical record.
  </action>
  <what-built>
    Matrix YAML promoted to GATED for gpt-5-mini cell. Baseline JSON refreshed with n=5 medians for all 3 cells in `configs/eval_matrix_refinement.yaml`. PROV-01 milestone anchor gate measured locally.
  </what-built>
  <how-to-verify>
    1. `git log --oneline -5` shows the two commits in order: matrix YAML promotion, then JSON data refresh.
    2. D-09-02 2-part gate check (re-scoped 2026-06-05): Part A (hard) — confirm `openai/gpt-5-mini × committed_itinerary_rate` ≥ 0.6 (PASS / FAIL); Part B (advisory) — confirm `openai/gpt-5-mini × refinement_minimal_edit` median ≥ 0.5 (PASS / accept-with-notes if FAIL). Part A failure blocks the PR; Part B failure does not.
    3. `python -c 'import json; d=json.load(open("configs/eval_baselines/refinement_cheaper.json")); print(d["providers"]["openai/gpt-4o-mini"]["scorers"]["refinement_minimal_edit"])'` median is unchanged vs the previous Phase 7 commit (no v2.0 anchor regression).
    4. `poetry run python scripts/check_baselines_fresh.py origin/main` exits 0 (D-09-11).
    5. `make eval-matrix-refinement-structural-check` exits 0 (CI hard gate stays green).
    6. The gpt-5-mini cell's `_observations` text contains the literal phrase `"Phase 9 PROV-01 GATED"`.
    7. The YAML comment for the gpt-5-mini cell contains `"D-09-12 / PROV-01: GATED"`.
  </how-to-verify>
  <resume-signal>Type "approved: gate PASSED" (D-09-02 Part A — committed_itinerary_rate ≥ 0.6 — clears; continue to Plan 09-02) OR "approved: gate FAILED — opening blocker" (Part A < 0.6; PR cannot ship per re-scoped D-09-02; we open `09-PROV-01-BLOCKER.md` and re-route).</resume-signal>
  <acceptance_criteria>
    - Config: `grep -c "D-09-12 / PROV-01: GATED" configs/eval_matrix_refinement.yaml` returns ≥ 1
    - Config: `grep -c "Phase 9 PROV-01 GATED" configs/eval_baselines/refinement_cheaper.json` returns ≥ 1
    - Behavior / staleness: `poetry run python scripts/check_baselines_fresh.py origin/main` exits 0
    - Behavior / structural: `make eval-matrix-refinement-structural-check` exits 0
    - Behavior / no-regression: the `openai/gpt-4o-mini × refinement_cheaper × refinement_minimal_edit` median in the refreshed JSON is ≥ the value in the previous commit (no v2.0 anchor regression)
    - Behavior / milestone gate (PR-blocking per re-scoped D-09-02, 2026-06-05 — Option A): Part A (hard) `openai/gpt-5-mini × committed_itinerary_rate ≥ 0.6`; if < 0.6 the plan TERMINATES and `09-PROV-01-BLOCKER.md` is updated with the failure rationale + quantified gap. Part B (advisory) `openai/gpt-5-mini × refinement_minimal_edit median ≥ 0.5`; if < 0.5 the plan logs accept-with-notes in SUMMARY.md and continues.
    - Git: matrix YAML promotion and JSON refresh are SEPARATE commits per `feedback_small_focused_commits`
  </acceptance_criteria>
  <verify>
    <automated>poetry run python scripts/check_baselines_fresh.py origin/main && make eval-matrix-refinement-structural-check  # D-09-02 Part A (hard) gate is committed_itinerary_rate ≥ 0.6 — measured from eval_reports/*/openai--gpt-5-mini--run-*.json aggregate, not from refinement_cheaper.json; see SUMMARY.md "Empirical Gate Result" section for the n=5 value
    <human-check>Median values confirmed against per-cell eval_reports/*/summary.json output; no regression on gpt-4o-mini anchor cell.</human-check>
  </verify>
  <done>YAML promoted to GATED for gpt-5-mini cell, JSON refreshed with n=5 medians for all 3 cells, milestone anchor gate measured against the re-scoped D-09-02 2-part gate (Part A hard `committed_itinerary_rate ≥ 0.6`; Part B advisory `refinement_minimal_edit median ≥ 0.5`), v2.0 anchor unchanged.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Local dev → OpenAI API | Probe script + matrix runs send representative tool-call queries with real OPENAI_API_KEY |
| Local dev → Cloud SQL | Matrix runs go through cloud-sql-proxy :5433 to the prod-equivalent pgvector |
| Disk → git repo | Probe artifact + baseline JSON committed to .planning/ and configs/ |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-09-01-T1 | Information Disclosure | scripts/probe_gpt5_capture.py + 09-PROV-01-PROBE.md | mitigate | Probe writes ONLY `additional_kwargs.keys()`, `response_metadata` (sanitized), `content` shape, and the verdict line — NEVER `OPENAI_API_KEY` value or `Authorization:` headers. Acceptance criterion greps for `sk-[A-Za-z0-9]{20,}` in the artifact and asserts zero matches. |
| T-09-01-T2 | Information Disclosure | CI live-provider key leak | mitigate | Phase 9 gates execute LOCALLY only (D-09-10). No GitHub Actions workflow file added or modified in Plan 09-01. CI continues with structural-check + check_baselines_fresh.py only. |
| T-09-01-T3 | Tampering | Adapter capture/replay mutating prior-turn state | mitigate | Unit test `test_openai_reasoning_adapter` asserts `capture_reasoning_state` does NOT mutate the input AIMessage's `additional_kwargs` (compares `dict(msg.additional_kwargs)` before and after). Adapter returns a new StatePayload dict from capture. |
</threat_model>

<verification>
- All Phase 8 conformance tests still pass (5/5 + 1 new openai real-adapter sibling test = 6 expected under `-m reasoning_conformance`)
- All Phase 8 unit tests still pass (`pytest tests/unit/test_agent_graph.py -v` = 47 passed)
- New `tests/unit/test_adapters.py::test_openai_reasoning_adapter` passes
- `scripts/check_baselines_fresh.py origin/main` exits 0
- `make eval-matrix-refinement-structural-check` exits 0 (CI hard gate)
- `make test` exits 0 (no regression sweep)
- PR-blocking gate per re-scoped D-09-02 (2026-06-05 Option A): Part A (hard) `openai/gpt-5-mini × committed_itinerary_rate` mean ≥ 0.6 (measured from `eval_reports/*/openai--gpt-5-mini--run-*.json` aggregates); Part B (advisory) `refinement_minimal_edit` median ≥ 0.5 in the refreshed `configs/eval_baselines/refinement_cheaper.json`
- v2.0 anchor unchanged: `openai/gpt-4o-mini × refinement_minimal_edit` median in the refreshed JSON is ≥ the previous-commit value
- `app/agent/adapters/openai_gpt5.py` imports satisfy D-09-07: no imports from sibling adapter files
</verification>

<success_criteria>
PROV-01 is met when:
1. `OpenAIReasoningAdapter` class lives in `app/agent/adapters/openai_gpt5.py` and is registered at `ADAPTERS["openai"]` (replacing `NoOpAdapter`).
2. The openai parametrize sibling case in `tests/integration/test_reasoning_state_roundtrip.py` passes with the real `OpenAIReasoningAdapter`.
3. `tests/unit/test_adapters.py::test_openai_reasoning_adapter` passes.
4. `make eval-matrix-refinement RUNS=5` locally with OPENAI_API_KEY + cloud-sql-proxy yields `gpt-5-mini × refinement_cheaper` satisfying the re-scoped D-09-02 2-part gate — Part A (hard) `committed_itinerary_rate ≥ 0.6`; Part B (advisory) `refinement_minimal_edit median ≥ 0.5` (re-scoped 2026-06-05 per user-approved Option A; see `09-PROV-01-BLOCKER.md` Resolution).
5. `configs/eval_matrix_refinement.yaml` gpt-5-mini cell carries the `GATED` comment.
6. `configs/eval_baselines/refinement_cheaper.json` gpt-5-mini cell records the n=5 medians + the PROV-01 GATED `_observations` text.
7. Probe artifact `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` committed with a verdict line.
8. v2.0 `openai/gpt-4o-mini × refinement_cheaper` cell unchanged or improved.
9. No imports across sibling adapter files (D-09-07).
</success_criteria>

<output>
Create `.planning/phases/09-per-provider-state-preservation-implementations/09-01-SUMMARY.md` capturing:
- Probe verdict (Path A or B) + which adapter shape was implemented
- Unit + integration test outcomes (`pytest -v` final summary lines)
- Local n=5 medians for each of the 3 cells in `configs/eval_matrix_refinement.yaml` (verbatim from the eval_reports/*/summary.json)
- PR-blocking gate outcome (PASS = median 1.0; FAIL = blocker materialized) + link to `09-PROV-01-BLOCKER.md` if FAIL
- 3 atomic commits in the expected order (per `feedback_small_focused_commits`):
  1. `feat(09-01): probe gpt-5-mini AIMessage shape (PROV-01)` (probe script + artifact)
  2. `feat(09-01): OpenAIReasoningAdapter + swap ADAPTERS['openai'] (PROV-01)` (adapter + tests)
  3. `chore(09-01): promote gpt-5-mini cell to gated (PROV-01)` (YAML comment)
  4. `data(09-01): refresh refinement_cheaper baselines with PROV-01 n=5 medians (PROV-01)` (JSON refresh)
</output>