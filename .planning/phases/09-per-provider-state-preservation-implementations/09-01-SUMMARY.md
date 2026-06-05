---
phase: 09-per-provider-state-preservation-implementations
plan: 01
subsystem: agent
tags: [openai, gpt-5, langchain, responses-api, provider-adapter, reasoning-state, refinement]

# Dependency graph
requires:
  - phase: 08-reasoning-state-thread-through-contract-conformance-harness
    provides: ProviderAdapter ABC, ADAPTERS registry, graph.py capture/replay sites, 4-shape conformance fixture
provides:
  - OpenAIReasoningAdapter (first non-NoOp ADAPTERS entry; reads/writes additional_kwargs["reasoning_content"])
  - OpenAIReasoningChatModel (ChatOpenAI subclass; lifts Responses-API reasoning blocks into additional_kwargs)
  - gpt-5 dispatch path in build_chat_model gated by _is_openai_reasoning_model (gpt-4o-mini intentionally excluded)
  - GATED matrix cell for openai/gpt-5-mini × refinement_cheaper
  - Refreshed n=5 baseline JSON for all 3 refinement-matrix cells
  - BLOCKER doc 09-PROV-01-BLOCKER.md (D-09-02 milestone anchor gate FAILED)
affects: [09-02 deepseek, 09-03 anthropic, 09-04 gemini, v2.1 milestone close]

# Tech tracking
tech-stack:
  added: [Responses-API surface (langchain-openai 1.2.2 already pinned; no new deps)]
  patterns:
    - "Provider-specific ChatModel subclass overrides _generate/_agenerate to lift wire-shape fields into AIMessage.additional_kwargs so adapter contract reads via the documented path"
    - "Dispatch by model-family predicate (_is_openai_reasoning_model) inside build_chat_model — keeps v2.0 anchor (gpt-4o-mini) on plain ChatOpenAI"
    - "ADAPTERS registry: cell-by-cell mutation (Option A) keeps anthropic key absent until PROV-03 adds it to SUPPORTED_PROVIDERS"

key-files:
  created:
    - scripts/probe_gpt5_capture.py
    - .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md
    - app/agent/adapters/openai_gpt5.py
    - tests/unit/test_adapters.py
    - .planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-BLOCKER.md
  modified:
    - app/llm_factory.py
    - app/agent/adapters/__init__.py
    - tests/integration/test_reasoning_state_roundtrip.py
    - configs/eval_matrix_refinement.yaml
    - configs/eval_baselines/refinement_cheaper.json

key-decisions:
  - "Probe verdict was Path B (subclass required) — langchain-openai 1.2.2 Chat-Completions wrapper for gpt-5-mini surfaces only reasoning_tokens counter, never the text. Switched gpt-5 family onto Responses API via use_responses_api=True."
  - "_is_openai_reasoning_model() predicate scopes the subclass to chat_model.startswith('gpt-5') only — gpt-4o-mini stays on plain ChatOpenAI to honor CLAUDE.md v2.0 anchor rule. Empirically confirmed no regression on gpt-4o-mini cell."
  - "ADAPTERS registry mutated cell-by-cell (Option A) — Option B explicit literal would KeyError on 'anthropic' before PROV-03 lands."
  - "Milestone anchor gate D-09-02 (gpt-5-mini × refinement_minimal_edit median = 1.0) FAILED at median = 0.0 — Phase 9 PR blocked pending Option-A/B/C/D triage in 09-PROV-01-BLOCKER.md."

patterns-established:
  - "Responses-API reasoning lift pattern: subclass overrides _generate/_agenerate, scans content for {'type': 'reasoning'} blocks, shallow-copies into additional_kwargs['reasoning_content']. Sibling providers (DeepSeek-reasoner, Gemini thought_signature) reuse this shape."
  - "Conformance harness extension pattern: new sibling test test_reason_02_openai_real_adapter swaps OpenAIReasoningAdapter into ADAPTERS['scripted'] and scripts an AIMessage with the provider-native key — does NOT modify the locked test_reason_02_four_shape_roundtrip body (D-08-13 / canonical_refs lock)."
  - "Local-only empirical gate: matrix run is LOCAL (D-09-10) — no GitHub Actions surface added. CI continues with structural-check + check_baselines_fresh.py only."

requirements-completed: []  # PROV-01 NOT marked complete — milestone anchor gate FAILED per D-09-02. Will be set once blocker is resolved.

# Metrics
duration: ~30min (Task 3 / eval matrix wall-clock; plan total ~60min including Tasks 1+2 from prior conversation)
completed: 2026-06-04
---

# Phase 9 Plan 01: OpenAI GPT-5 Adapter Summary

**OpenAIReasoningAdapter + OpenAIReasoningChatModel ship cleanly via Path B (Responses API + reasoning-block lift); milestone anchor gate D-09-02 FAILED at gpt-5-mini median 0.0 — Phase 9 PR is blocked pending gate-restructure decision.**

## Performance

- **Duration (Task 3 only):** ~30 min (matrix wall-clock)
- **Started (Task 3):** 2026-06-04T20:35Z (matrix kickoff)
- **Completed (Task 3):** 2026-06-04T21:06Z (last gpt-5-mini run wrote)
- **Tasks:** 3/3 mechanically executed; gate result is FAILED, plan NOT scope-complete
- **Files modified:** 10 (5 new, 5 edited — see key-files)

## Accomplishments

- Probe-then-build (D-09-03) executed: probe artifact committed, verdict Path B locked the implementation shape.
- OpenAIReasoningChatModel ships in app/llm_factory.py: _generate / _agenerate override, _lift_reasoning_blocks shallow-copies reasoning content blocks into AIMessage.additional_kwargs["reasoning_content"].
- OpenAIReasoningAdapter ships in app/agent/adapters/openai_gpt5.py with full ProviderAdapter contract: capture/replay round-trip via additional_kwargs["reasoning_content"], mutation-safe.
- ADAPTERS["openai"] swapped from NoOpAdapter to OpenAIReasoningAdapter (cell-by-cell, Option A).
- 5 unit tests + 1 new integration sibling test pass; no regression on Phase-8 47-test suite or reasoning_conformance harness.
- Matrix YAML promoted to GATED for the gpt-5-mini cell.
- Empirical gate run locally: 3 cells × 5 runs against live OpenAI + DeepSeek APIs + cloud-sql-proxy.
- Baseline JSON refreshed with n=5 medians for all 3 cells; staleness gate (`scripts/check_baselines_fresh.py origin/main`) exits 0; structural gate (`make eval-matrix-refinement-structural-check`) exits 0.

## Task Commits

Each task was committed atomically per `feedback_small_focused_commits`:

1. **Task 1: Probe gpt-5-mini AIMessage shape (PROV-01)** — `c2f8537` (feat)
2. **Task 2: OpenAIReasoningAdapter + swap ADAPTERS['openai'] (PROV-01)** — `954d761` (feat; includes Path B llm_factory subclass)
3. **Task 3a: Promote gpt-5-mini cell to gated (PROV-01)** — `eb0ebe4` (chore; YAML-only)
4. **Task 3b: Refresh refinement_cheaper baselines with PROV-01 n=5 medians (PROV-01)** — `7532522` (data; JSON-only)
5. **This SUMMARY + BLOCKER** — pending commit at end of executor run.

## Files Created/Modified

**New:**
- `scripts/probe_gpt5_capture.py` — One-shot Path A/B probe script (~50 LOC; build_chat_model + invoke + markdown report).
- `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` — Probe artifact with langchain-openai==1.2.2 / additional_kwargs={refusal} / Verdict: subclass required.
- `app/agent/adapters/openai_gpt5.py` — OpenAIReasoningAdapter ProviderAdapter implementation (~85 LOC).
- `tests/unit/test_adapters.py` — 5 unit tests covering capture/replay/mutation-safety contracts.
- `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-BLOCKER.md` — D-09-02 gate-failure triage doc with 4 resolution options.

**Modified:**
- `app/llm_factory.py` — Added OpenAIReasoningChatModel subclass (lines 59-138), _is_openai_reasoning_model predicate (lines 152-158), gpt-5 dispatch branch (lines 307-316).
- `app/agent/adapters/__init__.py` — Import OpenAIReasoningAdapter, mutate ADAPTERS["openai"] post-comprehension (Option A per D-09-07).
- `tests/integration/test_reasoning_state_roundtrip.py` — New sibling test test_reason_02_openai_real_adapter (scripts AIMessage with additional_kwargs["reasoning_content"], asserts replay writes provider-native key).
- `configs/eval_matrix_refinement.yaml` — gpt-5-mini cell comment block updated from "Phase 7 logged-not-gated" to "Phase 9 / D-09-12 / PROV-01: GATED" per D-09-12.
- `configs/eval_baselines/refinement_cheaper.json` — n=5 medians refreshed for all 3 cells; gpt-5-mini cell `_observations` records GATE FAILED + links to BLOCKER.

## Decisions Made

See key-decisions in frontmatter. Key technical choices:

- **Path B over Path A:** probe found langchain-openai 1.2.2's Chat-Completions wrapper does NOT expose reasoning text on AIMessage at all (only a `reasoning_tokens` counter in usage). Path A (read from additional_kwargs without library change) was impossible; Path B (Responses-API subclass) was the only viable shape.
- **Shallow-copy lift, not move:** _lift_reasoning_blocks COPIES reasoning blocks to additional_kwargs rather than moving them. This keeps LangChain's Responses-API outbound serializer (which round-trips blocks in `content`) working on the wire while the adapter contract still has access via the documented additional_kwargs path.
- **Hard scope to gpt-5 family:** `_is_openai_reasoning_model` only matches `chat_model.startswith("gpt-5")`. gpt-4o-mini, gpt-4o, and gpt-4-turbo all stay on plain ChatOpenAI. CLAUDE.md memory `project_reasoning_models_break_agent_loop` notes gpt-4o-mini is the locked v2.0 anchor — changing its wire path is out of scope.
- **Option A registry mutation:** kept `ADAPTERS = {p: NoOpAdapter() for p in SUPPORTED_PROVIDERS}` + `ADAPTERS["openai"] = OpenAIReasoningAdapter()` instead of rewriting to an explicit literal. Plan 09-03 adds "anthropic" to SUPPORTED_PROVIDERS; a literal would have KeyError'd before 09-03 ships.

## Empirical Gate Result (D-09-02)

**Matrix:** `configs/eval_matrix_refinement.yaml` × `refinement_cheaper` × n=5 × temp=1.0 × `REFINEMENT_STRUCTURED_PLAN_ENABLED=true`.
**Output dir:** `eval_reports/2026-06-05T03-35-57Z/` (UTC timestamp; local 2026-06-04 evening).

| Cell                       | refinement_minimal_edit per-run | median | max | committed_itinerary_rate | Gate     |
| -------------------------- | ------------------------------- | ------ | --- | ------------------------ | -------- |
| openai/gpt-5-mini (GATED)  | [0.0, 0.0, 0.0, 0.0, 0.0]       | **0.0** | 0.0 | 0.4 (2/5)                | **FAILED** (target median 1.0 per D-09-02) |
| openai/gpt-4o-mini (ref)   | [0.0, 0.5, 0.0, 0.0, 0.0]       | 0.0    | 0.5 | 1.0 (5/5)                | n/a (no regression vs prior 0.0/0.5) |
| deepseek/deepseek-chat (ref) | [0.0, 0.0, 0.0, 0.0, 0.0]     | 0.0    | 0.0 | 0.0 (0/5)                | n/a (no change; PROV-02 territory) |

**Per-run summary for the gated cell:** see `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-BLOCKER.md` for full table including tool_calls, revision_reasons, latency, and final_reply head per run.

**No regression on v2.0 anchor:** openai/gpt-4o-mini `refinement_minimal_edit` median and max are identical to the prior baseline; the same per-run distribution shape (one 0.5, four 0.0) is preserved, confirming the gpt-5 Path B subclass does NOT spill onto gpt-4o-mini.

**Why the gate failed:** Path B subclass demonstrably works at the wire level (reasoning blocks reach additional_kwargs; agent commits on 2/5 runs vs Phase-7 falsifier's 0/5). What does NOT work:
1. The 2 committed runs choose wholesale stop replacements over minimal edits → `refinement_minimal_edit` returns 0.0 even on success.
2. The 3 non-committed runs hit max_steps via `low_similarity` revision-loop retries.

Root cause is downstream of provider state preservation: refinement edit-minimization is prompt/critique-loop territory, not adapter territory. See BLOCKER for Option A (recommended: re-scope gate) and Options B/C/D (technical mitigations).

## Deviations from Plan

### Auto-fixed Issues

None during this conversation — Tasks 1, 2, 3a, and the empirical-gate run executed exactly as written in the PLAN. The plan's `<action>` for Task 3 anticipated the FAIL branch and instructed: "if median ≠ 1.0, STOP, open a phase blocker note." That branch was taken cleanly.

### Plan-Anticipated Outcome (NOT a deviation)

The PLAN's Task 3 action explicitly handles the gate-failure path:

> "**D-09-02 PR-blocking gate:** if `openai/gpt-5-mini × refinement_minimal_edit` median ≠ 1.0 (i.e., not 5/5 commits scoring 1.0), STOP. The entire Phase 9 PR cannot ship per D-09-02. Open a phase blocker note (`.planning/phases/09-.../09-PROV-01-BLOCKER.md`) capturing the median value, the per-run rationale_stop_alignment / refinement_minimal_edit / revision_reasons distribution, and re-route to either (a) probe Path B if Path A was tried and the median is < 5/5, or (b) flag for v2.1 replan per D-09-02 + memory `project_reasoning_models_break_agent_loop`."

Path B was already chosen (probe verdict), so route (a) is not applicable. Route (b) requires user input on the four triage options documented in the BLOCKER.

---

**Total deviations:** 0.
**Impact on plan:** Mechanical scope delivered; empirical gate FAILED. Plan NOT marked requirements-complete pending user triage decision on D-09-02 re-scope.

## Issues Encountered

The empirical gate failure itself is the headline issue. See BLOCKER for full diagnosis. Summary: provider state preservation works (Path B subclass + adapter wire correctly, 2/5 commits succeed where Phase-7 had 0/5), but the milestone gate as written (`refinement_minimal_edit median == 1.0`) measures refinement edit-minimization, not state preservation. The v2.0 anchor itself sits at median 0.0 / max 0.5 under the same scorer — holding gpt-5-mini to median 1.0 is asymmetric with the anchor.

## Test results (n=5 unit + 6 conformance + 47 graph)

- `pytest tests/unit/test_adapters.py -v` → 5 passed
- `pytest tests/integration/test_reasoning_state_roundtrip.py -m reasoning_conformance -v` → 6 passed (5 from Phase 8 + 1 new openai sibling)
- `pytest tests/unit/test_agent_graph.py -v` → 47 passed (Phase 8 baseline, no regression)
- `poetry run python scripts/check_baselines_fresh.py origin/main` → exit 0
- `make eval-matrix-refinement-structural-check` → exit 0

## User Setup Required

None — `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, and `cloud-sql-proxy --port 5433` were all live in this conversation. The plan's `user_setup` block is informational; no new env vars or dashboard config needed.

## Next Phase Readiness

**BLOCKED.** Per D-09-02 the entire Phase 9 PR cannot ship until the milestone anchor gate resolves. The OpenAIReasoningAdapter is technically complete and correct, but the gate as written is unmet. Plan 09-02 (DeepSeek), 09-03 (Anthropic), and 09-04 (Gemini) should NOT start until the user picks a triage option:

- **Option A (recommended):** Re-scope D-09-02 to soft `refinement_minimal_edit` threshold + hard `committed_itinerary_rate` threshold; ship PROV-01 as scope-complete and move edit-minimization to a follow-on plan.
- **Option B:** Add gpt-5-specific imperative preamble to `build_refinement_prompt_message` and re-run the matrix.
- **Option C:** Raise `MAX_PLAN_STEPS` for gpt-5 family.
- **Option D:** Tighten `LOW_SIMILARITY_THRESHOLD` to break the critique-commit conflict.

Full triage discussion in `09-PROV-01-BLOCKER.md`.

## Self-Check: PASSED

- File `scripts/probe_gpt5_capture.py` — FOUND
- File `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md` — FOUND
- File `app/agent/adapters/openai_gpt5.py` — FOUND
- File `tests/unit/test_adapters.py` — FOUND
- File `configs/eval_matrix_refinement.yaml` (GATED comment) — FOUND
- File `configs/eval_baselines/refinement_cheaper.json` (n=5 refresh) — FOUND
- File `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-BLOCKER.md` — FOUND
- Commit `c2f8537` (Task 1 probe) — FOUND
- Commit `954d761` (Task 2 adapter+subclass) — FOUND
- Commit `eb0ebe4` (Task 3a YAML promote) — FOUND
- Commit `7532522` (Task 3b baseline refresh) — FOUND

---
*Phase: 09-per-provider-state-preservation-implementations*
*Plan: 09-01*
*Completed (mechanical): 2026-06-04 — milestone gate FAILED; awaiting user triage*
