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
  - BLOCKER doc 09-PROV-01-BLOCKER.md (D-09-02 milestone anchor gate — re-scoped 2026-06-05 per user-approved Option A; Part A hard 0.4 vs ≥0.6 still fails, Part B advisory 0.0 vs ≥0.5 still fails)
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
  - "Milestone anchor gate D-09-02 RE-SCOPED 2026-06-05 per user-approved Option A: original strict `refinement_minimal_edit median = 1.0` (was a Phase-6 baseline-saturation artifact) became 2-part gate — Part A (hard) `committed_itinerary_rate ≥ 0.6`; Part B (advisory) `refinement_minimal_edit median ≥ 0.5`. Empirical n=5 against re-scoped gate: Part A = 0.4 (2/5) STILL FAILS by 0.2; Part B = 0.0 STILL FAILS but advisory. Phase 9 PR remains blocked pending user ship/re-run decision."

patterns-established:
  - "Responses-API reasoning lift pattern: subclass overrides _generate/_agenerate, scans content for {'type': 'reasoning'} blocks, shallow-copies into additional_kwargs['reasoning_content']. Sibling providers (DeepSeek-reasoner, Gemini thought_signature) reuse this shape."
  - "Conformance harness extension pattern: new sibling test test_reason_02_openai_real_adapter swaps OpenAIReasoningAdapter into ADAPTERS['scripted'] and scripts an AIMessage with the provider-native key — does NOT modify the locked test_reason_02_four_shape_roundtrip body (D-08-13 / canonical_refs lock)."
  - "Local-only empirical gate: matrix run is LOCAL (D-09-10) — no GitHub Actions surface added. CI continues with structural-check + check_baselines_fresh.py only."

requirements-completed: []  # PROV-01 NOT marked complete — re-scoped D-09-02 (2026-06-05 Option A) Part A (hard) still fails at 0.4 vs ≥0.6 needed. Will be set once Part A clears (either via re-run with tighter CI or via additional triage path).

# Metrics
duration: ~30min (Task 3 / eval matrix wall-clock; plan total ~60min including Tasks 1+2 from prior conversation; +~10min for the 2026-06-05 re-scope pass)
completed: 2026-06-04 (mechanical execution); 2026-06-05 (D-09-02 re-scope landed); plan still HELD pending user ship/re-run decision against re-scoped gate
---

# Phase 9 Plan 01: OpenAI GPT-5 Adapter Summary

**OpenAIReasoningAdapter + OpenAIReasoningChatModel ship cleanly via Path B (Responses API + reasoning-block lift). D-09-02 milestone anchor gate was re-scoped 2026-06-05 per user-approved Option A from strict `refinement_minimal_edit median = 1.0` to a 2-part gate (Part A hard `committed_itinerary_rate ≥ 0.6`; Part B advisory `refinement_minimal_edit median ≥ 0.5`). Against the re-scoped gate the n=5 result is Part A = 0.4 (FAILS by 0.2, i.e. 1 more commit out of 5 would clear it) and Part B = 0.0 (FAILS advisory). Phase 9 PR remains held pending user decision: (1) ship with documented Part A gap, (2) re-run at higher n to tighten the confidence interval, or (3) apply a separate technical mitigation (Option B/C/D in BLOCKER.md).**

## Performance

- **Duration (Task 3 only):** ~30 min (matrix wall-clock)
- **Started (Task 3):** 2026-06-04T20:35Z (matrix kickoff)
- **Completed (Task 3):** 2026-06-04T21:06Z (last gpt-5-mini run wrote)
- **Tasks:** 3/3 mechanically executed (Tasks 1, 2, 3a, 3b committed atomically); post-execution D-09-02 re-scope applied 2026-06-05 per user-approved Option A; against re-scoped gate Part A (hard) still fails at 0.4 vs ≥0.6; plan NOT scope-complete (HELD pending user decision)
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
5. **Original SUMMARY + BLOCKER draft** — `4333407` (docs)
6. **STATE.md blocker note** — `f27d352` (docs)
7. **D-09-02 re-scope (CONTEXT, PLAN, YAML, ROADMAP)** — `b072806` (docs; user-approved Option A on 2026-06-05)
8. **SUMMARY + BLOCKER updated for re-scoped gate** — pending commit at end of this executor run.

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

## Empirical Gate Result (D-09-02 — re-scoped 2026-06-05 per user-approved Option A)

**Matrix:** `configs/eval_matrix_refinement.yaml` × `refinement_cheaper` × n=5 × temp=1.0 × `REFINEMENT_STRUCTURED_PLAN_ENABLED=true`.
**Output dir:** `eval_reports/2026-06-05T03-35-57Z/` (UTC timestamp; local 2026-06-04 evening).

### Re-scoped gate definition

| Part | Threshold | Hard/Advisory | Source                                                                                  |
| ---- | --------- | ------------- | --------------------------------------------------------------------------------------- |
| A    | `committed_itinerary_rate ≥ 0.6` | **HARD (PR-blocking)** | eval_reports/*/openai--gpt-5-mini--run-*.json `aggregate.committed_itinerary_rate`, averaged across n=5 |
| B    | `refinement_minimal_edit median ≥ 0.5` | Advisory (logged-but-not-blocking) | configs/eval_baselines/refinement_cheaper.json `providers["openai/gpt-5-mini"].scorers.refinement_minimal_edit.median` |

### Measurement

| Cell                       | refinement_minimal_edit per-run | median | committed_itinerary_rate per-run | mean | Part A (≥0.6) | Part B (≥0.5) |
| -------------------------- | ------------------------------- | ------ | -------------------------------- | ---- | ------------- | ------------- |
| openai/gpt-5-mini (GATED)  | [0.0, 0.0, 0.0, 0.0, 0.0]       | 0.0    | [0.0, 0.0, 1.0, 0.0, 1.0]        | **0.4** | **FAIL** (0.4 vs ≥0.6 — short by 0.2 / 1 commit out of 5) | FAIL (0.0 vs ≥0.5; advisory) |
| openai/gpt-4o-mini (ref)   | [0.0, 0.5, 0.0, 0.0, 0.0]       | 0.0    | [1.0, 1.0, 1.0, 1.0, 1.0]        | 1.0  | n/a (v2.0 anchor; no regression vs prior 0.0/0.5) | n/a |
| deepseek/deepseek-chat (ref) | [0.0, 0.0, 0.0, 0.0, 0.0]     | 0.0    | [0.0, 0.0, 0.0, 0.0, 0.0]        | 0.0  | n/a (PROV-02 territory; no change) | n/a |

**Gate verdict (re-scoped):** Part A (hard) **FAILS** at 0.4 vs ≥0.6 needed — 1 more commit out of 5 would clear it. Part B (advisory) fails at 0.0 vs ≥0.5 needed — but does not block the PR. Per the re-scoped D-09-02, Phase 9 PR cannot ship until Part A clears.

### v2.0 anchor non-regression

openai/gpt-4o-mini `refinement_minimal_edit` median and max are identical to the prior baseline; the same per-run distribution shape (one 0.5, four 0.0) is preserved, confirming the gpt-5 Path B subclass does NOT spill onto gpt-4o-mini. `committed_itinerary_rate` = 1.0 (5/5) for gpt-4o-mini also confirms dispatch correctly excludes it from the subclass.

### Why Part A is at 0.4 (per-run diagnosis)

- **run-0, run-1, run-3:** hit max_steps via `low_similarity` revision-loop retries; `final_reply` = "I hit the planning step limit. Here is the best plan I had so far." No commit.
- **run-2, run-4:** agent committed successfully (`committed_itinerary_rate` = 1.0 for these runs).

Path B subclass demonstrably works at the wire level (reasoning blocks reach additional_kwargs; agent commits on 2/5 runs vs Phase-7 falsifier's 0/5). The remaining 3-step-limit failures are downstream of provider state preservation — the documented `project_critique_commit_conflict` interaction between `low_similarity` revision-loop and `commit_itinerary` decisiveness — and would not be expected to clear via further adapter work.

### Why Part B is at 0.0

Even when the agent commits (run-2, run-4), it chooses wholesale stop replacements over minimal edits, so `refinement_minimal_edit` returns 0.0 on the committed runs. This is prompt/critique-loop territory (the user-approved Option A explicitly carves edit-minimization out of the Phase 9 charter and defers it to v2.1 phase 2 prompt-rubric refinement), which is why Part B is advisory rather than hard.

### Variance caveat

The Part A gap is 1 commit out of 5 (0.2 absolute on a binary outcome). At n=5 a single flip from run-0/1/3 step-limited to committed would put the rate at 3/5 = 0.6 exactly. The confidence interval is wide; a re-run at n=10 or n=20 to tighten the CI is a plausible next step before declaring the gap "real". See `09-PROV-01-BLOCKER.md` Resolution section for the user's options.

## Deviations from Plan

### Auto-fixed Issues

None during the original execution — Tasks 1, 2, 3a, and the empirical-gate run executed exactly as written in the PLAN. The plan's `<action>` for Task 3 anticipated the FAIL branch and instructed: "if median ≠ 1.0, STOP, open a phase blocker note." That branch was taken cleanly.

### Post-execution re-scope (2026-06-05, user-approved)

After the original Tasks 1–3 landed and the BLOCKER was filed, the user reviewed Options A/B/C/D in `09-PROV-01-BLOCKER.md` and approved Option A. The D-09-02 gate was re-scoped from strict `refinement_minimal_edit median = 1.0` to the 2-part shape documented at the top of this SUMMARY. The re-scope is recorded in CONTEXT.md D-09-02, plan 09-01 PLAN.md, the matrix YAML comment, and ROADMAP.md (commit `b072806`). The original strict wording is preserved in `09-PROV-01-BLOCKER.md` for the historical record.

### Plan-Anticipated Outcome (NOT a deviation)

The PLAN's Task 3 action (now updated to reflect the re-scope) explicitly handles the gate-failure path. Pre-re-scope the route was "(a) probe Path B if Path A was tried" (n/a — Path B was already chosen by probe verdict) or "(b) flag for v2.1 replan per D-09-02". The user-approved Option A is a calibrated variant of route (b): re-shape the gate to measure what the adapter actually delivers (committed_itinerary_rate) rather than what is downstream prompt/critique-loop territory (edit-distance).

---

**Total deviations:** 0 from PLAN execution; the gate-restructure is a post-execution decision recorded as a calibrated reading of the empirical data.
**Impact on plan:** Mechanical scope delivered; empirical gate (re-scoped) Part A still fails at 0.4 vs ≥0.6 needed. Plan NOT marked requirements-complete pending user choice between (i) ship-with-gap, (ii) re-run at higher n, (iii) Option B prompt tweak, or (iv) Option C/D mechanical tweaks.

## Issues Encountered

The Part A gap (0.4 vs ≥0.6) is the headline issue. See BLOCKER Resolution for the four user options. Summary: provider state preservation works — Path B subclass + adapter wire correctly, 2/5 commits succeed where Phase-7 had 0/5 — but the agent-loop convergence rate on gpt-5-mini sits at 0.4 (mean of 5 binary outcomes: 0, 0, 1, 0, 1) which is short of the re-scoped Part A threshold by exactly 1 commit out of 5. At n=5 the confidence interval is wide enough that "0.4 is real" vs "0.4 is variance and the true rate is ≥0.6" cannot be distinguished without more data. Part B at 0.0 is documented as advisory because the v2.0 anchor itself sits at median 0.0 / max 0.5 under the same scorer — holding gpt-5-mini to a higher bar than the anchor would be asymmetric.

## Test results (n=5 unit + 6 conformance + 47 graph)

- `pytest tests/unit/test_adapters.py -v` → 5 passed
- `pytest tests/integration/test_reasoning_state_roundtrip.py -m reasoning_conformance -v` → 6 passed (5 from Phase 8 + 1 new openai sibling)
- `pytest tests/unit/test_agent_graph.py -v` → 47 passed (Phase 8 baseline, no regression)
- `poetry run python scripts/check_baselines_fresh.py origin/main` → exit 0
- `make eval-matrix-refinement-structural-check` → exit 0

## User Setup Required

None — `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, and `cloud-sql-proxy --port 5433` were all live in this conversation. The plan's `user_setup` block is informational; no new env vars or dashboard config needed.

## Next Phase Readiness

**STILL BLOCKED — but at a calibrated bar.** User approved Option A on 2026-06-05 and the D-09-02 gate was re-scoped to the 2-part shape above. Against the re-scoped gate, Part A (hard, `committed_itinerary_rate ≥ 0.6`) still fails at 0.4 (2/5 commits), short by 0.2 — the equivalent of 1 more committed run out of 5. Part B (advisory, `refinement_minimal_edit median ≥ 0.5`) still fails at 0.0 but does not block. The OpenAIReasoningAdapter is technically complete and correct; the gap is in agent-loop convergence, not provider state preservation.

Per D-09-02 the Phase 9 PR cannot ship until Part A clears. Plan 09-02 (DeepSeek), 09-03 (Anthropic), and 09-04 (Gemini) should NOT start until the user picks a path forward:

- **(i) Ship with documented Part A gap:** accept 0.4 vs 0.6 as the empirical reality; relax Part A further or note PROV-01 as accept-with-notes per D-06-09 precedent. Lowest cost; weakest gate.
- **(ii) Re-run at n=10 or n=20:** tighten the confidence interval to test whether 0.4 is real or n=5 variance. Estimated incremental spend ≈ $0.20–$0.80 for additional gpt-5-mini runs. Highest information-per-dollar before any code change. Recommended.
- **(iii) Option B prompt tweak:** add a gpt-5-specific imperative preamble to `build_refinement_prompt_message` (analog of D-07-10's gpt-4o-mini partial-recovery). Falsifiable in one matrix re-run. Modest risk of re-coupling prompt to scorer.
- **(iv) Option C/D mechanical tweaks:** raise `MAX_PLAN_STEPS` for gpt-5 family, or tighten `LOW_SIMILARITY_THRESHOLD`. Risk of regressing the v2.0 anchor — needs careful matrix run against gpt-4o-mini cell.

Full discussion + original strict gate wording preserved in `09-PROV-01-BLOCKER.md`.

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
