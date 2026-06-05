# Phase 9 / PROV-01 — Milestone Anchor Gate BLOCKER

**Opened:** 2026-06-04
**Plan:** 09-01 (openai-gpt5-adapter)
**Decision driver:** D-09-02 (PR-blocking gate on PROV-01) — RE-SCOPED 2026-06-05 per user-approved Option A
**Status:** PARTIALLY RESOLVED — gate re-scoped (resolves the "gate as written is asymmetric with anchor" structural issue), but Part A (hard) of the new gate still FAILS at 0.4 vs ≥0.6 needed. Plan 09-01 is HELD pending user choice between (i) ship-with-documented-gap, (ii) re-run at higher n, or (iii)/(iv) one of the original technical mitigations below.

## Resolution (2026-06-05)

User reviewed Options A, B, C, D below and approved **Option A** on 2026-06-05. The D-09-02 gate has been re-scoped from strict `refinement_minimal_edit median = 1.0` to a 2-part gate:

| Part | Threshold                                  | Hard/Advisory       |
| ---- | ------------------------------------------ | ------------------- |
| A    | `committed_itinerary_rate ≥ 0.6`           | **HARD (PR-blocking)** |
| B    | `refinement_minimal_edit median ≥ 0.5`     | Advisory            |

Re-scope commit: `b072806` (touches CONTEXT D-09-02 + D-09-12, PLAN 09-01, YAML matrix comment, ROADMAP Phase 9 SC #1).

### Empirical result against the re-scoped gate

| Part | Threshold | Measured (n=5) | Status                                   |
| ---- | --------- | -------------- | ---------------------------------------- |
| A    | ≥ 0.6     | **0.4** (2/5)  | **FAILS** by 0.2 — 1 more commit out of 5 would clear it |
| B    | ≥ 0.5     | 0.0            | FAILS (advisory; does not block PR)      |

Per the re-scoped D-09-02, Phase 9 PR still cannot ship until Part A clears.

### What the user must decide

Approving Option A re-shaped the gate; it did NOT bring the empirical data over the new threshold. The Part A gap is small (1 commit out of 5) and at n=5 the confidence interval is wide. Four paths forward, ordered by recommended priority:

1. **(ii) Re-run at n=10 or n=20 to tighten the confidence interval.** Highest information-per-dollar before any code change. Estimated incremental spend ≈ $0.20–$0.80 for additional gpt-5-mini runs (each gpt-5-mini run is 120–335s on this matrix; n=15 incremental ≈ 30–80 min wall-clock). If the true rate is ≥0.6 the gate clears at n=15 with high probability; if the true rate is ≤0.4 the gate confirms its failure and the user moves to one of (iii)/(iv) with better data.
2. **(i) Ship with documented Part A gap (accept-with-notes precedent from D-06-09).** Lowest cost; weakest gate. PROV-01 ships as "accept-with-notes: committed_itinerary_rate = 0.4 vs ≥0.6 target; n=5 sample, 1 commit short, CI not tightened". Mark in SUMMARY.md.
3. **(iii) Option B prompt tweak.** Add a gpt-5-specific imperative preamble to `build_refinement_prompt_message` (analog of D-07-10's gpt-4o-mini partial-recovery). Falsifiable in one matrix re-run. Modest risk of re-coupling prompt to scorer (D-07-04's Phase-7 decoupling work).
4. **(iv) Option C/D mechanical tweaks.** Raise `MAX_PLAN_STEPS` for gpt-5 family (Option C) or tighten `LOW_SIMILARITY_THRESHOLD` (Option D). Risk of regressing the v2.0 anchor — needs careful re-run against gpt-4o-mini cell.

**No new matrix runs without orchestrator/user approval** (same spend-gate as before).

### Original gate wording (preserved for archaeology)

The pre-2026-06-05 strict D-09-02 wording is preserved verbatim in the **TL;DR** + **Empirical measurement** + **Triage paths** sections below. These sections are NOT updated — they record the historical state of the blocker on 2026-06-04 before the user re-scoped the gate.

---

## Original BLOCKER content (2026-06-04 — pre-re-scope)


## TL;DR

The OpenAIReasoningAdapter + OpenAIReasoningChatModel (Path B) **landed correctly and is wired** — but the **milestone anchor gate FAILED**.

Gate target: `openai/gpt-5-mini × refinement_cheaper × refinement_minimal_edit` median = `1.0` (5/5 minimal-edit commits).
Gate result: median = **0.0** (5/5 runs scored 0.0). `committed_itinerary_rate` = 0.4 (2/5 runs committed at all).

Per D-09-02 the entire Phase 9 PR cannot ship until this blocker is resolved.

## Empirical measurement (n=5, 2026-06-04, local)

| Run    | refinement_minimal_edit | committed_itinerary_rate | tool_calls | revision_reasons                                   | latency (s) | final_reply |
| ------ | ----------------------- | ------------------------ | ---------- | -------------------------------------------------- | ----------- | ----------- |
| run-0  | 0.0                     | 0.0                      | 8          | ['low_similarity']                                 | 120.3       | "I hit the planning step limit. Here is the best plan I had so far." |
| run-1  | 0.0                     | 0.0                      | 8          | ['low_similarity']                                 | 173.0       | "I hit the planning step limit. Here is the best plan I had so far." |
| run-2  | 0.0                     | 1.0                      | 8          | ['low_similarity', 'low_similarity', 'neighborhood_no_match'] | 335.3       | "Here's your itinerary: 1. Il Borgo …" |
| run-3  | 0.0                     | 0.0                      | 8          | ['low_similarity']                                 | 219.8       | "I hit the planning step limit. Here is the best plan I had so far." |
| run-4  | 0.0                     | 1.0                      | 12         | ['low_similarity']                                 | 178.1       | "Here's your itinerary: 1. Hazie's …" |
| **median** | **0.0** (target 1.0) | 0.0                      | 8          | —                                                  | —           | —           |

Source: `eval_reports/2026-06-05T03-35-57Z/openai--gpt-5-mini--refinement_cheaper--run-*.json`
Aggregated into: `configs/eval_baselines/refinement_cheaper.json` (committed in `data(09-01): refresh refinement_cheaper baselines with PROV-01 n=5 medians (PROV-01)`).

## Reference cells (no regression)

| Cell                       | refinement_minimal_edit median | prior baseline | Δ        |
| -------------------------- | ------------------------------ | -------------- | -------- |
| openai/gpt-4o-mini (v2.0)  | 0.0                            | 0.0            | 0.0 (no regression — distribution identical: [0.0, 0.5, 0.0, 0.0, 0.0]) |
| deepseek/deepseek-chat      | 0.0                            | 0.0            | 0.0 (no change; DeepSeek's decisiveness gap is PROV-02 territory) |

The Path B subclass routes `_is_openai_reasoning_model(chat_model)`-positive models through `OpenAIReasoningChatModel(use_responses_api=True)`. `gpt-4o-mini` is intentionally excluded — its baseline is unchanged, confirming the v2.0 anchor stays on plain ChatOpenAI.

## Diagnosis: where the loop actually breaks

The Path B subclass DOES work as designed at the wire level:

1. `OpenAIReasoningChatModel._generate` / `_agenerate` calls super (Responses API).
2. LangChain returns `AIMessage.content` as a list including `{"type": "reasoning", ...}` blocks.
3. `_lift_reasoning_blocks` copies them to `AIMessage.additional_kwargs["reasoning_content"]`.
4. `OpenAIReasoningAdapter.capture_reasoning_state` reads them and returns the payload.
5. `graph.py` stashes them at `additional_kwargs["_reasoning_state"]` (Phase 8 contract).
6. Next turn, `replay_reasoning_state` writes them back onto the most-recent AIMessage.
7. LangChain's Responses-API serializer round-trips reasoning blocks on the wire.

Empirical evidence Path B is doing real work:
- gpt-5-mini commit rate **rose from 2/5 in Phase 7 to 2/5 + 3 step-limited (no longer 0/5)** — actually identical here, but the failure mode shifted: in Phase 7 the agent never committed at all; now 2/5 commit and 3/5 fail with `low_similarity` revision retries before max_steps.
- Latencies are 2-3× longer than gpt-4o-mini — consistent with the Responses-API reasoning round-trip.

What's NOT working:
1. **Refinement edit-minimization:** even when the agent commits (run-2, run-4), the committed plan differs structurally from the prior committed plan. `refinement_minimal_edit` measures token-level edit-distance against the prior plan; score 0.0 means the agent rewrote the whole stop list instead of swapping the one stop the user asked to make cheaper. The reasoning state survives, but the agent's interpretation of "make stop N cheaper" still produces a wholesale re-plan.
2. **Step-limit termination on 3/5 runs:** revision_reasons=['low_similarity'] dominates — the critique loop fires when semantic_search recall is thin, and gpt-5-mini retries with similarly-narrow queries until max_steps. This is the documented `project_critique_commit_conflict` interaction — reasoning state preservation didn't dissolve it.

## Triage paths

### Option A (RECOMMENDED): defer milestone gate → soft threshold + committed_rate

Restructure D-09-02 from strict `refinement_minimal_edit median = 1.0` to a 2-part gate:
- Part A (hard): `committed_itinerary_rate ≥ 0.6` (currently 0.4 — close)
- Part B (advisory): `refinement_minimal_edit median ≥ 0.5` (currently 0.0 — far)

Rationale: the v2.0 anchor `gpt-4o-mini` itself sits at median 0.0 / max 0.5 in the current scorer (D-07-05 + D-07-07 tightening). Holding gpt-5-mini to a higher bar than the anchor is the wrong shape. The Phase-9 charter is "preserve provider reasoning state cross-turn" — Path B demonstrably does that. Refinement edit-minimization is prompt/critique-loop territory, not adapter territory.

This option ships the OpenAIReasoningAdapter as scope-delivered for PROV-01 and re-routes the edit-minimization work to a separate plan (likely v2.1 phase 2: prompt-rubric refinement).

### Option B: tighten prompt to enforce minimal edit on refinement turns

Re-open `build_refinement_prompt_message` in `app/agent/io.py` and add an imperative preamble specifically for gpt-5 family: "Reuse `place_id` and `slot` index of every stop you are NOT changing exactly as listed; only emit the single targeted stop in `commit_itinerary` with its replacement." This was D-07-10's partial-recovery move on gpt-4o-mini; it shifted distribution from [0.0×5] to [0.0×3, 0.5×2]. Re-running n=5 with this preamble for gpt-5 would tell us if it lifts median to 0.5+.

Risk: re-couples prompt to scorer (D-07-04's Phase-7 decoupling work). Not ideal.

### Option C: raise max_steps for gpt-5 family

The 3/5 step-limited runs hit max_steps on `low_similarity` retries. Raising `MAX_PLAN_STEPS` from current value (likely 8-10) to 15-20 for `_is_openai_reasoning_model` models might let the agent finish its critique loop. Risk: latency budget already at 120-335s; this would push it higher.

### Option D: shrink LOW_SIMILARITY_THRESHOLD

`project_critique_commit_conflict` notes the critique-loop ↔ commit conflict on weak-search scenarios. Tightening to 0.55 (already discussed in `project_w10_migration_necessary_not_sufficient` for Gemini) might stop the retry loop. Risk: regresses gpt-4o-mini convergence.

## Recommendation

**Option A** — re-scope D-09-02 to match what the adapter actually delivers. The Path B implementation is technically correct; the gate as written conflates "provider state preservation" with "refinement quality." Restructure the gate, ship PROV-01 as scope-complete, and move edit-minimization to a follow-on plan.

If the user rejects re-scoping, **Option B** is the lowest-risk technical move (small, prompt-only change, falsifiable in one matrix run).

## Files referenced

- Implementation (already committed): `app/agent/adapters/openai_gpt5.py`, `app/llm_factory.py` (lines 59-138), `app/agent/adapters/__init__.py` (lines 121-122)
- Probe artifact: `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-PROBE.md`
- Matrix YAML (already promoted to GATED): `configs/eval_matrix_refinement.yaml` (lines 25-31)
- Baseline (refreshed with this run's medians): `configs/eval_baselines/refinement_cheaper.json`
- Per-run JSONs: `eval_reports/2026-06-05T03-35-57Z/openai--gpt-5-mini--refinement_cheaper--run-{0..4}.json`
- Summary: `eval_reports/2026-06-05T03-35-57Z/summary.json`

## Resume signal

**Updated 2026-06-05:** User chose Option A (re-scope gate); gate has been re-scoped (commit `b072806`). Against the re-scoped gate Part A (hard) still FAILS at 0.4 vs ≥0.6. User must now choose between:
- "approved: re-run at n=10" — most-recommended next step
- "approved: re-run at n=20" — if n=10 still indeterminate
- "approved: ship-with-gap" — PROV-01 ships as accept-with-notes per D-06-09 precedent
- "approved: Option B" — gpt-5-specific imperative preamble in `build_refinement_prompt_message`
- "approved: Option C" — raise `MAX_PLAN_STEPS` for gpt-5 family
- "approved: Option D" — tighten `LOW_SIMILARITY_THRESHOLD`

The executor cannot self-resolve. No matrix runs without user approval.

**Original signal (pre-2026-06-05):** User to choose Option A, B, C, or D (or another path) — the gate cannot be self-resolved by the executor.
