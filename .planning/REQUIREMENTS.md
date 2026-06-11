# Requirements: City Concierge — v2.2 Reasoning-Model Decisiveness

**Defined:** 2026-06-11
**Core Value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Milestone goal:** Make reasoning models decisive on the tool loop. v2.1 proved reasoning state round-trips byte-correctly through all four provider adapters; the models still burn their step budget exploring and never (or rarely) call `commit_itinerary` (gpt-5-mini 2/5, deepseek-reasoner 0/5 vs the gpt-4o-mini anchor's 5/5).

**Decision 3 (resolved at milestone start):** prod latency budget is ~30s/turn. Reasoning models target gate-passage as documented alternates; `openai/gpt-4o-mini` stays the prod anchor. Decisiveness (step-count reduction) is the dominant latency lever and benefits the anchor directly.

**Seed:** `milestones/v2.2-MILESTONE-SEED.md` (2026-06-10 post-Phase-9 harness analysis — load-bearing findings there are settled; do not re-derive).

## v2.2 Requirements

### Instrumentation & Falsifier (INST)

- [ ] **INST-01**: Eval runs record steps-to-first-commit-consideration per run (when the model first weighs calling `commit_itinerary`, or never)
- [ ] **INST-02**: Eval runs record per-step viable-candidate counts (results meeting the viability bar — cosine threshold + matching `primary_type` — for each requested stop)
- [ ] **INST-03**: Eval runs record whether SYSTEM_PROMPT rule 8's commit precondition ("one viable option per requested stop") was objectively met at each step the model kept searching
- [ ] **INST-04**: Eval runs record per-turn latency decomposition (LLM call time vs sequential tool-execution time, per plan step)
- [ ] **INST-05**: The milestone falsifier is executable as a single report: an intervention "works" iff gpt-5-mini commit rate ≥ 0.6 at n=5 AND gpt-4o-mini holds ≥ its honest baseline (no anchor regression)

### Comparison-Floor Anchors (ANCH)

- [ ] **ANCH-01**: anthropic/claude-sonnet-4-6 honest n=5 baseline measured and written via `write_baselines.py` (billing top-up per `docs/baseline_regen.md` promotion path)
- [ ] **ANCH-02**: gemini/gemini-3.1-pro-preview first honest n=5 baseline measured and written via `write_baselines.py` (quota resolution; single scored run hit commit-rate 1.0 — first evidence the Phase-9 adapter fixed it)
- [ ] **ANCH-03**: `_DEFERRED_BASELINE_CELLS` cleared and the 6-cell matrix comparison floor complete (all cells honest n=5, none partial/quarantined)

### Decisiveness Experiment Arms (DEC)

Joint experiments over three coupled levers — prompt contract, critique pressure, state richness — never one lever in isolation. Arms judged with Phase-10 honest gates, n=5, temp=1.0, against the ANCH comparison floor.

- [ ] **DEC-01**: Viability-contract arm: explicit viability definition in the commit precondition ("a result above X cosine with matching primary_type IS viable — do not keep searching past it"), without violating the Phase-7 prompt/rubric-decoupling CI grep gate
- [ ] **DEC-02**: Forced-commit-at-step-N arm: graph-level, model-independent mechanism that ends exploration with a commit from best-so-far candidates
- [ ] **DEC-03**: Critique-recalibration arm co-tuned with DEC-01 (threshold below 0.55 and/or `low_similarity` scoped to pre-candidate steps only — never tuned in isolation per `critique-loop-and-commit-tool-conflict`)
- [ ] **DEC-04**: Parallel tool execution in `act()` — the all-provider latency arm; tool calls within one plan step execute concurrently with results order-stable
- [ ] **DEC-05**: Arm verdicts documented per the INST-05 falsifier: winning arm (or honest null result) recorded with per-arm n=5 numbers for gpt-5-mini, deepseek-reasoner, and the gpt-4o-mini anchor

### Richer State Replay (REPLAY) — conditional

Entry gate: only if all DEC arms plateau below the INST-05 falsifier bar. Justified by A/B, not by assumption.

- [ ] **REPLAY-01**: Multi-message reasoning-state replay A/B: every in-window tool-calling AIMessage gets its `_reasoning_state` replayed (vs current most-recent-only at `graph.py:307-312`), measured against the DEC plateau
- [ ] **REPLAY-02**: Content-block preservation A/B: `_prune_for_llm` preserves pre-cutoff AIMessage list-content reasoning blocks (vs current `str()` collapse at `graph.py:227`), measured against the DEC plateau

### Gate Promotion & Baseline Regen (PROMO)

- [ ] **PROMO-01**: Winning arm's baselines regenerated honest n=5 via `write_baselines.py` for all matrix cells (refuses partial/quarantined per D-11-14)
- [ ] **PROMO-02**: Reasoning-model gates promoted from logged-not-gated to enforced in `configs/eval_gates.yaml` where measured data earns it (and explicitly left logged where it doesn't)
- [ ] **PROMO-03**: Latency report vs the ~30s/turn prod budget: anchor latency with the winning arm documented (decomposition from INST-04), prod driver recommendation ratified or revised

## Future Requirements

Deferred. Tracked but not in the v2.2 roadmap.

### Prod promotion

- **PROD-01**: Promote a reasoning model to the prod MLflow alias — blocked by Decision 3 (~30s/turn budget) until per-call reasoning cost drops; revisit if a winning arm lands a reasoning model under budget
- **PROD-02**: Streaming responses to cut perceived latency — separate hosting concern (carried from v2.0 out-of-scope)

## Out of Scope

Explicitly excluded (locked in the v2.2 seed). Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Replacing LangGraph / custom imperative loop | Premature; ARCH-FUT-01 stays a contingency triggered only if all DEC arms fail |
| Multi-agent planner-executor split | Same contingency class; no evidence current loop shape is the blocker |
| New scorers | Phase 10 EVAL-03 re-derived honest gates; use them |
| Provider-shopping | Confirmed dead end (`project_agent_loses_reasoning_state_all_providers`) — failure mode is architectural/behavioral, not vendor-specific |
| Sub-10s "frontier chat" latency parity | Ungrounded single-pass generation is a different computation; grounded multi-stop verification has a tool-call floor (Decision 3) |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INST-01 | Phase 12 | Pending |
| INST-02 | Phase 12 | Pending |
| INST-03 | Phase 12 | Pending |
| INST-04 | Phase 12 | Pending |
| INST-05 | Phase 12 | Pending |
| ANCH-01 | Phase 12 | Pending |
| ANCH-02 | Phase 12 | Pending |
| ANCH-03 | Phase 12 | Pending |
| DEC-01 | Phase 13 | Pending |
| DEC-02 | Phase 13 | Pending |
| DEC-03 | Phase 13 | Pending |
| DEC-04 | Phase 13 | Pending |
| DEC-05 | Phase 13 | Pending |
| REPLAY-01 | Phase 14 | Pending |
| REPLAY-02 | Phase 14 | Pending |
| PROMO-01 | Phase 15 | Pending |
| PROMO-02 | Phase 15 | Pending |
| PROMO-03 | Phase 15 | Pending |

**Coverage:**
- v2.2 requirements: 18 total
- Mapped to phases: 18 (100%)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-11*
*Last updated: 2026-06-11 — traceability table completed during roadmap creation (v2.2 roadmap, Phases 12-15)*
