# Roadmap: City Concierge

## Milestones

- ✅ **v1.0 Knowledge Graph** — Phase 1 (shipped 2026-05-14, PR merged into main)
- ✅ **v2.0 Production Readiness** — Phases 2-6 (shipped 2026-06-03, PR #100 at `14e01dd`) — see [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)
- ✅ **v2.1 Reasoning-Model Compat** — Phases 7-11 (shipped 2026-06-11, PRs #103/#105/#106) — see [milestones/v2.1-ROADMAP.md](milestones/v2.1-ROADMAP.md)
- 🚧 **v2.2 Reasoning-Model Decisiveness** — Phases 12-15 (started 2026-06-11)

## Phases

<details>
<summary>✅ v1.0 Knowledge Graph (Phase 1) — SHIPPED 2026-05-14</summary>

- [x] Phase 1: Knowledge Graph Layer — `place_relations` edge table + idempotent five-edge-type builder + real `kg_traverse` tool (W7)

*Reference: `implementation_plan/james/w7_knowledge_graph.md`*

</details>

<details>
<summary>✅ v2.0 Production Readiness (Phases 2-6) — SHIPPED 2026-06-03</summary>

- [x] Phase 2: Model Override (1/1 plans) — completed 2026-05-22
- [x] Phase 3: Eval Harness Extension (12/12 plans) — completed 2026-05-22
- [x] Phase 4: Category Compliance Fix (7/7 plans) — completed 2026-05-23 (PR #97)
- [x] Phase 5: Rationale-Stop Alignment Fix (2/2 plans) — completed 2026-05-27
- [x] Phase 6: Minimal-Edit Refinement (7/7 plans + 5 follow-ups) — completed 2026-06-03 (PR #100)

*Full details: [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)*

</details>

<details>
<summary>✅ v2.1 Reasoning-Model Compat (Phases 7-11) — SHIPPED 2026-06-11</summary>

- [x] Phase 7: Prompt/Rubric Decoupling (7/7 plans) — completed 2026-06-04
- [x] Phase 8: Reasoning-State Thread-Through Contract + Conformance Harness (5/5 plans) — completed 2026-06-04
- [x] Phase 9: Per-Provider State Preservation Implementations (5/5 plans) — completed 2026-06-05 (PR #103)
- [x] Phase 10: Eval Harness Honesty (9/9 plans) — completed 2026-06-11 (PR #105)
- [x] Phase 11: Cross-Model Baseline Regen + Matrix Expansion (9/9 plans) — completed 2026-06-11 (PR #106)

*Full details: [milestones/v2.1-ROADMAP.md](milestones/v2.1-ROADMAP.md)*

</details>

### 🚧 v2.2 Reasoning-Model Decisiveness (In Progress)

**Milestone Goal:** Make reasoning models decisive on the tool loop — pass gpt-5-mini commit rate ≥ 0.6 at n=5 with no gpt-4o-mini anchor regression, and reduce per-turn latency via decisiveness (step-count).

- [x] **Phase 12: Decisiveness Instrumentation + Comparison Floor** — Per-run telemetry, executable falsifier, and the v2.2 comparison floor confirmed (BOTH anthropic AND gemini cells deferred at user decision — D-12-09; non-deferred cells honest n=5) (completed 2026-06-12)
- [x] **Phase 13: Decisiveness Experiment Arms** — Four coupled experiment arms (viability contract, forced-commit, critique recalibration, parallel tools) judged jointly against the falsifier (completed 2026-06-12; honest null result — no arm cleared INST-05; Phase 14 entry gate OPEN)
- [ ] **Phase 14: Richer State Replay** — CONDITIONAL: multi-message reasoning-state replay and content-block preservation, entered only if all Phase 13 arms plateau below the falsifier bar
- [ ] **Phase 15: Gate Promotion + Baseline Regen** — Winning arm's honest n=5 baselines regenerated, reasoning-model gates promoted from logged-not-gated where data earns it, latency report vs ~30s/turn prod budget

## Phase Details

### Phase 12: Decisiveness Instrumentation + Comparison Floor

**Goal**: The eval harness emits per-run decisiveness telemetry and the v2.2 comparison floor is confirmed — the matrix minus BOTH deferred cells (anthropic AND gemini, per D-12-09) — so every experiment arm in Phase 13 can be judged objectively against the same falsifier
**Depends on**: Phase 11 (honest baselines infrastructure, `write_baselines.py`, `eval_gates.yaml` gates)
**Requirements**: INST-01, INST-02, INST-03, INST-04, INST-05, ANCH-02, ANCH-03
**Success Criteria** (what must be TRUE):

  1. `eval_matrix.py` output includes per-run fields: steps-to-first-commit-consideration, per-step viable-candidate counts, and rule-8 precondition met/not-met flag — readable in the run JSON without post-processing
  2. Per-turn latency decomposition (LLM call time vs sequential tool-execution time per plan step) is recorded in each run JSON
  3. A single `make eval-falsifier` (or equivalent) report answers: did gpt-5-mini hit ≥ 0.6 commit rate at n=5, and did gpt-4o-mini hold ≥ its honest baseline? — pass/fail with per-model numbers
  4. `gemini/gemini-3.1-pro-preview` n=5 baseline is **DEFERRED** at user decision (D-12-09, 2026-06-11) — no quota/billing top-up, same treatment as anthropic ANCH-01. Gemini joins anthropic in `_DEFERRED_BASELINE_CELLS` with a deferral note; both cells stay logged-not-gated. The v2.2 comparison floor is the matrix minus BOTH deferred cells (anthropic AND gemini); every other (non-deferred) matrix cell is honest n=5 (ANCH-02 satisfied as deferred-with-note; ANCH-03 reinterpreted per D-12-09). Promotion path for both preserved in `docs/baseline_regen.md`.**Plans**: 4 plans

**Wave 1**

- [x] 12-01-in-graph-step-telemetry-PLAN.md — INST-04: always-on in-graph per-step LLM-call + tool-execution timing on ItineraryState
- [x] 12-03-falsifier-report-PLAN.md — INST-05: make eval-falsifier — pooled gpt-5-mini commit rate vs 0.6 + anchor non-regression, exit 0/1/2
- [x] 12-04-comparison-floor-deferral-bookkeeping-PLAN.md — ANCH-02/03: record gemini + anthropic deferrals (D-12-09); confirm non-deferred cells honest n=5

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 12-02-harness-derived-decisiveness-fields-PLAN.md — INST-01/02/03: harness-side first-commit-step, viable-candidate counts, rule-8 precondition fields

**External dependency**: ANCH-02 (gemini n=5) is **DEFERRED** at user decision D-12-09 (2026-06-11) — no quota/billing top-up; gemini is NOT a phase-completion requirement. ANCH-01 (anthropic n=5) was deferred at milestone start (no billing top-up; user decision 2026-06-11). Both deferred cells stay logged-not-gated with their `_DEFERRED_BASELINE_CELLS` entries intact; neither blocks INST-01..05 work or Phase 12 completion.

### Phase 13: Decisiveness Experiment Arms

**Goal**: Four coupled experiment arms are implemented, run at n=5 temp=1.0 against the Phase-12 comparison floor, and their verdicts are documented — revealing whether any arm clears the falsifier bar or all plateau below it
**Depends on**: Phase 12 (telemetry fields in runs, falsifier report executable, 6-cell comparison floor complete)
**Requirements**: DEC-01, DEC-02, DEC-03, DEC-04, DEC-05
**Success Criteria** (what must be TRUE):

  1. The viability-contract arm (DEC-01) ships without touching the text or structure of any prompt section covered by the Phase-7 CI grep gate — the grep gate stays green
  2. The forced-commit-at-step-N arm (DEC-02) is a graph-level mechanism that triggers independently of model identity — confirmed by a unit test that fires it on a mock that never calls `commit_itinerary`
  3. The parallel-tool-execution arm (DEC-04) runs all tool calls within one `act()` plan step concurrently with results order-stable — measurable gpt-4o-mini latency reduction at n=5 recorded in run JSON
  4. The critique-recalibration arm (DEC-03) is co-tuned with DEC-01 (not tuned in isolation), with the `LOW_SIMILARITY_THRESHOLD` change direction and the `low_similarity` scoping decision both documented before any threshold change lands
  5. DEC-05 arm-verdict document records per-arm n=5 commit-rate numbers for gpt-5-mini, deepseek-reasoner, and gpt-4o-mini anchor, and explicitly states which arm (if any) cleared the INST-05 falsifier bar — or records an honest null result

**Plans**: 7 plans + 3 gap-closure plans

**Wave 1**

- [x] 13-01-viability-predicate-and-telemetry-PLAN.md — Shared viability predicate (app/agent/viability.py) + commit_forced/forced_commit_step state fields + arm_flags run-JSON self-description
- [x] 13-02-viability-contract-prompt-PLAN.md — DEC-01 additive rule-8 viability addendum (flag-gated, both-flag-states prompt locks)

**Wave 2** *(blocked on Wave 1)*

- [x] 13-03-dec03-doc-and-critique-scoping-PLAN.md — DEC-03 decision doc FIRST, then env-overridable threshold + flag-gated low_similarity scoping (co-tuned with DEC-01)
- [x] 13-04-graph-arms-forced-commit-and-parallel-PLAN.md — A2 forced-commit-at-step-N branch + A3 parallel act() + A1 prompt wiring in graph.py
- [x] 13-05-arm-matrix-config-and-falsifier-PLAN.md — configs/eval_matrix_arm.yaml (3 models x 2 scenarios) + falsifier --matrix-config + forced-split reader + Makefile arm targets

**Wave 3** *(live runs — checkpoints, real API spend)*

- [x] 13-06-run-judged-arms-PLAN.md — Run A1/A2/A3 smoke-first at n=5 temp=1.0; record verdict sections in docs/decisiveness_arm_verdicts.md

**Wave 4**

- [x] 13-07-a4-combo-and-closing-verdict-PLAN.md — A4 conditional combo decision (D-13-01, <=4-run cap) + closing INST-05 verdict + bookkeeping

**Gap closure** *(post-verification: gaps_found 4/5 — repairs CR-01, CR-02, SC-3; honest null result unchanged)*

- [x] 13-08-cr01-forced-commit-synthesizer-fix-PLAN.md — CR-01: fix viability.py typed-path PlaceHit→dict + synthesizer rationale so the A2 forced-commit branch works; non-mocked regression test; annotate A2 verdict (mechanism was inoperative; 0.500 model-initiated stands; forced untested at n=5)
- [x] 13-09-cr02-falsifier-split-reader-fix-PLAN.md — CR-02: fix eval_falsifier split reader to read queries[i].deterministic; fixture to real EvalRunReport shape + regression test; annotate verdict that pasted 0/0 was a tool bug (hand-computed tables correct)
- [ ] 13-10-sc3-respecify-and-flag-hygiene-PLAN.md — SC-3 zero-spend respecify (criterion 3 → absolute latency for future baseline; constraint annotated) + WR-09 env_flag DRY helper + WR-02 VIABILITY_CONTRACT_ENABLED single-read co-tuning fix

### Phase 14: Richer State Replay (CONDITIONAL)

**Goal**: Multi-message reasoning-state replay and content-block preservation are A/B-tested against the Phase-13 plateau baseline, producing evidence that either justifies promotion to the winning configuration or confirms the decisiveness gap requires architectural rethinking (ARCH-FUT-01 trigger)
**Depends on**: Phase 13 (DEC-05 verdict — entry gate is "all DEC arms plateau below the INST-05 falsifier bar")
**Requirements**: REPLAY-01, REPLAY-02
**Entry gate (CONDITIONAL)**: This phase executes ONLY if Phase 13's DEC-05 arm-verdict document records that no arm cleared the INST-05 falsifier bar (gpt-5-mini commit rate ≥ 0.6 at n=5 with no anchor regression). If any arm clears the bar, Phase 14 is skipped and the roadmap proceeds directly to Phase 15.
**Success Criteria** (what must be TRUE):

  1. Multi-message `_reasoning_state` replay A/B (REPLAY-01) is measured at n=5 against the DEC plateau: the commit-rate delta vs the best DEC arm is reported, not assumed — positive or negative result is valid
  2. Content-block preservation through `_prune_for_llm` A/B (REPLAY-02) is measured at n=5 against the DEC plateau: the delta is reported alongside an explanation of whether `str()` collapse was causing observable loss in run JSONs
  3. The combined REPLAY result either (a) clears the INST-05 falsifier bar and Phase 15 begins, or (b) is documented as a plateau, triggering explicit ARCH-FUT-01 evaluation before Phase 15 scope is finalized

**Plans**: TBD

### Phase 15: Gate Promotion + Baseline Regen

**Goal**: The winning arm's honest baselines are written for all matrix cells, reasoning-model gates are promoted from logged-not-gated to enforced where the data earns it, and the prod latency budget analysis is documented — closing the milestone with a ratified prod-driver recommendation
**Depends on**: Phase 13 (or Phase 14 if entered) — winning arm identified and merged
**Requirements**: PROMO-01, PROMO-02, PROMO-03
**Success Criteria** (what must be TRUE):

  1. `scripts/write_baselines.py` successfully writes honest n=5 baselines for all matrix cells under the winning arm configuration — no partial/quarantined cells in the output (per D-11-14)
  2. `configs/eval_gates.yaml` is updated: reasoning-model entries are promoted to `enforced` where measured commit-rate data meets the gate threshold, and entries that fall short explicitly retain `logged` with a note in the file
  3. The latency report (decomposed from INST-04 data) documents actual per-turn LLM-call time + tool-execution time for gpt-4o-mini under the winning arm, explicitly comparing against the ~30s/turn prod budget, with a written prod-driver recommendation (ratify gpt-4o-mini anchor OR revise with justification)

**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Knowledge Graph | v1.0 | — | Complete | 2026-05-14 |
| 2. Model Override | v2.0 | 1/1 | Complete | 2026-05-22 |
| 3. Eval Harness Extension | v2.0 | 12/12 | Complete | 2026-05-22 |
| 4. Category Compliance Fix | v2.0 | 7/7 | Complete | 2026-05-23 |
| 5. Rationale-Stop Alignment Fix | v2.0 | 2/2 | Complete | 2026-05-27 |
| 6. Minimal-Edit Refinement | v2.0 | 7/7 | Complete | 2026-06-03 |
| 7. Prompt/Rubric Decoupling | v2.1 | 7/7 | Complete | 2026-06-04 |
| 8. Reasoning-State Contract + Harness | v2.1 | 5/5 | Complete | 2026-06-04 |
| 9. Per-Provider State Preservation Impls | v2.1 | 5/5 | Complete | 2026-06-05 |
| 10. Eval Harness Honesty | v2.1 | 9/9 | Complete | 2026-06-11 |
| 11. Cross-Model Baseline Regen + Matrix | v2.1 | 9/9 | Complete | 2026-06-11 |
| 12. Decisiveness Instrumentation + Comparison Floor | v2.2 | 5/5 | Complete    | 2026-06-12 |
| 13. Decisiveness Experiment Arms | v2.2 | 9/10 | In Progress|  |
| 14. Richer State Replay (CONDITIONAL) | v2.2 | 0/TBD | Not started | - |
| 15. Gate Promotion + Baseline Regen | v2.2 | 0/TBD | Not started | - |

---

*Last updated: 2026-06-12 — Phase 13 complete (7/7 plans); honest null result — no arm cleared INST-05 bar; Phase 14 (Richer State Replay) entry gate OPEN per DEC-05 closing verdict.*
