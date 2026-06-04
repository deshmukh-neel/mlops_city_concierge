# Roadmap: City Concierge

## Milestones

- ✅ **v1.0 Knowledge Graph** — Phase 1 (shipped 2026-05-14, PR merged into main)
- ✅ **v2.0 Production Readiness** — Phases 2-6 (shipped 2026-06-03, PR #100 at `14e01dd`) — see [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)
- 🚧 **v2.1 Reasoning-Model Compat** — Phases 7-10 (active, started 2026-06-03)

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

### 🚧 v2.1 Reasoning-Model Compat (Phases 7-10)

Empirical anchor gate: `gpt-5-mini × refinement_cheaper × prod × flag-on` commits 3 stops in median 5/5 runs at temp=1.0 (currently 0/1).
Without this, every new reasoning model the field ships in 2026 is permanently unusable on this codebase.

- [ ] **Phase 7: Prompt/Rubric Decoupling** — Behavioral rules move from prompt body to scorer; no regression on v2.0 anchor; serves as falsifier for Phase 8 architectural diagnosis
- [ ] **Phase 8: Reasoning-State Thread-Through Contract + Conformance Harness** — Typed `ProviderAdapter` contract + per-provider conformance tests + `_prune_for_llm` refactor; doubles as harness-swap decision gate
- [ ] **Phase 9: Per-Provider State Preservation Implementations** — One sub-phase each: gpt-5 family → DeepSeek reasoner → Claude Sonnet 4.6 (+ Anthropic wiring) → Gemini 3 (experimental); milestone anchor gate lands here
- [ ] **Phase 10: Cross-Model Baseline Regen + Matrix Expansion** — Rebuild all baselines honestly post-fail-open; add three new cross-model anchors; lock per-family merge gates in CI

## Phase Details

### Phase 7: Prompt/Rubric Decoupling

**Goal**: The refinement scorer enforces behavioral rules that were previously baked into the prompt, so the prompt is task-description-only and any reasoning model can follow it without being confused by prescription-as-description.
**Depends on**: Phase 6 (v2.0 shipped — `REFINEMENT_STRUCTURED_PLAN_ENABLED` flag, `build_refinement_prompt_message` helper, and the v2.0 eval baseline are prerequisites)
**Requirements**: PROMPT-01, PROMPT-02, PROMPT-03, PROMPT-04, PROMPT-05
**Success Criteria** (what must be TRUE):

  1. `SYSTEM_PROMPT` rule 10 and `_REFINEMENT_PREAMBLE` contain no behavioral prescriptions — no "keep same stop count", "do not ask clarifying questions", "preserve `place_id` byte-for-byte", "same primary_type on replacement" in the prompt body; a grep for those phrases returns zero matches.
  2. `refinement_minimal_edit` scorer enforces same-stop-count, byte-equal preserved `place_id`s for untouched stops, and same Google Place `primary_type` on the replacement as scorer logic, with unit tests covering each rule.
  3. A `/chat` refinement turn ("make stop 2 cheaper") with `REFINEMENT_STRUCTURED_PLAN_ENABLED=on` returns a full itinerary with the requested edit applied and all other stops byte-identical to the committed plan.
  4. `openai/gpt-4o-mini × refinement_cheaper` median score is >= the pre-Phase-7 v2.0 baseline (no regression on the v2.0 prod anchor; PROMPT-04 gate).
  5. `gpt-5-mini × refinement_cheaper` median score is > 0 across 5 runs at temp=1.0 — any non-zero confirms prompt-coupling materially contributed; a flat 0/5 confirms architectural state-loss dominates and Phase 9 scope stays at full (PROMPT-05 falsifier signal).

**Plans:** 7 plans
Plans:
**Wave 1**

- [ ] 07-01-prompt-rewrite-PLAN.md — Delete SYSTEM_PROMPT rule 10 and rewrite `_REFINEMENT_PREAMBLE` as task-only (PROMPT-01, PROMPT-02)
- [ ] 07-02-scratch-payload-extend-PLAN.md — Extend `prior_committed_stops` scratch entries with `primary_type` and update `ExpectedRefinement` docstring (PROMPT-03)
- [ ] 07-03-gpt5-mini-matrix-entry-PLAN.md — Add `openai/gpt-5-mini` as logged-not-gated entry in `configs/eval_matrix_refinement.yaml` (PROMPT-05 wiring)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 07-04-scorer-category-extend-PLAN.md — Extend `refinement_minimal_edit` Branch 5 with target-slot `primary_type` check per D-07-07 (PROMPT-03)
- [ ] 07-05-scorer-tests-and-grep-gate-PLAN.md — Unit tests: PROMPT-02 grep gate + six new `TestRefinementMinimalEdit` methods + prod-threading scratch assertion (PROMPT-02, PROMPT-03)
- [ ] 07-06-chat-refinement-integration-test-PLAN.md — Functional test in `TestChatRefinementInjection` driving real LangGraph + `/chat` with scripted-LLM commit (PROMPT-01)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 07-07-rebaseline-and-falsifier-PLAN.md — Snapshot pre-Phase-7 baseline; re-run `make eval-matrix-refinement RUNS=5`; evaluate PROMPT-04 vs snapshot + PROMPT-05 falsifier signal (PROMPT-04, PROMPT-05)

### Phase 8: Reasoning-State Thread-Through Contract + Conformance Harness

**Goal**: A typed `ProviderAdapter` contract exists and is proven to round-trip reasoning state through `graph.invoke`, or the harness-swap decision gate fires and v2.1 replans around a custom imperative loop.
**Depends on**: Phase 7 (prompt decoupled so the conformance signal is not confounded by prompt-coupling noise)
**Requirements**: REASON-01, REASON-02, REASON-03, REASON-04, REASON-05, REASON-06
**Success Criteria** (what must be TRUE):

  1. A typed `ProviderAdapter` abstract interface exists in `app/agent/` (or `app/llm_factory/`) with stable `capture_reasoning_state(message)` and `replay_reasoning_state(payload, state)` methods; adding a fifth provider shape is an interface extension, not a rewrite.
  2. The contract type-stubs cover all four state shapes: OpenAI `reasoning_content` (string), Anthropic `thinking` blocks (signed), DeepSeek `reasoning_content` (string), Gemini `thought_signature` (bytes); each shape has a dedicated unit test.
  3. `_prune_for_llm` delegates state preservation to `ProviderAdapter.replay_reasoning_state` for reasoning providers; the gpt-4o-mini non-reasoning path passes through unchanged; both paths have regression unit tests.
  4. `tests/integration/test_reasoning_state_roundtrip.py` exists, runs a 2-turn agent loop against mocked provider responses, asserts the reasoning state field present on the turn-1 `AIMessage` is present in the turn-2 outbound payload, and is quarantined (does not gate prod merges unless explicitly opted in).
  5. The conformance test passes end-to-end **including through `graph.invoke`** for at least the gpt-5 family provider (REASON-05 — harness-swap decision gate). If the isolated conformance test passes but `graph.invoke` drops state, this criterion is explicitly marked FAILED, a Phase 8 blocker is filed, and v2.1 replans around a custom imperative loop before Phase 9 starts. This branch point is not a footnote — it gates whether Phase 9 proceeds as written.
  6. After the `_prune_for_llm` refactor, all v2.0 baselines (`openai/gpt-4o-mini × refinement_cheaper` and all other committed baselines) do not regress; the existing staleness CI hard gate (`scripts/check_baselines_fresh.py`) continues to pass (REASON-06 no-regression gate).

**Plans**: TBD

### Phase 9: Per-Provider State Preservation Implementations

**Goal**: Each of the four reasoning provider families (gpt-5, DeepSeek, Claude, Gemini 3) has a working `ProviderAdapter` implementation that preserves reasoning state across turns; the milestone anchor gate (`gpt-5-mini × refinement_cheaper` at 5/5) is met; the v2.0 anchor path is untouched throughout.
**Depends on**: Phase 8 (typed `ProviderAdapter` contract and conformance harness in place; REASON-05 harness-swap gate passed — if Phase 8 REASON-05 was a blocker and v2.1 replanned around a custom imperative loop, Phase 9 implementations target that loop instead of the LangGraph path)
**Requirements**: PROV-01, PROV-02, PROV-03, PROV-04, PROV-05
**Success Criteria** (what must be TRUE):

  1. `ProviderAdapter` implementation for OpenAI gpt-5 family is merged and the milestone anchor gate is met: `gpt-5-mini × refinement_cheaper × prod × flag-on` commits 3 stops in median 5/5 runs at temp=1.0 (PROV-01).
  2. `ProviderAdapter` implementation for DeepSeek reasoner is merged; `deepseek-reasoner × refinement_cheaper` median >= 0.6 across 5 runs at temp=1.0 (PROV-02; lower bar, exploratory).
  3. `ProviderAdapter` implementation for Anthropic Claude is merged AND `claude` is added to `SUPPORTED_PROVIDERS` in `app/llm_factory.py` with a new `build_chat_model` branch and `langchain-anthropic` in `pyproject.toml`; `claude-sonnet-4-6 × refinement_cheaper` median >= 1.0 across 5 runs at temp=1.0 (PROV-03).
  4. `ProviderAdapter` implementation for Gemini 3 is merged as experimental (no merge gate); `thought_signature` round-trips cleanly through a 2-turn loop in the conformance harness; a known-issues note accompanies the commit and Gemini 3 is absent from the prod matrix (PROV-04).
  5. Each provider sub-phase ships as an independently revertable commit; reverting any one sub-phase leaves the remaining adapters and the v2.0 `openai/gpt-4o-mini` anchor fully functional in prod; verified by running `make test` after each revert (PROV-05).

**Plans**: TBD

### Phase 10: Cross-Model Baseline Regen + Matrix Expansion

**Goal**: All eval baselines are regenerated honestly under DB-up conditions with the Phase-7 prompt and Phase-9 adapters in place; three new cross-model anchors are in the matrix; per-family merge gates are documented and enforced in CI so future code changes that regress any anchor fail before merge.
**Depends on**: Phase 9 (all provider adapters shipped; prompt decoupled; no fail-open saturation in the baselines)
**Requirements**: BASE-01, BASE-02, BASE-03, BASE-04
**Success Criteria** (what must be TRUE):

  1. All `configs/eval_baselines/*.json` are regenerated under DB-up conditions (Cloud SQL or local Postgres reachable) with the Phase-7-decoupled prompt and Phase-9 provider adapters active; the regen procedure is documented in a runbook (e.g. `docs/baseline_regen.md`) and the fail-open-saturated v2.0 baselines are replaced (BASE-01).
  2. `configs/eval_matrix*.yaml` includes `gpt-5-mini`, `claude-sonnet-4-6`, and `deepseek-reasoner` as cross-model matrix entries alongside `openai/gpt-4o-mini`; running `make eval-matrix` against the updated config produces results for all four providers without errors (BASE-02).
  3. Per-family merge gates are documented in a single source-of-truth file (e.g. `docs/eval_gates.md`) and enforced via named Makefile targets and a CI step; the `gpt-5-mini × refinement_cheaper` anchor gate is one of them and fires on a synthetic regression (BASE-03).
  4. A staleness check analogous to `scripts/check_baselines_fresh.py` covers the new cross-model baselines; a code change touching the agent loop without regenerating the new baselines causes CI to fail, verified by a dry-run test of the staleness script (BASE-04).

**Plans**: TBD

## Progress

| Phase                                          | Milestone | Plans Complete | Status      | Completed  |
| ---------------------------------------------- | --------- | -------------- | ----------- | ---------- |
| 1. Knowledge Graph                             | v1.0      | —              | Complete    | 2026-05-14 |
| 2. Model Override                              | v2.0      | 1/1            | Complete    | 2026-05-22 |
| 3. Eval Harness Extension                      | v2.0      | 12/12          | Complete    | 2026-05-22 |
| 4. Category Compliance Fix                     | v2.0      | 7/7            | Complete    | 2026-05-23 |
| 5. Rationale-Stop Alignment Fix                | v2.0      | 2/2            | Complete    | 2026-05-27 |
| 6. Minimal-Edit Refinement                     | v2.0      | 7/7            | Complete    | 2026-06-03 |
| 7. Prompt/Rubric Decoupling                    | v2.1      | 0/7            | Pending     | -          |
| 8. Reasoning-State Contract + Harness          | v2.1      | 0/TBD          | Pending     | -          |
| 9. Per-Provider State Preservation Impls       | v2.1      | 0/TBD          | Pending     | -          |
| 10. Cross-Model Baseline Regen + Matrix        | v2.1      | 0/TBD          | Pending     | -          |

---

*Last updated: 2026-06-04 — Phase 7 (Prompt/Rubric Decoupling) planned: 7 plans across 3 waves.*
