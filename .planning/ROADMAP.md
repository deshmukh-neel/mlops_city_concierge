# Roadmap: City Concierge

## Milestones

- ✅ **v1.0 Knowledge Graph** — Phase 1 (shipped 2026-05-14, PR merged into main)
- ✅ **v2.0 Production Readiness** — Phases 2-6 (shipped 2026-06-03, PR #100 at `14e01dd`) — see [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)
- ✅ **v2.1 Reasoning-Model Compat** — Phases 7-11 (shipped 2026-06-11, PRs #103/#105/#106) — see [milestones/v2.1-ROADMAP.md](milestones/v2.1-ROADMAP.md)
- ✅ **v2.2 Reasoning-Model Decisiveness** — Phases 12-15 (closed 2026-06-15; honest null INST-05; gpt-4o-mini anchor ratified; ARCH-FUT-01 deferred) — see [milestones/v2.2-ROADMAP.md](milestones/v2.2-ROADMAP.md)
- 🚧 **v2.3 Adaptive Data Loop** — Phases 16-19 (active; productionizes the `coverage_agent` loop to learn from real USER queries). Gate: Phase 16 FALSIFY-01 PASSED 2026-06-15 (hit@5 delta +1.000). Phases 17-19 not yet planned.

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

<details>
<summary>✅ v2.2 Reasoning-Model Decisiveness (Phases 12-15) — CLOSED 2026-06-15</summary>

**Outcome:** Honest null result on INST-05 — no arm cleared gpt-5-mini ≥ 0.6 commit rate at n=5 across Phases 13/14/15 (best A2 = 0.500 pooled). gpt-4o-mini anchor re-ratified (omakase median 1.000, gate ≥ 0.8 enforced); gpt-5-mini demoted to logged. ARCH-FUT-01 (custom imperative loop) deferred as tracked debt. Canonical record: `docs/promotion_decision.md`.

- [x] Phase 12: Decisiveness Instrumentation + Comparison Floor (5/5 plans) — completed 2026-06-12
- [x] Phase 13: Decisiveness Experiment Arms (10/10 plans) — completed 2026-06-12
- [x] Phase 14: Richer State Replay — CONDITIONAL (5/5 plans) — completed 2026-06-12
- [x] Phase 15: Gate Promotion + Baseline Regen (4/4 plans) — completed 2026-06-15

*Full details: [milestones/v2.2-ROADMAP.md](milestones/v2.2-ROADMAP.md)*

</details>

<details open>
<summary>🚧 v2.3 Adaptive Data Loop (Phases 16-19) — ACTIVE</summary>

**Goal:** Productionize the `coverage_agent.py` retrieval-gap loop so it learns from real USER queries (not ingestion hits), gated by a falsifier that proves a positive before→after hit@k delta before the full build. Never writes shared prod `places_raw` — all ingest goes to an isolated sandbox DB.

- [x] Phase 16: Loop Falsifier + Sandbox Provisioning (3/3 plans) — completed 2026-06-15 (FALSIFY-01 + LOOP-00). **Hard gate PASSED: hit@5 delta +1.000 (0/5 → 5/5), exit 0, prod untouched.**
- [x] Phase 17: Query Logging (LOG) — log `/chat` user queries to Cloud SQL as the loop's learning signal (foundational requirement; thin-sliced in 16). **2 plans (2 waves) — planned 2026-06-16.** (completed 2026-06-16)
  - [x] 17-01-PLAN.md — create the `user_query_log` table via Alembic (migration + apply + schema verify) [D-02/D-03/D-04]
  - [x] 17-02-PLAN.md — fire-and-forget `log_user_query` write fn + BackgroundTasks wiring in `chat()` + unit/integration tests [D-01/D-02/D-04]
- [ ] Phase 18: Gap Mining (GAP) — real demand/supply gap miner (replaces Phase 16's hardcoded gap constant). **4 plans (4 waves) — planned 2026-06-17.**
  - [x] 18-01-sandbox-prereqs-PLAN.md — apply user_query_log migration to sandbox + DEMAND_DATABASE_URL in .env.example + deterministic demand seed helper [D-04/D-05]
  - [x] 18-02-demand-extraction-PLAN.md — gather_demand() over user_query_log + lexical/LLM extraction + get_demand_conn two-DB plumbing [D-01/D-05; GAP-01]
  - [ ] 18-03-gap-scoring-cli-PLAN.md — find_demand_gaps (D-02 ranking) + exact seed-format emit + gap_mine_main CLI/MLflow + cold-start no-op [D-02/D-03/D-04; GAP-02/03/04]
  - [ ] 18-04-tests-make-docs-PLAN.md — smoke/functional/integration tests + make gap-mine + CLAUDE/AGENTS/copilot docs sync [D-03/D-04; GAP-03/04]
- [ ] Phase 19: Productionized Loop + Metric (LOOP-01..03 + METRIC) — full Make-targeted ingest→embed→metric loop + productionized hit@k/recall@k scorer. *Not yet planned.*

**Success gate:** FALSIFY-01 (Phase 16) was the milestone go/no-go — a strictly-positive delta proves the loop can add retrievable places. PASSED, so Phases 17-19 are cleared to proceed.

</details>

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
| 13. Decisiveness Experiment Arms | v2.2 | 10/10 | Complete    | 2026-06-12 |
| 14. Richer State Replay (CONDITIONAL) | v2.2 | 5/5 | Complete    | 2026-06-12 |
| 15. Gate Promotion + Baseline Regen | v2.2 | 4/4 | Complete    | 2026-06-15 |
| 16. Loop Falsifier + Sandbox Provisioning | v2.3 | 3/3 | Complete | 2026-06-15 |
| 17. Query Logging (LOG) | v2.3 | 2/2 | Complete    | 2026-06-16 |
| 18. Gap Mining (GAP) | v2.3 | 2/4 | In Progress|  |
| 19. Productionized Loop + Metric | v2.3 | — | Not planned | — |

---

*Last updated: 2026-06-15 — v2.3 Adaptive Data Loop ACTIVE. Phase 16 (Loop Falsifier) COMPLETE — FALSIFY-01 hard gate PASSED (hit@5 delta +1.000). Phase 17 (LOG) PLANNED 2026-06-16 (2 plans). Phases 18-19 scoped but not yet planned. Next: `/gsd-discuss-phase 17`. v2.2 archived in milestones/v2.2-ROADMAP.md.*
