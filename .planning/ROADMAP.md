# Roadmap: City Concierge

## Milestones

- ✅ **v1.0 Knowledge Graph** — Phase 1 (shipped 2026-05-14, PR merged into main)
- ✅ **v2.0 Production Readiness** — Phases 2-6 (shipped 2026-06-03, PR #100 at `14e01dd`) — see [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)
- ✅ **v2.1 Reasoning-Model Compat** — Phases 7-11 (shipped 2026-06-11, PRs #103/#105/#106) — see [milestones/v2.1-ROADMAP.md](milestones/v2.1-ROADMAP.md)

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

## Progress

| Phase                                          | Milestone | Plans Complete | Status      | Completed  |
| ---------------------------------------------- | --------- | -------------- | ----------- | ---------- |
| 1. Knowledge Graph                             | v1.0      | —              | Complete    | 2026-05-14 |
| 2. Model Override                              | v2.0      | 1/1            | Complete    | 2026-05-22 |
| 3. Eval Harness Extension                      | v2.0      | 12/12          | Complete    | 2026-05-22 |
| 4. Category Compliance Fix                     | v2.0      | 7/7            | Complete    | 2026-05-23 |
| 5. Rationale-Stop Alignment Fix                | v2.0      | 2/2            | Complete    | 2026-05-27 |
| 6. Minimal-Edit Refinement                     | v2.0      | 7/7            | Complete    | 2026-06-03 |
| 7. Prompt/Rubric Decoupling                    | v2.1      | 7/7 | Complete   | 2026-06-04 |
| 8. Reasoning-State Contract + Harness          | v2.1      | 5/5 | Complete    | 2026-06-04 |
| 9. Per-Provider State Preservation Impls       | v2.1      | 5/5 | Complete   | 2026-06-05 |
| 10. Eval Harness Honesty                       | v2.1      | 9/9 | Complete    | 2026-06-11 |
| 11. Cross-Model Baseline Regen + Matrix        | v2.1      | 9/9 | Complete    | 2026-06-11 |

---

*Last updated: 2026-06-11 — Phase 11 gap-closure: verification found 3/6 truths (CR-01 None-abstain + CR-02 fail-open gate). Code fixes already committed (fbd1174..054a20c); the remaining gap is the contaminated empirical baselines. Plan 11-09 re-measures only the category_compliance-contaminated cells live at n=5 under the now-fixed abstain semantics — the clean gpt-4o-mini omakase anchor and the anthropic/gemini deferrals are NOT re-run.*
