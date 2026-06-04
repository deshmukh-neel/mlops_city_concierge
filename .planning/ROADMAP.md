# Roadmap: City Concierge

## Milestones

- ✅ **v1.0 Knowledge Graph** — Phase 1 (shipped 2026-05-14, PR merged into main)
- ✅ **v2.0 Production Readiness** — Phases 2-6 (shipped 2026-06-03, PR #100 at `14e01dd`) — see [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)
- 📋 **v2.1 Reasoning-Model Compat** — drafted, not started (4 phases)

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

### 📋 v2.1 Reasoning-Model Compat (drafted, not started)

Empirical anchor gate: gpt-5-mini × refinement_cheaper commits 3 stops in median 5/5 runs at temp=1.0.
Without this, every new model the field ships in 2026 is permanently unusable on this codebase.

- [ ] Phase 7: Reasoning-state thread-through in `_prune_for_llm` + message-history reconstruction
- [ ] Phase 8: Prompt/rubric decoupling (move behavioral rules from prompt → scorer)
- [ ] Phase 9: Anthropic provider wiring (add `claude` to `SUPPORTED_PROVIDERS`)
- [ ] Phase 10: Honest cross-model baseline regen + matrix expansion

Run `/gsd-new-milestone v2.1` to formalize when ready.

## Progress

| Phase                            | Milestone | Plans Complete | Status   | Completed  |
| -------------------------------- | --------- | -------------- | -------- | ---------- |
| 1. Knowledge Graph               | v1.0      | —              | Complete | 2026-05-14 |
| 2. Model Override                | v2.0      | 1/1            | Complete | 2026-05-22 |
| 3. Eval Harness Extension        | v2.0      | 12/12          | Complete | 2026-05-22 |
| 4. Category Compliance Fix       | v2.0      | 7/7            | Complete | 2026-05-23 |
| 5. Rationale-Stop Alignment Fix  | v2.0      | 2/2            | Complete | 2026-05-27 |
| 6. Minimal-Edit Refinement       | v2.0      | 7/7            | Complete | 2026-06-03 |

---

*Last updated: 2026-06-03 — v2.0 milestone shipped via PR #100.*
