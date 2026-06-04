# Milestones

Historical record of shipped milestones for City Concierge.

---

## v2.0 — Production Readiness

**Shipped:** 2026-06-03 (PR #100 → main `14e01dd`)
**Phases:** 2-6 (5 phases, 29 plans + 5 follow-up commits)
**Timeline:** 2026-05-21 → 2026-06-03 (~14 days)
**Git range:** `ad8ca84` (PR #94 merge) → `14e01dd` — 193 commits, 107 files, +18672 / −501 LOC

**Delivered:** Moved the City Concierge agent from demo-ready to hostable. Built the reproducible cross-model eval infrastructure that lets every subsequent fix be scored against committed baselines, then fixed the three agent-behavior bugs surviving the PR #94 reliability merge.

**Key accomplishments:**
- `RAG_MODEL_OVERRIDE` env var lets any candidate model wire through `/chat` without touching the shared MLflow `production` alias (Phase 2 / OVR-01..06)
- Reproducible cross-model eval harness: `category_compliance` + `rationale_stop_alignment` scorers, multi-turn threading, `eval_matrix.py`, Makefile targets, three committed baselines, scripted-LLM CI mode, hard CI gate on baseline staleness (Phase 3 / EVAL-01..10)
- Category compliance fix: tool calls inject `primary_type_family` per slot via `_inject_primary_type_family` in `act()`; rationales describe committed places, not requested categories (Phase 4 / CAT-01..04 + RAT-01 + RAT-03)
- Closure-swap placeholder bleed eliminated at construction site via `_candidates_to_matches` inherit-or-synthesize, pinned by unit + functional regression (Phase 5 / RAT-02)
- Minimal-edit refinement: `committed_stops` round-trip + `build_refinement_prompt_message` shared helper used by both `/chat` and the eval runner; `REFINEMENT_STRUCTURED_PLAN_ENABLED` feature flag (default OFF); CI structural-check hard gate via `make eval-matrix-refinement-structural-check` (Phase 6 / REF-01..04)
- Architectural reasoning-model limit empirically confirmed: gpt-5-mini / gpt-5.4-mini / DeepSeek reasoner / Claude / Gemini 3 all fail on this codebase's tool-loop tasks because `_prune_for_llm` drops `reasoning_content` across turns. Locked `openai/gpt-4o-mini` as v2.0 prod anchor; scoped v2.1 milestone to fix.

**Accepted-with-notes:**
- D-06-09 part 2 (no-regression on baseline): pre-Phase-6 1.0 baselines were Phase-4 fail-open false positives now exposed by an actually-committing agent. Real behavior improved; the measurement shifted. Architectural fix scoped to v2.1.

**Archives:**
- [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md) — full phase details + decisions
- [milestones/v2.0-REQUIREMENTS.md](milestones/v2.0-REQUIREMENTS.md) — final traceability + deferred-to-v2.1

---

## v1.0 — Knowledge Graph

**Shipped:** 2026-05-14
**Phases:** 1 (W7 Knowledge Graph)

**Delivered:** Replaced the W2 `kg_traverse` stub with a real implementation backed by a `place_relations` edge table, seeded with five free/computed edge types, tracked in MLflow via `kg_enabled`. Closure-aware itinerary swap and tool-call JSON-safety reliability fixes shipped as follow-ups on `main` (PR #94 merge `ad8ca84`, 2026-05-20).

*Reference: `implementation_plan/james/w7_knowledge_graph.md`*

---
