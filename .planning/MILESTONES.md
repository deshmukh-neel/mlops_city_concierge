# Milestones

## v2.1 Reasoning-Model Compat (Shipped: 2026-06-11)

**Phases completed:** 5 phases (7-11), 35 plans, 48 tasks
**Timeline:** 2026-06-03 → 2026-06-11 | 156 files changed, +24,097/−6,420 vs v2.0
**PRs:** #103 (phases 7-9), #105 (phase 10), #106 (phase 11)

**Key accomplishments:**

- **Prompt/rubric decoupling (Phase 7):** behavioral rules moved from SYSTEM_PROMPT rule 10 and the 18-line refinement preamble into the `refinement_minimal_edit` scorer (D-07-07 four-cell `primary_type` matrix), locked by a CI grep gate; the PROMPT-05 falsifier resolved decisively — gpt-5-mini flat 0/5 proved architectural state-loss dominates over prompt-coupling.
- **Reasoning-state contract (Phase 8):** typed `ProviderAdapter` ABC with `capture_reasoning_state`/`replay_reasoning_state`, wired POST-PRUNE into the agent graph, plus a 9-test conformance harness (`make test-reasoning-conformance`, now a required CI step).
- **Per-provider adapters (Phase 9):** four isolated implementations — OpenAI gpt-5 family (Responses API), DeepSeek reasoner, Anthropic Claude, Gemini 3 (experimental) — with zero cross-adapter imports and a revertability audit; PROV-01..03 shipped-with-gap (residual decisiveness deferred to v2.2).
- **Eval-harness honesty (Phase 10):** fail-open scoring paths closed, per-case error records (`make_error_record`), per-family gates re-derived from honest data, probe redaction made fail-closed.
- **Honest baselines + CI enforcement (Phase 11):** all baselines regenerated at n=5 under DB-up conditions via the new `write_baselines.py` tool (refuses partial/quarantined cells); 3 cross-model anchors added to the matrix; live-key-free `--baselines-mode` gate + staleness watch-set enforced as required CI steps; gpt-4o-mini anchor held at commit-rate median 1.0 throughout.

**Known deferred items at close:** 5 (see v2.1-MILESTONE-AUDIT.md tech_debt — anthropic n=5 billing, gemini n=5 quota, gpt-5-mini aspirational gate, v2.2 decisiveness gap, Phase 9 verifier-doc process note)

---

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
