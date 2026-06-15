# Milestones

## v2.2 — Reasoning-Model Decisiveness (Shipped: 2026-06-15)

**Phases completed:** 4 phases (12-15), 24 plans, 26 tasks
**Timeline:** 2026-06-11 → 2026-06-15 (~4 days) | 188 files changed, +23,970/−8,173 vs v2.1
**PRs:** #110 (phase 15) and the phase 12-14 branches merged ahead of it
**Git range:** `beadb44` (v2.1 tag) → `3d7adc8` — 173 commits

**Delivered:** An honest negative result. v2.1 proved reasoning state round-trips byte-correctly through all four provider adapters; v2.2 set out to make those models *decisive* — to call `commit_itinerary` instead of burning the step budget exploring. After instrumenting the loop, running four joint experiment arms, and entering the conditional replay phase, **no intervention cleared the falsifier bar** (gpt-5-mini ≥ 0.6 commit rate at n=5). The decisiveness gap is architectural, not tunable under the ~30s/turn budget. `openai/gpt-4o-mini` is re-ratified as the prod anchor; ARCH-FUT-01 (custom imperative loop) is deferred as tracked debt.

**Key accomplishments:**

- **Decisiveness instrumentation + executable falsifier (Phase 12 / INST-01..05):** per-run telemetry — steps-to-first-commit-consideration, per-step viable-candidate counts, rule-8 precondition objectively-met flag, and per-turn latency decomposition (LLM-call vs sequential tool-execution time) — all in the run JSON without post-processing; `make eval-falsifier` answers pass/fail in one report.
- **Four joint experiment arms, honest null (Phase 13 / DEC-01..05):** viability-contract prompt addendum, forced-commit-at-step-N graph mechanism, critique recalibration (co-tuned, not isolated), and parallel tool execution in `act()` — all run at n=5 temp=1.0 against the comparison floor. Best result A2 = 0.500 pooled; no arm cleared the 0.6 bar. Gap closure fixed the forced-commit synthesizer (CR-01) and the falsifier split reader (CR-02).
- **Conditional replay phase entered and also plateaued (Phase 14 / REPLAY-01/02):** R1 multi-message `_reasoning_state` replay hit 0.500 (identical to A2, delta ±0.000); R2 content-block preservation was *refuted in the breaking direction* — gpt-5-mini 10/10 deterministic provider 400s, proving the `str()` collapse was load-bearing for the Responses API path. R3/valve correctly skipped per D-14-01 preconditions.
- **Gate promotion + honest baseline regen (Phase 15 / PROMO-01..03):** gpt-4o-mini re-ratified to `enforced` (omakase median 1.000 flag-off, ≥ 0.8 gate holds); gpt-5-mini demoted to `logged` (0.500 < 0.600). Baselines regenerated honest flag-off n=5 for 6 runnable cells with corrected provenance (the prior `refinement_cheaper` 1.000 was a flag-ON arm artifact). Latency report: gpt-4o-mini omakase median 47s — budget NOT met, documented honestly.
- **Canonical decision record:** `docs/promotion_decision.md` cross-links both immutable verdict docs (`docs/decisiveness_arm_verdicts.md`, `docs/replay_arm_verdicts.md`) — the milestone's evidence chain from instrumentation through ARCH-FUT-01 deferral.

**Known deferred items at close (tracked debt, not gaps):**

- ARCH-FUT-01 (custom imperative agent loop) — deferred, user-ratified at the D-14-08 checkpoint; trigger is the Phases 13-14 evidence chain
- anthropic + gemini n=5 baselines — deferred (D-12-09, no billing top-up); stay logged-not-gated with `_DEFERRED_BASELINE_CELLS` entries intact
- Prod-default `FORCED_COMMIT_STEP=6` flip — flagged but NOT implemented (D-15-07, likely-deferred)
- `refinement_cheaper` baseline (committed 0.0 for gpt-4o-mini) is stale vs the post-retrieval-fix ~0.8 rate — a clean follow-up PR

**Archives:**

- [milestones/v2.2-ROADMAP.md](milestones/v2.2-ROADMAP.md) — full phase details + decisions
- [milestones/v2.2-REQUIREMENTS.md](milestones/v2.2-REQUIREMENTS.md) — final traceability (17/17, 2 deferred-with-note)
- [milestones/v2.2-MILESTONE-AUDIT.md](milestones/v2.2-MILESTONE-AUDIT.md) — audit PASSED (17/17 reqs, 6/6 flows, integration complete)

---

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
