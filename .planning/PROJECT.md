# City Concierge

## What This Is

City Concierge is a tool-calling RAG agent for San Francisco place recommendations and multi-stop itinerary planning. It's grounded in a Google Places dataset (~5,800 SF places in `places_raw`) plus pgvector embeddings, served by a FastAPI `/chat` endpoint backed by a LangGraph agent loop. The agent driver is selected from MLflow Model Registry (Opus 4.7 / GPT-4o / Gemini 2.5).

This is the USF MSDS 603 (MLOps) capstone codebase. Infrastructure is GCP: Cloud Run for the API, Cloud SQL Postgres 18 (pgvector + plain edge tables) for retrieval and KG, a shared MLflow tracking server on GCE for experiments and the registry.

## Core Value

The ONE thing that must work: a user asks for a constraint-heavy multi-stop itinerary in natural language ("date night in North Beach, 3 stops, under $$$, walking distance"), and the agent returns a coherent plan grounded in real places — geographically anchored, temporally consistent, and constraint-satisfying — with a booking deep-link.

## Context

The agentic redesign shipped as workstreams W0 → W8 (`implementation_plan/james/`) and W7 (knowledge graph) was the v1.0 GSD milestone. Post-W7, two reliability rounds shipped on `main`: the closure-aware itinerary swap (replacing the temporal_coherence caveat with silent walking-distance swaps + accept/decline/alternative `/chat` routing) and the json-safety fix for tool_call args mutated across plan steps.

Five live runs against the omakase Mission/Japantown query revealed the next class of bugs — category compliance, rationale-stop alignment, refinement-turn explosions, and the lack of any reproducible way to measure agent quality without invading the shared MLflow `production` alias. **v2.0 Production Readiness shipped that work on 2026-06-03 (PR #100 → `14e01dd`):** the eval harness + model-override env var + three agent-behavior fixes, all measured against committed cross-model baselines with a CI hard gate on baseline staleness. The Phase 6 minimal-edit refinement probe also empirically confirmed an architectural limit (`_prune_for_llm` drops `reasoning_content` across turns), which scopes v2.1: reasoning-model compat.

## Current Milestone: v2.1 Reasoning-Model Compat

**Goal:** Unblock reasoning models (gpt-5 family, Claude Sonnet 4.6, DeepSeek reasoner, Gemini 3) on the agent loop so the v2.0 anchor (`openai/gpt-4o-mini`) is no longer the only viable prod path.

**Empirical anchor gate:** `gpt-5-mini × refinement_cheaper × prod × flag-on` commits 3 stops in median 5/5 runs at temp=1.0 (currently 0/1).

**Target features (phases 7-10):**
- Prompt/rubric decoupling so the rubric stops describing the prompt the locked anchor was tuned against (Phase 7)
- Reasoning-state thread-through: typed provider-adapter contract + conformance harness; doubles as the LangGraph-vs-imperative-loop decision gate (Phase 8)
- Per-provider state preservation implementations, one sub-phase per provider in order gpt-5 → DeepSeek reasoner → Claude (absorbs Anthropic wiring) → Gemini 3 (Phase 9)
- Honest cross-model baseline regen + matrix expansion under DB-up conditions (Phase 10)

**Key context:** Phase 7 is sequenced first because it's a falsifier for the architectural work in Phase 8 *and* the only phase that independently improves the v2.0 prod anchor. Phase 8's conformance harness is a decision gate: if reasoning state plumbs through provider adapters but fails when run through `graph.invoke`, v2.1 replans around a custom imperative loop instead of finishing on LangGraph. Anti-scope: replace LangGraph entirely (unless Phase 8 forces it), multi-agent split, new scorers.

## Current State

**Shipped milestone:** v2.0 Production Readiness (2026-06-03, PR #100 → main `14e01dd`).
**Active milestone:** v2.1 Reasoning-Model Compat (started 2026-06-03; phases 7-10).
**Current codebase scale:** ~107 files touched and 18k LOC added across v2.0; agent driver locked to `openai/gpt-4o-mini` until v2.1 ships reasoning-content thread-through.

**v2.1 progress:**
- Phase 7 (Prompt/Rubric Decoupling) — shipped 2026-06-03 (PR #101).
- Phase 8 (Reasoning-State Thread-Through Contract + Conformance Harness) — shipped 2026-06-04 (branch `gsd/phase-08-...`). D-08-11 Branch A acceptance: REASON-05 gate PASSED — LangGraph's `add_messages` reducer preserves `additional_kwargs` end-to-end through `graph.invoke`. Phase 9 proceeds on LangGraph; no v2.1.1 imperative-loop replan triggered. Typed `ProviderAdapter` contract live; `NoOpAdapter` byte-identical to pre-Phase-8 for the locked `openai/gpt-4o-mini` anchor (pinned by `tests/unit/fixtures/reason_04_prune_baseline.json`).
- Phase 10 (Eval Harness Honesty) — complete 2026-06-11 (branch `gsd/phase-10-eval-harness-honesty`, verification 6/6 after gap-closure wave 10-07..10-09 closed CR-01..CR-05). ERROR-status records replace fail-open scoring; `late_night_closure_cascade` quarantined via `baseline_eligible` flag wired through real summary.json; per-family merge gates in `configs/eval_gates.yaml` enforced by `make eval-gates-check` against the real nested summary shape; per-provider live-probe fixtures with fail-closed redaction; gpt-5 dispatch + ScriptedChatModel ainvoke test debt closed.

**v2.0 delivered:**
- Reproducible cross-model eval harness with committed baselines, multi-turn threading, scripted-LLM CI mode, and a hard CI gate on baseline staleness.
- `RAG_MODEL_OVERRIDE` env var so any candidate model can be wired through `/chat` without touching the shared MLflow `production` alias.
- Category-compliance fix: tool calls inject `primary_type_family` per slot; rationales describe the committed place's actual category.
- Closure-swap placeholder bleed eliminated at construction site, pinned by two-layer regression coverage.
- Minimal-edit refinement behind `REFINEMENT_STRUCTURED_PLAN_ENABLED` feature flag; `committed_stops` round-trip + shared `build_refinement_prompt_message` helper used by both `/chat` and the eval runner.

**Flagged for separate investigation (carried from v2.0):**
- Over-aggressive closure detection on Mission queries: pre-Phase-3 D-07 check confirmed `place_is_open` is correct; remaining anomaly is likely hours-data drift or `_per_stop_closure_status` overly cautious. (CLO-01 in v2.0 archive.)

## Requirements

### Validated

- ✓ FastAPI `/chat` endpoint backed by LangGraph agent loop — W2 (PR #71)
- ✓ pgvector retrieval tools (`semantic_search`, `nearby`, `get_details`) over `place_documents` view — W1 (PR #65, #66)
- ✓ Parallel `place_embeddings_v2` table with cleaned chunks + structured neighborhood/landmarks — W0a (PR #58)
- ✓ Self-correction loop (constraint relaxation on empty/low-quality results) — W3 (PR #74)
- ✓ Booking handoff stub (`propose_booking` deep-links to Resy/Tock/Maps) — W4 (PR #75)
- ✓ Infra hardening (cold starts, tracing, secrets, cost telemetry) — W0 (PR #60); MLflow auth proxy deferred
- ✓ MLflow registry-driven model selection at startup — pre-W0
- ✓ Alembic migrations + `app.db_url` resolver for local + Cloud SQL — pre-W0
- ✓ Shared MLflow tracking server on GCP — pre-W0
- ✓ `place_relations` edge table + idempotent five-edge-type builder + real `kg_traverse` tool — W7 (v1.0 GSD milestone)
- ✓ Live map + Directions routing overlay (ordered itinerary on a real Google Map) — W8 series
- ✓ Closure-aware itinerary swap: `swap_closed_stops` node, `ClosureContext` round-tripped via opaque `conversation_state` field on `/chat`, accept/decline/alternative early-return paths, SQL-layer `excluded_place_ids` enforcement across `semantic_search`/`nearby`/`kg_traverse` — PR #94, merge commit `ad8ca84` (2026-05-20)
- ✓ Bare-number reply ("3") parses as `num_stops` when the prior assistant turn was a how-many-stops question — PR #94 commit `74e9c88`
- ✓ Tool-call args stay JSON-serializable across plan steps (`_inject_closure_exclusions` returns dict-shaped filters; `act()` never mutates `AIMessage.tool_calls[i]["args"]`) — PR #94 commit `be541a3`
- ✓ Rationale-stop alignment fix: closure-swap candidate rationale is now derived via inherit-or-deterministic-fallback (no LLM call inside `swap.py`); legacy `"Walking-distance alternative for X"` placeholder cannot reach `final_reply` either by new construction or by carryover from pre-fix `conversation_state` — Phase 5 (RAT-02), commits `a20e225` + `ff3e073`
- ✓ `RAG_MODEL_OVERRIDE` env var: any candidate model can be wired through `/chat` (`version:N` or `alias:NAME`) without touching the shared MLflow `production` alias — v2.0 Phase 2 (OVR-01..OVR-06)
- ✓ Reproducible cross-model eval harness: `category_compliance` + `rationale_stop_alignment` scorers, multi-turn threading via `EvalQuery.turns`, `EvalMatrixConfig` + `MatrixEntry`, `scripts/eval_matrix.py`, three committed baselines, scripted-LLM CI mode, hard CI gate on baseline staleness (`scripts/check_baselines_fresh.py`) — v2.0 Phase 3 (EVAL-01..EVAL-10)
- ✓ Category compliance: agent injects `SearchFilters.primary_type_family` per named slot via `_inject_primary_type_family` in `act()`; rationales describe the committed place's actual category via `rationale_misaligned` revision dispatch; `category_compliance_strict` scorer holds the floor — v2.0 Phase 4 (CAT-01..CAT-04 + RAT-01 + RAT-03 folded in per D-04-09)
- ✓ Minimal-edit refinement: `ConversationState.committed_stops` round-trips between turns; `build_refinement_prompt_message` shared helper used by both `/chat` and the eval runner; `REFINEMENT_STRUCTURED_PLAN_ENABLED` feature flag (default OFF); CI structural-check hard gate via `make eval-matrix-refinement-structural-check` — v2.0 Phase 6 (REF-01..REF-04, D-06-09 part 1 PASSES on 5-run live)

### Active (v2.1 — Reasoning-Model Compat)

- [ ] **Phase 7 — Prompt/rubric decoupling:** rewrite `_REFINEMENT_PREAMBLE` + `SYSTEM_PROMPT` rule 10 to describe TASK ("user wants to swap one stop; return the new full itinerary") not BEHAVIOR ("keep same stop count; do not ask clarifying questions"); behavioral rules move into the scorer where they belong. Sequenced first as a falsifier for Phase 8 *and* because it independently improves the v2.0 anchor today.
- [ ] **Phase 8 — Reasoning-state thread-through (contract + harness):** typed provider-adapter contract for round-tripping reasoning state (`reasoning_content` / `thought_signature` / encrypted reasoning / `thinking` blocks); per-provider conformance test harness running a 2-turn agent loop and asserting state field shows up in turn 2's outbound payload; `_prune_for_llm` refactor that delegates state preservation to the provider adapter. **Doubles as harness-swap decision gate** — if isolated tests pass but `graph.invoke` drops state, v2.1 replans around a custom imperative loop.
- [ ] **Phase 9 — Per-provider state preservation impls:** one sub-phase per provider in order: gpt-5 family → DeepSeek reasoner → Claude Sonnet 4.6 (absorbs Anthropic provider wiring: add `claude` to `SUPPORTED_PROVIDERS` in `app/llm_factory.py`, new `build_chat_model` branch, `langchain-anthropic` dep) → Gemini 3 (experimental, no gate). Each sub-phase independently shippable and revertable; v2.0 anchor path untouched.
- [ ] **Phase 10 — Cross-model baseline regen + matrix expansion:** rebuild all `configs/eval_baselines/*.json` honestly (post-fail-open era); add gpt-5-mini, claude-sonnet-4-6, deepseek-reasoner as cross-model anchors; lock new per-family merge gates and enforce in CI.

### Out of Scope (v2.0 — archived)

- DSPy / managed eval UI — eval harness is intentionally a Python script, not a managed service
- Per-PR ephemeral MLflow aliases — `RAG_MODEL_OVERRIDE` proved sufficient
- Full LLM-as-judge rubric on rationale quality — deterministic checks worked; revisit only if they plateau
- Streaming responses, retry policies, rate limiting — separate hosting concerns

### Out of Scope (Project)

- Booking automation (Playwright against Resy/Tock) — separate future PR behind `BOOKING_AUTOMATION_ENABLED`
- Dropping v1 embeddings — gated on W6 evals showing v2 is non-regressing
- LLM-extracted edges (`OPERATED_BY`, `MENTIONED_WITH`, `SAME_CHEF`) — deferred until editorial scrape lands and cheap edges prove their value
- Apache AGE / openCypher in Postgres — not supported on Cloud SQL

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Plain SQL edge tables for the KG | Apache AGE not available on Cloud SQL; pure SQL is indexable, joinable, no new extension | Locked (W7 spec) |
| Seed five free/computed edge types only; defer LLM-extracted edges | Cheap edges first; justify LLM cost with retrieval-quality wins | Locked (W7 spec) |
| `SIMILAR_VECTOR` computed from v2 embeddings only | v2 is the active embedding table post-W0a | Locked (W7 spec); revisit on v1 deprecation |
| No FK on `dst_place_id` | Landmarks may live outside `places_raw`; JOIN through `place_documents` filters at query time | Locked (W7 spec) |
| GSD scope = W7 only as Phase 1 | Existing `implementation_plan/james/` is the source of truth for prior workstreams | Decided 2026-05-14 during scaffold |
| Closure handling moves from "ship plan + caveat" to "silent swap or one user question" | Caveat path was worst-of-both-worlds (broken plan + ugly warning); user testing showed swap is preferred | Locked (fix/agent-reliability-review) |
| `conversation_state` is opaque to the frontend, stored in `useRef`, validated server-side | Stateless `/chat` needs to round-trip closure history; `useRef` avoids stale-closure capture in empty-deps `handleSend` | Locked (fix/agent-reliability-review) |
| Eval harness ships BEFORE agent-behavior fixes in v2.0 | Without it, fix evaluation is one-run-at-a-time eyeballing (failure mode that surfaced this milestone) | Decided 2026-05-21 during v2.0 scoping |
| Model selection for testing uses an env-var override, not MLflow alias surgery | Shared `production` alias has caused incidents (Gemini 3 rollback); env-var keeps prod safe | Decided 2026-05-21 during v2.0 scoping |
| PR #94 merges before v2.0 phase 1 starts; re-baseline confirms which bugs are real | Branch-vs-main confusion during initial v2.0 scoping mis-attributed which bugs existed where; re-running 5 omakase scenarios against merged main is the only way to know what v2.0 must actually fix | Decided 2026-05-21; merged at `ad8ca84` |
| Drop "step-budget tuning" from v2.0 scope (originally 6 phases → 5 phases) | Post-merge re-baseline showed bug #7 (step-limit on revision turns) was a misattributed JSON-safety crash now fixed in PR #94; the actual step-budget issue does not reproduce on merged main | Decided 2026-05-21 after re-baseline |
| D-04-09: RAT-01 + RAT-03 fold into Phase 4; Phase 5 narrows to RAT-02 only | Refinement-turn rationale integrity is structurally the same problem as category compliance (CAT-02), and `rationale_misaligned` revision dispatch handles both. Phase 5's RAT-02 (closure-swap placeholder bleed) is a structurally different bug | ✓ Good — locked 2026-05-22, both phases shipped clean |
| D-04-13 / D-04-14: When the gated scorer is added in the same phase, substitute an absolute floor for the undefined delta-vs-old-baseline | Delta-vs-baseline is undefined when there is no old baseline; documenting the trade-off as a locked decision is more honest than gaming the threshold | ✓ Good — both gates passed cleanly |
| Phase 6 D-06-09 part 2 (no-regression) accepted with notes; real fix scoped to v2.1 | Pre-Phase-6 1.0 baselines were Phase-4 fail-open false positives; the new fail-loud measurement makes the "regression" appear, but real agent behavior improved. The architectural fix (reasoning-content thread-through) requires v2.1 work | ⚠️ Revisit during v2.1 reasoning-content phase |
| Lock agent driver to `openai/gpt-4o-mini` for v2.0 prod anchor | gpt-5-mini / gpt-5.4-mini / DeepSeek reasoner / Claude / Gemini 3 all fail on this codebase's tool-loop tasks because `_prune_for_llm` drops `reasoning_content` across turns. Locked anchor is a feature given current architecture | Decided 2026-06-03 (Phase 6 probe); revisit when v2.1 ships |
| v2.1 milestone drafted: reasoning-state thread-through + prompt/rubric decoupling + Anthropic wiring + honest baseline regen | Without these, every new reasoning model the field ships in 2026 is unusable on this codebase. Empirical anchor gate: gpt-5-mini × refinement_cheaper commits 3 stops in median 5/5 runs at temp=1.0 | ✓ Promoted to active 2026-06-03 |
| v2.1 phase order reversed from draft: prompt/rubric decoupling (Phase 7) sequenced BEFORE reasoning-state thread-through | Decoupling is a falsifier for the architectural diagnosis (if reasoning models still 0/5 after decoupling, state-loss is confirmed; if they move off 0, prompt-coupling was a bigger factor and Phase 9 scope shrinks). Also the only phase that pays back on the v2.0 anchor even if v2.1 stalls | — Pending (decided 2026-06-03) |
| v2.1 draft Phase 1 (reasoning-state thread-through) split into Phase 8 (contract + conformance harness) and Phase 9 (per-provider impls, one sub-phase each) | W10 prior attempt landed three different "fixes" before finding what worked; per-provider variance is too high for one phase. Splitting gives independent ship/revert per provider and lets Phase 8 double as the harness-swap decision gate | — Pending (decided 2026-06-03) |
| Stay on LangGraph for v2.1 unless Phase 8's conformance harness shows `graph.invoke` itself drops the round-tripped state | The W10 failure was at the langchain-openai library boundary, not at LangGraph's reducer. Swapping the harness now would pay full architectural cost for a problem we haven't proven the harness causes. Phase 8 is the cheapest place to test the harness in isolation | ✓ Resolved 2026-06-04 — Phase 8 REASON-05 gate PASSED: `add_messages` reducer preserves `additional_kwargs["_reasoning_state"]` through `graph.invoke`. LangGraph retained for v2.1; no v2.1.1 replan triggered |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-11 — Phase 10 complete: eval harness honesty (error-status records, gate enforcement, quarantine wiring, live-probe fixtures); v2.1 advances to Phase 11 baseline regen.*
