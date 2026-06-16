# City Concierge

## What This Is

City Concierge is a tool-calling RAG agent for San Francisco place recommendations and multi-stop itinerary planning. It's grounded in a Google Places dataset (~5,800 SF places in `places_raw`) plus pgvector embeddings, served by a FastAPI `/chat` endpoint backed by a LangGraph agent loop. The agent driver is selected from MLflow Model Registry (Opus 4.7 / GPT-4o / Gemini 2.5).

This is the USF MSDS 603 (MLOps) capstone codebase. Infrastructure is GCP: Cloud Run for the API, Cloud SQL Postgres 18 (pgvector + plain edge tables) for retrieval and KG, a shared MLflow tracking server on GCE for experiments and the registry.

## Core Value

The ONE thing that must work: a user asks for a constraint-heavy multi-stop itinerary in natural language ("date night in North Beach, 3 stops, under $$$, walking distance"), and the agent returns a coherent plan grounded in real places — geographically anchored, temporally consistent, and constraint-satisfying — with a booking deep-link.

## Context

The agentic redesign shipped as workstreams W0 → W8 (`implementation_plan/james/`) and W7 (knowledge graph) was the v1.0 GSD milestone. Post-W7, two reliability rounds shipped on `main`: the closure-aware itinerary swap (replacing the temporal_coherence caveat with silent walking-distance swaps + accept/decline/alternative `/chat` routing) and the json-safety fix for tool_call args mutated across plan steps.

Five live runs against the omakase Mission/Japantown query revealed the next class of bugs — category compliance, rationale-stop alignment, refinement-turn explosions, and the lack of any reproducible way to measure agent quality without invading the shared MLflow `production` alias. **v2.0 Production Readiness shipped that work on 2026-06-03 (PR #100 → `14e01dd`):** the eval harness + model-override env var + three agent-behavior fixes, all measured against committed cross-model baselines with a CI hard gate on baseline staleness. The Phase 6 minimal-edit refinement probe also empirically confirmed an architectural limit (`_prune_for_llm` drops `reasoning_content` across turns), which scopes v2.1: reasoning-model compat.

## Current State

**Active milestone:** v2.3 Adaptive Data Loop (Phases 16-19) — productionizes the `coverage_agent.py` retrieval-gap loop to learn from real USER queries. **Phase 16 (Loop Falsifier) COMPLETE 2026-06-15: FALSIFY-01 hard gate PASSED** (hit@5 delta +1.000, exit 0, against an isolated `city_concierge_sandbox` DB, prod untouched) — the milestone go/no-go is cleared. **Phase 17 (LOG) COMPLETE 2026-06-16: 9/9 must-haves verified** — `user_query_log` table (Alembic migration chained to head `e0cd7069bc8f`, 7 cols) + fire-and-forget `log_user_query` write path wired into `chat()` via `BackgroundTasks`; full suite green (1516 passed), zero behavior change to the existing loop. Phases 18 (GAP), 19 (LOOP-01..03 + METRIC) are scoped (see 16-CONTEXT.md) but not yet planned. Next: `/gsd-discuss-phase 18`.

**Last shipped milestone:** v2.2 Reasoning-Model Decisiveness (2026-06-15; PR #110 + phase 12-14 branches; 4 phases (12-15), 24 plans, 26 tasks; 188 files changed, +24k/−8.2k vs v2.1). Audit: 17/17 requirements, integration COMPLETE, 6/6 flows, status PASSED — see `milestones/v2.2-MILESTONE-AUDIT.md`.

**The v2.2 result is an honest negative.** v2.1 proved reasoning state round-trips byte-correctly through all four provider adapters; v2.2 set out to make those models *decisive* (call `commit_itinerary` instead of burning the step budget exploring). It instrumented the loop, ran four joint experiment arms, and entered the conditional replay phase — and **no intervention cleared the falsifier bar** (gpt-5-mini ≥ 0.6 commit rate at n=5; best result A2 = 0.500 pooled). The decisiveness gap is architectural, not tunable under the ~30s/turn budget. `openai/gpt-4o-mini` is re-ratified as the prod anchor (omakase median 1.000 flag-off, gate ≥ 0.8 enforced); gpt-5-mini stays logged-not-gated. ARCH-FUT-01 (replacing the LangGraph loop with a custom imperative loop) is deferred as tracked debt, user-ratified at the D-14-08 checkpoint. Canonical record: `docs/promotion_decision.md`, cross-linking the two immutable verdict docs (`docs/decisiveness_arm_verdicts.md`, `docs/replay_arm_verdicts.md`).

**Agent driver:** `openai/gpt-4o-mini` (anchor held commit-rate median 1.0 throughout v2.1 and v2.2). Reasoning-state loss is FIXED (adapters + conformance harness in CI); reasoning-model *decisiveness* is an open architectural question, now bounded by evidence rather than left as a hypothesis.

**v2.3 in flight:** Phase 16 falsified the loop mechanism (it can add places that weren't there and make them retrievable). Phase 17 delivered the loop's learning signal: every main-path `/chat` demand query is now logged to Cloud SQL (`user_query_log`) fire-and-forget. Remaining v2.3 threads to scope into Phases 18-19: the real demand/supply gap miner (GAP, mines `user_query_log` to replace the hardcoded gap constant), and the productionized ingest→embed→metric loop + hit@k scorer (LOOP/METRIC). Carried-over deferred threads from v2.2 (anthropic/gemini baselines, stale `refinement_cheaper` baseline, ARCH-FUT-01) remain open — see Active requirements + Key Decisions.

**v2.2 delivered:**
- Decisiveness instrumentation + executable falsifier: per-run telemetry (steps-to-first-commit-consideration, per-step viable-candidate counts, rule-8 precondition flag) + per-turn latency decomposition; `make eval-falsifier` is a single pass/fail report (Phase 12 / INST-01..05).
- Four joint experiment arms run at n=5 temp=1.0 — viability contract, forced-commit-at-step-N, co-tuned critique recalibration, parallel tool execution — judged against the comparison floor; honest null, no arm cleared 0.6 (Phase 13 / DEC-01..05).
- Conditional replay phase entered and plateaued: R1 multi-message replay = 0.500 (= A2); R2 content-block preservation refuted in the breaking direction (str() collapse was load-bearing for the Responses API path) (Phase 14 / REPLAY-01/02).
- Gate promotion + honest baseline regen: gpt-4o-mini enforced, gpt-5-mini logged; 6 runnable cells re-baselined flag-off n=5 with corrected provenance; latency report documents the 47s median vs the ~30s budget (Phase 15 / PROMO-01..03).

**v2.1 delivered:**
- Prompt/rubric decoupling: behavioral rules live in the `refinement_minimal_edit` scorer (D-07-07 `primary_type` matrix), locked out of prompts by a CI grep gate; PROMPT-05 falsifier resolved (gpt-5-mini flat 0/5 → state-loss dominated, Phase 9 stayed full scope).
- Typed `ProviderAdapter` reasoning-state contract wired POST-PRUNE into the agent graph + 9-test conformance harness as a required CI step; LangGraph retained (REASON-05 gate passed).
- Four per-provider adapters (gpt-5 Responses API, DeepSeek reasoner, Anthropic, Gemini 3 experimental) with import isolation and a revertability audit.
- Eval-harness honesty: fail-open scoring closed, per-case error records, scenario quarantine, per-family gates re-derived from honest data, fail-closed probe redaction.
- Honest n=5 baselines via `scripts/write_baselines.py` (refuses partial/quarantined cells); 3 cross-model anchors in the matrix; live-key-free `--baselines-mode` gate + extended staleness watch-set as required CI steps; `docs/baseline_regen.md` runbook.

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

- ✓ Prompt/rubric decoupling: task-only `_REFINEMENT_PREAMBLE` + SYSTEM_PROMPT rule 10 deleted; behavioral rules enforced by `refinement_minimal_edit` scorer; CI grep gate — v2.1 Phase 7 (PROMPT-01..05; 04 via documented override, 05 falsifier resolved)
- ✓ Typed `ProviderAdapter` reasoning-state contract + 9-test conformance harness (required CI step); `_prune_for_llm` delegates to adapters; LangGraph retained per REASON-05 gate — v2.1 Phase 8 (REASON-01..06)
- ✓ Per-provider state preservation: OpenAI gpt-5 (Responses API), DeepSeek reasoner, Anthropic (full provider wiring), Gemini 3 (experimental); import isolation + revertability audit — v2.1 Phase 9 (PROV-01..05; 01–03 shipped-with-gap, decisiveness → v2.2)
- ✓ Eval-harness honesty: error-status records, `baseline_eligible` quarantine, per-family gates vs real summary shape, fail-closed probe redaction — v2.1 Phase 10 (EVAL-01..06)
- ✓ Honest n=5 baselines via `write_baselines.py`; 3 cross-model matrix anchors; `--baselines-mode` live-key-free CI gate (fail-closed); staleness watch-set covers `app/llm_factory.py` + matrix configs; `docs/baseline_regen.md` runbook — v2.1 Phase 11 (BASE-01..04)

- ✓ Decisiveness instrumentation + executable falsifier: per-run telemetry (first-commit step, viable-candidate counts, rule-8 precondition flag) + per-turn latency decomposition; `make eval-falsifier` single pass/fail report — v2.2 Phase 12 (INST-01..05)
- ✓ Joint decisiveness experiment arms (viability contract, forced-commit-at-step-N, co-tuned critique recalibration, parallel tool execution) judged at n=5 against the comparison floor — **honest null, no arm cleared the 0.6 bar** — v2.2 Phase 13 (DEC-01..05)
- ✓ Conditional richer state replay (multi-message `_reasoning_state` replay, content-block preservation), A/B-measured against the DEC plateau — both plateaued; R2 refuted in the breaking direction — v2.2 Phase 14 (REPLAY-01/02)
- ✓ Gate promotion + honest baseline regen: gpt-4o-mini re-ratified `enforced`, gpt-5-mini demoted to `logged`; 6 runnable cells re-baselined flag-off n=5 with corrected provenance; latency report vs ~30s budget — v2.2 Phase 15 (PROMO-01..03)

- ✓ Query logging (LOG): net-new `user_query_log` table via Alembic migration `d1be72aea7d4` (chained to head `e0cd7069bc8f`, 7 cols, raw message text stored verbatim per D-04 private-capstone posture); fire-and-forget `log_user_query` sync INSERT helper (fail-open, 100% unit coverage) wired into `chat()` via FastAPI `BackgroundTasks`; autouse `_neutralize_query_log` test-isolation fixture + scheduling spy + integration round-trip; zero behavior change — v2.3 Phase 17 (D-01..04)

### Active (between milestones — scope next via `/gsd-new-milestone`)

Open threads carried out of v2.2 (none are committed scope yet):

- [ ] Reasoning-model decisiveness — confirmed architectural; ARCH-FUT-01 (custom imperative loop replacing LangGraph) is the deferred contingency, trigger = the Phases 13-14 evidence chain
- [ ] anthropic n=5 + gemini n=5 baselines — deferred at user billing decision (D-11-20, D-12-09); promotion path in `docs/baseline_regen.md`
- [ ] `refinement_cheaper` gpt-4o-mini baseline (committed 0.0) is stale vs the post-retrieval-fix ~0.8 rate — clean follow-up baseline regen
- [ ] Prod-default `FORCED_COMMIT_STEP=6` flip — flagged but not implemented (D-15-07)

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
| v2.1 phase order reversed from draft: prompt/rubric decoupling (Phase 7) sequenced BEFORE reasoning-state thread-through | Decoupling is a falsifier for the architectural diagnosis (if reasoning models still 0/5 after decoupling, state-loss is confirmed; if they move off 0, prompt-coupling was a bigger factor and Phase 9 scope shrinks). Also the only phase that pays back on the v2.0 anchor even if v2.1 stalls | ✓ Good — falsifier resolved exactly as designed: gpt-5-mini stayed flat 0/5 post-decoupling, confirming state-loss dominated; Phase 9 ran at full scope |
| v2.1 draft Phase 1 (reasoning-state thread-through) split into Phase 8 (contract + conformance harness) and Phase 9 (per-provider impls, one sub-phase each) | W10 prior attempt landed three different "fixes" before finding what worked; per-provider variance is too high for one phase. Splitting gives independent ship/revert per provider and lets Phase 8 double as the harness-swap decision gate | ✓ Good — Phase 8 gate kept LangGraph; all 4 adapters shipped independently with revertability audit |
| `category_compliance` abstains (`None`) on zero committed stops instead of fail-open 1.0 | Decisiveness-failing providers were inflating medians; abstain = no signal, excluded from aggregation. The first fix crashed at the `float()` consumer — caught by phase-11 code review, fixed + contaminated baseline cells re-measured live | ✓ Good — locked 2026-06-11 (D-11-03 + CR-01 gap closure) |
| Anthropic demoted to logged-not-gated; gemini deferral retained | API credits exhausted mid-regen (billing, not code); promotion path documented in `docs/baseline_regen.md`. Gemini's single scored refinement run (commit-rate 1.0) is first evidence the Phase-9 adapter fixed it | — Anthropic deferral extended into v2.2 (user declined top-up at milestone start); gemini promoted to v2.2 scope (ANCH-02) |
| Baselines only ever written by `scripts/write_baselines.py`; matrix execution stays sequential (D-11-14) | Hand-rolled baselines caused the v2.0 fail-open saturation; the tool refuses partial/quarantined cells mechanically. Parallel runs would contaminate latency measurements — scoped temp matrix configs are the sanctioned subset-rerun mechanism | ✓ Good — locked 2026-06-11 |
| v2.2 Decision 3: prod latency budget ~30s/turn — reasoning models are documented alternates, not prod-promotion candidates; gpt-4o-mini stays anchor | Sub-10s "frontier chat" latency is a single ungrounded forward pass; this agent's 50-90s is 12-18 sequential grounded tool calls. Decisiveness (step-count) + parallel tool execution are the levers that close the gap for the anchor; reasoning models' 20-30s/call thinking cost can't fit ~30s/turn regardless of convergence | Decided 2026-06-11 at v2.2 start; bounds investment in experiment/replay phases |
| Stay on LangGraph for v2.1 unless Phase 8's conformance harness shows `graph.invoke` itself drops the round-tripped state | The W10 failure was at the langchain-openai library boundary, not at LangGraph's reducer. Swapping the harness now would pay full architectural cost for a problem we haven't proven the harness causes. Phase 8 is the cheapest place to test the harness in isolation | ✓ Resolved 2026-06-04 — Phase 8 REASON-05 gate PASSED: `add_messages` reducer preserves `additional_kwargs["_reasoning_state"]` through `graph.invoke`. LangGraph retained for v2.1; no v2.1.1 replan triggered |
| INST-05 falsifier defined as a single executable report: intervention "works" iff gpt-5-mini commit rate ≥ 0.6 at n=5 AND gpt-4o-mini holds ≥ its honest baseline | A milestone chasing a behavior needs one objective, pre-registered pass/fail criterion or it drifts into post-hoc rationalization. `make eval-falsifier` makes the bar non-negotiable and machine-checked | ✓ Good — locked 2026-06-11; every v2.2 arm judged against it, no goalpost-moving |
| D-12-09: gemini n=5 baseline deferred at user billing decision, joining anthropic; comparison floor = matrix minus BOTH cells | No quota/billing top-up; gemini's single scored run already hit 1.0 (measurement debt, not unknown risk). Deferring keeps the floor honest (non-deferred cells stay n=5) without blocking the falsifier | — Both cells stay logged-not-gated; revisit when budget allows |
| Phase 14 (richer state replay) gated on all DEC arms plateauing below the falsifier bar | Replay is more expensive and more architectural than prompt/graph tuning; entering it speculatively would burn spend on an unproven lever. The conditional gate forced the cheaper arms to fail first | ✓ Good — gate opened correctly (DEC null); replay ran and also plateaued, converting a hypothesis into evidence |
| v2.2 closes as an honest null: no arm cleared the falsifier; gpt-4o-mini anchor re-ratified; ARCH-FUT-01 deferred as tracked debt | Four experiment arms + two replay arms all plateaued at/below 0.500 vs the 0.6 bar. The decisiveness gap is architectural, not tunable under the ~30s/turn budget. Ratifying the anchor and deferring the loop rewrite (vs forcing a weak promotion) is the evidence-driven call | ✓ Good — user-ratified at D-14-08; canonical record `docs/promotion_decision.md`. Trigger for ARCH-FUT-01 reconsideration: the Phases 13-14 evidence chain |
| gpt-4o-mini gate promoted to `enforced` (≥ 0.8 omakase); gpt-5-mini demoted to `logged`; baselines re-measured flag-off n=5 with corrected provenance | The prior `refinement_cheaper` 1.000 was a flag-ON arm artifact, not honest prod config; promoting a gate on a contaminated number repeats the v2.0 fail-open mistake. Enforce only what honest flag-off data earns | ✓ Good — locked 2026-06-15 (Phase 15); 6 runnable cells re-baselined; `refinement_cheaper` 0.0 baseline noted stale, flagged for follow-up regen |

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
*Last updated: 2026-06-16 after v2.3 Phase 17 (Query Logging / LOG) — `user_query_log` table + fire-and-forget `/chat` write path shipped, 9/9 must-haves verified. v2.2 milestone (Reasoning-Model Decisiveness) was an honest null; gpt-4o-mini anchor re-ratified; ARCH-FUT-01 deferred as tracked debt.*
