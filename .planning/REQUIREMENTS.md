# Requirements: City Concierge — v2.1 Reasoning-Model Compat

**Defined:** 2026-06-03
**Core Value:** Constraint-heavy multi-stop SF itinerary from a natural-language request, grounded in real places, with a booking deep-link.
**Milestone Goal:** Unblock reasoning models (gpt-5 family, Claude Sonnet 4.6, DeepSeek reasoner, Gemini 3) on the agent loop so the v2.0 anchor (`openai/gpt-4o-mini`) is no longer the only viable prod path.
**Milestone Empirical Anchor Gate:** `gpt-5-mini × refinement_cheaper × prod × flag-on` commits 3 stops in median 5/5 runs at temp=1.0 (currently 0/1).

## v2.1 Requirements

Each requirement maps to exactly one phase. Categories align 1:1 with phases for clean traceability.

### Prompt/Rubric Decoupling (Phase 7)

- [ ] **PROMPT-01**: The `/chat` refinement turn returns a full itinerary with the requested edit applied when the user asks to swap or modify a single stop. (Observable user behavior; previously enforced by prompt rules, now enforced by scorer.)
- [ ] **PROMPT-02**: `SYSTEM_PROMPT` rule 10 and `_REFINEMENT_PREAMBLE` are rewritten to describe the TASK ("user wants to swap one stop in this itinerary; return the new full itinerary"), with no behavioral prescriptions ("keep same stop count", "do not ask clarifying questions", "preserve `place_id` byte-for-byte", "same primary_type on replacement") in the prompt body.
- [ ] **PROMPT-03**: The `refinement_minimal_edit` scorer enforces the behavioral rules previously baked into the prompt — same stop count, byte-equal preserved `place_id`s except the target, same Google Place `primary_type` on the replacement — as scorer logic, not via prompt-coupled string heuristics.
- [ ] **PROMPT-04**: `openai/gpt-4o-mini × refinement_cheaper` median is ≥ pre-Phase-7 baseline on the existing v2.0 matrix. (No regression on the v2.0 prod anchor.)
- [ ] **PROMPT-05**: After Phase 7 ships, `gpt-5-mini × refinement_cheaper` median is > 0 across 5 runs at temp=1.0. (Falsifier signal: any non-zero is evidence prompt-coupling materially contributes; a flat 0/5 confirms architectural state-loss dominates and Phase 9 scope stays at full.)

### Reasoning-State Thread-Through Contract (Phase 8)

- [x] **REASON-01**: A typed `ProviderAdapter` contract exists in `app/agent/` (or `app/llm_factory/`) with a stable interface for inbound-capture and outbound-replay of provider reasoning state across one plan→act→plan turn boundary.
- [x] **REASON-02**: The contract supports at least four state field shapes: OpenAI `reasoning_content` (string), Anthropic `thinking` blocks (signed), DeepSeek `reasoning_content` (string), Gemini `thought_signature` (bytes). Adding a fifth shape is an interface extension, not a rewrite.
- [x] **REASON-03**: A per-provider conformance test harness (e.g. `tests/integration/test_reasoning_state_roundtrip.py`) runs a 2-turn agent loop against a mocked provider response, asserts the state field captured on the turn-1 `AIMessage` is present in the turn-2 outbound payload, and runs in CI as a quarantined integration test (does not gate prod merges unless explicitly enabled).
- [x] **REASON-04**: `_prune_for_llm` delegates state preservation to the `ProviderAdapter` contract for reasoning providers; non-reasoning providers (gpt-4o-mini, etc.) pass through unchanged. The delegation is unit-tested with regression coverage for the gpt-4o-mini happy path.
- [x] **REASON-05**: The conformance test passes for at least one reasoning provider (gpt-5 family) end-to-end, **including through `graph.invoke`**. *This is the harness-swap decision gate*: if isolated tests pass but `graph.invoke` drops state, surface as a Phase 8 blocker and replan v2.1 around a custom imperative loop.
- [x] **REASON-06**: After the `_prune_for_llm` refactor lands, `openai/gpt-4o-mini × refinement_cheaper` and all other v2.0 baselines do not regress. (CI hard gate — same staleness mechanism as v2.0 Phase 3 `EVAL-10`.)

### Per-Provider State Preservation Implementations (Phase 9)

- [x] **PROV-01**: `ProviderAdapter` implementation for OpenAI gpt-5 family lands. **Milestone anchor gate**: `gpt-5-mini × refinement_cheaper × prod × flag-on` commits 3 stops in median 5/5 runs at temp=1.0.
- [x] **PROV-02**: `ProviderAdapter` implementation for DeepSeek reasoner lands. Gate: `deepseek-reasoner × refinement_cheaper` median ≥ 0.6 across 5 runs at temp=1.0. (Lower bar; reasoner is exploratory and may need follow-up work.)
- [x] **PROV-03**: `ProviderAdapter` implementation for Anthropic Claude lands AND `claude` is added to `SUPPORTED_PROVIDERS` in `app/llm_factory.py` with a new `build_chat_model` branch and a `langchain-anthropic` dependency in `pyproject.toml`. Gate: `claude-sonnet-4-6 × refinement_cheaper` median ≥ 1.0 across 5 runs at temp=1.0.
- [x] **PROV-04**: `ProviderAdapter` implementation for Gemini 3 lands as **experimental** (no merge gate). `thought_signature` round-trips cleanly through a 2-turn loop via the conformance harness. Ships with a known-issues note and is not in the prod matrix until a follow-up phase upgrades it.
- [x] **PROV-05**: Each provider sub-phase ships as an independently revertable commit. Reverting any one sub-phase leaves the others and the v2.0 anchor (`openai/gpt-4o-mini`) functional in prod.

### Eval Harness Honesty (Phase 10 — re-scoped 2026-06-10)

- [x] **EVAL-01**: Eval runs whose turn-0 or turn-1 raises an exception produce an ERROR-status record excluded from score aggregation and surfaced in `summary.json` as an error count — never a fail-open 1.0 or fail-loud 0.0. Closes the three infra-failure scoring paths proven by `eval_reports/2026-06-05T21-14-30Z/` (all 25 cells were quota/temperature failures yet medians read 1.0).
- [ ] **EVAL-02**: `late_night_closure_cascade` runs prod threading (text-only history, mirroring `/chat`) or is explicitly quarantined from baselines and merge gates, with the decision recorded next to the scenario config.
- [x] **EVAL-03**: Per-family merge gates are re-derived from honest anchor data (the documented strict `refinement_minimal_edit == 1.0` gate is unsatisfiable — anchor median is 0.0/max 0.5 post-Phase-7) and enforced by an executable Makefile target that exits non-zero on regression.
- [ ] **EVAL-04**: A test asserts baseline JSON provider cells match matrix YAML entries in both directions, modulo documented deferrals (initial test shipped in PR #104).
- [x] **EVAL-05**: A per-provider live-probe Make target (~$0.01/call) is the documented mandatory pre-matrix step; captured real-wire responses are checked in as fixtures consumed by adapter/conformance tests.
- [x] **EVAL-06**: The `build_chat_model` gpt-5 dispatch branch (`use_responses_api=True`) has factory-level tests; `ScriptedChatModel` is exercised via `ainvoke`; the blocking sync `vibe_check` LLM call inside the async graph is made non-blocking or flag-documented as eval-only.

### Cross-Model Baseline Regen + Matrix Expansion (Phase 11)

- [ ] **BASE-01**: All `configs/eval_baselines/*.json` are regenerated under DB-up conditions (Cloud SQL or local Postgres reachable) with the Phase-7-decoupled prompt and Phase-9 provider adapters in place. The fail-open-saturated v2.0 baselines (documented in `project_phase4_d_04_14_locked` and `project_phase6_d_06_09_root_cause`) are replaced, and the regen procedure is documented in a runbook.
- [ ] **BASE-02**: The eval matrix (`configs/eval_matrix*.yaml`) includes `gpt-5-mini`, `claude-sonnet-4-6`, and `deepseek-reasoner` as cross-model entries alongside the existing `openai/gpt-4o-mini` anchor.
- [ ] **BASE-03**: Per-family merge gates are documented in a single source-of-truth doc (e.g. `docs/eval_gates.md` or similar) and enforced via Makefile targets + CI. The `gpt-5-mini × refinement_cheaper` anchor gate is one of them.
- [ ] **BASE-04**: A staleness check (analogous to `scripts/check_baselines_fresh.py`) covers the new cross-model baselines so a code change touching the agent loop without regenerating the new baselines fails CI.

## Future Requirements

Deferred to v2.2 or later. Tracked but not in current roadmap.

### Provider Coverage

- **PROV-FUT-01**: Promote Gemini 3 from experimental to gated provider once `thought_signature` integration stabilizes across LangChain releases.
- **PROV-FUT-02**: Add Kimi K2 / Moonshot back to the matrix if a `ChatMoonshot` reasoning-content round-trip path emerges (currently library-blocked per `project_agent_loses_reasoning_state_all_providers`).
- **PROV-FUT-03**: Add OpenAI o-series (o3, o4-mini) as a distinct provider adapter — encrypted reasoning blocks are a different shape than gpt-5 family `reasoning_content`.

### Architecture

- **ARCH-FUT-01**: Replace LangGraph with a custom imperative agent loop. *Triggered only if Phase 8's harness decision gate (`REASON-05`) shows `graph.invoke` itself drops reasoning state.*
- **ARCH-FUT-02**: Multi-agent planner-executor split. *Out of scope for v2.1; revisit if single-model adapters plateau on complex tasks.*

### Eval Infrastructure

- **EVAL-FUT-01**: LLM-as-judge rubric for rationale-quality scoring. *Current deterministic scorers cover v2.1 needs; revisit only if they plateau.*

## Out of Scope (v2.1)

Explicitly excluded. Documented to prevent scope creep mid-milestone.

| Feature | Reason |
|---------|--------|
| Replace LangGraph entirely in v2.1 | Premature; Phase 8's conformance harness is the cheapest place to test whether the framework is actually lossy. Only triggered by `REASON-05` failure. |
| Multi-agent / planner-executor split | Single-model fix is cheaper to test; defer until the per-provider adapter pattern has shipped. |
| New scorers beyond the rule-extraction in `PROMPT-03` | Current scorers are fine once rubric/prompt coupling is broken. Adding new scorers in v2.1 confounds the signal. |
| Streaming responses, retry policies, rate limiting | Separate hosting concerns; orthogonal to reasoning-state compat. |
| DSPy / managed eval UI | v2.0 already declared this out of scope; eval harness stays a Python script. |
| Booking automation (Playwright vs. Resy/Tock) | Carried out of scope from project-level; gated behind `BOOKING_AUTOMATION_ENABLED` for a future PR. |
| Apache AGE / openCypher in Postgres | Carried out of scope from project-level; Cloud SQL does not support AGE. |
| LLM-extracted KG edges (`OPERATED_BY`, `MENTIONED_WITH`, `SAME_CHEF`) | Carried out of scope from project-level; awaits editorial scrape. |
| Dropping v1 embeddings | Gated on v2 non-regression evals (separate from v2.1 reasoning-model work). |

## Traceability

Updated during roadmap creation (Phase 10 of this workflow).

| Requirement | Phase | Status |
|-------------|-------|--------|
| PROMPT-01 | Phase 7 | Pending |
| PROMPT-02 | Phase 7 | Pending |
| PROMPT-03 | Phase 7 | Pending |
| PROMPT-04 | Phase 7 | Pending |
| PROMPT-05 | Phase 7 | Pending |
| REASON-01 | Phase 8 | Complete |
| REASON-02 | Phase 8 | Complete |
| REASON-03 | Phase 8 | Complete |
| REASON-04 | Phase 8 | Complete |
| REASON-05 | Phase 8 | Complete |
| REASON-06 | Phase 8 | Complete |
| PROV-01 | Phase 9 | Complete |
| PROV-02 | Phase 9 | Complete |
| PROV-03 | Phase 9 | Complete |
| PROV-04 | Phase 9 | Complete |
| PROV-05 | Phase 9 | Complete |
| EVAL-01 | Phase 10 | Complete |
| EVAL-02 | Phase 10 | Pending |
| EVAL-03 | Phase 10 | Complete |
| EVAL-04 | Phase 10 | Pending |
| EVAL-05 | Phase 10 | Complete |
| EVAL-06 | Phase 10 | Complete |
| BASE-01 | Phase 11 | Pending |
| BASE-02 | Phase 11 | Pending |
| BASE-03 | Phase 11 | Pending |
| BASE-04 | Phase 11 | Pending |

**Coverage:**
- v2.1 requirements: 26 total
- Mapped to phases: 26
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-03 — v2.1 Reasoning-Model Compat milestone start*
*Last updated: 2026-06-10 — Phase 10 re-scoped to Eval Harness Honesty (EVAL-01..06); BASE-01..04 moved to new Phase 11*
