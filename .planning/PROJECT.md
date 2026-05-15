# City Concierge

## What This Is

City Concierge is a tool-calling RAG agent for San Francisco place recommendations and multi-stop itinerary planning. It's grounded in a Google Places dataset (~5,800 SF places in `places_raw`) plus pgvector embeddings, served by a FastAPI `/chat` endpoint backed by a LangGraph agent loop. The agent driver is selected from MLflow Model Registry (Opus 4.7 / GPT-4o / Gemini 2.5).

This is the USF MSDS 603 (MLOps) capstone codebase. Infrastructure is GCP: Cloud Run for the API, Cloud SQL Postgres 18 (pgvector + plain edge tables) for retrieval and KG, a shared MLflow tracking server on GCE for experiments and the registry.

## Core Value

The ONE thing that must work: a user asks for a constraint-heavy multi-stop itinerary in natural language ("date night in North Beach, 3 stops, under $$$, walking distance"), and the agent returns a coherent plan grounded in real places — geographically anchored, temporally consistent, and constraint-satisfying — with a booking deep-link.

## Context

The agentic redesign is captured in `implementation_plan/james/` as workstreams W0 → W7. W0–W4 are merged (infra hardening, v2 embeddings, retrieval tools, agent loop, self-correction, booking handoff stub). W5 and W6 are in flight in parallel branches. **W7 — the knowledge graph layer (`place_relations` + real `kg_traverse` tool) — is the active milestone for this GSD initialization.**

The W2 agent currently has a `kg_traverse` stub that returns "not yet available." W7 makes it real by building five edge types from data Google already gave us in `source_json` (landmarks, containing places, neighborhoods) plus computed edges (haversine NEAR, top-k SIMILAR_VECTOR over v2 embeddings). Storage is plain SQL edge tables — Apache AGE was ruled out because it isn't supported on Cloud SQL for Postgres.

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

### Active (W7 — Knowledge Graph)

- [ ] `place_relations` edge table migrated into Cloud SQL Postgres
- [ ] Idempotent builder that seeds five edge types: NEAR, SAME_NEIGHBORHOOD, CONTAINED_IN, NEAR_LANDMARK, SIMILAR_VECTOR
- [ ] Real `kg_traverse(place_id, relation_type, k)` tool replacing the W2 stub, returning `RelatedPlace` typed results joined through `place_documents`
- [ ] `kg_enabled` MLflow param logged on each agent run for A/B comparison
- [ ] Agent system prompt updated with `relation_type` selection guidance
- [ ] Unit + gated integration tests for builder and tool
- [ ] `make build-relations` target

### Out of Scope (W7)

- LLM-extracted edges (`OPERATED_BY`, `MENTIONED_WITH`, `SAME_CHEF`) — deferred until editorial scrape lands and cheap edges prove their value
- Editorial-source edges from teammate's Eater/Infatuation scrape — schema is forward-compatible but blocked on that table
- Apache AGE / openCypher in Postgres — not supported on Cloud SQL
- FK on `dst_place_id` — intentionally absent so landmark targets outside `places_raw` can be stored

### Out of Scope (Project)

- Booking automation (Playwright against Resy/Tock) — separate future PR behind `BOOKING_AUTOMATION_ENABLED`
- DSPy / managed eval UI — flagged in W6 future-watch, not adopted
- Dropping v1 embeddings — gated on W6 evals showing v2 is non-regressing

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Plain SQL edge tables for the KG | Apache AGE not available on Cloud SQL; pure SQL is indexable, joinable, no new extension | Locked (W7 spec) |
| Seed five free/computed edge types only; defer LLM-extracted edges | Cheap edges first; justify LLM cost with retrieval-quality wins | Locked (W7 spec) |
| `SIMILAR_VECTOR` computed from v2 embeddings only | v2 is the active embedding table post-W0a | Locked (W7 spec); revisit on v1 deprecation |
| No FK on `dst_place_id` | Landmarks may live outside `places_raw`; JOIN through `place_documents` filters at query time | Locked (W7 spec) |
| GSD scope = W7 only as Phase 1 | Existing `implementation_plan/james/` is the source of truth for prior workstreams | Decided 2026-05-14 during scaffold |

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
*Last updated: 2026-05-14 after initialization*
