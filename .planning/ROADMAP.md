# Roadmap

**Project:** City Concierge
**Active milestone:** W7 — Knowledge Graph Layer
**Granularity:** Coarse (single phase for the W7 PR)

## Phases

### Phase 1: Knowledge Graph Layer (`place_relations` + real `kg_traverse`)
**Goal:** Replace the W2 `kg_traverse` stub with a real implementation backed by a `place_relations` edge table, seeded with five free/computed edge types from existing data, and tracked in MLflow via a `kg_enabled` param so eval can A/B with the KG on or off.
**Branch:** `feature/agent-w7-knowledge-graph`
**Requirements:** KG-01, KG-02, BLD-01, BLD-02, BLD-03, BLD-04, BLD-05, BLD-06, BLD-07, TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05, TOOL-06, MLOPS-01, TEST-01, TEST-02
**Success Criteria:**
1. Migration `002_place_relations.sql` applied cleanly to Cloud SQL; `place_relations` table + indexes + `neighborhood_of()` helper exist.
2. `make build-relations` against the SF dataset produces non-empty edge counts for all five `relation_type` values; per-type counts match the expected ordering (`NEAR` largest, then `SAME_NEIGHBORHOOD`, `SIMILAR_VECTOR`, `NEAR_LANDMARK`, `CONTAINED_IN`); a second run produces zero new rows (idempotent).
3. `kg_traverse` from a known SF `place_id` (e.g., `ChIJExYUW8Z_j4AREJB4F5tJJto`) returns ordered, well-shaped `RelatedPlace` results for `NEAR` (ascending distance) and `SIMILAR_VECTOR` (descending cosine), and the LangGraph agent can call it end-to-end via `/chat`.
4. Unit tests (`tests/unit/test_kg_traverse.py`) and gated integration tests (`tests/integration/test_build_place_relations.py`) pass; `make test` and `make lint` are green.
5. `scripts/log_model_to_mlflow.py` writes `kg_enabled` as a logged param on a fresh run, visible in the shared MLflow tracking server.

---
*Reference: full implementation skeleton in `implementation_plan/james/w7_knowledge_graph.md` (treat as the W7 spec — already research- and design-complete).*
