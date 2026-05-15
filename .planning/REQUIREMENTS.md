# Requirements

## v1 Requirements (W7 Knowledge Graph milestone)

### Schema

- [ ] **KG-01**: A `place_relations` edge table exists in Cloud SQL with columns `(src_place_id, dst_place_id, relation_type, weight, metadata, source, built_at)`, primary key `(src, dst, type)`, indexes on `(src, type)`, `(dst, type)`, and `(type)`. `src_place_id` has FK to `places_raw(place_id) ON DELETE CASCADE`. `dst_place_id` has no FK. *(W7 §Files / migration 002)*
- [ ] **KG-02**: A `neighborhood_of(jsonb) → text` immutable plpgsql helper exists, mirroring `compose_embedding_text_v2`'s `_neighborhood_from_address_components` logic. *(W7 §Files / helper)*

### Builder

- [ ] **BLD-01**: `scripts/build_place_relations.py` is idempotent (re-running produces no PK growth) and supports `--only RELATION_TYPE[,...]` to rebuild a subset. *(W7 §Files / builder)*
- [ ] **BLD-02**: Builder seeds `NEAR` edges via haversine ≤ 800m between two `OPERATIONAL` places with non-null lat/lng, weight = distance in meters, source = `haversine`. *(W7 §sub-builders)*
- [ ] **BLD-03**: Builder seeds `SAME_NEIGHBORHOOD` edges between distinct places sharing a non-empty `neighborhood_of(source_json)`, source = `address_components`. *(W7 §sub-builders)*
- [ ] **BLD-04**: Builder seeds `CONTAINED_IN` edges from `source_json.containingPlaces[].id`, source = `source_json`. *(W7 §sub-builders)*
- [ ] **BLD-05**: Builder seeds `NEAR_LANDMARK` edges from `source_json.addressDescriptor.landmarks[]` with weight = `travelDistanceMeters` and metadata containing `displayName` + `types`, source = `source_json`. *(W7 §sub-builders)*
- [ ] **BLD-06**: Builder seeds `SIMILAR_VECTOR` edges as top-K (K=10) cosine ≥ 0.65 over `place_embeddings_v2`, weight = cosine, source = `vector_topk`. *(W7 §SIMILAR_VECTOR)*
- [ ] **BLD-07**: `make build-relations` invokes the builder. *(W7 §Makefile)*

### Tool

- [ ] **TOOL-01**: `app/tools/graph.py` exposes `kg_traverse(place_id, relation_type, k) → list[RelatedPlace]`, joining `place_relations` to `place_documents` so destinations missing from `places_raw` are dropped silently. *(W7 §app/tools)*
- [ ] **TOOL-02**: `RelatedPlace` extends `PlaceHit` with `relation_type: str`, `weight: Optional[float]`, `relation_metadata: dict`. *(W7 §app/tools)*
- [ ] **TOOL-03**: Result ordering is ascending by weight for `NEAR`, descending by weight for `SIMILAR_VECTOR`, unordered for the rest. *(W7 §SQL ORDER BY)*
- [ ] **TOOL-04**: Unknown `relation_type` raises `ValueError` (validated against `VALID_RELATIONS` set). *(W7 §app/tools)*
- [ ] **TOOL-05**: `app/agent/tools.py` `kg_traverse` stub is replaced by a thin wrapper delegating to `app.tools.graph.kg_traverse`. *(W7 §agent/tools.py)*
- [ ] **TOOL-06**: Agent `SYSTEM_PROMPT` (W2) gains a paragraph explaining when to pick each `relation_type`. *(W7 §SYSTEM_PROMPT addition)*

### MLOps

- [ ] **MLOPS-01**: `scripts/log_model_to_mlflow.py` logs `kg_enabled: bool` (default `True`) on each agent run so eval can A/B with KG on/off. *(W7 §MLflow)*

### Tests

- [ ] **TEST-01**: `tests/unit/test_kg_traverse.py` covers: ValueError on unknown relation_type; NEAR ordering ascending by weight; SIMILAR_VECTOR ordering descending by weight; `RelatedPlace` includes `relation_type` + `weight`; missing-`dst` rows are dropped via the JOIN. *(W7 §Tests)*
- [ ] **TEST-02**: `tests/integration/test_build_place_relations.py` (gated by `APP_ENV=integration`) on a 10-place fixture verifies: full builder runs; `NEAR` is symmetric within tolerance; `CONTAINED_IN` populates from `source_json`; `SIMILAR_VECTOR` weight ∈ [0,1] above threshold; second run is idempotent (no row growth). *(W7 §Tests)*

## v2 Requirements (deferred)

- v2 builders for LLM-extracted edges (`OPERATED_BY`, `MENTIONED_WITH`, `SAME_CHEF`) writing with `source = 'editorial_llm'` and a confidence in `metadata` — gated on editorial scrape landing.
- Editorial-source edges from teammate's Eater/Infatuation table — schema is forward-compatible; blocked on table availability.
- Per-source cap (`LIMIT 50` per `src_place_id`) on `NEAR` edges if a dense cluster (e.g., Union Square) generates pathological counts during first build.

## Out of Scope

- Apache AGE / openCypher in Postgres — not supported on Cloud SQL.
- FK on `dst_place_id` — intentional: landmark/containing-place targets may not exist in `places_raw`.
- Promotion of v1 → v2 embeddings — that's a W6 evals decision, not a W7 concern.
- Booking automation — separate future PR (W4 future-watch).

## Traceability

| REQ-ID | Phase |
|--------|-------|
| KG-01, KG-02 | Phase 1 |
| BLD-01 — BLD-07 | Phase 1 |
| TOOL-01 — TOOL-06 | Phase 1 |
| MLOPS-01 | Phase 1 |
| TEST-01, TEST-02 | Phase 1 |
