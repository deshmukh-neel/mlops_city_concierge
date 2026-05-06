# Agentic City Concierge — Implementation Plan (james)

## Why this exists

The current system is a single-pass RAG chain (`app/chain.py:13-47`) that does one vector-similarity retrieval (`app/retriever.py:38-52`) and stuffs the results into an LLM prompt. It throws away most of the rich metadata in `places_raw` (`scripts/db/init.sql:26-48`) — `price_level`, `rating`, `regular_opening_hours`, `business_status`, `latitude/longitude`, `types[]`, `editorial_summary` — none of which participate in retrieval. Every call is one-shot. There is no filtering, no re-querying, no multi-step reasoning, no actions.

This makes the product structurally incapable of the things a productionized concierge needs to do well:
- Constraint-heavy queries ("under $$$, open Sunday night, walking distance from North Beach")
- Multi-stop itineraries with geographic and temporal coherence
- Graceful recovery from empty / low-quality results
- Any kind of action (booking, sharing, calendar holds)
- A data flywheel that improves the system from real usage

It is also indistinguishable from "Opus 4.7 + web search" on the tasks Opus is good at, while losing on the tasks Opus structurally can't do — the wrong side of that comparison.

## What this plan delivers

Six workstreams that convert the system into a tool-calling agent grounded in your structured DB, with the MLOps loop that lets it improve over time. Each workstream is its own file, its own branch, its own PR.

| # | Workstream | File | Branch | Depends on | Status |
|---|---|---|---|---|---|
| W0  | Infra hardening (cold starts, tracing, secrets, MLflow auth, cost telemetry) | [w0_infra.md](w0_infra.md) | `feature/agent-w0-infra` | — | ✅ Merged ([#60](https://github.com/deshmukh-neel/mlops_city_concierge/pull/60)) — MLflow auth proxy (§3) deferred |
| W0a | Cleaner embeddings on a parallel `place_embeddings_v2` table | [w0a_embeddings_v2.md](w0a_embeddings_v2.md) | `feature/agent-w0a-embeddings-v2` | — | ✅ Merged ([#58](https://github.com/deshmukh-neel/mlops_city_concierge/pull/58)) — promotion gated on W6 evals |
| W1  | Unified place view + filterable retrieval tools | [w1_retrieval_tools.md](w1_retrieval_tools.md) | `feature/agent-w1-retrieval-tools` | W0a | ✅ Merged ([#65](https://github.com/deshmukh-neel/mlops_city_concierge/pull/65)) — price_level enum hotfix in [#66](https://github.com/deshmukh-neel/mlops_city_concierge/pull/66) |
| W2  | Agent loop + ItineraryState + `/chat` endpoint | [w2_agent_graph.md](w2_agent_graph.md) | `feature/agent-w2-agent-graph` | W1 (and ideally W0 for tracing) | 🚧 Not started |
| W3  | Self-correction (within W2's graph) | [w3_self_correction.md](w3_self_correction.md) | `feature/agent-w3-self-correction` | W2 | 🚧 Not started |
| W4  | Booking handoff stub (`propose_booking`) | [w4_booking_stub.md](w4_booking_stub.md) | `feature/agent-w4-booking-stub` | W2 | 🚧 Not started |
| W5  | Coverage-gap ingestion agent | [w5_coverage_agent.md](w5_coverage_agent.md) | `feature/agent-w5-coverage-agent` | — (independent) | 🚧 Not started |
| W6  | Eval-loop agent (RAGAS retrieval + custom itinerary checks) | [w6_eval_agent.md](w6_eval_agent.md) | `feature/agent-w6-eval-agent` | W2 | 🚧 Not started |
| W7  | Knowledge graph (`place_relations`) + `kg_traverse` tool | [w7_knowledge_graph.md](w7_knowledge_graph.md) | `feature/agent-w7-knowledge-graph` | W0a, W1 | 🚧 Not started |

Suggested merge order: **W0a → W0 → W1 → W2 → W3 → W4** (cleaner embeddings first so all downstream retrieval rides on better vectors; user-facing demo path with infra in front), then **W5, W6, W7** in any order (MLOps story + KG layer). W0 and W5 are independent of the agent code and can land any time. If W0 is delayed, W2's tracing wiring degrades gracefully to a no-op (Langfuse env vars unset → empty callbacks list). W7 is technically optional but cheap to land once W0a + W1 are in.

## Architecture

```
POST /chat ──► LangGraph agent ──► tools:
                   │                ├─ semantic_search(query, filters)
                   │                ├─ nearby(place_id, radius_m, filters)
                   │                ├─ get_details(place_id)
                   │                ├─ kg_traverse(place_id, relation_type)  [stub in W2; real in W7]
                   │                ├─ propose_booking(place_id, ...)
                   │                └─ web_search(query)                       [optional, off by default]
                   │
                   └─► ItineraryState (Pydantic) ──► {reply, places, ragLabel}

Retrieval reads from:                                 KG reads from:
  place_documents     (joins place_embeddings   v1)     place_relations
  place_documents_v2  (joins place_embeddings_v2)         (NEAR, SAME_NEIGHBORHOOD,
  selected by EMBEDDING_TABLE env var (W0a).               CONTAINED_IN, NEAR_LANDMARK,
                                                           SIMILAR_VECTOR — built by
                                                           scripts/build_place_relations.py)
```

The agent driver is the existing MLflow-registry-selected model (Opus 4.7 / GPT-4o / Gemini 2.5) — same swap mechanism as today, no new infra. Tools are deterministic Python functions wrapping SQL or external APIs. State is a Pydantic model so the LLM revises structured fields rather than regenerating prose.

## Cross-cutting decisions

- **Editorial data is unknown shape.** A teammate is scraping Eater + Infatuation and pushing to Cloud SQL with embeddings; the schema isn't finalized. W1 introduces a `place_documents` SQL view so retrieval tools never see the underlying tables directly. When the editorial table lands, the view definition is updated; the agent code does not change.
- **v1/v2 embeddings live side-by-side.** W0a creates a parallel `place_embeddings_v2` table with cleaned chunks (no URLs, no embedded numbers, structured neighborhood + landmarks added). The app picks which one to read via `EMBEDDING_TABLE`. v1 stays untouched until W6 evals show v2 is non-regressing on retrieval quality; flipping the env var is the promotion mechanism. v1 may be dropped in a small follow-up after evals.
- **Knowledge graph is real, not deferred.** W7 builds `place_relations` edges from data Google already gave us in `source_json` (`addressDescriptor.landmarks[]`, `containingPlaces[]`) plus computed edges (`NEAR` haversine, `SAME_NEIGHBORHOOD`, `SIMILAR_VECTOR` over v2 vectors). Storage is plain SQL edge tables — Apache AGE is not available on Cloud SQL. LLM-extracted edges (`OPERATED_BY`, `MENTIONED_WITH`) are deferred to a follow-up PR; this plan only covers free / computed edges.
- **Recommendations are constraint-aware, multi-stop, geographically anchored.** The agent asks the user how many stops they want (default 3), assigns hard-coded duration defaults per `primary_type`, computes per-stop arrival times with walking math, and applies shared constraints (price, rating, rating-count floor) across every stop. The walking budget caps total distance so "plan a date" doesn't return three places spread across SF. Quality floor `min_user_rating_count = 50` prevents single-rater outliers (the Pasadena Velasco failure mode found on 2026-05-04) from surfacing.
- **Eval has three surfaces.** W6 splits evals into RAGAS retrieval metrics, a deterministic itinerary checker (geographic + temporal + constraint correctness, plus a hallucination check), and a cheap-LLM-judge taste rubric. RAGAS measures retrieval quality; the checker measures planning correctness; the judge measures things you can't compute. All metrics land in MLflow as run metrics, gating alias promotion. Test set is hybrid: ~20 hand-written queries that mirror real user intents + ~50 RAGAS-generated for breadth.
- **MLflow params expand to track every config knob.** `embedding_table`, `retrieval_mode`, `agent_strategy`, `kg_enabled`, `default_num_stops`, `walking_budget_m`, `min_user_rating_count` all log per run. Without these, A/B comparisons aren't apples-to-apples.
- **Booking automation is out of scope.** W4 ships an *assisted-handoff* stub — a deep-link to Resy/Tock/Google Maps with the booking pre-filled in the URL where supported. No credentials, no Playwright, no ToS risk. A future PR (not in this plan) adds Playwright behind `BOOKING_AUTOMATION_ENABLED` for your test account only.
- **Routing / Calendar / Weather use columns, not chunks.** Embedding text deliberately drops lat/lng and Maps URLs. Routing uses `places_raw.latitude/longitude` to construct Google Maps directions URLs at request time. Calendar event payloads use `name + address + lat/lng + opening hours` columns. Weather calls hit lat/lng → external weather API. None of these flows read from the embedding text — the embedding's only job is to surface the right `place_id`.
- **Frontend already expects the contract we're building.** `frontend/src/api/chat.js:28` calls `POST /chat` with `{message, history}` and expects `{reply, places, ragLabel}`. The current `/predict` (`app/main.py:183-196`) does not match. W2 makes `/chat` the primary endpoint and keeps `/predict` as a thin compatibility shim.
- **Agent driver = MLflow-registry model.** No parallel model selection path. `load_registered_rag_chain` (`app/main.py:75-110`) and `parse_active_model_config` are reused as-is to pick the agent driver. The eval-loop's *judge* uses a cheap small model (`gpt-4o-mini` / `gemini-2.5-flash`) selected by `EVAL_JUDGE_MODEL` — separate from the candidate driver to avoid self-judging bias.
- **New top-level dependencies:** `langgraph` (orchestration; W2), `pydantic-ai` (type-checked tool definitions; W2/W4), `langfuse` (per-request tracing; W0), `ragas` (retrieval evals; W6). LangChain 0.2 is already pinned (`pyproject.toml:25`). DSPy and a managed eval UI (Langfuse / Braintrust) are flagged as future directions in W6 but not added in this plan. We deliberately use LangGraph for the agent *loop* and Pydantic AI for *tool definitions* — each library where it's strongest, with a one-function adapter between them in `app/agent/tools.py`.

## Future-watch — concerns documented separately

Design concerns flagged during planning that aren't load-bearing today live in [FUTURE_WATCH.md](FUTURE_WATCH.md). That file covers things like the LangGraph ↔ Pydantic AI adapter possibly being obsolete by implementation time, the v1/v2 embedding script duplication needing cleanup once v2 wins, and the deferred supervisor / constraint-extractor patterns. Check it before starting any workstream — saves rediscovering known issues mid-implementation.

## Conventions for these plan files

- Each workstream's file contains: **Goal**, **Branch**, **Files to create/modify with full skeletons**, **Tests**, **Manual verification**, **Risks / open questions**.
- Skeletons include type signatures, Pydantic models, SQL fragments, and prompt outlines. They are intentionally close to copy-pasteable so an implementer (you, a teammate, or a future agent) can move fast without re-deriving design intent.
- File paths are absolute-ish from repo root (e.g. `app/agent/graph.py`).
- Line-number references point to the codebase as of the planning commit so they may drift slightly after merges; if a reference is wrong, the surrounding context still applies.

## Verification at the end of W0a–W4

```bash
make dev                 # start db + app
make migrate             # apply v2 + place_documents migrations (W0a, W1)
make embed-v2            # populate place_embeddings_v2 (W0a)
EMBEDDING_TABLE=place_embeddings_v2 make dev   # serve from cleaned embeddings
python scripts/seed.py   # seed sample data
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "plan a date night in north beach: 3 stops — dinner around 7, drinks then dessert, all under $$$", "history": []}'
```

Expected: `places` array with 3 stops in/near North Beach, all `OPERATIONAL`, all `price_level <= 3`, all `user_rating_count >= 50`, each stop within walking distance of the previous (cumulative ≤ ~2400m), arrival times spaced by previous-stop duration + walking time, first stop open at 7pm, later stops open at *their* arrival times. The `reply` string narrates the plan, surfaces planned durations ("Dinner ~90 min, drinks ~60 min, dessert ~30 min"), and ends with a booking deep-link from W4. Agent should make 4+ tool calls in the LangGraph trace.

Failure-mode test:
```bash
curl -X POST http://localhost:8000/chat \
  -d '{"message": "find a 3-star vegan ethiopian place in pacific heights open at 4am", "history": []}'
```

Expected (with W3 merged): agent recognizes empty / low-quality results, drops constraints progressively, returns the closest matches with a transparent explanation, or asks a clarifying question. Does **not** return a hallucinated answer or an empty `places` array silently.

After W5: `python scripts/coverage_agent.py --dry-run` against a seeded `place_query_hits` table proposes new seed queries that fill measurable gaps.

After W6: `python scripts/log_model_to_mlflow.py --eval` on a candidate config runs RAGAS retrieval evals + the deterministic itinerary checker + a cheap LLM-judge taste rubric, logs all three metric sets to MLflow, and refuses to alias-promote on a regression of any critical metric.

After W7: `python scripts/build_place_relations.py` populates the KG; `kg_traverse(place_id, relation_type)` is callable from the agent and from eval. Edge-count summary should show all five relation types non-empty for the SF dataset.
