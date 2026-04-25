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

| # | Workstream | File | Branch | Depends on |
|---|---|---|---|---|
| W0 | Infra hardening (cold starts, tracing, secrets, MLflow auth, cost telemetry) | [w0_infra.md](w0_infra.md) | `feature/agent-w0-infra` | — |
| W1 | Unified place view + filterable retrieval tools | [w1_retrieval_tools.md](w1_retrieval_tools.md) | `feature/agent-w1-retrieval-tools` | — |
| W2 | Agent loop + ItineraryState + `/chat` endpoint | [w2_agent_graph.md](w2_agent_graph.md) | `feature/agent-w2-agent-graph` | W1 (and ideally W0 for tracing) |
| W3 | Self-correction (within W2's graph) | [w3_self_correction.md](w3_self_correction.md) | `feature/agent-w3-self-correction` | W2 |
| W4 | Booking handoff stub (`propose_booking`) | [w4_booking_stub.md](w4_booking_stub.md) | `feature/agent-w4-booking-stub` | W2 |
| W5 | Coverage-gap ingestion agent | [w5_coverage_agent.md](w5_coverage_agent.md) | `feature/agent-w5-coverage-agent` | — (independent) |
| W6 | Eval-loop agent | [w6_eval_agent.md](w6_eval_agent.md) | `feature/agent-w6-eval-agent` | W2 |

Suggested merge order: **W0 → W1 → W2 → W3 → W4** (user-facing demo path with infra in front), then **W5 and W6** in parallel (MLOps story). W0 and W5 are independent of the agent code and can land any time. If W0 is delayed, W2's tracing wiring degrades gracefully to a no-op (Langfuse env vars unset → empty callbacks list).

## Architecture

```
POST /chat ──► LangGraph agent ──► tools:
                   │                ├─ semantic_search(query, filters)
                   │                ├─ nearby(place_id, radius_m, filters)
                   │                ├─ get_details(place_id)
                   │                ├─ kg_traverse(...)              [stub; lights up when KG lands]
                   │                ├─ propose_booking(place_id, ...)
                   │                └─ web_search(query)              [optional, off by default]
                   │
                   └─► ItineraryState (Pydantic) ──► {reply, places, ragLabel}
```

The agent driver is the existing MLflow-registry-selected model (Opus 4.7 / GPT-4o / Gemini 2.5) — same swap mechanism as today, no new infra. Tools are deterministic Python functions wrapping SQL or external APIs. State is a Pydantic model so the LLM revises structured fields rather than regenerating prose.

## Cross-cutting decisions

- **Editorial data is unknown shape.** A teammate is scraping Eater + Infatuation and pushing to Cloud SQL with embeddings; the schema isn't finalized. W1 introduces a `place_documents` SQL view so retrieval tools never see the underlying tables directly. When the editorial table lands, the view definition is updated; the agent code does not change.
- **Knowledge graph is downstream of editorial.** A `kg_traverse` tool stub is reserved in W2 and returns "not yet available" until the KG lands. No KG construction is in this plan.
- **Booking automation is out of scope.** W4 ships an *assisted-handoff* stub — a deep-link to Resy/Tock/Google Maps with the booking pre-filled in the URL where supported. No credentials, no Playwright, no ToS risk. A future PR (not in this plan) adds Playwright behind `BOOKING_AUTOMATION_ENABLED` for your test account only.
- **Frontend already expects the contract we're building.** `frontend/src/api/chat.js:28` calls `POST /chat` with `{message, history}` and expects `{reply, places, ragLabel}`. The current `/predict` (`app/main.py:183-196`) does not match. W2 makes `/chat` the primary endpoint and keeps `/predict` as a thin compatibility shim.
- **Agent driver = MLflow-registry model.** No parallel model selection path. `load_registered_rag_chain` (`app/main.py:75-110`) and `parse_active_model_config` are reused as-is to pick the agent driver.
- **New top-level dependencies:** `langgraph` (orchestration; W2), `pydantic-ai` (type-checked tool definitions; W2/W4), `langfuse` (per-request tracing; W0). LangChain 0.2 is already pinned (`pyproject.toml:25`). DSPy and a managed eval UI (Langfuse / Braintrust) are flagged as future directions in W6 but not added in this plan. We deliberately use LangGraph for the agent *loop* and Pydantic AI for *tool definitions* — each library where it's strongest, with a one-function adapter between them in `app/agent/tools.py`.

## Conventions for these plan files

- Each workstream's file contains: **Goal**, **Branch**, **Files to create/modify with full skeletons**, **Tests**, **Manual verification**, **Risks / open questions**.
- Skeletons include type signatures, Pydantic models, SQL fragments, and prompt outlines. They are intentionally close to copy-pasteable so an implementer (you, a teammate, or a future agent) can move fast without re-deriving design intent.
- File paths are absolute-ish from repo root (e.g. `app/agent/graph.py`).
- Line-number references point to the codebase as of the planning commit so they may drift slightly after merges; if a reference is wrong, the surrounding context still applies.

## Verification at the end of W1–W4

```bash
make dev                 # start db + app
make migrate             # apply place_documents view migration (W1)
python scripts/seed.py   # seed sample data
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "plan a date night in north beach: dinner around 7, drinks within walking distance, both under $$$", "history": []}'
```

Expected: `places` array with 2 stops in/near North Beach, both `OPERATIONAL`, both `price_level <= 3`, second stop within ~800m of first, first stop open at 7pm, second open after dinner. The `reply` string narrates the plan and ends with a booking deep-link from W4. Agent should make 3+ tool calls in the LangGraph trace.

Failure-mode test:
```bash
curl -X POST http://localhost:8000/chat \
  -d '{"message": "find a 3-star vegan ethiopian place in pacific heights open at 4am", "history": []}'
```

Expected (with W3 merged): agent recognizes empty / low-quality results, drops constraints progressively, returns the closest matches with a transparent explanation, or asks a clarifying question. Does **not** return a hallucinated answer or an empty `places` array silently.

After W5: `python scripts/coverage_agent.py --dry-run` against a seeded `place_query_hits` table proposes new seed queries that fill measurable gaps.

After W6: `python scripts/log_model_to_mlflow.py --eval` on a candidate config logs metrics + comparison report to MLflow and refuses to alias-promote on a regression.
