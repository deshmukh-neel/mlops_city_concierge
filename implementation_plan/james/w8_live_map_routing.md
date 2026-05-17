# W8 — Live Google Map + multimodal itinerary routing

**Branch (per PR):** `feature/agent-w8a-coord-propagation`, `feature/agent-w8b-live-map`, `feature/agent-w8c-directions-retiming`
**Depends on:** W2 (`/chat` agent graph, `ItineraryState`, `Stop`, `state_to_cards`), W3 (critique/revision loop — `critique/checks.py`, `revision.py`)
**Unblocks:** real spatial UX; future itinerary streaming (deferred, see §Deferred)

## Why this exists

The frontend renders a **fake CSS-grid map** (`frontend/src/components/MapView.jsx`): hardcoded
`STREET_LINES` / `BLOCKS` / `NEIGHBORHOODS` and pins positioned by a synthetic
`spreadMapPos()` in `frontend/src/api/chat.js`. No real geography, no route.

Worse, the frontend calls **`POST /predict`** (legacy single-pass RAG, unordered
`sources`, no coordinates) instead of **`POST /chat`** (the W2/W3 agent that produces
an *ordered, time-chained* `ItineraryState.stops` — i.e. an actual itinerary). A
route is only meaningful for an ordered itinerary, so the frontend is wired to the
one endpoint that cannot be routed.

The backend also has no concept of real travel time: `app/agent/planning.py`
`walking_time_min()` is `haversine ÷ 80 m/min` — walking-only, road-blind,
transit-blind. `arrival_time` (chained off that estimate) is load-bearing: the
`place_open_at_arrival` critique (`app/agent/critique/checks.py`) gates itinerary
validity and drives `shift_arrival_time` revisions (`app/agent/revision.py`).

## What this delivers

A real Google Map with pins at true coordinates and a Google **Directions**-rendered
route with a Walk / Transit / Drive toggle that shows real per-leg and total travel
times. The agent's *final* `arrival_time`s are reconciled against Google Directions
once the plan is committed (Tiered design), so displayed times are truthful without
making the revision loop API-bound.

## Architecture decisions (resolved via BIG CHANGE review)

- **Endpoint:** frontend switches `/predict → /chat`. The agent is the product;
  `state_to_cards()` already returns the `{reply, places, ragLabel}` contract.
  Delete the `/predict` adapter (`toPlace`/`spreadMapPos`/`slugify`).
- **Tiered timing:** the revision loop keeps the cheap haversine estimate as an
  *internal sanity gate* (no API in the loop). Exactly **one** Google Directions
  call per *finalized* plan re-computes real `arrival_time`s; the open-at-arrival
  critique re-runs **once** on the real times.
- **Re-timing placement:** a dedicated graph node **after** the revision loop
  settles, before `final_reply` — not inside the loop, not in the HTTP handler
  (preserves the `app/agent` vs `app.main` boundary documented in `io.py`).
- **Frontend state:** `places` modeled as a keyed map (`place_id → place`) owned by
  `App.jsx`, plus a `planFinalized` flag (already required to gate the route).
  v1 is non-streaming; this is the seam that makes streaming additive later.
- **Maps lib:** `@vis.gl/react-google-maps` (Google-endorsed, React 18 OK).
  `AdvancedMarker` requires a Map ID. `APIProvider` owns the loader — **no**
  `<script>` in `index.html` (double-load bug).
- **Displayed route = Google Directions** (`DirectionsService`/`DirectionsRenderer`),
  not a hand-drawn polyline — this is *why* Google Maps over MapLibre (real
  transit/drive engine).

## Scope

### PR1 — `w8a` backend coordinate propagation (small, safe, no behavior change)
- `app/agent/state.py`: add `latitude`/`longitude` to `PlaceCard` (`Stop` has them).
- `app/agent/io.py`: `state_to_cards()` passes coords through; order preserved.
- Tests: extend `tests/unit/test_io.py` — coords survive, stop order == output
  order, null-coord stop yields null lat/lng (not dropped, not crashed).

### PR2 — `w8b` frontend live map + `/chat` switch
- Add Vitest + React Testing Library to `frontend/` (no FE test setup exists today).
- `frontend/src/api/chat.js`: delete `/predict` adapter; thin `/chat` adapter;
  normalize coords + preserve order. Adapter unit tests.
- `App.jsx`: keyed `places` map + `planFinalized` flag; clear/merge semantics;
  reducer unit-tested. Implement real `handlePlaceClick` (pan/zoom + highlight
  via shared map instance) — replaces the no-op TODO.
- `MapView.jsx`: full rewrite — `APIProvider`/`Map`/`AdvancedMarker` + `FitBounds`
  (`useMap`). Extract shared `<PlaceTooltip>` (reused by map marker + list).
  Lazy-load (`React.lazy`); `APIProvider` mounts only when a Maps key exists AND
  ≥1 place. No key → static fallback panel, SDK never fetched. Stable `place_id`
  marker keys + memoized marker list.
- New `RouteOverlay.jsx`: `DirectionsService`/`DirectionsRenderer`, Walk/Transit/
  Drive toggle, total + per-leg time readout. Renders only when `planFinalized` &&
  ≥2 placed stops. Directions result cached per `(itinerary-hash, mode)`
  (≤3 FE calls/plan, instant re-toggle). Waypoints via `useMemo` on
  `(ordered place_ids, mode)`.
- Smoke tests (Google lib mocked): N markers, polyline/route for N≥2,
  keyless-degrade panel.
- Env/compose: `VITE_GOOGLE_MAPS_API_KEY` + `VITE_GOOGLE_MAPS_MAP_ID` in
  `.env.example`, `frontend/.env.development`, `docker-compose.yml` frontend svc.
- Docs: Google Cloud setup (enable **Maps JavaScript API** + **Directions API**;
  create Map ID; browser key restricted to HTTP referrers incl.
  `http://localhost:5173/*` + Cloud Run domain — **separate** key from the
  backend Places key). README/CLAUDE.md note: UI now drives the agent (`/chat`).

### PR3 — `w8c` backend Directions tool + re-timing node
- New `app/tools/directions.py` — mirrors `app/tools/retrieval.py` structure
  (typed result models, key via `get_settings()`, async `httpx.AsyncClient`,
  ~3s timeout). Mode-aware (walk/transit/drive). Reuses `planning.py`
  `haversine_m` as the offline fallback on timeout/error.
- New post-revision graph node in `app/agent/graph.py`: after the loop settles,
  ≤1 Directions call rewrites `arrival_time`s for the chosen mode, then the
  open-at-arrival critique re-runs once on real data. Async, tight timeout,
  haversine fallback — never blocks/hangs `/chat`.
- Tests (four layers per project convention):
  - unit `tests/unit/test_tools_directions.py` — HTTP mocked, mode matrix +
    failure→haversine fallback.
  - smoke — extend `tests/unit/test_agent_graph.py` (node present, ≤1 call).
  - functional — `tests/unit/test_chat_functional.py`: committed plan → real
    `arrival_time`s overwrite estimate; **and** a fixture where slower real
    transit time pushes arrival past a venue's close → critique re-check flips
    pass→fail and the reply reflects it.
  - integration — gated test hitting real Directions (skipped unless key +
    `APP_ENV=integration`, mirroring existing integration gating).

## Out of scope / Deferred

- **Itinerary streaming** (pins populate incrementally as the agent proposes/
  revises mid-loop). Requires an SSE/WebSocket endpoint + `graph.astream` +
  per-iteration emission — a net-new transport layer, its own workstream. The
  keyed-state seam (PR2) makes this *additive*, not a rewrite. Explicitly v-next.
- Road-snapped backend pathing beyond what Directions returns.
- "Open in Google Maps" deep links (user chose match-current-behavior).
- Real pins on `/predict` (would need retriever SQL + `RecommendationSource`
  changes; the UI is moving off `/predict` anyway).

## Process

Planning via the user's CLAUDE.md BIG CHANGE interactive review (completed:
Architecture / Code Quality / Tests / Performance — all converged to the
recommended option). Execution per PR uses `superpowers:test-driven-development`
and `superpowers:verification-before-completion` (run `make test` / vitest before
any completion claim). Per-PR atomic, single-line commits (user preference). User
merges PRs (do not `gh pr merge`).

**Status:** Planned 2026-05-17. Phased: w8a → w8b → w8c.

- **w8a** (coord propagation) — ✅ built on `feature/agent-w8a-coord-propagation`,
  57 backend tests green. PR pending (user opens/merges).
- **w8b** (live map + `/chat` switch) — ✅ built on `feature/agent-w8b-live-map`
  (branched off w8a), 17 frontend tests green, production build + code-split
  verified. **Visual verification pending a Google Maps browser key**
  (`docs/google_maps_setup.md`). Pre-existing Bars/Dinner filter pills are
  no-ops against real Google `primary_type`s — deferred (filter taxonomy rework
  is out of scope for "add the map"). PR pending.
- **w8c** (Directions tool + re-timing node) — not started.
