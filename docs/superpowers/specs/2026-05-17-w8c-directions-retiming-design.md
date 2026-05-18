# W8c — Backend Directions Tool + Re-timing Node — Design

**Date:** 2026-05-17
**Branch:** `feature/agent-w8c-directions-retiming` (off `main` @ 87eb33b, post-W8b PR #87)
**Spec source:** `implementation_plan/james/w8_live_map_routing.md` §PR3
**Depends on:** W2 (`/chat` agent graph, `ItineraryState`, `Stop`), W3 (critique/revision
loop — `critique/checks.py`, `revision.py`), W8a (coords on `PlaceCard`), W8b (frontend
live map + `/chat` switch, both merged to `main`).

## Problem

The agent's `arrival_time`s are chained off a haversine ÷ 80 m/min estimate
(`app/agent/planning.py`) — walking-only, road-blind, transit-blind. `arrival_time`
is load-bearing: the `temporal_coherence` critique (`app/agent/critique/checks.py`)
gates itinerary validity. Displayed arrival times can therefore be untruthful, and
the validity gate runs on a fiction.

W8b put a real Google Directions route on the map (frontend, JS SDK). W8c makes the
backend's *final* `arrival_time`s truthful by reconciling them against Google
Directions **once per finalized plan** (Tiered design — no API in the revision loop),
then re-running the open-at-arrival check on real data.

## Resolved decisions (BIG CHANGE interactive review)

| # | Decision | Choice |
|---|----------|--------|
| Key | Backend Directions key | New dedicated `google_directions_api_key` in `Settings` (separate from frontend browser key and from `GOOGLE_PLACES_API_KEY`) |
| API | Which Google API | **Routes API v2** (`routes.googleapis.com/directions/v2:computeRoutes`) |
| Flip | What happens when real time flips temporal pass→fail | Append caveat to `final_reply` via existing `_final_with_caveats()` — **no** LLM, **no** loop re-entry |
| Wiring | Graph integration | New `retime` node between `critique` and `END`; `route_after_critique` returns `"retime"` (was `END`) when done |
| Mode | Which travel mode the backend re-times | **WALK** — matches the walking-centric constraint model and the haversine fallback units; frontend toggle stays client-side |

## Architecture & data flow

### Files

```
NEW   app/tools/directions.py                      — Routes API v2 client, async, typed, fallback
NEW   tests/unit/test_tools_directions.py
NEW   tests/integration/test_directions_live.py    — gated real API
EDIT  app/config.py                                — google_directions_api_key field
EDIT  app/agent/planning.py                        — chain_arrival_times() pure helper
EDIT  app/agent/graph.py                            — retime node + rewire route_after_critique
EDIT  .env.example                                 — GOOGLE_DIRECTIONS_API_KEY (commented, blank)
EDIT  tests/unit/test_agent_planning.py            — chain_arrival_times unit tests
EDIT  tests/unit/test_agent_graph.py               — smoke: node present, ≤1 call, passthrough
EDIT  tests/unit/test_chat_functional.py           — functional: real overwrite + pass→fail flip
```

### `app/tools/directions.py`

Mirrors `app/tools/retrieval.py` *structure* (typed Pydantic models, key via
`get_settings()`, module-level functions, `__all__`) — **not** its sync DB pattern.
This tool is natively async (`httpx.AsyncClient`), per spec.

- `class DirectionsLeg(BaseModel)`: `duration_s: int`, `distance_m: float`
- `class DirectionsResult(BaseModel)`: `legs: list[DirectionsLeg]`,
  `total_duration_s: int`, `mode: str`, `source: Literal["google", "haversine_fallback"]`
- `_MODE_TO_ROUTES_TRAVELMODE: dict[str, str]` = `{"walk":"WALK","transit":"TRANSIT","drive":"DRIVE"}`
- `DEFAULT_MODE = "walk"`
- `async def route_legs(stops: list[tuple[float, float]], mode: str = DEFAULT_MODE) -> DirectionsResult`
  - Unknown `mode` → `ValueError` **before any network call** (fast, testable).
  - Empty `google_directions_api_key` → `_haversine_fallback(stops)`, **no HTTP attempted**.
  - WALK path: one POST to `computeRoutes` — origin=`stops[0]`, destination=`stops[-1]`,
    intermediates=`stops[1:-1]`, `travelMode` mapped, header
    `X-Goog-FieldMask: routes.legs.duration,routes.legs.distanceMeters`.
  - `httpx.AsyncClient(timeout=3.0)`.
  - On timeout / non-200 / malformed / missing `routes` → `_haversine_fallback(stops)`.
    **Never raises to the caller.**
  - TRANSIT/DRIVE with intermediates → route **per-leg point-to-point**, stitch legs
    (mirrors the W8b RouteOverlay prod-bug fix). Not on the canonical `/chat` path
    (backend re-times WALK only); exists for the integration test and future use.
- `_haversine_fallback(stops)` → `DirectionsResult(source="haversine_fallback")` with
  per-leg minutes = `planning.haversine_m(a, b) / WALKING_SPEED_M_PER_MIN`.

### `app/agent/planning.py`

`def chain_arrival_times(stops: list[Stop], leg_durations_min: list[float]) -> list[Stop]`
— pure function. `stops[0].arrival_time` preserved (it's the user's start time); for
`i ≥ 1`: `arrival[i] = arrival[i-1] + planned_duration_min[i-1] + leg_durations_min[i-1]`.
Raises `ValueError` if `stops[0].arrival_time` is None (matches `next_arrival_time`'s
contract). All itinerary-time math stays in `planning.py`; `next_arrival_time` is left
unchanged (different call shape, used by `commit.py`).

### `retime` graph node — data flow

```
critique (state.done=True, has stops)
   │  route_after_critique → "retime"   (was END)
   ▼
retime node (async):
   1. Guard: not state.done OR <2 stops with both coords → return {}  (pure passthrough)
   2. coords = [(s.latitude, s.longitude) for s in state.stops if both set]
      len(coords) < 2 → return {}        (haversine arrival_time survives untouched)
   3. result = await route_legs(coords, mode="walk")   (≤1 API call, 3s cap, fallback)
      # route_legs is natively async (httpx) — call directly, NOT via to_thread
      # (contrast act() which wraps sync tools in asyncio.to_thread)
   4. leg_min = [leg.duration_s / 60 for leg in result.legs]
      retimed = planning.chain_arrival_times(state.stops, leg_min)
   5. Re-run ONLY temporal_coherence (the open-at-arrival check) on `retimed`
      — not full itinerary_violations(); geographic/walking/hallucination
      checks are coord/id-based and unaffected by re-timing. One DB round-trip,
      the same coalesced unnest query the loop already used. If its score now
      drops below CRITIQUE_THRESHOLDS["temporal_coherence"]:
        final_reply = _final_with_caveats(state.final_reply, ["temporal_coherence"])
        (existing prose helper from revision.py — no LLM, no loop re-entry)
   6. return {"stops": retimed, "final_reply": maybe_caveated}
   ▼
END → /chat handler → state_to_cards() → frontend
```

### Graph wiring

```python
g.add_node("retime", retime)
g.add_edge("retime", END)
# route_after_critique now returns:
#   "plan"    if not state.done   (revision loop continues, unchanged)
#   "retime"  if state.done       (was END; retime self-guards routability → {} passthrough)
g.add_conditional_edges("critique", route_after_critique, {"plan": "plan", "retime": "retime"})
```

`retime` self-guards (returns `{}`) on every non-routable path — clarifying-question
replies, single-stop plans, coordless plans — so behavior is unchanged there.

### Boundary preservation

`directions.py` lives in `app/tools/` (alongside `retrieval.py`) and is called only
from the graph node, never from `app.main` — preserving the `app/agent` vs `app.main`
separation `io.py` documents.

## Code quality decisions

1. **`chain_arrival_times` in `planning.py`** (not inlined in `graph.py`, not by
   generalizing `next_arrival_time`) — keeps `graph.py` orchestration-only, all
   itinerary-time math in one home, additive/pure/low-risk.
2. **Single `_MODE_TO_ROUTES_TRAVELMODE` map + `DEFAULT_MODE`** in `directions.py`;
   unknown mode raises `ValueError` before any network call. Backend never speaks JS
   enums.
3. **`google_directions_api_key: str = ""`** mirroring `openai_api_key`. Empty key is
   a first-class "use fallback" signal — unit tests get the fallback path for free
   without mocking httpx for the no-key case.
4. **`route_legs` is natively async**, awaited directly in the `retime` node — *not*
   via `asyncio.to_thread` (which `act()` uses for sync tools). A one-line comment in
   the node documents the asymmetry to prevent a future double-async mistake.

## Test plan (four layers per project convention)

### Unit — `tests/unit/test_tools_directions.py` (HTTP-mocked, no network)

| Test | Asserts |
|---|---|
| `test_models_construct` | `DirectionsLeg`/`DirectionsResult` build |
| `test_route_legs_walk_parses_legs` | Mocked 200 → legs parsed, `source=="google"`, `mode=="walk"` |
| `test_route_legs_transit_maps_travelmode` | Posted JSON carries `travelMode:"TRANSIT"` |
| `test_route_legs_drive_maps_travelmode` | `travelMode:"DRIVE"` |
| `test_unknown_mode_raises_before_network` | `ValueError`, post **never called** |
| `test_missing_key_uses_fallback` | Empty key → `source=="haversine_fallback"`, no HTTP |
| `test_timeout_falls_back` | `httpx.TimeoutException` → fallback, no raise |
| `test_non_200_falls_back` | 500 → fallback |
| `test_malformed_body_falls_back` | 200 missing `routes` → fallback |
| `test_fallback_durations_match_haversine` | Fallback leg minutes == `haversine_m ÷ 80` |
| `test_transit_with_intermediates_routes_per_leg` | mode=transit + 3 stops → N-1 posts, stitched |

Plus in `tests/unit/test_agent_planning.py`: `chain_arrival_times` — empty list,
single stop (no-op), N stops chain correctly, `stops[0].arrival_time` preserved,
None `stops[0].arrival_time` raises.

### Smoke — extend `tests/unit/test_agent_graph.py`

| Test | Asserts |
|---|---|
| `test_retime_node_present` | Graph has `retime`; `route_after_critique` → `"retime"` when done |
| `test_retime_at_most_one_directions_call` | Call-counted `route_legs` → ≤1 for a 3-stop plan |
| `test_retime_passthrough_when_not_routable` | Single-stop/coordless → `route_legs` never called, `arrival_time` unchanged |
| `test_retime_passthrough_on_clarifying_reply` | Finalize-without-stops → no-op |

### Functional — extend `tests/unit/test_chat_functional.py` (real graph, `_ScriptedLLM`, `route_legs` mocked)

| Test | Asserts |
|---|---|
| `test_chat_retimes_arrival_with_real_directions` | 2-stop plan, mocked longer leg → `places[1].arrival_time` == re-chained value (≠ haversine estimate) |
| `test_chat_retiming_flips_temporal_pass_to_fail` | Stop 2 closes 21:00; haversine arrival 20:50 (passes loop); mocked transit-slow arrival 21:10 → re-run `temporal_coherence` fails → `reply` ends with caveat naming the stop |
| `test_chat_directions_failure_keeps_haversine_reply` | `route_legs` internally fails → fallback → `/chat` 200, **no** spurious caveat, arrival == haversine |

`temporal_coherence`'s DB path is stubbed the same way `test_critique_checks.py` mocks
`get_conn` — no real Postgres.

### Integration — `tests/integration/test_directions_live.py` (gated)

`pytestmark = pytest.mark.skipif(os.getenv("APP_ENV","test") != "integration" or not
os.getenv("GOOGLE_DIRECTIONS_API_KEY"), reason="Set APP_ENV=integration and
GOOGLE_DIRECTIONS_API_KEY to run live Directions tests.")` — mirrors
`tests/integration/test_db.py:19-21`. One test: real `route_legs` for two known SF
coords in WALK → `source=="google"`, ≥1 leg, `0 < total_duration_s < 7200`.

## Performance

- **≤1 Directions POST per finalized `/chat`** (WALK = single multi-leg request);
  3s `httpx` timeout is the hard latency ceiling; degrades to haversine in ≤3s.
  Smoke test enforces the ≤1 invariant.
- **Field mask** (`routes.legs.duration,routes.legs.distanceMeters`) → minimal
  response, no polyline/steps. Frontend draws its own route; backend only needs
  durations. Main perf lever, free.
- **One extra DB query** for the re-run `temporal_coherence` — already coalesced
  (single `unnest` round-trip), only on the finalized routable success path.
  No caching: the re-run exists *because* `arrival_time` changed; a cached result
  is stale by construction.
- No N+1s, no unbounded loops, no memory growth. Added budget = ≤3s capped Directions
  + one fast indexed DB query, only on finalized routable plans, graceful degradation
  everywhere.

## Out of scope (per W8 spec §Deferred)

Itinerary streaming; road-snapped pathing beyond Directions; "Open in Google Maps"
deep links; real pins on `/predict`; user-constraint travel-mode parsing (backend
re-times WALK only — mode parsing would be a UserConstraints change outside W8c).

## Process

`superpowers:test-driven-development` then `superpowers:verification-before-completion`
(run `make test` before any completion claim). Per-PR atomic single-line commits
(pre-commit runs ruff — do not run it manually). User merges the PR (no `gh pr merge`).
On merge: update `implementation_plan/james/README.md` status row + the
`w8_live_map_routing.md` §Status footer (w8c row).
