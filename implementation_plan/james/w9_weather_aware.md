# W9 — Weather-aware itinerary recommendations

**Branch:** `feature/agent-w9-weather-aware` (not started)
**Depends on:** **W8c** (needs real per-stop `arrival_time`s from the Directions
re-timing node — weather only matters *at the time the user is there*), W3
(critique/revision loop)
**Status:** Planned 2026-05-17. Deferred — not started. Recorded so the idea is
durable and the W8c dependency is explicit. Does **not** touch W8 PRs.

## Why this exists

A concierge that routes you to an outdoor courtyard bar during a rainstorm is
giving structurally bad advice that plain RAG can't catch. This is exactly the
real-world reasoning an agent (vs. "Opus + web search") justifies.

## Why it fits cleanly (and why it's still its own workstream)

Architecturally this is the **same shape as W8c**: a post-commit external signal
that can trigger the existing W3 critique → revision loop. A `weather_appropriate`
check behaves like the existing `place_open_at_arrival` check — fail an
outdoor-heavy itinerary when rain is forecast at those arrival times, and the
existing revision machinery re-plans toward indoor options. **No new
architecture.**

It is *not* folded into W8c because W8c is already a large backend change
touching the critique loop; stacking weather on top repeats a scope-creep
pattern. Separate workstream, separate review.

## The actual hard part (flagged up front)

The weather API call is trivial. The real risk is **"is this place
rain-sensitive?"** — Google `primary_type` does not encode "courtyard /
rooftop / outdoor seating." Options to resolve (decide during W9 planning):
- a `types[]` / editorial-summary heuristic,
- an LLM-extracted `outdoor` tag (offline, like W7's deferred LLM edges),
- accept coarse signal (e.g. only down-rank explicit park/beach/garden types).

This metadata problem — not the weather provider — is what W9 planning must
solve first.

## Tentative scope (to be refined in plan-phase)

- A weather client (`app/tools/weather.py`) mirroring `app/tools/retrieval.py` /
  the W8c `directions.py` conventions; keyed by location + arrival-time window.
- An `outdoor` / rain-sensitivity signal on places (approach TBD per above).
- A `weather_appropriate` critique check + revision hint, reusing W3.
- One forecast lookup per finalized plan (same bounding discipline as W8c
  Directions: cheap gate in-loop, real signal once post-commit).
- Four-layer tests per project convention, incl. a functional test where a
  rain forecast flips an outdoor itinerary's critique pass→fail.

## Out of scope (tentative)

- Hour-by-hour multi-day forecasting; historical climate.
- Frontend weather UI (backend reasoning first; surfacing it is a later slice).
