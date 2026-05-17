# W8b verification gate — real Google APIs + real `/chat`

Mocked tests (17 green) prove component *logic*. They prove nothing about the
real map rendering, real coordinates, or real Directions. **W8b is not "done"
until this checklist passes against real APIs.** PR3 (w8c) is blocked on it.

## Prerequisites (you provision — I cannot)

1. **Google Maps browser key + Map ID** — follow `docs/google_maps_setup.md`
   (enable Maps JavaScript API **and** Directions API; referrer-restrict the
   key to `http://localhost:5173/*`).
2. **A working backend** so `/chat` returns a real itinerary with coordinates:
   - Either the deployed Cloud Run backend (default `VITE_API_URL` already
     points at it — simplest), or
   - a local backend (`make dev`) with OpenAI/Gemini + DB configured.

## Setup (local frontend against deployed backend — least friction)

```bash
cat >> frontend/.env.development.local <<'EOF'
VITE_GOOGLE_MAPS_API_KEY=your-browser-key
VITE_GOOGLE_MAPS_MAP_ID=your-map-id
EOF

cd frontend && npm run dev      # restart if already running — Vite won't hot-reload env
```

Open http://localhost:5173 and ask: **"Plan me a 3-stop night out in the
Mission — dinner then two bars."**

## The gate — every item must pass

| # | Check | Pass criteria |
|---|---|---|
| 1 | Map renders | Real SF streets visible, not a grey/fallback panel |
| 2 | Pins placed | One numbered pin per stop, at plausible real locations |
| 3 | Pin order | Pin numbers match the itinerary order in the chat reply |
| 4 | Hover tooltip | Hovering a pin shows name / type / rating / address |
| 5 | Click-to-focus | Clicking a chat place or a pin pans+zooms the map to it |
| 6 | Route drawn | A route line connects the stops **in itinerary order** |
| 7 | Mode toggle | Walk / Transit / Drive changes the route and the time readout |
| 8 | Cache | Re-selecting a previously-viewed mode is instant (no reflow/flicker) |
| 9 | Total time | The "N min · M legs" readout is non-zero and plausible |
| 10 | Multi-turn | A follow-up ("make it 4 stops") updates pins + route, no stale pins |
| 11 | Keyless fallback | Remove the key, restart dev → explicit "Map unavailable" panel, no SDK fetched (Network tab shows no maps.googleapis.com) |

## If something fails

Don't mark verified. Capture: which row, a screenshot, and the browser
console + Network tab. Common causes:
- Blank map / `RefererNotAllowedMapError` → key referrer restriction missing
  `http://localhost:5173/*`.
- Markers missing, map greys → Map ID missing/invalid (Advanced Markers need
  cloud styling).
- `REQUEST_DENIED` on route → Directions API not enabled on the key.
- Pins but no route → fewer than 2 stops had coordinates (editorial stops have
  null lat/lng by design — ask for a query that returns Google-Places stops).

## Done

When all 11 pass, W8b is verified. Note it in
`implementation_plan/james/w8_live_map_routing.md` status and unblock PR3.
