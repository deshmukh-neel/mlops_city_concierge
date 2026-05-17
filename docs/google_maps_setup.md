# Google Maps setup (frontend live map + itinerary routing)

The frontend renders a real Google Map with the agent's itinerary pins and a
Directions-powered route (Walk / Transit / Drive). This needs a **browser**
Google Maps key — **separate from the backend's Google Places key** — plus a
Map ID. Without them the map degrades to an explicit "Map unavailable" panel and
the Maps SDK is never fetched, so frontend-only contributors are not blocked.

## What to enable (Google Cloud Console)

In the project's Google Cloud console → **APIs & Services → Library**, enable:

1. **Maps JavaScript API** — renders the map + Advanced Markers.
2. **Directions API** — the Walk/Transit/Drive route + travel times.

> The backend's Places ingestion key and this browser key are different keys
> with different restrictions. Do **not** reuse the backend key in the browser.

## Create the Map ID

`AdvancedMarker` requires cloud-based map styling. Console → **Google Maps
Platform → Map Management → Create Map ID**:

- Map type: **JavaScript**
- Rendering: **Vector** (required for Advanced Markers)
- **Tilt: OFF**, **Rotation: OFF** — this is a route-reading map; a flat
  top-down view keeps the ordered itinerary legible and keeps the `fitBounds`
  framing predictable. (Independent toggles; leave them off even though
  rendering is Vector. Can be enabled later without code changes if ever needed.)

Copy the Map ID (looks like `a1b2c3d4e5f6...`). For quick local prototyping you
may instead use the literal `DEMO_MAP_ID` (the code defaults to this if
`VITE_GOOGLE_MAPS_MAP_ID` is unset), but create a real one for anything shared.

## Create + restrict the browser key

Console → **APIs & Services → Credentials → Create credentials → API key**.
Then **restrict it** (the key ships in client JS — referrer restriction is the
protection, not secrecy):

- **Application restrictions → Websites (HTTP referrers)**, add:
  - `http://localhost:5173/*` (Vite dev) — **add this now**
  - the deployed frontend origin `/*` — **only when a deployed frontend
    exists.** There is currently **no deployed frontend** (no CI builds
    `frontend/`; it runs locally or points at the deployed *backend*). Adding a
    referrer for a non-existent origin does nothing — defer it. When the
    frontend is deployed, add its real origin here in the console (no code
    change). Until then, requests from any non-localhost origin will fail with
    `RefererNotAllowedMapError` — that is the restriction working, not a bug.
- **API restrictions → Restrict key** → select **Maps JavaScript API** and
  **Directions API** only.

> Do **not** interpret "no deployed origin yet" as "no restrictions." An
> unrestricted browser key is world-usable from your shipped JS. Restrict to
> localhost-only now; widen later.

## Wire it in

**Local dev (untracked override — never commit a real key):**

```bash
cat >> frontend/.env.development.local <<'EOF'
VITE_GOOGLE_MAPS_API_KEY=your-browser-key
VITE_GOOGLE_MAPS_MAP_ID=your-map-id
EOF
# Vite does not hot-reload env — restart `npm run dev` after editing.
```

`*.local` is loaded last by Vite and is gitignored. Delete the file to revert
to the no-key fallback.

**Docker Compose:** put the same two vars in your root `.env`; the `frontend`
service forwards them (see `docker-compose.yml`).

**Deployed frontend:** provide both as build/runtime env on the host that
serves `frontend/` (they must be present at Vite build time, prefixed `VITE_`).

## Verify

1. Restart the dev server, open `http://localhost:5173`.
2. Ask the agent to plan a multi-stop night out.
3. Pins appear at real coordinates; once the plan is final a route is drawn
   with a Walk/Transit/Drive toggle and total travel time. Toggling a mode the
   first time fetches; re-toggling a seen mode is instant (cached).

If you see "Map unavailable", the key env var didn't reach the running build —
recheck the `VITE_` prefix and that you restarted the dev server.

## Notes / cost

- Directions calls are billable. The app bounds them: ≤1 backend call per
  finalized plan (W8c) + ≤3 frontend calls per plan (one per mode, then cached).
- Map JS bundle is code-split and lazy-loaded — only fetched when the map tab
  is viewed and a key exists.
- See `implementation_plan/james/w8_live_map_routing.md` for the full design.
