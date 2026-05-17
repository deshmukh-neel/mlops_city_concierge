# Chat API Contract (frontend ↔ backend)

This is the source of truth for the HTTP contract the City Concierge frontend
calls. Backend code: `app/main.py`, `app/agent/state.py`, `app/agent/io.py`.

If you find this doc disagreeing with the code, the code wins — but please
open an issue or fix the doc in the same PR.

Last updated for: W5 (coverage-gap ingestion agent). Reflects everything merged W0 → W5. Note (2026-05-16): the `PlaceCard` `address`/`rating`/`price_level` fields were always in the documented contract but the backend shipped them as `null` until the `fix/placecard-address-rating-price` fix — they are now actually populated. No shape change.

---

## Endpoint inventory

| Method + path | Purpose | Status |
|---|---|---|
| `POST /chat` | Agent-driven itinerary (W2/W3) | ✅ Active — primary endpoint |
| `POST /predict` | Legacy single-pass RAG (pre-W2) | ⚠️ Compatibility shim — currently used by frontend; migrate to `/chat` |
| `GET /health` | App + LLM config status | ✅ Active |
| `GET /health/db` | DB connectivity check | ✅ Active |
| `GET /root` | Hello-world | ✅ Active |

Base URL is configured via `VITE_API_URL` (see `frontend/src/api/chat.js`).
Local dev default: `http://localhost:8000`. Production: Cloud Run service URL.

---

## `POST /chat` — primary endpoint

The agent plans a multi-stop itinerary, grounded in the structured DB.
Conversation history is stateless on the backend — the frontend sends the full
history each turn.

### Request

```json
{
  "message": "plan a date night in north beach: dinner around 7, drinks then dessert, all under $$$",
  "history": [
    { "role": "user", "content": "earlier user message" },
    { "role": "assistant", "content": "earlier assistant reply" }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `message` | string | Current user turn. |
| `history` | array | Prior turns. Empty on the first turn. Each item has `role: "user" \| "assistant"` and `content: string`. |

### Response

```json
{
  "reply": "I've put together a North Beach date night...",
  "places": [PlaceCard, ...],
  "ragLabel": "openai:gpt-4o"
}
```

| Field | Type | Notes |
|---|---|---|
| `reply` | string | The agent's final natural-language reply. May be a clarifying question instead of an itinerary — see "Multi-turn flows" below. |
| `places` | array of PlaceCard | Committed stops, in itinerary order. Empty array when the reply is a clarifying question or an error apology. |
| `ragLabel` | string | Model identifier (`provider:model`) for UI badge. Falls back to `"unknown"` if the active model config can't be read. |

### `PlaceCard` shape

```ts
{
  place_id:         string,         // Google Places ID; stable across calls
  name:             string,
  address:          string | null,
  rating:           number | null,  // 0-5
  price_level:      number | null,  // 0-4 (Google scale)
  primary_type:     string | null,  // e.g. "restaurant", "cocktail_bar"
  arrival_time:     string | null,  // ISO 8601 datetime, agent-planned
  rationale:        string,         // 1-2 sentence justification
  booking_url:      string | null,  // see Booking section below (W4)
  booking_provider: string | null   // see Booking section below (W4)
}
```

`place_id` is the join key for everything: maps, future detail pages, the
booking link. Always include it in any UI keys / cache lookups.

### Multi-turn flows

The agent sometimes returns a clarifying question instead of an itinerary
(e.g. "How many stops would you like?"). In that case:

- `places` is `[]`
- `reply` contains the question
- The frontend appends both the user's message and the assistant's reply to
  `history` and waits for the next user turn

There is no special status code or flag — the empty `places` array is the
signal. Keep the chat UI layout flexible enough to render an
itinerary-less assistant turn.

### Errors

| Status | Body shape | When |
|---|---|---|
| 503 | `{"detail": "Agent graph unavailable: ..."}` | MLflow registry was unreachable when the app booted. The agent isn't loaded. |
| 422 | `{"detail": [...FastAPI validation errors...]}` | Request body doesn't match `ChatRequest`. |
| 500 | `{"detail": "..."}` | Unhandled exception in the graph. Should be rare; surface as a generic "something went wrong" toast. |

The frontend should treat 503 as a long-lived "AI offline" state — it persists
until the backend restarts. Polling `/health` is the cheap way to check.

---

## Booking integration (W4)

When the agent commits an itinerary, every stop is auto-enriched with a
`booking_url` + `booking_provider`. The frontend renders these as a
link/button per card.

This is a **handoff**, not a confirmed booking. The user still completes the
reservation on the provider's site. We cannot guarantee table availability —
**the button MUST NOT say "Book now" or "Reserve."** Suggested labels:

| `booking_provider` | Label | Notes |
|---|---|---|
| `"resy"` | "Open in Resy" | Opens Resy with date + party size pre-filled. |
| `"tock"` | "Open in Tock" | Opens Tock with date + size + time pre-filled. |
| `"opentable"` | "Open in OpenTable" | Opens OpenTable with covers + dateTime pre-filled. |
| `"unknown"` | "Visit website" | Provider not detected; opens the venue's website. The user finds the reservations page themselves. |
| `"google_maps"` | "View on Google Maps" | No website on file; opens Google Maps. |
| `null` | (no button) | Enrichment failed. Extremely rare. |

### Frontend usage

```jsx
{place.booking_url && (
  <a
    href={place.booking_url}
    target="_blank"
    rel="noopener noreferrer"
  >
    {labelFor(place.booking_provider)}
  </a>
)}

const LABELS = {
  resy:        'Open in Resy',
  tock:        'Open in Tock',
  opentable:   'Open in OpenTable',
  unknown:     'Visit website',
  google_maps: 'View on Google Maps',
};
const labelFor = (p) => LABELS[p] ?? 'Visit website';
```

Always open in a new tab (`target="_blank" rel="noopener noreferrer"`) —
booking flows are multi-step and the user shouldn't lose the chat.

---

## `POST /predict` — legacy contract

The frontend currently calls `/predict` (`frontend/src/api/chat.js`). It's a
single-pass RAG endpoint that predates the agent. New work should target
`/chat`; `/predict` stays as a compatibility shim until the frontend migrates.

### Request

```json
{ "query": "Best tacos in the Mission", "limit": 5 }
```

### Response

```json
{
  "response": "Here are some recommendations...",
  "sources": [
    {
      "name":         "...",
      "rating":       4.5,
      "address":      "...",
      "similarity":   0.83,
      "place_id":     "...",
      "primary_type": "restaurant"
    }
  ]
}
```

### Errors

| Status | When |
|---|---|
| 503 | RAG chain wasn't loaded at boot (MLflow unreachable). |
| 422 | Body doesn't match `RecommendationRequest`. |

---

## `/chat` migration guide (for the frontend dev doing the swap)

| Old (`/predict`) | New (`/chat`) | Notes |
|---|---|---|
| `query` | `message` | |
| `limit` | _(removed)_ | Agent decides stop count. Pass-through to nothing. |
| _(none)_ | `history` | Send the full prior conversation each turn. |
| `response` (string) | `reply` (string) | Same idea. |
| `sources[]` | `places[]` | Different shape — see `PlaceCard` above. |
| `sources[].name` | `places[].name` | Same. |
| `sources[].rating` | `places[].rating` | Same. |
| `sources[].address` | `places[].address` | Same. |
| `sources[].similarity` | _(not exposed)_ | Internal to retrieval. If you want a "match strength" badge, derive from agent rationale or skip. |
| `sources[].place_id` | `places[].place_id` | Same. |
| `sources[].primary_type` | `places[].primary_type` | Same. |
| _(none)_ | `places[].arrival_time` | New: itinerary order + planned time. Render as "7:00 PM" near the place name. |
| _(none)_ | `places[].rationale` | New: 1-2 sentence justification. Render below the name. |
| _(none)_ | `places[].booking_url` + `booking_provider` | New (W4). See Booking section. |
| _(none)_ | `ragLabel` | Already in `/predict` adapter as a derived field; now comes from the backend directly. |

The mock-data path (`MOCK_PLACES` in `frontend/src/api/chat.js`) can stay —
it predates both endpoints and is unrelated to backend wire format.

---

## `GET /health`

```json
// when ready
{ "status": "ok", "llm_provider": "openai", "chat_model": "gpt-4o" }

// when MLflow registry was unreachable at boot
{ "status": "degraded", "rag_chain": "unavailable" }
```

Use this for an "AI offline" banner. Cheap and unauthenticated.

## `GET /health/db`

```json
// 200
{ "status": "ok" }

// 500 if the DB pool is down
```

Mainly for ops / readiness probes. The frontend doesn't need to call this.

---

## CORS

`app/main.py` configures CORS via the `cors_allow_origins` setting. Allowed
methods: `GET`, `POST`, `OPTIONS`. Allowed headers: `*`.

If you serve the frontend from a new origin (preview deploy, custom domain),
update the backend allowlist before going live or browser requests will fail
with a CORS error.

---

## What's NOT yet in the contract

These are deliberately not exposed today. If the frontend starts depending on
them, surface that need so we can plan it.

- **Trace IDs** — Langfuse traces (W0) are emitted server-side but not
  echoed back in response headers. If we ever want a "View trace" debug
  link, we'd add `X-Trace-Id`.
- **Knowledge-graph fields** — `kg_traverse` returns a "not yet available"
  stub today; W7 will populate place relations. No frontend impact until then.
- **Booking automation status** — `automation_available` exists on the
  backend `BookingProposal` model but is always `false` today. Future
  Playwright PR may flip it; the field isn't on `PlaceCard` yet.
- **Per-stop walking distance** — the agent computes walking math internally
  but doesn't surface "0.3 mi from previous stop" on the card. Add to
  `PlaceCard` if/when designs need it.
- **Streaming** — `/chat` is request/response. No SSE / chunked streaming.
  Acceptable for short replies; revisit if median latency exceeds ~3s.
