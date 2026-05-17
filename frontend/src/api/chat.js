/**
 * api/chat.js
 *
 * Calls the FastAPI agent endpoint:
 *   POST /chat
 *   Body: { message: string, history: { role, content }[] }
 *
 * Backend returns (see app/main.py ChatResponse):
 *   {
 *     reply: string,
 *     places: PlaceCard[],   // ordered itinerary stops (app/agent/state.py)
 *     ragLabel: string
 *   }
 *
 * PlaceCard carries latitude/longitude (W8a) and the array order IS the
 * itinerary/route order — the map and route overlay rely on both. This module
 * adapts PlaceCard into the shape the right-panel UI components render. It does
 * NOT fabricate coordinates; a stop without coords keeps null lat/lng.
 *
 * Set VITE_API_URL to the FastAPI base URL (defaults to localhost:8000; in dev
 * the Vite proxy forwards /chat).
 */

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/**
 * Send a chat message to the FastAPI /chat endpoint and adapt the response
 * into the UI's chat shape.
 *
 * @param {string} message
 * @param {{ role: string, content: string }[]} [history]
 * @returns {Promise<{ reply: string, places: object[], ragLabel?: string }>}
 */
export async function sendMessage(message, history = []) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
  })

  if (!res.ok) {
    let detail = ''
    try {
      const errBody = await res.json()
      detail = errBody?.detail ? ` — ${errBody.detail}` : ''
    } catch {
      /* ignore parse errors */
    }
    throw new Error(`API error ${res.status}${detail}`)
  }

  const data = await res.json()
  const cards = Array.isArray(data?.places) ? data.places : []

  return {
    reply: formatReply(data?.reply ?? ''),
    places: cards.map(toUiPlace),
    ragLabel: data?.ragLabel || undefined,
  }
}

/** Turn plain text from the agent into HTML-safe content with line breaks preserved. */
function formatReply(text) {
  if (typeof text !== 'string') return ''
  return escapeHtml(text).replace(/\n/g, '<br>')
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

/**
 * Map a backend PlaceCard to the place shape the right panel + map render.
 * Array index is itinerary order — `num` reflects route position. Coordinates
 * pass through untouched (null stays null; never fabricated).
 */
function toUiPlace(card, idx) {
  const ratingNum = typeof card?.rating === 'number' ? card.rating : null
  const lat = typeof card?.latitude === 'number' ? card.latitude : null
  const lng = typeof card?.longitude === 'number' ? card.longitude : null

  return {
    id: card?.place_id || `stop-${idx}`,
    placeId: card?.place_id || null,
    num: String(idx + 1),
    order: idx,
    name: card?.name || 'Unknown place',
    type: prettyType(card?.primary_type),
    category: card?.primary_type || 'place',
    desc: card?.rationale || card?.address || '',
    address: card?.address || '',
    tags: [],
    rating: ratingNum != null ? ratingNum.toFixed(1) : '—',
    priceLevel: typeof card?.price_level === 'number' ? card.price_level : null,
    distance: '',
    status: 'open',
    hours: card?.arrival_time ? `arrive ${formatArrival(card.arrival_time)}` : '',
    arrivalTime: card?.arrival_time || null,
    bookingUrl: card?.booking_url || null,
    bookingProvider: card?.booking_provider || null,
    latitude: lat,
    longitude: lng,
    featured: idx === 0,
  }
}

/** "italian_restaurant" -> "Italian Restaurant". */
function prettyType(primaryType) {
  if (!primaryType || typeof primaryType !== 'string') return ''
  return primaryType
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

/** ISO datetime -> "7:30 PM"; falls back to the raw string if unparseable. */
function formatArrival(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return String(iso)
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}

// ─── Initial chat state (no mock places — the real agent populates them) ─────

export const MOCK_PLACES = []

export const INITIAL_MESSAGES = [
  {
    id: 'welcome',
    role: 'assistant',
    content:
      "Hey! I'm your San Francisco city guide. Ask me to plan a night out, " +
      'find hidden gems, or build an itinerary for any neighborhood.<br><br>' +
      'I pull real-time data from Google Places, so everything is current. ' +
      'What are you in the mood for?',
    time: formatTime(new Date()),
  },
]

export function formatTime(date) {
  return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}
