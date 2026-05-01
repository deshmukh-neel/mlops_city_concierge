/**
 * api/chat.js
 *
 * Replace VITE_API_URL in your .env with your FastAPI base URL.
 * e.g. VITE_API_URL=http://localhost:8000
 *
 * Calls the FastAPI RAG endpoint:
 *   POST /predict
 *   Body: { query: string, limit?: number }
 *
 * Backend returns (see app/main.py RecommendationResponse):
 *   {
 *     response: string,
 *     sources: { name, rating, address, similarity }[]
 *   }
 *
 * This module adapts that shape into what the chat UI expects:
 *   { reply, places, ragLabel }
 */

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/**
 * Send a chat message to the FastAPI /predict endpoint and adapt the
 * response into the UI's chat shape.
 *
 * @param {string} message
 * @param {{ role: string, content: string }[]} _history  (unused; kept for signature compatibility)
 * @param {{}} [_opts]  (reserved for future per-request options)
 * @returns {Promise<{ reply: string, places: object[], ragLabel?: string }>}
 */
export async function sendMessage(message, _history = [], _opts = {}) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: message }),
  })

  if (!res.ok) {
    let detail = ''
    try {
      const errBody = await res.json()
      detail = errBody?.detail ? ` — ${errBody.detail}` : ''
    } catch (_) { /* ignore parse errors */ }
    throw new Error(`API error ${res.status}${detail}`)
  }

  const data = await res.json()
  const sources = Array.isArray(data?.sources) ? data.sources : []

  return {
    reply: formatReply(data?.response ?? ''),
    places: sources.map(toPlace),
    ragLabel: sources.length
      ? `${sources.length} live result${sources.length === 1 ? '' : 's'} · Google Places`
      : undefined,
  }
}

/** Turn plain text from the LLM into HTML-safe content with line breaks preserved. */
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

/** Map a backend RecommendationSource to the place shape used by the right panel + map. */
function toPlace(source, idx) {
  const name = source?.name || 'Unknown place'
  const ratingNum = typeof source?.rating === 'number' ? source.rating : null
  const similarityPct =
    typeof source?.similarity === 'number'
      ? `${Math.round(source.similarity * 100)}% match`
      : null

  const tags = []
  if (similarityPct) tags.push({ label: similarityPct, highlight: idx === 0 })

  return {
    id: slugify(name) + '-' + idx,
    num: String(idx + 1),
    name,
    type: '',
    category: 'dinner',
    desc: source?.address || '',
    tags,
    rating: ratingNum != null ? ratingNum.toFixed(1) : '—',
    distance: '',
    status: 'open',
    hours: '',
    featured: idx === 0,
    mapPos: spreadMapPos(idx),
  }
}

function slugify(str) {
  return String(str)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '') || 'place'
}

/** Deterministically spread pins around the placeholder map so they don't overlap. */
function spreadMapPos(idx) {
  const cols = 4
  const col = idx % cols
  const row = Math.floor(idx / cols)
  const x = 22 + col * 18 + (row % 2) * 6
  const y = 30 + row * 16
  return { x: `${Math.min(x, 80)}%`, y: `${Math.min(y, 78)}%` }
}

// ─── Mock data (used when API is unavailable / for development) ───────────────

export const MOCK_PLACES = [
  {
    id: 'la-lengua',
    num: '★', name: 'La Lengua', type: 'Mexican Restaurant', category: 'dinner',
    desc: 'Low-key neighborhood staple on 20th. Tacos, mole, and no frills — exactly what the Mission used to be.',
    tags: [
      { label: 'Neighborhood gem', highlight: true },
      { label: 'Cash friendly' },
      { label: 'No reservations' },
    ],
    rating: '4.6', distance: '0.3 mi', status: 'open', hours: 'until 10pm',
    featured: true,
    mapPos: { x: '26%', y: '56%' },
  },
  {
    id: 'foreign-cinema',
    num: '★', name: 'Foreign Cinema', type: 'Cal-Mediterranean', category: 'dinner',
    desc: 'Outdoor courtyard where they project classic films. Romantic without trying too hard. Great natural wine selection.',
    tags: [
      { label: 'Date-night pick', highlight: true },
      { label: 'Film projections' },
      { label: 'Courtyard' },
    ],
    rating: '4.5', distance: '0.5 mi', status: 'busy', hours: 'until 11pm',
    featured: false,
    mapPos: { x: '62%', y: '42%' },
  },
  {
    id: 'trick-dog',
    num: '1', name: 'Trick Dog', type: 'Cocktail Bar', category: 'bar',
    desc: 'Rotating thematic cocktail menu changes every few months. Always a conversation piece. Go here first.',
    tags: [
      { label: 'Must visit', highlight: true },
      { label: 'Inventive cocktails' },
    ],
    rating: '4.6', distance: '0.2 mi', status: 'open', hours: 'until 2am',
    featured: true,
    mapPos: { x: '38%', y: '52%' },
  },
  {
    id: 'true-laurel',
    num: '2', name: 'True Laurel', type: 'Cocktail Bar · Natural Wine', category: 'bar',
    desc: 'Intimate, low-lit, and impressively curated. From the same team as Lazy Bear.',
    tags: [{ label: 'Natural wine' }, { label: 'Intimate' }],
    rating: '4.7', distance: '0.4 mi', status: 'open', hours: 'until 1am',
    featured: false,
    mapPos: { x: '48%', y: '44%' },
  },
  {
    id: 'elixir',
    num: '3', name: 'Elixir', type: 'Neighborhood Dive Bar', category: 'bar',
    desc: "One of SF's oldest bars. Wooden bar, dark corners, totally unpretentious.",
    tags: [{ label: 'Historic' }, { label: 'No-frills' }],
    rating: '4.4', distance: '0.6 mi', status: 'open', hours: 'until 2am',
    featured: false,
    mapPos: { x: '30%', y: '40%' },
  },
  {
    id: 'abv',
    num: '4', name: 'ABV', type: 'Bar + Kitchen', category: 'bar',
    desc: "Good for a late snack and cocktail. Menu is concise and thoughtful.",
    tags: [{ label: 'Late night food' }, { label: 'Full bar' }],
    rating: '4.5', distance: '0.7 mi', status: 'open', hours: 'until 2am',
    featured: false,
    mapPos: { x: '56%', y: '56%' },
  },
  {
    id: 'virgils',
    num: '5', name: "Virgil's Sea Room", type: 'Eclectic Dive Bar', category: 'bar',
    desc: 'Loud, dark, nautical decor, and a genuinely fun crowd. End-of-night energy.',
    tags: [{ label: 'End of night' }, { label: 'Lively' }],
    rating: '4.3', distance: '0.8 mi', status: 'open', hours: 'until 2am',
    featured: false,
    mapPos: { x: '42%', y: '65%' },
  },
]

export const INITIAL_MESSAGES = [
  {
    id: 'welcome',
    role: 'assistant',
    content: "Hey! I'm your San Francisco city guide. Ask me to plan a night out, find hidden gems, or build an itinerary for any neighborhood.<br><br>I pull real-time data from Google Places, so everything is current. What are you in the mood for?",
    time: formatTime(new Date()),
  },
]

export function formatTime(date) {
  return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}
