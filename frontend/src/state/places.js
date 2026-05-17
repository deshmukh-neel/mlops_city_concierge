/**
 * places.js — single source of truth for the itinerary place set.
 *
 * Modeled as an ordered, keyed collection so that:
 *  - the map, list, and route all derive from ONE structure;
 *  - replacing the set on each agent turn is O(n) and order-stable;
 *  - future streaming (incremental stop updates) is purely additive — push
 *    into the same structure, no consumer changes (see W8 §Deferred).
 *
 * `order` comes from the adapter (array index = itinerary/route order). We key
 * by id for stable React keys + dedupe, but always render/route by `order`.
 */

/** Empty state: no plan, nothing routed yet. */
export const emptyPlacesState = Object.freeze({
  byId: {},
  order: [], // array of ids, in itinerary order
  planFinalized: false,
})

/**
 * Replace the entire place set from an agent response.
 *
 * A turn that returns places is treated as a finalized plan (the agent only
 * emits committed stops via state_to_cards). A turn with no places (e.g. the
 * agent asking a clarifying question) leaves the existing set intact and
 * un-finalizes — we don't wipe a good plan because of a follow-up question.
 *
 * @param {typeof emptyPlacesState} _prev
 * @param {object[]} places  adapter-shaped places (must have id + order)
 */
export function replacePlaces(_prev, places) {
  if (!Array.isArray(places) || places.length === 0) {
    return { ..._prev, planFinalized: false }
  }
  const sorted = places
    .filter((p) => p != null && p.id != null)
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
  const byId = {}
  const order = []
  for (const p of sorted) {
    byId[p.id] = p
    order.push(p.id)
  }
  return { byId, order, planFinalized: order.length > 0 }
}

/** Clear everything (used by the chat "clear" action). */
export function clearPlaces() {
  return { ...emptyPlacesState }
}

/** Ordered array view for rendering (map pins, list, route waypoints). */
export function selectOrderedPlaces(state) {
  return state.order.map((id) => state.byId[id]).filter(Boolean)
}

/** Stops that have real coordinates — the only ones a map/route can use. */
export function selectRoutablePlaces(state) {
  return selectOrderedPlaces(state).filter(
    (p) => typeof p.latitude === 'number' && typeof p.longitude === 'number',
  )
}
