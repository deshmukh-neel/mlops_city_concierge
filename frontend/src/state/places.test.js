import { describe, expect, it } from 'vitest'
import {
  clearPlaces,
  emptyPlacesState,
  replacePlaces,
  selectOrderedPlaces,
  selectRoutablePlaces,
} from './places'

const P = (id, order, extra = {}) => ({ id, order, name: id, ...extra })

describe('places keyed-state reducer', () => {
  it('starts empty and not finalized', () => {
    expect(emptyPlacesState.order).toEqual([])
    expect(emptyPlacesState.planFinalized).toBe(false)
  })

  it('replacePlaces keys by id and preserves itinerary order', () => {
    const s = replacePlaces(emptyPlacesState, [P('b', 1), P('a', 0), P('c', 2)])
    expect(s.order).toEqual(['a', 'b', 'c'])
    expect(selectOrderedPlaces(s).map((p) => p.id)).toEqual(['a', 'b', 'c'])
    expect(s.planFinalized).toBe(true)
  })

  it('a turn with places finalizes the plan; replacing swaps the whole set', () => {
    const s1 = replacePlaces(emptyPlacesState, [P('a', 0), P('b', 1)])
    const s2 = replacePlaces(s1, [P('x', 0)])
    expect(s2.order).toEqual(['x'])
    expect(s2.byId.a).toBeUndefined()
    expect(s2.planFinalized).toBe(true)
  })

  it('a turn with NO places keeps the prior set but un-finalizes', () => {
    const s1 = replacePlaces(emptyPlacesState, [P('a', 0)])
    const s2 = replacePlaces(s1, [])
    expect(s2.order).toEqual(['a']) // not wiped
    expect(s2.planFinalized).toBe(false) // route hidden until re-finalized
  })

  it('clearPlaces resets to empty', () => {
    const s1 = replacePlaces(emptyPlacesState, [P('a', 0)])
    expect(clearPlaces().order).toEqual([])
    expect(clearPlaces().planFinalized).toBe(false)
  })

  it('selectRoutablePlaces drops stops without numeric coords', () => {
    const s = replacePlaces(emptyPlacesState, [
      P('a', 0, { latitude: 37.7, longitude: -122.4 }),
      P('b', 1, { latitude: null, longitude: null }),
      P('c', 2, { latitude: 37.8, longitude: -122.41 }),
    ])
    expect(selectRoutablePlaces(s).map((p) => p.id)).toEqual(['a', 'c'])
  })

  it('ignores null / id-less entries defensively', () => {
    const s = replacePlaces(emptyPlacesState, [P('a', 0), null, { order: 1 }])
    expect(s.order).toEqual(['a'])
  })
})
