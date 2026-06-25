import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// useMap() must return a STABLE reference (the real hook returns the same map
// instance across renders); a fresh object each render makes RouteOverlay's
// [map] effect dep churn -> infinite render loop. Mock shape mirrors
// MapView.test.jsx (__esModule + default) — an ESM-only package mocked without
// these hangs module resolution under Node ESM interop.
const STABLE_MAP = { id: 'map' }
vi.mock('@vis.gl/react-google-maps', () => ({
  __esModule: true,
  useMap: () => STABLE_MAP,
  default: ({ children }) => children,
}))

import RouteOverlay from './RouteOverlay'

const STOPS_3 = [
  { id: 'a', placeId: 'penny-roma', num: '1', name: 'Penny Roma', latitude: 37.7592504, longitude: -122.4110591 },
  { id: 'b', placeId: 'trick-dog', num: '2', name: 'Trick Dog', latitude: 37.7592244, longitude: -122.4111927 },
  { id: 'c', placeId: 'true-laurel', num: '3', name: 'True Laurel', latitude: 37.7595236, longitude: -122.4114663 },
]

let routeCalls
let lastRenderer

function legResult(min) {
  return { routes: [{ legs: [{ duration: { value: min * 60 } }] }] }
}

beforeEach(() => {
  routeCalls = []
  lastRenderer = null
  window.google = {
    maps: {
      TravelMode: { WALKING: 'WALKING', TRANSIT: 'TRANSIT', DRIVING: 'DRIVING' },
      DirectionsService: class {
        route(req, cb) {
          routeCalls.push(req)
          // Real DirectionsService.route is always async — model that so the
          // callback fires after the effect returns (avoids act() deadlock).
          setTimeout(() => {
            // Mirror the real API: TRANSIT rejects any request with waypoints.
            if (req.travelMode === 'TRANSIT' && req.waypoints?.length) {
              cb(null, 'INVALID_REQUEST')
              return
            }
            cb(legResult(req.travelMode === 'WALKING' ? 5 : 3), 'OK')
          }, 0)
        }
      },
      DirectionsRenderer: class {
        constructor() {
          this.map = null
          this.directions = null
        }
        setMap(m) {
          this.map = m
        }
        getMap() {
          return this.map
        }
        setDirections(d) {
          this.directions = d
          lastRenderer = this
        }
      },
    },
  }
})

afterEach(() => {
  delete window.google
  vi.restoreAllMocks()
})

describe('RouteOverlay transit handling (real-API constraint)', () => {
  it('WALKING uses a single waypointed request for a 3-stop itinerary', async () => {
    render(<RouteOverlay stops={STOPS_3} />)
    await waitFor(() => expect(screen.getByText(/min ·/)).toBeInTheDocument())
    const walkCalls = routeCalls.filter((c) => c.travelMode === 'WALKING')
    expect(walkCalls).toHaveLength(1)
    expect(walkCalls[0].waypoints).toHaveLength(1) // the middle stop
  })

  it('keeps the renderer bound to the map across re-renders (not only at creation)', async () => {
    // Regression: the renderer was setMap()'d only inside the
    // `if (!rendererRef.current)` create-once block. If the ref outlived its
    // map binding (HMR, or map null at creation), the renderer drew onto no
    // map — summary updated (pure JS) but the polyline was invisible
    // (browser diag showed rendererMap:null with a valid 132-pt overview_path).
    // Invariant: whenever setDirections() runs, the renderer is bound to the
    // map — including effect runs where the renderer ref already exists.
    const { rerender } = render(<RouteOverlay stops={STOPS_3} />)
    await waitFor(() => expect(screen.getByText(/min ·/)).toBeInTheDocument())
    expect(lastRenderer?.getMap()).toBe(STABLE_MAP)

    // Force the renderer to lose its map (models the HMR/orphan condition),
    // then trigger another effect run via a mode change.
    lastRenderer.setMap(null)
    fireEvent.click(screen.getByRole('button', { name: /Drive/i }))
    await waitFor(() => {
      const drive = routeCalls.filter((c) => c.travelMode === 'DRIVING')
      expect(drive.length).toBeGreaterThan(0)
    })
    // After the redraw the renderer must be re-bound — not left on null.
    expect(lastRenderer.getMap()).toBe(STABLE_MAP)
  })

  it('TRANSIT issues per-leg point-to-point requests (NO waypoints) and sums them', async () => {
    render(<RouteOverlay stops={STOPS_3} />)
    await waitFor(() => expect(screen.getByText(/min ·/)).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /Transit/i }))

    await waitFor(() => {
      const transitCalls = routeCalls.filter((c) => c.travelMode === 'TRANSIT')
      // 3 stops -> 2 legs -> 2 separate requests, each with ZERO waypoints.
      expect(transitCalls).toHaveLength(2)
      expect(
        transitCalls.every((c) => !c.waypoints || c.waypoints.length === 0),
      ).toBe(true)
    })
    // Summed: 2 legs * 3 min = 6 min, 2 legs. Must NOT show an error.
    expect(screen.getByText(/6 min · 2 legs/)).toBeInTheDocument()
    expect(screen.queryByText(/unavailable/i)).not.toBeInTheDocument()
  })
})
