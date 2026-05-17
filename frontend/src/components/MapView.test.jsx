import React from 'react'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Mock the Google Maps lib so smoke tests need no API key / network.
vi.mock('@vis.gl/react-google-maps', () => {
  const Passthrough = ({ children }) => <div>{children}</div>
  return {
    APIProvider: ({ children }) => <div data-testid="api-provider">{children}</div>,
    Map: ({ children }) => <div data-testid="map">{children}</div>,
    AdvancedMarker: ({ children, title }) => (
      <div data-testid="marker" data-title={title}>
        {children}
      </div>
    ),
    Pin: ({ children }) => <span data-testid="pin">{children}</span>,
    useMap: () => null,
    __esModule: true,
    default: Passthrough,
  }
})

// RouteOverlay uses useMap()/google.maps; stub it to a no-op for MapView smoke.
vi.mock('./RouteOverlay', () => ({
  default: () => <div data-testid="route-overlay" />,
}))

import MapView from './MapView'

const PLACES = [
  { id: 'a', num: '1', name: 'Trick Dog', latitude: 37.759, longitude: -122.41 },
  { id: 'b', num: '2', name: 'True Laurel', latitude: 37.752, longitude: -122.42 },
  { id: 'c', num: '3', name: 'Editorial (no coords)', latitude: null, longitude: null },
]

const ENV = import.meta.env

function setKey(value) {
  if (value == null) delete ENV.VITE_GOOGLE_MAPS_API_KEY
  else ENV.VITE_GOOGLE_MAPS_API_KEY = value
}

afterEach(() => {
  setKey(null)
  vi.restoreAllMocks()
})

describe('MapView smoke', () => {
  it('shows the keyless fallback panel when no API key is set', () => {
    setKey(null)
    render(<MapView places={PLACES} planFinalized />)
    expect(screen.getByText(/Map unavailable/i)).toBeInTheDocument()
    expect(screen.getByText('VITE_GOOGLE_MAPS_API_KEY')).toBeInTheDocument()
    expect(screen.queryByTestId('map')).not.toBeInTheDocument()
  })

  it('renders one marker per routable place (drops null-coord stops)', () => {
    setKey('test-key')
    render(<MapView places={PLACES} planFinalized />)
    // 2 stop markers (a, b) — 'c' has null coords and is excluded.
    const markers = screen.getAllByTestId('marker')
    const titled = markers.filter((m) => m.getAttribute('data-title'))
    expect(titled.map((m) => m.getAttribute('data-title'))).toEqual([
      'Trick Dog',
      'True Laurel',
    ])
  })

  it('mounts the route overlay when finalized with >= 2 routable stops', () => {
    setKey('test-key')
    render(<MapView places={PLACES} planFinalized />)
    expect(screen.getByTestId('route-overlay')).toBeInTheDocument()
  })

  it('does NOT render the route overlay before the plan is finalized', () => {
    setKey('test-key')
    render(<MapView places={PLACES} planFinalized={false} />)
    expect(screen.queryByTestId('route-overlay')).not.toBeInTheDocument()
  })

  it('does NOT render the route overlay with fewer than 2 routable stops', () => {
    setKey('test-key')
    render(<MapView places={[PLACES[0]]} planFinalized />)
    expect(screen.queryByTestId('route-overlay')).not.toBeInTheDocument()
  })
})
