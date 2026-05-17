import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useMap } from '@vis.gl/react-google-maps'

/**
 * Draws the itinerary route with Google Directions and a Walk / Transit / Drive
 * toggle, showing real total + per-leg travel time. Rendered only when the plan
 * is finalized and >= 2 routable stops exist (gated by MapView).
 *
 * Directions results are cached per (itinerary signature, mode) so re-toggling
 * a previously-seen mode costs zero API calls (Performance #1): worst case 3
 * calls per plan, then instant.
 */

const MODES = [
  { id: 'WALKING', label: 'Walk', icon: '🚶' },
  { id: 'TRANSIT', label: 'Transit', icon: '🚇' },
  { id: 'DRIVING', label: 'Drive', icon: '🚗' },
]

const s = {
  panel: {
    position: 'absolute',
    top: '16px',
    left: '50%',
    transform: 'translateX(-50%)',
    background: 'var(--white)',
    border: '1px solid var(--border)',
    borderRadius: '10px',
    boxShadow: '0 4px 16px rgba(0,0,0,.12)',
    padding: '8px 10px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    zIndex: 30,
    fontSize: '12px',
  },
  toggle: { display: 'flex', gap: '2px', background: 'var(--cream-dark)', borderRadius: '7px', padding: '3px' },
  btn: {
    border: 'none', background: 'transparent', cursor: 'pointer',
    padding: '5px 10px', borderRadius: '5px', fontSize: '12px',
    fontFamily: 'var(--font-body)', color: 'var(--warm-gray)',
  },
  btnActive: { background: 'var(--white)', color: 'var(--charcoal)', boxShadow: '0 1px 3px rgba(0,0,0,.1)' },
  total: { fontWeight: 600, color: 'var(--charcoal)', whiteSpace: 'nowrap' },
  err: { color: 'var(--rust)', whiteSpace: 'nowrap' },
}

function itinerarySignature(stops) {
  return stops.map((p) => p.placeId || `${p.latitude},${p.longitude}`).join('|')
}

export default function RouteOverlay({ stops }) {
  const map = useMap()
  const [mode, setMode] = useState('WALKING')
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)

  const rendererRef = useRef(null)
  const serviceRef = useRef(null)
  const cacheRef = useRef(new Map()) // `${sig}::${mode}` -> DirectionsResult

  const sig = useMemo(() => itinerarySignature(stops), [stops])

  // Build the request once per (itinerary, mode); memo avoids spurious refetch.
  const request = useMemo(() => {
    if (stops.length < 2) return null
    const origin = { lat: stops[0].latitude, lng: stops[0].longitude }
    const destination = {
      lat: stops[stops.length - 1].latitude,
      lng: stops[stops.length - 1].longitude,
    }
    const waypoints = stops.slice(1, -1).map((p) => ({
      location: { lat: p.latitude, lng: p.longitude },
      stopover: true,
    }))
    return { origin, destination, waypoints, mode }
  }, [sig, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!map || !request || !window.google?.maps) return

    if (!serviceRef.current) {
      serviceRef.current = new window.google.maps.DirectionsService()
    }
    if (!rendererRef.current) {
      rendererRef.current = new window.google.maps.DirectionsRenderer({
        suppressMarkers: true, // our AdvancedMarkers already mark the stops
        preserveViewport: true,
      })
      rendererRef.current.setMap(map)
    }

    const cacheKey = `${sig}::${mode}`
    const cached = cacheRef.current.get(cacheKey)

    const render = (result) => {
      rendererRef.current.setDirections(result)
      const legs = result.routes?.[0]?.legs ?? []
      const totalSec = legs.reduce((t, l) => t + (l.duration?.value ?? 0), 0)
      setSummary({ totalMin: Math.round(totalSec / 60), legs: legs.length })
      setError(null)
    }

    if (cached) {
      render(cached)
      return
    }

    let cancelled = false
    serviceRef.current.route(
      {
        origin: request.origin,
        destination: request.destination,
        waypoints: request.waypoints,
        travelMode: window.google.maps.TravelMode[mode],
        optimizeWaypoints: false, // itinerary order is intentional
      },
      (result, status) => {
        if (cancelled) return
        if (status === 'OK' && result) {
          cacheRef.current.set(cacheKey, result)
          render(result)
        } else {
          setError(status === 'ZERO_RESULTS' ? 'No route for this mode' : 'Route unavailable')
          setSummary(null)
        }
      },
    )
    return () => {
      cancelled = true
    }
  }, [map, request, sig, mode])

  // Clean the rendered route off the map on unmount.
  useEffect(() => {
    return () => {
      if (rendererRef.current) rendererRef.current.setMap(null)
    }
  }, [])

  return (
    <div style={s.panel}>
      <div style={s.toggle}>
        {MODES.map((m) => (
          <button
            key={m.id}
            style={{ ...s.btn, ...(mode === m.id ? s.btnActive : {}) }}
            onClick={() => setMode(m.id)}
            aria-pressed={mode === m.id}
          >
            {m.icon} {m.label}
          </button>
        ))}
      </div>
      {error ? (
        <span style={s.err}>{error}</span>
      ) : summary ? (
        <span style={s.total}>
          {summary.totalMin} min · {summary.legs} leg{summary.legs === 1 ? '' : 's'}
        </span>
      ) : (
        <span style={{ color: 'var(--warm-gray)' }}>Routing…</span>
      )}
    </div>
  )
}
