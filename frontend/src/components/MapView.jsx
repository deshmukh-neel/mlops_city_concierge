import React, { useEffect, useMemo, useState } from 'react'
import {
  APIProvider,
  Map,
  AdvancedMarker,
  Pin,
  useMap,
} from '@vis.gl/react-google-maps'
import PlaceTooltip from './PlaceTooltip'
import RouteOverlay from './RouteOverlay'

const SF_CENTER = { lat: 37.7749, lng: -122.4194 }

// Read env at render (not module scope) so the key can't be baked in before
// tests set it, and so a late-injected key is picked up.
const mapsApiKey = () => import.meta.env.VITE_GOOGLE_MAPS_API_KEY
const mapsMapId = () => import.meta.env.VITE_GOOGLE_MAPS_MAP_ID || 'DEMO_MAP_ID'

const s = {
  container: { flex: 1, position: 'relative', overflow: 'hidden' },
  fallback: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
    background: '#E8E4DA',
    color: 'var(--warm-gray)',
    fontSize: '13px',
    textAlign: 'center',
    padding: '40px',
  },
  fallbackIcon: { fontSize: '34px', opacity: 0.4 },
  fallbackCode: {
    fontFamily: 'monospace',
    fontSize: '12px',
    background: 'var(--white)',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    padding: '6px 10px',
    color: 'var(--charcoal)',
  },
  tooltipWrap: {
    position: 'absolute',
    transform: 'translate(-50%, -120%)',
    pointerEvents: 'none',
    zIndex: 20,
  },
}

/** Pan/zoom the map when a place is selected from the list or a pin. */
function FocusController({ focusId, places }) {
  const map = useMap()
  useEffect(() => {
    if (!map || !focusId) return
    const p = places.find((x) => x.id === focusId)
    if (!p || typeof p.latitude !== 'number') return
    map.panTo({ lat: p.latitude, lng: p.longitude })
    if ((map.getZoom() ?? 0) < 15) map.setZoom(15)
  }, [map, focusId, places])
  return null
}

/** Fit the viewport to all routable markers whenever the set changes. */
function FitBounds({ places }) {
  const map = useMap()
  useEffect(() => {
    if (!map || places.length === 0 || !window.google?.maps) return
    if (places.length === 1) {
      map.setCenter({ lat: places[0].latitude, lng: places[0].longitude })
      map.setZoom(15)
      return
    }
    const bounds = new window.google.maps.LatLngBounds()
    places.forEach((p) => bounds.extend({ lat: p.latitude, lng: p.longitude }))
    map.fitBounds(bounds, 64)
  }, [map, places])
  return null
}

function GoogleMapInner({ places, routable, onPinClick, planFinalized, focusId }) {
  const [hoverId, setHoverId] = useState(null)

  // Memoized so unchanged sets don't re-create markers (Performance #4).
  const markers = useMemo(
    () =>
      routable.map((p) => (
        <AdvancedMarker
          key={p.id}
          position={{ lat: p.latitude, lng: p.longitude }}
          title={p.name}
          onClick={() => onPinClick?.(p.id)}
          onMouseEnter={() => setHoverId(p.id)}
          onMouseLeave={() => setHoverId(null)}
        >
          <Pin
            background="var(--rust, #B5532F)"
            borderColor="#7A3620"
            glyphColor="#fff"
          >
            {p.num}
          </Pin>
        </AdvancedMarker>
      )),
    [routable, onPinClick],
  )

  const hovered = hoverId ? routable.find((p) => p.id === hoverId) : null

  return (
    <Map
      mapId={mapsMapId()}
      defaultCenter={SF_CENTER}
      defaultZoom={13}
      gestureHandling="greedy"
      disableDefaultUI={false}
      style={{ width: '100%', height: '100%' }}
    >
      {markers}
      <FitBounds places={routable} />
      <FocusController focusId={focusId} places={routable} />
      {planFinalized && routable.length >= 2 ? (
        <RouteOverlay stops={routable} />
      ) : null}
      {hovered ? (
        <AdvancedMarker
          position={{ lat: hovered.latitude, lng: hovered.longitude }}
          clickable={false}
        >
          <div style={s.tooltipWrap}>
            <PlaceTooltip place={hovered} />
          </div>
        </AdvancedMarker>
      ) : null}
    </Map>
  )
}

/**
 * Real Google Map. Degrades to an explicit panel when no API key is set so
 * frontend-only contributors aren't blocked and the SDK is never fetched
 * (Performance #3 — this component is lazy-loaded by RightPanel).
 */
export default function MapView({
  places = [],
  onPinClick,
  planFinalized = false,
  focusId = null,
}) {
  const routable = useMemo(
    () =>
      places.filter(
        (p) => typeof p.latitude === 'number' && typeof p.longitude === 'number',
      ),
    [places],
  )

  const apiKey = mapsApiKey()
  if (!apiKey) {
    return (
      <div style={s.fallback}>
        <div style={s.fallbackIcon}>🗺️</div>
        <div>Map unavailable — no Google Maps key configured.</div>
        <div style={s.fallbackCode}>VITE_GOOGLE_MAPS_API_KEY</div>
        <div>Set it in <code>frontend/.env.development.local</code> to see the live map.</div>
      </div>
    )
  }

  return (
    <div style={s.container}>
      <APIProvider apiKey={apiKey}>
        <GoogleMapInner
          places={places}
          routable={routable}
          onPinClick={onPinClick}
          planFinalized={planFinalized}
          focusId={focusId}
        />
      </APIProvider>
    </div>
  )
}
