import React from 'react'

/**
 * Shared place summary used by both the map marker hover-card and (optionally)
 * list affordances. Single source of truth for "how a place is summarized"
 * (Code Quality #2 — DRY between map + list).
 */

const s = {
  card: {
    background: 'var(--white)',
    border: '1px solid var(--border)',
    borderRadius: '8px',
    padding: '10px 14px',
    minWidth: '170px',
    maxWidth: '240px',
    boxShadow: '0 4px 16px rgba(0,0,0,.12)',
    fontSize: '12px',
  },
  name: { fontWeight: 600, color: 'var(--charcoal)', marginBottom: '2px' },
  type: { color: 'var(--warm-gray)', fontSize: '11px' },
  meta: { color: 'var(--rust)', fontSize: '11px', marginTop: '4px' },
  addr: { color: 'var(--warm-gray)', fontSize: '11px', marginTop: '4px', lineHeight: 1.4 },
}

export default function PlaceTooltip({ place }) {
  if (!place) return null
  const ratingKnown = place.rating && place.rating !== '—'
  return (
    <div style={s.card} role="tooltip">
      <div style={s.name}>
        {place.num ? `${place.num}. ` : ''}
        {place.name}
      </div>
      {place.type ? <div style={s.type}>{place.type}</div> : null}
      <div style={s.meta}>
        {ratingKnown ? `★ ${place.rating}` : 'Rating n/a'}
        {place.hours ? ` · ${place.hours}` : ''}
      </div>
      {place.address ? <div style={s.addr}>{place.address}</div> : null}
    </div>
  )
}
