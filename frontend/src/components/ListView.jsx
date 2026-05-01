import React from 'react'
import PlaceCard from './PlaceCard'

const s = {
  list: {
    flex: 1, overflowY: 'auto',
    padding: '20px 28px',
    display: 'flex', flexDirection: 'column', gap: '12px',
  },
  sectionLabel: {
    fontSize: '10px', fontWeight: 500,
    textTransform: 'uppercase', letterSpacing: '0.8px',
    color: 'var(--warm-gray)',
    padding: '4px 0 2px',
    display: 'flex', alignItems: 'center', gap: '8px',
  },
  sectionLine: {
    flex: 1, height: '1px', background: 'var(--border)',
  },
  empty: {
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    flex: 1, gap: '8px',
    color: 'var(--warm-gray)', fontSize: '13px', fontWeight: 300,
    paddingTop: '60px',
  },
  emptyIcon: { fontSize: '32px', opacity: 0.4 },
}

export default function ListView({ places = [], onPlaceClick }) {
  if (places.length === 0) {
    return (
      <div style={s.list}>
        <div style={s.empty}>
          <div style={s.emptyIcon}>🗺️</div>
          <span>Ask the concierge to find somewhere</span>
        </div>
      </div>
    )
  }

  const dinnerPlaces = places.filter(p => p.category === 'dinner')
  const barPlaces    = places.filter(p => p.category === 'bar')
  const otherPlaces  = places.filter(p => !['dinner', 'bar'].includes(p.category))

  return (
    <div style={s.list}>
      {dinnerPlaces.length > 0 && (
        <>
          <SectionLabel>Dinner · Start here</SectionLabel>
          {dinnerPlaces.map(place => (
            <PlaceCard
              key={place.id}
              place={place}
              featured={place.featured}
              onClick={() => onPlaceClick?.(place.id)}
            />
          ))}
        </>
      )}

      {barPlaces.length > 0 && (
        <>
          <SectionLabel>The Bar Crawl</SectionLabel>
          {barPlaces.map(place => (
            <PlaceCard
              key={place.id}
              place={place}
              featured={place.featured}
              onClick={() => onPlaceClick?.(place.id)}
            />
          ))}
        </>
      )}

      {otherPlaces.length > 0 && (
        <>
          <SectionLabel>Other Spots</SectionLabel>
          {otherPlaces.map(place => (
            <PlaceCard
              key={place.id}
              place={place}
              onClick={() => onPlaceClick?.(place.id)}
            />
          ))}
        </>
      )}
    </div>
  )
}

function SectionLabel({ children }) {
  return (
    <div style={s.sectionLabel}>
      {children}
      <div style={s.sectionLine} />
    </div>
  )
}
