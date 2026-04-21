import React, { useState } from 'react'
import MapView from './MapView'
import ListView from './ListView'

const FILTER_OPTIONS = ['All', 'Bars', 'Dinner', 'Open now']

const s = {
  panel: {
    flex: 1, display: 'flex', flexDirection: 'column',
    background: 'var(--cream)', overflow: 'hidden',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '0 28px', height: '56px',
    borderBottom: '1px solid var(--border)',
    background: 'var(--cream)',
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: '16px' },
  toggle: {
    display: 'flex', gap: '2px',
    background: 'var(--cream-dark)',
    borderRadius: '8px', padding: '3px',
  },
  toggleBtn: {
    display: 'flex', alignItems: 'center', gap: '6px',
    padding: '6px 14px', borderRadius: '6px', border: 'none',
    fontFamily: 'var(--font-body)', fontSize: '12px', fontWeight: 500,
    cursor: 'pointer', transition: 'all .2s',
    color: 'var(--warm-gray)', background: 'transparent',
  },
  toggleBtnActive: {
    background: 'var(--white)', color: 'var(--charcoal)',
    boxShadow: '0 1px 3px rgba(0,0,0,.08)',
  },
  meta: {
    fontSize: '12px', color: 'var(--warm-gray)',
    display: 'flex', alignItems: 'center', gap: '16px',
  },
  metaCount: { fontWeight: 500, color: 'var(--charcoal)' },
  filters: { display: 'flex', gap: '6px', alignItems: 'center' },
  filterPill: {
    fontSize: '11px', padding: '4px 10px', borderRadius: '20px',
    border: '1px solid var(--border)', background: 'var(--white)',
    cursor: 'pointer', color: 'var(--warm-gray)',
    transition: 'all .2s', fontFamily: 'var(--font-body)',
  },
  filterPillActive: {
    background: 'var(--charcoal)', color: 'var(--white)',
    borderColor: 'var(--charcoal)',
  },
  footer: {
    padding: '10px 28px',
    borderTop: '1px solid var(--border)',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    fontSize: '11px', color: 'var(--warm-gray)',
    background: 'var(--cream)',
  },
  footerLeft: { display: 'flex', alignItems: 'center', gap: '5px' },
  powered: { fontSize: '10px' },
  poweredStrong: { color: 'var(--moss)' },
}

function MapIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2">
      <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/>
      <line x1="8" y1="2" x2="8" y2="18"/>
      <line x1="16" y1="6" x2="16" y2="22"/>
    </svg>
  )
}
function ListIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2">
      <line x1="8" y1="6" x2="21" y2="6"/>
      <line x1="8" y1="12" x2="21" y2="12"/>
      <line x1="8" y1="18" x2="21" y2="18"/>
      <line x1="3" y1="6" x2="3.01" y2="6"/>
      <line x1="3" y1="12" x2="3.01" y2="12"/>
      <line x1="3" y1="18" x2="3.01" y2="18"/>
    </svg>
  )
}
function ClockIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  )
}

export default function RightPanel({ places = [], onPlaceClick, lastRefreshed = '4 min ago' }) {
  const [view, setView] = useState('map')       // 'map' | 'list'
  const [activeFilter, setActiveFilter] = useState('All')

  // Filter logic
  const filteredPlaces = places.filter(p => {
    if (activeFilter === 'All') return true
    if (activeFilter === 'Bars') return p.category === 'bar'
    if (activeFilter === 'Dinner') return p.category === 'dinner'
    if (activeFilter === 'Open now') return p.status !== 'closed'
    return true
  })

  return (
    <div style={s.panel}>
      {/* Header */}
      <div style={s.header}>
        <div style={s.headerLeft}>
          {/* View toggle */}
          <div style={s.toggle}>
            {[
              { id: 'map', label: 'Map', Icon: MapIcon },
              { id: 'list', label: 'List', Icon: ListIcon },
            ].map(({ id, label, Icon }) => (
              <button
                key={id}
                style={{ ...s.toggleBtn, ...(view === id ? s.toggleBtnActive : {}) }}
                onClick={() => setView(id)}
              >
                <Icon />
                {label}
              </button>
            ))}
          </div>

          {/* Result count + context */}
          <div style={s.meta}>
            <span style={s.metaCount}>{filteredPlaces.length} places</span>
            <span>Mission District · Tonight</span>
          </div>
        </div>

        {/* Filters */}
        <div style={s.filters}>
          {FILTER_OPTIONS.map(f => (
            <FilterPill
              key={f}
              label={f}
              active={activeFilter === f}
              onClick={() => setActiveFilter(f)}
            />
          ))}
        </div>
      </div>

      {/* Map / List */}
      {view === 'map'
        ? <MapView places={filteredPlaces} onPinClick={onPlaceClick} />
        : <ListView places={filteredPlaces} onPlaceClick={onPlaceClick} />
      }

      {/* Footer */}
      <div style={s.footer}>
        <div style={s.footerLeft}>
          <ClockIcon />
          Last refreshed {lastRefreshed}
        </div>
        <div style={s.powered}>
          RAG pipeline ·{' '}
          <strong style={s.poweredStrong}>FastAPI</strong>
          {' '}+ Google Places API +{' '}
          <strong style={s.poweredStrong}>Claude</strong>
        </div>
      </div>
    </div>
  )
}

function FilterPill({ label, active, onClick }) {
  const [hovered, setHovered] = useState(false)
  const isActive = active

  return (
    <button
      style={{
        ...s.filterPill,
        ...(isActive ? s.filterPillActive : {}),
        ...(!isActive && hovered
          ? { borderColor: 'var(--moss)', color: 'var(--moss)' }
          : {}),
      }}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {label}
    </button>
  )
}
