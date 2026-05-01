import React, { useState } from 'react'

const s = {
  card: {
    background: 'var(--white)', borderRadius: '10px',
    border: '1px solid var(--border)',
    padding: '16px 18px',
    display: 'flex', gap: '16px', alignItems: 'flex-start',
    cursor: 'pointer', transition: 'all .2s',
    position: 'relative', overflow: 'hidden',
  },
  cardActive: {
    borderColor: 'var(--moss)',
    boxShadow: '0 4px 16px rgba(0,0,0,.07)',
    transform: 'translateY(-1px)',
  },
  accentBar: {
    position: 'absolute', left: 0, top: 0, bottom: 0, width: '3px',
    background: 'var(--moss)', transition: 'opacity .2s',
  },
  num: {
    width: '26px', height: '26px', borderRadius: '50%',
    background: 'var(--moss)', color: 'var(--white)',
    fontSize: '12px', fontWeight: 600,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    flexShrink: 0,
  },
  numRust: { background: 'var(--rust)' },
  body: { flex: 1, minWidth: 0 },
  name: { fontSize: '14px', fontWeight: 500, color: 'var(--charcoal)', marginBottom: '2px' },
  type: {
    fontSize: '11px', color: 'var(--warm-gray)',
    textTransform: 'uppercase', letterSpacing: '0.4px', marginBottom: '6px',
  },
  desc: { fontSize: '12.5px', color: 'var(--warm-gray)', lineHeight: 1.5, fontWeight: 300 },
  tags: { display: 'flex', gap: '4px', flexWrap: 'wrap', marginTop: '8px' },
  tag: {
    fontSize: '10px', padding: '2px 8px', borderRadius: '20px',
    background: 'var(--cream)', border: '1px solid var(--border)',
    color: 'var(--warm-gray)',
  },
  tagHighlight: {
    background: 'var(--sage)', borderColor: 'var(--sage)', color: 'var(--moss)',
  },
  meta: {
    display: 'flex', flexDirection: 'column',
    alignItems: 'flex-end', gap: '6px', flexShrink: 0,
  },
  rating: {
    fontSize: '12px', color: 'var(--rust)', fontWeight: 600,
    display: 'flex', alignItems: 'center', gap: '3px',
  },
  distance: { fontSize: '11px', color: 'var(--warm-gray)' },
  statusOpen: {
    fontSize: '10px', padding: '2px 7px', borderRadius: '3px',
    background: '#E8F5E8', color: '#3D7A3D',
  },
  statusBusy: {
    fontSize: '10px', padding: '2px 7px', borderRadius: '3px',
    background: '#FFF3E0', color: '#B07A30',
  },
}

/**
 * @param {Object} props
 * @param {{ id, num, name, type, desc, tags, rating, distance, status, category }} props.place
 * @param {boolean} props.featured
 * @param {function} props.onClick
 */
export default function PlaceCard({ place, featured = false, onClick }) {
  const [hovered, setHovered] = useState(false)
  const isHighlighted = featured || hovered

  return (
    <div
      style={{ ...s.card, ...(isHighlighted ? s.cardActive : {}) }}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Left accent bar */}
      <div style={{ ...s.accentBar, opacity: isHighlighted ? 1 : 0 }} />

      {/* Number badge */}
      <div style={{ ...s.num, ...(place.category === 'dinner' ? s.numRust : {}) }}>
        {place.num}
      </div>

      {/* Body */}
      <div style={s.body}>
        <div style={s.name}>{place.name}</div>
        <div style={s.type}>{place.type}</div>
        <div style={s.desc}>{place.desc}</div>
        <div style={s.tags}>
          {place.tags?.map((tag, i) => (
            <span key={i} style={{ ...s.tag, ...(tag.highlight ? s.tagHighlight : {}) }}>
              {tag.label}
            </span>
          ))}
        </div>
      </div>

      {/* Meta */}
      <div style={s.meta}>
        <div style={s.rating}>★ {place.rating}</div>
        <div style={s.distance}>{place.distance}</div>
        <div style={place.status === 'busy' ? s.statusBusy : s.statusOpen}>
          {place.status === 'busy' ? `Busy · ${place.hours}` : `Open · ${place.hours}`}
        </div>
      </div>
    </div>
  )
}
