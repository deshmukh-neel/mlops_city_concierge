import React, { useState } from 'react'

const s = {
  container: { flex: 1, position: 'relative', overflow: 'hidden' },
  mapBg: {
    width: '100%', height: '100%',
    backgroundColor: '#E8E4DA',
    position: 'relative', overflow: 'hidden',
  },
  grid: {
    position: 'absolute', inset: 0,
    backgroundImage: `
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px)
    `,
    backgroundSize: '60px 60px',
    opacity: 0.6,
  },
  neighborhoodLabel: {
    position: 'absolute',
    fontFamily: 'var(--font-display)',
    fontStyle: 'italic',
    fontSize: '14px',
    color: 'rgba(74,92,63,0.35)',
    letterSpacing: '0.5px',
    pointerEvents: 'none',
    userSelect: 'none',
  },
  streetLine: {
    position: 'absolute',
    left: '5%', right: '5%',
    height: '0',
  },
  streetLabel: {
    position: 'absolute',
    fontSize: '9px', color: '#9A9488',
    fontWeight: 500, letterSpacing: '0.5px',
    textTransform: 'uppercase',
  },
  block: {
    position: 'absolute',
    background: '#D8D2C8', borderRadius: '2px',
  },
  pin: {
    position: 'absolute',
    transform: 'translate(-50%, -100%)',
    cursor: 'pointer',
    filter: 'drop-shadow(0 3px 6px rgba(0,0,0,.25))',
    transition: 'transform .2s',
  },
  pinHovered: { transform: 'translate(-50%, -100%) scale(1.15)' },
  pinBody: {
    width: '32px', height: '32px',
    borderRadius: '50% 50% 50% 0',
    transform: 'rotate(-45deg)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: 'white', fontWeight: 700, fontSize: '12px',
  },
  pinNum: { transform: 'rotate(45deg)', display: 'inline-block' },
  tooltip: {
    position: 'absolute', bottom: '44px', left: '50%',
    transform: 'translateX(-50%)',
    background: 'var(--white)', border: '1px solid var(--border)',
    borderRadius: '8px', padding: '10px 14px',
    whiteSpace: 'nowrap', minWidth: '160px',
    boxShadow: '0 4px 16px rgba(0,0,0,.12)',
    pointerEvents: 'none',
    fontSize: '12px', zIndex: 10,
  },
  ttName: { fontWeight: 500, color: 'var(--charcoal)', marginBottom: '2px' },
  ttType: { color: 'var(--warm-gray)', fontSize: '11px' },
  ttRating: { color: 'var(--rust)', fontSize: '11px', marginTop: '2px' },
  legend: {
    position: 'absolute', bottom: '20px', left: '20px',
    background: 'var(--white)', border: '1px solid var(--border)',
    borderRadius: '8px', padding: '12px 16px',
    fontSize: '11px', color: 'var(--charcoal)',
  },
  legendTitle: {
    fontWeight: 600, marginBottom: '8px',
    fontSize: '10px', letterSpacing: '0.5px',
    textTransform: 'uppercase', color: 'var(--warm-gray)',
  },
  legendItem: { display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '5px' },
  legendDot: { width: '10px', height: '10px', borderRadius: '50%' },
}

const STREET_LINES = [
  { top: '33%', label: 'Mission St', labelLeft: '42%', thickness: '2px' },
  { top: '50%', label: 'Valencia St', labelLeft: '44%', thickness: '1.5px' },
  { top: '22%', label: 'Cesar Chavez St', labelLeft: '40%', thickness: '1px' },
  { top: '66%', label: '24th St', labelLeft: '48%', thickness: '1px' },
]

const BLOCKS = [
  { width: '80px', height: '50px', top: '36%', left: '18%' },
  { width: '60px', height: '40px', top: '36%', left: '32%' },
  { width: '100px', height: '45px', top: '55%', left: '40%' },
  { width: '70px', height: '55px', top: '38%', left: '58%' },
  { width: '90px', height: '40px', top: '25%', left: '48%' },
  { width: '50px', height: '60px', top: '60%', left: '22%' },
  { width: '110px', height: '35px', top: '44%', left: '68%' },
]

const NEIGHBORHOODS = [
  { label: 'The Mission', top: '38%', left: '38%' },
  { label: 'Castro', top: '18%', left: '12%' },
  { label: 'Bernal Heights', top: '60%', left: '62%' },
]

function MapPin({ place, onPinClick }) {
  const [hovered, setHovered] = useState(false)
  const color = place.category === 'dinner' ? 'var(--rust)' : 'var(--moss)'

  return (
    <div
      style={{
        ...s.pin,
        left: place.mapPos.x, top: place.mapPos.y,
        ...(hovered ? s.pinHovered : {}),
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onPinClick?.(place.id)}
    >
      {hovered && (
        <div style={s.tooltip}>
          <div style={s.ttName}>{place.name}</div>
          <div style={s.ttType}>{place.type}</div>
          <div style={s.ttRating}>★ {place.rating} · {place.status === 'busy' ? 'Busy' : 'Open'} {place.hours}</div>
        </div>
      )}
      <div style={{ ...s.pinBody, background: color }}>
        <span style={s.pinNum}>{place.num}</span>
      </div>
    </div>
  )
}

export default function MapView({ places = [], onPinClick }) {
  const bars = places.filter(p => p.category === 'bar')
  const dinner = places.filter(p => p.category === 'dinner')

  return (
    <div style={s.container}>
      <div style={s.mapBg}>
        {/* Grid */}
        <div style={s.grid} />

        {/* Neighborhood labels */}
        {NEIGHBORHOODS.map(n => (
          <div key={n.label} style={{ ...s.neighborhoodLabel, top: n.top, left: n.left }}>
            {n.label}
          </div>
        ))}

        {/* Streets */}
        {STREET_LINES.map(st => (
          <div key={st.label} style={{ ...s.streetLine, top: st.top, borderTop: `${st.thickness} solid #C8C0B4` }}>
            <span style={{ ...s.streetLabel, top: '4px', left: st.labelLeft }}>{st.label}</span>
          </div>
        ))}

        {/* Blocks */}
        {BLOCKS.map((b, i) => (
          <div key={i} style={{ ...s.block, ...b }} />
        ))}

        {/* Pins */}
        {places.map(place => (
          <MapPin key={place.id} place={place} onPinClick={onPinClick} />
        ))}

        {/* Legend */}
        <div style={s.legend}>
          <div style={s.legendTitle}>Tonight's Route</div>
          <div style={s.legendItem}>
            <div style={{ ...s.legendDot, background: 'var(--moss)' }} />
            Bars ({bars.length})
          </div>
          <div style={s.legendItem}>
            <div style={{ ...s.legendDot, background: 'var(--rust)' }} />
            Dinner ({dinner.length})
          </div>
        </div>
      </div>
    </div>
  )
}
