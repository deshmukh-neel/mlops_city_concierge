import React from 'react'

const styles = {
  nav: {
    position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '0 32px', height: '56px',
    background: 'var(--cream)',
    borderBottom: '1px solid var(--border)',
  },
  logo: {
    fontFamily: 'var(--font-display)',
    fontSize: '20px', fontWeight: 600, letterSpacing: '-0.3px',
    color: 'var(--charcoal)',
    display: 'flex', alignItems: 'center', gap: '8px',
    textDecoration: 'none',
  },
  logoDot: {
    width: '6px', height: '6px', borderRadius: '50%',
    background: 'var(--rust)', flexShrink: 0,
  },
  links: {
    display: 'flex', alignItems: 'center', gap: '28px',
    fontSize: '13px', fontWeight: 400, color: 'var(--warm-gray)',
  },
  link: {
    textDecoration: 'none', color: 'inherit',
    transition: 'color .2s',
  },
  badge: {
    fontSize: '11px', padding: '4px 12px', borderRadius: '20px',
    background: 'var(--moss)', color: 'var(--white)',
    fontWeight: 500, letterSpacing: '0.3px',
  },
  status: {
    display: 'flex', alignItems: 'center', gap: '6px',
    fontSize: '12px', color: 'var(--moss-light)',
  },
  statusDot: {
    width: '6px', height: '6px', borderRadius: '50%',
    background: '#6CBF6C',
    animation: 'pulse 2s infinite',
  },
}

export default function TopNav({ indexedCount = 847 }) {
  return (
    <nav style={styles.nav}>
      <a href="/" style={styles.logo}>
        <div style={styles.logoDot} />
        CITY CONCIERGE
      </a>

      <div style={styles.links}>
        {['Explore', 'Neighborhoods', 'Events', 'About'].map(label => (
          <a key={label} href="#" style={styles.link}
            onMouseEnter={e => e.target.style.color = 'var(--charcoal)'}
            onMouseLeave={e => e.target.style.color = 'var(--warm-gray)'}
          >
            {label}
          </a>
        ))}
        <span style={styles.badge}>SF Concierge</span>
      </div>

      <div style={styles.status}>
        <div style={styles.statusDot} />
        Live · {indexedCount.toLocaleString()} places indexed
      </div>
    </nav>
  )
}
