import React from 'react'

const s = {
  row: {
    display: 'flex', gap: '10px',
    animation: 'fadeUp .3s ease both',
  },
  rowUser: { flexDirection: 'row-reverse' },
  avatar: {
    width: '28px', height: '28px', borderRadius: '50%',
    flexShrink: 0, display: 'flex', alignItems: 'center',
    justifyContent: 'center', fontSize: '11px', fontWeight: 600,
    marginTop: '2px',
  },
  avatarBot:  {
    background: 'var(--sage)',
    color: 'var(--moss)',
    fontFamily: 'var(--font-display)',
    fontSize: '16px',
    lineHeight: 1,
    overflow: 'hidden',
  },
  avatarUser: { background: 'var(--charcoal)', color: 'var(--white)' },
  bridgeEmoji: {
    display: 'inline-block',
    fontSize: '18px',
    lineHeight: 1,
    transformOrigin: 'center bottom',
    animation: 'bridgeFloat 3.4s ease-in-out infinite',
  },
  bubbleWrap: { maxWidth: '320px' },
  ragBadge: {
    display: 'inline-flex', alignItems: 'center', gap: '5px',
    fontSize: '10px', padding: '3px 8px', borderRadius: '20px',
    background: 'var(--sage)', color: 'var(--moss)', fontWeight: 500,
    marginBottom: '6px',
  },
  bubble: {
    padding: '12px 15px', borderRadius: '12px',
    fontSize: '13.5px', lineHeight: 1.6, fontWeight: 300,
  },
  bubbleBot: {
    background: 'var(--cream)', color: 'var(--charcoal)',
    borderRadius: '4px 12px 12px 12px',
  },
  bubbleUser: {
    background: 'var(--charcoal)', color: 'var(--white)',
    borderRadius: '12px 4px 12px 12px',
  },
  time: {
    fontSize: '10px', color: 'var(--warm-gray)',
    marginTop: '4px', padding: '0 4px',
  },
  timeUser: { textAlign: 'right' },
  // Pill styles (for place references inside bot messages)
  pill: {
    display: 'inline-block', background: 'var(--white)',
    border: '1px solid var(--border)', borderRadius: '4px',
    padding: '2px 8px', fontSize: '12px', margin: '2px',
    cursor: 'pointer', color: 'var(--charcoal)',
    transition: 'all .15s',
  },
  pillNum: { color: 'var(--rust)', fontWeight: 600, marginRight: '4px' },
  typing: {
    display: 'flex', gap: '4px', alignItems: 'center', padding: '8px 0',
  },
  typingDot: {
    width: '6px', height: '6px', borderRadius: '50%',
    background: 'var(--moss-light)',
    animation: 'bounce 1.2s infinite',
  },
}

/** Renders either a typing indicator or a message bubble */
export default function ChatMessage({ message, onPlacePillClick }) {
  const isUser = message.role === 'user'
  const isTyping = message.role === 'typing'

  if (isTyping) {
    return (
      <div style={s.row}>
        <div style={{ ...s.avatar, ...s.avatarBot }}>
          <span style={s.bridgeEmoji} role="img" aria-label="Golden Gate Bridge">🌉</span>
        </div>
        <div>
          {message.ragLabel && (
            <div style={s.ragBadge}>
              <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--moss)', display: 'inline-block' }} />
              {message.ragLabel}
            </div>
          )}
          <div style={s.typing}>
            {[0, 200, 400].map((delay, i) => (
              <div key={i} style={{ ...s.typingDot, animationDelay: `${delay}ms` }} />
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ ...s.row, ...(isUser ? s.rowUser : {}) }}>
      <div style={{ ...s.avatar, ...(isUser ? s.avatarUser : s.avatarBot) }}>
        {isUser ? 'Y' : (
          <span style={s.bridgeEmoji} role="img" aria-label="Golden Gate Bridge">🌉</span>
        )}
      </div>
      <div style={isUser ? {} : s.bubbleWrap}>
        {/* RAG badge for bot messages */}
        {!isUser && message.ragLabel && (
          <div style={s.ragBadge}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--moss)', display: 'inline-block' }} />
            {message.ragLabel}
          </div>
        )}

        <div style={{ ...s.bubble, ...(isUser ? s.bubbleUser : s.bubbleBot) }}>
          {/* Render content: plain string or array of nodes */}
          {renderContent(message.content, onPlacePillClick)}
        </div>

        <div style={{ ...s.time, ...(isUser ? s.timeUser : {}) }}>
          {message.time}
        </div>
      </div>
    </div>
  )
}

/** Renders message content — supports {type:'text'} and {type:'pill', label, id} nodes */
function renderContent(content, onPlacePillClick) {
  if (typeof content === 'string') {
    return <span dangerouslySetInnerHTML={{ __html: content }} />
  }
  if (Array.isArray(content)) {
    return content.map((node, i) => {
      if (node.type === 'text') {
        return <span key={i} dangerouslySetInnerHTML={{ __html: node.value }} />
      }
      if (node.type === 'pill') {
        return (
          <PlacePill
            key={i}
            label={node.label}
            num={node.num}
            onClick={() => onPlacePillClick?.(node.id)}
          />
        )
      }
      return null
    })
  }
  return null
}

function PlacePill({ label, num, onClick }) {
  const [hovered, setHovered] = React.useState(false)
  return (
    <span
      style={{
        ...s.pill,
        ...(hovered ? { background: 'var(--moss)', color: 'var(--white)', borderColor: 'var(--moss)' } : {}),
      }}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {num && <span style={s.pillNum}>{num}</span>}
      {label}
    </span>
  )
}
