import React, { useRef, useEffect, useState } from 'react'
import ChatMessage from './ChatMessage'

const SUGGESTION_CHIPS = [
  '🌉 Date night in the Mission',
  '☕ Best coffee, Hayes Valley',
  '🍜 Ramen spots open late',
  '🥾 Hike + lunch in Marin',
]

const s = {
  panel: {
    width: '480px', flexShrink: 0,
    display: 'flex', flexDirection: 'column',
    background: 'var(--white)',
    borderRight: '1px solid var(--border)',
  },
  header: {
    padding: '20px 24px 16px',
    borderBottom: '1px solid var(--border)',
  },
  headerTop: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
  },
  title: {
    fontFamily: 'var(--font-display)',
    fontSize: '18px', fontWeight: 600, lineHeight: 1.2,
    color: 'var(--charcoal)',
  },
  subtitle: {
    fontSize: '12px', color: 'var(--warm-gray)',
    marginTop: '3px', fontWeight: 300,
  },
  clearBtn: {
    fontSize: '11px', color: 'var(--warm-gray)',
    background: 'none', border: '1px solid var(--border)',
    padding: '4px 10px', borderRadius: '4px',
    cursor: 'pointer', transition: 'all .2s',
    fontFamily: 'var(--font-body)',
  },
  chips: {
    display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '12px',
  },
  chip: {
    fontSize: '11px', padding: '5px 11px', borderRadius: '20px',
    border: '1px solid var(--border)', background: 'var(--cream)',
    cursor: 'pointer', color: 'var(--warm-gray)',
    transition: 'all .2s', whiteSpace: 'nowrap',
    fontFamily: 'var(--font-body)',
  },
  messages: {
    flex: 1, overflowY: 'auto',
    padding: '20px 24px',
    display: 'flex', flexDirection: 'column', gap: '18px',
    scrollBehavior: 'smooth',
  },
  inputArea: {
    padding: '16px 24px',
    borderTop: '1px solid var(--border)',
    background: 'var(--white)',
  },
  inputRow: {
    display: 'flex', gap: '8px', alignItems: 'flex-end',
  },
  textarea: {
    flex: 1, border: '1px solid var(--border)', borderRadius: '10px',
    padding: '10px 14px', fontFamily: 'var(--font-body)',
    fontSize: '13.5px', color: 'var(--charcoal)', background: 'var(--cream)',
    resize: 'none', outline: 'none', minHeight: '42px', maxHeight: '100px',
    lineHeight: 1.5, transition: 'border-color .2s',
  },
  sendBtn: {
    width: '42px', height: '42px', borderRadius: '10px', border: 'none',
    background: 'var(--moss)', cursor: 'pointer',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    flexShrink: 0, transition: 'background .2s', color: 'white',
  },
  inputMeta: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    marginTop: '8px', fontSize: '11px', color: 'var(--warm-gray)',
  },
  ragIndicator: {
    display: 'flex', alignItems: 'center', gap: '5px',
    fontSize: '11px', color: 'var(--moss-light)',
  },
  ragDot: {
    width: '5px', height: '5px', borderRadius: '50%',
    background: 'var(--moss-light)', animation: 'pulse 2s infinite',
  },
}

export default function ChatPanel({ messages, onSend, onClear, onPlacePillClick, isLoading }) {
  const [input, setInput] = useState('')
  const [sendHovered, setSendHovered] = useState(false)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleSend() {
    const text = input.trim()
    if (!text || isLoading) return
    onSend(text)
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  function handleInput(e) {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px'
  }

  function handleChipClick(chip) {
    // Strip leading emoji + space
    const text = chip.replace(/^[\p{Emoji}\s]+/u, '').trim()
    setInput(text)
    textareaRef.current?.focus()
  }

  return (
    <div style={s.panel}>
      {/* Header */}
      <div style={s.header}>
        <div style={s.headerTop}>
          <div>
            <div style={s.title}>Your SF Guide</div>
            <div style={s.subtitle}>Powered by live Google Places data</div>
          </div>
          <button
            style={s.clearBtn}
            onClick={onClear}
            onMouseEnter={e => Object.assign(e.target.style, { background: 'var(--cream-dark)', color: 'var(--charcoal)' })}
            onMouseLeave={e => Object.assign(e.target.style, { background: 'none', color: 'var(--warm-gray)' })}
          >
            Clear chat
          </button>
        </div>
        <div style={s.chips}>
          {SUGGESTION_CHIPS.map(chip => (
            <button
              key={chip}
              style={s.chip}
              onClick={() => handleChipClick(chip)}
              onMouseEnter={e => Object.assign(e.target.style, { background: 'var(--moss)', color: 'var(--white)', borderColor: 'var(--moss)' })}
              onMouseLeave={e => Object.assign(e.target.style, { background: 'var(--cream)', color: 'var(--warm-gray)', borderColor: 'var(--border)' })}
            >
              {chip}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div style={s.messages}>
        {messages.map((msg, i) => (
          <ChatMessage key={msg.id ?? i} message={msg} onPlacePillClick={onPlacePillClick} />
        ))}
        {isLoading && (
          <ChatMessage
            message={{ role: 'typing', ragLabel: 'Searching nearby · Mission District' }}
          />
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={s.inputArea}>
        <div style={s.inputRow}>
          <textarea
            ref={textareaRef}
            style={s.textarea}
            placeholder="Ask anything about SF — neighborhoods, food, events…"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            rows={1}
            onFocus={e => e.target.style.borderColor = 'var(--moss)'}
            onBlur={e => e.target.style.borderColor = 'var(--border)'}
          />
          <button
            style={{ ...s.sendBtn, background: sendHovered ? 'var(--charcoal)' : 'var(--moss)' }}
            onClick={handleSend}
            disabled={isLoading}
            onMouseEnter={() => setSendHovered(true)}
            onMouseLeave={() => setSendHovered(false)}
          >
            <SendIcon />
          </button>
        </div>
        <div style={s.inputMeta}>
          <div style={s.ragIndicator}>
            <div style={s.ragDot} />
            RAG pipeline active · Places updated 4 min ago
          </div>
          <span>⌘↵ to send</span>
        </div>
      </div>
    </div>
  )
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  )
}
