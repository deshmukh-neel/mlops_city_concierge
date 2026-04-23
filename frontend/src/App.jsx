import React, { useState, useCallback } from 'react'
import TopNav from './components/TopNav'
import ChatPanel from './components/ChatPanel'
import RightPanel from './components/RightPanel'
import { sendMessage, MOCK_PLACES, INITIAL_MESSAGES, formatTime } from './api/chat'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

const styles = {
  app: {
    display: 'flex',
    height: '100vh',
    paddingTop: '56px',
    overflow: 'hidden',
  },
}

export default function App() {
  const [messages, setMessages]   = useState(INITIAL_MESSAGES)
  const [places, setPlaces]       = useState(MOCK_PLACES)   // swap to [] when wired to API
  const [isLoading, setIsLoading] = useState(false)

  // ── Send a user message ──────────────────────────────────────────────────
  const handleSend = useCallback(async (text) => {
    const userMsg = {
      id: Date.now(),
      role: 'user',
      content: text,
      time: formatTime(new Date()),
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      if (USE_MOCK) {
        // Dev mode: echo a placeholder response
        await new Promise(r => setTimeout(r, 1200))
        const botMsg = {
          id: Date.now() + 1,
          role: 'assistant',
          content: "Got it! In production this response comes from your FastAPI + RAG pipeline. The map and list on the right will update with live Google Places data.",
          time: formatTime(new Date()),
        }
        setMessages(prev => [...prev, botMsg])
      } else {
        // Production: call FastAPI
        const history = messages.map(m => ({ role: m.role, content: typeof m.content === 'string' ? m.content : '' }))
        const data = await sendMessage(text, history)

        const botMsg = {
          id: Date.now() + 1,
          role: 'assistant',
          content: data.reply,
          ragLabel: data.ragLabel,
          time: formatTime(new Date()),
        }
        setMessages(prev => [...prev, botMsg])
        if (data.places?.length) setPlaces(data.places)
      }
    } catch (err) {
      console.error('Chat error:', err)
      setMessages(prev => [...prev, {
        id: Date.now() + 2,
        role: 'assistant',
        content: 'Something went wrong connecting to the server. Please try again.',
        time: formatTime(new Date()),
      }])
    } finally {
      setIsLoading(false)
    }
  }, [messages])

  // ── Clear chat ───────────────────────────────────────────────────────────
  const handleClear = useCallback(() => {
    setMessages(INITIAL_MESSAGES)
    setPlaces([])
  }, [])

  // ── Highlight place from chat pill or map pin click ──────────────────────
  const handlePlaceClick = useCallback((placeId) => {
    console.log('Place selected:', placeId)
    // TODO: pan map to pin, or highlight list card
  }, [])

  return (
    <>
      <TopNav indexedCount={847} />
      <div style={styles.app}>
        <ChatPanel
          messages={messages}
          onSend={handleSend}
          onClear={handleClear}
          onPlacePillClick={handlePlaceClick}
          isLoading={isLoading}
        />
        <RightPanel
          places={places}
          onPlaceClick={handlePlaceClick}
          lastRefreshed="4 min ago"
        />
      </div>
    </>
  )
}
