import React, { useCallback, useReducer, useState } from 'react'
import TopNav from './components/TopNav'
import ChatPanel from './components/ChatPanel'
import RightPanel from './components/RightPanel'
import { sendMessage, INITIAL_MESSAGES, formatTime } from './api/chat'
import {
  emptyPlacesState,
  replacePlaces,
  clearPlaces,
  selectOrderedPlaces,
} from './state/places'
import { buildHistory, assistantContentFor } from './state/chatTurn'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

const styles = {
  app: {
    display: 'flex',
    height: '100vh',
    paddingTop: '56px',
    overflow: 'hidden',
  },
}

// Single source of truth for the itinerary place set (keyed; streaming-ready).
function placesReducer(state, action) {
  switch (action.type) {
    case 'replace':
      return replacePlaces(state, action.places)
    case 'clear':
      return clearPlaces()
    default:
      return state
  }
}

export default function App() {
  const [messages, setMessages] = useState(INITIAL_MESSAGES)
  const [placesState, dispatch] = useReducer(placesReducer, emptyPlacesState)
  const [isLoading, setIsLoading] = useState(false)
  const [focusId, setFocusId] = useState(null)

  const places = selectOrderedPlaces(placesState)

  // ── Send a user message ──────────────────────────────────────────────────
  const handleSend = useCallback(async (text) => {
    const userMsg = {
      id: Date.now(),
      role: 'user',
      content: text,
      time: formatTime(new Date()),
    }

    // Capture history from the state that ALREADY includes this user message,
    // via the functional updater — not a stale closure over `messages`. This
    // is what fixes the "previous answer shows up on the next send" desync:
    // the agent now always receives the current turn's full prior context.
    let history = []
    setMessages((prev) => {
      const next = [...prev, userMsg]
      history = buildHistory(next)
      return next
    })
    setIsLoading(true)

    try {
      if (USE_MOCK) {
        await new Promise((r) => setTimeout(r, 1200))
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            role: 'assistant',
            content:
              'Got it! In production this response comes from the FastAPI agent. ' +
              'The map and list on the right update with the live itinerary.',
            time: formatTime(new Date()),
          },
        ])
      } else {
        const data = await sendMessage(text, history)

        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            role: 'assistant',
            // Never render an empty bubble: an empty reply means the agent
            // failed to produce a plan (e.g. hit the step limit) — say so.
            content: assistantContentFor(data.reply),
            ragLabel: data.ragLabel,
            time: formatTime(new Date()),
          },
        ])
        // Always dispatch: a turn with places finalizes the plan; a turn
        // without keeps the prior set but un-finalizes (route hides).
        dispatch({ type: 'replace', places: data.places ?? [] })
      }
    } catch (err) {
      console.error('Chat error:', err)
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 2,
          role: 'assistant',
          content:
            'Something went wrong connecting to the server. Please try again.',
          time: formatTime(new Date()),
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }, [])

  // ── Clear chat ───────────────────────────────────────────────────────────
  const handleClear = useCallback(() => {
    setMessages(INITIAL_MESSAGES)
    dispatch({ type: 'clear' })
    setFocusId(null)
  }, [])

  // ── Select a place (chat pill or map pin): focus it on the map ───────────
  const handlePlaceClick = useCallback((placeId) => {
    // New object identity each click so the map re-focuses even if the same
    // pin is clicked twice in a row.
    setFocusId({ id: placeId })
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
          planFinalized={placesState.planFinalized}
          focusId={focusId?.id ?? null}
          onPlaceClick={handlePlaceClick}
          lastRefreshed="just now"
        />
      </div>
    </>
  )
}
