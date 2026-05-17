/**
 * chatTurn.js — pure helpers for a single chat turn.
 *
 * Extracted from App.jsx so the two defects that caused
 * "query shows empty, then the previous answer appears on the next send"
 * are unit-testable in isolation:
 *
 *  1. History must be built from the messages list that ALREADY includes the
 *     just-sent user message, not a stale useCallback closure. App.jsx now
 *     computes history inside the functional setState updater and passes it
 *     here, so this helper just normalizes shape.
 *
 *  2. An empty agent reply (e.g. the agent hit the planning step limit and
 *     returned stops=0 + empty final_reply) must NOT render as a blank
 *     assistant bubble. assistantContentFor() guarantees non-empty content.
 */

const EMPTY_REPLY_FALLBACK =
  "I couldn't put a full plan together for that one — try narrowing it " +
  '(a neighborhood, a vibe, or fewer stops) and I’ll give it another go.'

/** Map UI messages to the {role, content} wire shape, coercing non-strings. */
export function buildHistory(messages) {
  if (!Array.isArray(messages)) return []
  return messages.map((m) => ({
    role: m.role,
    content: typeof m.content === 'string' ? m.content : '',
  }))
}

/**
 * Decide the assistant bubble content. Never returns empty/whitespace —
 * an empty reply means the agent failed to produce a plan; say so plainly
 * instead of showing a blank message (the reported bug).
 */
export function assistantContentFor(reply) {
  if (typeof reply === 'string' && reply.trim().length > 0) return reply
  return EMPTY_REPLY_FALLBACK
}
